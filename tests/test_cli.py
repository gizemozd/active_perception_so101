import json
import os
import subprocess
from pathlib import Path

import pytest

from active_perception_arms.config import CONDITIONS, TASKS, Experiment
from active_perception_arms.train import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_dry_run_all_tasks_conditions(task, condition, capsys):
    main(["--task", task, "--condition", condition, "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert not report["training_started"]
    assert report["experiment"]["task"] == task


def test_invalid_experiments():
    for params in (
        {"task": "bad"},
        {"num_envs": 0},
        {"clearance": 0},
        {"width": 16},
        {"episode_seconds": 1},
        {"task": "plug", "perturb_push": True},
    ):
        with pytest.raises(ValueError):
            Experiment(**params)


@pytest.mark.parametrize("script", sorted((ROOT / "scripts/slurm").glob("*.sbatch")))
def test_sbatch_shell_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_sbatch_training_dispatch_without_training():
    for idx, condition in (
        (0, "wrist"),
        (5, "static"),
        (8, "wrist_static"),
        (11, "initial"),
        (14, "scheduled"),
        (17, "active"),
    ):
        proc = subprocess.run(
            ["bash", "scripts/slurm/train_plug.sbatch"],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "SLURM_ARRAY_TASK_ID": str(idx),
                "PROJECT_ROOT": str(ROOT),
                "FIXED_VIEW": "2",
            },
            text=True,
            capture_output=True,
            check=True,
        )
        report = json.loads(proc.stdout[proc.stdout.index("{") :])
        assert not report["training_started"]
        assert report["experiment"]["condition"] == condition


def test_wilson_interval():
    from active_perception_arms.evaluate import wilson

    assert wilson(0, 100)[0] == 0
    assert wilson(100, 100)[1] == 1
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi


def test_static_search_dispatch_and_equal_transition_budget():
    proc = subprocess.run(
        ["bash", "scripts/slurm/static_search.sbatch"],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "SLURM_ARRAY_TASK_ID": "77",
            "PROJECT_ROOT": str(ROOT),
            "TASK": "push",
            "NUM_ENVS": "512",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert report["experiment"]["fixed_position"] != [-0.119, -0.187, 0.274]
    assert report["experiment"]["seed"] == 2
    assert report["iterations"] * 24 * 512 == 18432000
    assert not report["training_started"]


def test_resume_rejects_changed_environment_count(tmp_path, capsys):
    # Changing N while retaining the checkpoint iteration silently changes the
    # cumulative transition axis and remaining scientific interaction budget.
    cfg = Experiment(num_envs=256)
    (tmp_path / "experiment.json").write_text(json.dumps(cfg.to_dict()))
    (tmp_path / "runner.json").write_text(
        json.dumps(
            {
                "algorithm": {
                    "learning_rate": 3e-4,
                    "schedule": "adaptive",
                    "entropy_coef": 0.003,
                },
                "actor": {"distribution_cfg": {"init_std": 0.4}},
            }
        )
    )
    checkpoint = tmp_path / "model_9.pt"
    with pytest.raises(ValueError, match="num_envs"):
        main(["--task", "plug", "--num-envs", "128", "--resume", str(checkpoint), "--dry-run"])
    main(["--task", "plug", "--num-envs", "256", "--resume", str(checkpoint), "--dry-run"])
    assert not json.loads(capsys.readouterr().out)["training_started"]


def test_search_optimization_dispatch_and_resume_guard(tmp_path):
    from active_perception_arms.train import optimization_from_args, parser

    args = parser().parse_args(
        [
            "--task",
            "plug",
            "--job-type",
            "search",
            "--learning-rate",
            "0.0001",
            "--lr-schedule",
            "fixed",
            "--entropy-coef",
            "0.01",
            "--run-label",
            "lr-entropy",
        ]
    )
    settings = optimization_from_args(args)
    assert settings == {"learning_rate": 0.0001, "schedule": "fixed", "entropy_coef": 0.01}
    (tmp_path / "runner.json").write_text(json.dumps({"algorithm": settings}))
    args.resume = tmp_path / "model_99.pt"
    args.learning_rate = args.lr_schedule = args.entropy_coef = None
    assert optimization_from_args(args) == settings
    args.entropy_coef = 0.003
    with pytest.raises(ValueError, match="Resume optimization differs"):
        optimization_from_args(args)


@pytest.mark.parametrize("flag,value", [("--learning-rate", "0"), ("--entropy-coef", "nan")])
def test_invalid_optimizer_settings(flag, value):
    from active_perception_arms.train import optimization_from_args, parser

    with pytest.raises(ValueError, match="Invalid"):
        optimization_from_args(parser().parse_args(["--task", "plug", flag, value]))
