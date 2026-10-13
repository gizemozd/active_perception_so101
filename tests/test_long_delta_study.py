"""Budgets, original-task evaluation and validation-only curriculum decisions."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from active_perception_arms.config import Experiment
from active_perception_arms.evaluate import task_overrides
from active_perception_arms.occlusion import with_occlusion
from active_perception_arms.train import experiment_from_args, parser

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "long_delta_study", ROOT / "scripts/run_long_delta_study.py"
)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


@pytest.mark.parametrize("task", ["plug", "push"])
def test_matched_delta_memory_recipes_and_rounded_budget(task, tmp_path):
    plan = dict(python="python", num_envs=256)
    configs = []
    for condition in ("wrist_static", "active"):
        case = dict(task=task, kind="direct", condition=condition, seed=0, name="test")
        cmd = study.training_command(plan, case, tmp_path, 20_000_000)
        cfg = experiment_from_args(parser().parse_args(cmd[3:]))
        configs.append(cfg.to_dict())
        assert cfg.memory == "gru" and cfg.translation_control == "delta"
        assert cfg.control_revision == "pose_v1" and cfg.camera_interval == 4
        assert cfg.camera_start == "inspection_v1" and cfg.translation_step == 0.005
        count = int(cmd[cmd.index("--iterations") + 1]) * 256 * 24
        assert 20_000_000 <= count < 20_000_000 + 256 * 24
        if task == "push":
            assert cfg.push_workspace_revision == "arc_clear_v2"
            assert cfg.push_target_layout == "arc_v2"
    assert {k for k in configs[0] if configs[0][k] != configs[1][k]} == {"condition"}


def test_curriculum_original_evaluation_restores_variant_horizon_jitter_and_sweep():
    cfg = Experiment(
        task="plug",
        plug_variant="ym",
        episode_seconds=8,
        reset_xy_jitter=0.002,
        occlusion="dynamic",
        control_revision="pose_v1",
        translation_control="delta",
    )
    args = SimpleNamespace(plug_variant="all", episode_seconds=3.5, reset_xy_jitter=0.008)
    restored = with_occlusion(cfg, "dynamic", **task_overrides(args, cfg))
    original = Experiment(task="plug", episode_seconds=3.5, occlusion="dynamic")
    assert restored.plug_variant is None and restored.reset_xy_jitter == 0.008
    assert restored.episode_seconds == 3.5
    assert restored.dynamic_onset_range == original.dynamic_onset_range
    assert restored.dynamic_duration_range == original.dynamic_duration_range
    assert restored.translation_control == "delta" and restored.control_revision == "pose_v1"


def test_variant_resume_inherits_and_rejects_stage_change(tmp_path):
    cfg = Experiment(task="plug", plug_variant="ym", episode_seconds=8)
    (tmp_path / "experiment.json").write_text(json.dumps(cfg.to_dict()))
    checkpoint = tmp_path / "model.pt"
    base = ["--task", "plug", "--resume", str(checkpoint)]
    assert experiment_from_args(parser().parse_args(base)).plug_variant == "ym"
    with pytest.raises(ValueError, match="plug_variant"):
        experiment_from_args(parser().parse_args(base + ["--plug-variant", "all"]))


def test_privileged_plug_diagnostic_keeps_delta_gru_and_excludes_images():
    from dataclasses import replace

    import torch

    from active_perception_arms.environment import make_env, make_env_cfg
    from active_perception_arms.policy import runner_cfg

    cfg = experiment_from_args(
        parser().parse_args(
            [
                "--task",
                "plug",
                "--actor-observation",
                "privileged",
                "--memory",
                "gru",
                "--control-revision",
                "pose_v1",
                "--translation-control",
                "delta",
            ]
        )
    )
    assert not cfg.render_sensors and cfg.sensors == ()
    assert cfg.plug_control == "delta5mm"
    options = runner_cfg(cfg)
    assert options.actor.class_name == "RNNModel"
    assert tuple(options.obs_groups["actor"]) == ("privileged",)
    env_cfg = make_env_cfg(cfg)
    assert "privileged" in env_cfg.observations
    assert "wrist" not in env_cfg.observations and "external" not in env_cfg.observations
    assert "tool_clear" not in env_cfg.metrics
    assert "plug_xy_error" in env_cfg.metrics
    env = make_env(replace(cfg, num_envs=8), "cpu")
    try:
        obs, _ = env.reset(seed=150000)
        assert "privileged" in obs
        _, reward, *_ = env.step(torch.zeros(8, cfg.action_dim))
        assert torch.isfinite(reward).all()
    finally:
        env.close()


def test_worker_resumes_and_selects_validation_not_test(tmp_path, monkeypatch):
    case = dict(
        name="plug_wrist_static_s0", task="plug", seed=0, condition="wrist_static", kind="direct"
    )
    plan = dict(
        root=str(ROOT),
        output=str(tmp_path),
        source=str(tmp_path),
        python="python",
        source_sha256={},
        num_envs=8,
        cases=[case],
        milestones=[384, 768],
        validation_seed=130000,
        test_seed=140000,
        validation_episodes=8,
        test_episodes=8,
        selection="Maximum mean clean/dynamic validation success, earliest tie",
    )
    study.save(tmp_path / "plan.json", plan)
    commands = []

    def execute(cmd, **kwargs):
        commands.append(cmd)

        def value(flag):
            return cmd[cmd.index(flag) + 1]

        if "active_perception_arms.train" in cmd:
            directory = Path(value("--log-root")) / "run"
            directory.mkdir(parents=True, exist_ok=True)
            index = int(value("--iterations")) - 1
            (directory / f"model_{index}.pt").write_bytes(b"checkpoint")
            study.save(Path(value("--result")), dict(actual_transitions=(index + 1) * 192))
        else:
            index = int(Path(cmd[3]).stem.split("_")[-1])
            successes = 2 if index == 1 else 1
            if value("--split") == "test":
                successes = 0
            study.save(
                Path(value("--output")),
                dict(
                    episodes=8,
                    successes=successes,
                    success_rate=successes / 8,
                    termination_audit=dict(
                        physical_success_violations=0,
                        records=[
                            dict(success=i < successes, qpos_position_error_m=0.02)
                            for i in range(8)
                        ],
                    ),
                ),
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(study.subprocess, "run", execute)
    args = SimpleNamespace(plan=tmp_path / "plan.json", index=0)
    study.worker(args)
    train = [c for c in commands if "active_perception_arms.train" in c]
    assert len(train) == 2 and "--resume" not in train[0] and "--resume" in train[1]
    output = tmp_path / case["name"]
    selected = json.loads((output / "selection.json").read_text())
    assert selected["checkpoint"].endswith("model_1.pt")
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == "complete" and state["transitions"] == 768
    assert len(state["evaluations"]) == 6
    before = len(commands)
    study.worker(args)
    assert len(commands) == before  # completed jobs never duplicate training


def test_physical_audit_rejection_prevents_publication():
    report = dict(termination_audit=dict(physical_success_violations=1))
    with pytest.raises(ValueError, match="Physical success audit"):
        study.physical_metrics(report)


def test_extension_uses_validation_progress_and_keeps_camera_pairs(tmp_path):
    cases = []
    for task in ("plug", "push"):
        for seed in (0, 1, 2):
            for condition in ("wrist_static", "active"):
                case = dict(
                    name=f"{task}_{condition}_s{seed}",
                    task=task,
                    seed=seed,
                    condition=condition,
                    kind="direct",
                )
                cases.append(case)
                evaluations = []
                for budget in (10_000_000, 20_000_000):
                    for mode in ("clean", "dynamic"):
                        evaluations.append(
                            dict(
                                scope="original",
                                split="validation",
                                occlusion=mode,
                                transitions=study.updates(budget, 256) * 256 * 24,
                                rate=0.10 if budget == 10_000_000 else 0.13,
                                mean_error_mm=20,
                            )
                        )
                if task == "push":
                    for e in evaluations:
                        e["rate"] = 0.5  # no progress: do not extend pushing
                evaluations.append(
                    dict(
                        scope="original",
                        split="test",
                        occlusion="dynamic",
                        transitions=20_004_864,
                        rate=1.0,
                        mean_error_mm=0,
                    )
                )
                study.save(
                    tmp_path / case["name"] / "state.json",
                    dict(status="complete", evaluations=evaluations),
                )
    plan = dict(output=str(tmp_path), cases=cases, num_envs=256, extension_rule="validation only")
    decision = study.extension_decision(plan)
    assert decision["eligible_indices"] == list(range(6))
    assert decision["tasks"]["plug"]["extend"]
    assert not decision["tasks"]["push"]["extend"]
