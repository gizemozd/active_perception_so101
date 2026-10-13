"""Plot a short insertion-reward pilot and audit its saved evaluations."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    plan = json.loads((out / "plan.json").read_text())
    logs = list(Path(plan["log_root"]).glob("*/iterations.jsonl"))
    assert len(logs) == 1, logs
    history = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert [r["iteration"] for r in history] == list(range(1, len(history) + 1))
    assert len(history) <= plan["iterations"]
    assert all(
        np.isfinite(v)
        for r in history
        for k, v in r.items()
        if k.startswith("train/") and isinstance(v, (int, float))
    )
    fields = {
        "episode_return": "train/Train/mean_reward",
        "terminal_distance_m": "train/Episode_Metrics/goal_distance",
    }
    windows = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Training episode return", "Terminal distance metric (mm)"]
    for ax, (name, key), label, scale in zip(axes, fields.items(), labels, (1, 1000), strict=True):
        valid = [r for r in history if key in r]
        values = np.array([r[key] for r in valid])
        assert len(valid) >= 20, "Need two non-overlapping 10-update windows"
        windows[name] = dict(
            first10=float(values[:10].mean()),
            last10=float(values[-10:].mean()),
            first_window_updates=[valid[0]["iteration"], valid[9]["iteration"]],
            last_window_updates=[valid[-10]["iteration"], valid[-1]["iteration"]],
        )
        ax.plot([r["transitions"] / 1e6 for r in valid], values * scale, linewidth=1.5)
        ax.set(xlabel="Million transitions", ylabel=label)
        ax.grid(alpha=0.25)
    fig.suptitle("Plug insertion reward pilot: static + wrist, seed 0")
    fig.tight_layout()
    fig.savefig(out / "learning.png", dpi=160)
    plt.close(fig)
    evaluations, reports = [], []
    for update in plan["evaluation"]["updates"]:
        path = out / f"evaluation-{update}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        records = data["termination_audit"]["records"]
        assert len(records) == data["episodes"] == 128
        assert data["successes"] == sum(data["outcomes"]) == sum(r["success"] for r in records)
        for key in (
            "success_hold_violations",
            "success_distance_violations",
            "success_three_sample_violations",
            "success_qpos_above_distance_threshold",
        ):
            assert data["termination_audit"][key] == 0, (path, key)
        errors = np.array([r["qpos_position_error_m"] for r in records]) * 1000
        for variant, stats in data["per_variant"].items():
            selected = [r for r in records if data["variants"][r["episode"]] == variant]
            assert len(selected) == stats["episodes"] == 32
            assert sum(r["success"] for r in selected) == stats["successes"]
        checkpoint = Path(data["checkpoint"])
        with checkpoint.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        evaluations.append(
            dict(
                updates=update,
                transitions=update * 512 * 24,
                successes=data["successes"],
                episodes=data["episodes"],
                per_variant=data["per_variant"],
                mean_terminal_error_mm=float(errors.mean()),
                median_terminal_error_mm=float(np.median(errors)),
                checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_hash,
                physical_success_violations=0,
            )
        )
        reports.append(data)
    matches = None
    if len(reports) == 2:
        matches = reports[0]["initial_state_sha256"] == reports[1]["initial_state_sha256"]
        assert matches, "Checkpoint comparison requires identical initial physics and images"
    summary = dict(
        built_at=datetime.now(UTC).isoformat(),
        updates=len(history),
        transitions=history[-1]["transitions"],
        target_transitions=plan["transitions"],
        windows=windows,
        reward_increased=windows["episode_return"]["last10"] > windows["episode_return"]["first10"],
        evaluations=evaluations,
        matched_evaluation_initial_states=matches,
        updates_with_logged_held_success=[
            r["iteration"] for r in history if r.get("train/Episode_Metrics/success_rate", 0) > 0
        ],
        run_directory=str(logs[0].parent),
        limitations="One seed, short pilot; training return includes instantaneous +1000 bonuses. Training terminal-distance and success metrics average reset batches unevenly; no training episode success rate is inferred. Success evaluation requires three held samples. First checkpoint has one PPO update, not zero. Reward trend is descriptive and does not establish task competence.",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
