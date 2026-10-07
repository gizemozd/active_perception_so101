"""Regression for the installed RSL-RL/W&B writer compatibility boundary."""

import wandb

from active_perception_arms.telemetry import compatible_wandb_writer


def test_native_wandb_writer_works_with_current_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    writer = compatible_wandb_writer(tmp_path, {"wandb_project": "unit-test-no-upload"})
    try:
        assert wandb.run is not None
        writer.add_scalar("train/transitions", 24, 0)
        writer.add_scalar("train/reward", 0.1, 0)
        writer.flush()
        assert list(tmp_path.glob("events.out.tfevents.*"))
    finally:
        writer.close()
        writer.stop()
