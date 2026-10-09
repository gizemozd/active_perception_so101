import json
from dataclasses import asdict
from types import SimpleNamespace

import torch
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from active_perception_arms.config import Experiment
from active_perception_arms.evaluate import evaluate
from active_perception_arms.policy import runner_cfg


def test_untrained_checkpoint_evaluation_and_replay_roundtrip(tmp_path):
    # Random initialization is saved only as a fixture. No learning or optimizer.
    exp = Experiment(num_envs=1, width=32, height=32, episode_seconds=1.12, initial_seconds=0.04)
    options = asdict(runner_cfg(exp))
    (tmp_path / "experiment.json").write_text(json.dumps(exp.to_dict()))
    (tmp_path / "runner.json").write_text(json.dumps(options))
    obs = TensorDict(
        {
            "proprio": torch.zeros(1, 51),
            "critic": torch.zeros(1, 76),
            "wrist": torch.zeros(1, 3, 32, 32, dtype=torch.uint8),
            "external": torch.zeros(1, 3, 32, 32, dtype=torch.uint8),
        },
        [1],
    )
    actor_options = options["actor"]
    cls = resolve_callable(actor_options.pop("class_name"))
    actor = cls(obs, options["obs_groups"], "actor", exp.action_dim, **actor_options)
    checkpoint = tmp_path / "UNTRAINED_FIXTURE.pt"
    torch.save({"actor_state_dict": actor.state_dict()}, checkpoint)
    args = SimpleNamespace(
        checkpoint=checkpoint,
        episodes=1,
        num_envs=1,
        seed=20000,
        split="test",
        occlusion="clean",
        device="cpu",
        freeze_camera_after=None,
        hold_external_after=None,
        reset_memory=False,
        camera_trace_in=None,
        camera_trace_out=tmp_path / "trace.npz",
        output=tmp_path / "eval.json",
    )
    first = evaluate(args)
    assert first["episodes"] == 1 and not first["training_started"]
    args.camera_trace_in = args.camera_trace_out
    args.camera_trace_out = None
    args.reset_memory = True
    args.hold_external_after = 0.04
    args.audit_termination = True
    second = evaluate(args)
    assert second["episodes"] == 1
    assert second["interventions"]["camera_trace_in"]
    assert second["termination_audit"]["episodes"] == 1
    assert second["termination_audit"]["success_qpos_above_distance_threshold"] == 0
    stored = json.loads(args.output.read_text())
    assert len(stored["initial_physics_state_sha256"]) == 1
    assert set(stored["initial_sensor_sha256"]) == {"wrist", "external"}
