from types import SimpleNamespace

import numpy as np
import torch

from active_perception_arms import evaluation_audit


def test_termination_audit_reads_manager_sampling_before_reset(monkeypatch):
    # The manager sees <2mm for three steps, while the integrated qpos has moved
    # outside. The audit must preserve both samples, not forward/relabel success.
    state = SimpleNamespace(
        hold=torch.zeros(1, dtype=torch.long),
        cfg=SimpleNamespace(task="plug", success_state_sample="derived_substep"),
    )
    derived = torch.tensor([[0.0015, 0.0, 0.0]])
    qpos = torch.tensor([[0.0025, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    flags = {"success": torch.tensor([False]), "failure": torch.tensor([False])}
    original_rng = torch.get_rng_state().clone()
    calls = []

    def original_compute():
        calls.append(1)
        state.hold += 1
        flags["success"][:] = state.hold >= 3
        return flags["success"].clone()

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        termination_manager=SimpleNamespace(
            compute=original_compute,
            get_term=lambda key: flags[key],
            time_outs=torch.tensor([False]),
        ),
        sim=SimpleNamespace(
            mj_model=SimpleNamespace(joint=lambda key: SimpleNamespace(qposadr=np.array([0]))),
            data=SimpleNamespace(qpos=qpos, qvel=torch.zeros(1, 6), mocap_pos=torch.zeros(1, 1, 3)),
        ),
    )
    monkeypatch.setattr(
        evaluation_audit.mdp,
        "positions",
        lambda env: (None, derived, None, torch.zeros_like(derived)),
    )
    monkeypatch.setattr(evaluation_audit.mdp, "state", lambda env: state)
    audit = evaluation_audit.TerminationAudit(env)
    audit.reset_batch()
    for _ in range(3):
        result = env.termination_manager.compute()
    assert result.item()
    assert len(calls) == 3
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert torch.equal(qpos, torch.tensor([[0.0025, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))
    record = {k: value[0].tolist() for k, value in audit.snapshot.items()}
    state.hold[:] = 0
    derived[:] = 1  # Stand-in for automatic reset after the captured compute.
    assert record["success_hold_steps"] == 3
    summary = evaluation_audit.summarize_termination_audit([record], "plug")
    assert summary["success_distance_violations"] == 0
    assert summary["success_three_sample_violations"] == 0
    assert summary["success_qpos_above_distance_threshold"] == 1
    audit.close()
    assert env.termination_manager.compute is original_compute


def test_initial_state_hash_covers_actor_images_and_world_state():
    audit = object.__new__(evaluation_audit.TerminationAudit)
    audit.env = SimpleNamespace(
        num_envs=2,
        sim=SimpleNamespace(
            data=SimpleNamespace(
                qpos=torch.zeros(2, 7), qvel=torch.zeros(2, 6), mocap_pos=torch.zeros(2, 1, 3)
            )
        ),
    )
    obs = {"wrist": torch.zeros(2, 3, 4, 4, dtype=torch.uint8)}
    before = audit.initial_state_hashes(obs, ["wrist"])
    physics_before, images_before = audit.initial_hash_components(obs, ["wrist"])
    obs["wrist"][1, 0, 0, 0] = 1
    after = audit.initial_state_hashes(obs, ["wrist"])
    physics_after, images_after = audit.initial_hash_components(obs, ["wrist"])
    assert before[0] == after[0] and before[1] != after[1]
    assert physics_before == physics_after
    assert images_before["wrist"][0] == images_after["wrist"][0]
    assert images_before["wrist"][1] != images_after["wrist"][1]
