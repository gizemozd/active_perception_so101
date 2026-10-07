"""Regression checks for the original hidden-prong geometry and its Warp port."""

import math

import mujoco
import numpy as np
import pytest
import torch

from active_perception_arms import mdp, plug
from active_perception_arms.config import Experiment, saved_experiment, static_candidates
from active_perception_arms.environment import make_env_cfg
from active_perception_arms.native import NativeEnv, ScriptedPolicy


def test_legacy_geometry_and_equalized_inertia():
    reference = None
    for variant in plug.VARIANTS:
        model = plug.object_spec(variant).compile()
        body = model.body("object")
        inertia = np.r_[body.mass, body.ipos, body.iquat, body.inertia]
        if reference is None:
            reference = inertia
        np.testing.assert_array_equal(inertia, reference)
        np.testing.assert_allclose(model.geom("housing").size, [0.032, 0.021, 0.015])
        for tag, sign in (("a", 1), ("b", -1)):
            geom = model.geom(f"prong_{tag}")
            offset = plug.OFFSETS[variant]
            np.testing.assert_allclose(
                geom.pos, [offset[0] + sign * 0.011, offset[1], -0.006], atol=1e-8
            )
        assert model.geom("collar").contype == 1
    assert reference[0] == pytest.approx(0.04880934, abs=1e-8)
    socket = plug.fixture_spec().compile()
    # Two 12 mm square holes in a 100 x 70 x 14 mm plate.
    assert socket.ngeom == 5
    np.testing.assert_allclose(socket.geom("mid").size, [0.005, 0.006, 0.007])
    np.testing.assert_allclose(socket.geom("front").size, [0.050, 0.0145, 0.007])


@pytest.mark.parametrize("variant", plug.VARIANTS)
def test_correct_alignment_fits_and_centering_the_body_hits_socket(variant):
    env = NativeEnv(Experiment(plug_variant=variant, randomize=False))
    np.testing.assert_allclose(env.goal[:2], [0.025, 0.025] - np.array(plug.OFFSETS[variant]))
    adr = env.object_adr
    socket = set(np.flatnonzero(env.model.geom_bodyid == env.model.body("fixture/fixture").id))
    prongs = {env.model.geom(f"object/prong_{tag}").id for tag in ("a", "b")}

    def penetrating_contacts():
        mujoco.mj_forward(env.model, env.data)
        return [
            c
            for c in env.data.contact
            if c.dist < -1e-5
            and (
                (c.geom1 in socket and c.geom2 in prongs)
                or (c.geom2 in socket and c.geom1 in prongs)
            )
        ]

    env.data.qpos[adr : adr + 7] = [*env.goal, 1, 0, 0, 0]
    assert not penetrating_contacts()
    assert env.success()
    env.data.qpos[adr : adr + 2] = [0.025, 0.025]
    assert penetrating_contacts()
    assert not env.success()


@pytest.mark.parametrize("variant", plug.VARIANTS)
def test_native_script_with_original_contacts_and_short_horizon(variant):
    env = NativeEnv(Experiment(plug_variant=variant, randomize=False))
    policy = ScriptedPolicy(env)
    held = 0
    won = False
    for _ in range(math.ceil(env.cfg.episode_seconds / env.cfg.step_dt)):
        held = held + 1 if env.step(policy()) else 0
        won |= held >= 3
    assert won and held >= 3


def test_no_variant_motion_cue_before_contact():
    states = []
    for variant in plug.VARIANTS:
        env = NativeEnv(Experiment(plug_variant=variant, randomize=False))
        # Hold the same TCP, with both prongs clear of the plate.
        action = np.zeros(8)
        action[:3] = (
            2 * (env.tcp_target - plug.ACTION_LOW) / (np.array(plug.ACTION_HIGH) - plug.ACTION_LOW)
            - 1
        )
        for _ in range(30):
            env.step(action)
        states.append(env.data.qpos.copy())
    np.testing.assert_allclose(states, np.broadcast_to(states[0], (4, len(states[0]))), atol=1e-9)


def test_checkpoint_compatibility_and_strong_fixed_grid():
    cfg = Experiment()
    assert cfg.action_dim == 8 and cfg.manip_dim == 3
    assert cfg.occlusion == "clean" and cfg.episode_seconds == 3.5
    assert saved_experiment(cfg.to_dict()) == cfg
    old = cfg.to_dict()
    old.pop("task_revision")
    with pytest.raises(ValueError, match="centered-pin"):
        saved_experiment(old)
    assert len(static_candidates("plug")) == 26
    assert min(p[2] for p in static_candidates("plug")) == 0.08


def test_warp_balanced_variants_indexing_resets_and_scripted_success():
    from mjlab.envs import ManagerBasedRlEnv

    from active_perception_arms.warp_sanity import sync_native_state

    cfg = Experiment(num_envs=4, randomize=False, render_sensors=False)
    setup = make_env_cfg(cfg)
    setup.terminations = {}
    env = ManagerBasedRlEnv(setup, device="cpu")
    proxies = [NativeEnv(Experiment(plug_variant=v, randomize=False)) for v in plug.VARIANTS]
    policies = [ScriptedPolicy(proxy) for proxy in proxies]
    try:
        env.reset()
        # Catch MjLab 1.4's uncompiled original-spec IDs (-1), including mocaps.
        for entity in env.scene.entities.values():
            assert (entity.indexing.body_ids >= 0).all()
            if entity.indexing.mocap_id is not None:
                assert entity.indexing.mocap_id >= 0
        _, obj, _, goal = mdp.positions(env)
        torch.testing.assert_close(
            obj, torch.tensor([[0.0, 0.025, 0.035]]).expand(4, -1), atol=2e-5, rtol=0
        )
        torch.testing.assert_close(goal[:, :2], 0.025 - mdp.state(env).plug_offsets)
        torch.testing.assert_close(mdp.proprio(env), mdp.proprio(env)[0].expand(4, -1))
        # Per-world collision/render mesh centers must preserve every hidden offset.
        ids = env.scene["object"].indexing.geom_ids
        mesh_ids = [
            i.item() for i in ids if env.sim.mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        ]
        # Prongs are first two mesh slots; collar is the third.
        center = env.sim.data.geom_xpos[:, mesh_ids[:2]].mean(1) - obj
        torch.testing.assert_close(center[:, :2], mdp.state(env).plug_offsets, atol=1e-6, rtol=0)
        hold = torch.zeros(4, dtype=torch.long)
        won = torch.zeros(4, dtype=torch.bool)
        for step in range(math.ceil(cfg.episode_seconds / cfg.step_dt)):
            for index, proxy in enumerate(proxies):
                sync_native_state(proxy, env, step, index)
            actions = torch.as_tensor(np.stack([p() for p in policies]), dtype=torch.float32)
            env.step(actions)
            hold = torch.where(mdp.instantaneous_success(env), hold + 1, 0)
            won |= hold >= 3
        assert won.all() and (hold >= 3).all()
        before = env.scene["object"].data.root_link_pos_w.clone()
        env.reset(env_ids=torch.tensor([1, 3]))
        torch.testing.assert_close(env.scene["object"].data.root_link_pos_w[[0, 2]], before[[0, 2]])
        torch.testing.assert_close(
            mdp.state(env).plug_offsets, torch.tensor(list(plug.OFFSETS.values()))
        )
        term = env.action_manager.get_term("arms")
        term.camera_frozen = True
        camera = term.targets["camera_arm"].clone()
        gimbal = term.gimbal_target.clone()
        term.process_actions(torch.ones(4, 8))
        torch.testing.assert_close(camera, term.targets["camera_arm"])
        torch.testing.assert_close(gimbal, term.gimbal_target)
    finally:
        env.close()
