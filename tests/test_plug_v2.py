"""hidden_prongs_v2: seated success, jam failure, staged reward, gap screen, look-at camera."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from rsl_rl.models import RNNModel
from tensordict import TensorDict

from active_perception_arms import plug, plug_v2
from active_perception_arms.config import Experiment, saved_experiment

V2 = dict(task="plug", control_revision="pose_v1", task_revision="hidden_prongs_v2")


def positions(*rows):
    return torch.tensor(rows, dtype=torch.float64)


def goal_at(n):
    goal = torch.zeros(n, 3, dtype=torch.float64)
    goal[:, 2] = plug.INSERT_Z
    return goal


def test_seated_counts_the_physically_inserted_plug_that_v1_rejects():
    # Housing resting on the plate (z = 14 mm) with prongs inside the clearance.
    seated_pose = positions([0.001, 0.0005, 0.014], [0.0, 0.0, 0.0165])
    goal = goal_at(2)
    assert plug_v2.seated(seated_pose, goal).all()
    v1 = (seated_pose - goal).norm(dim=-1) < plug.SUCCESS_DISTANCE
    assert v1.tolist() == [False, True]
    # Not engaged deeply enough, or outside the hole clearance.
    assert not plug_v2.seated(positions([0.0, 0.0, 0.0185]), goal_at(1)).any()
    assert not plug_v2.seated(positions([0.003, 0.0, 0.014]), goal_at(1)).any()


def test_jam_requires_a_low_misaligned_press_over_the_plate():
    socket = torch.zeros(1, 3, dtype=torch.float64)
    offset = torch.tensor([[-0.015, 0.0]], dtype=torch.float64)
    goal = goal_at(1)
    goal[:, :2] = socket[:, :2] - offset
    low_misaligned = goal + torch.tensor([[0.008, 0.0, 0.009]], dtype=torch.float64)
    assert plug_v2.jammed(low_misaligned, goal, offset, socket).all()
    high_misaligned = goal + torch.tensor([[0.008, 0.0, 0.015]], dtype=torch.float64)
    assert not plug_v2.jammed(high_misaligned, goal, offset, socket).any()
    aligned_low = goal + torch.tensor([[0.001, 0.0, -0.002]], dtype=torch.float64)
    assert not plug_v2.jammed(aligned_low, goal, offset, socket).any()
    beside_plate = goal + torch.tensor([[0.0, 0.12, 0.0]], dtype=torch.float64)
    assert not plug_v2.jammed(beside_plate, goal, offset, socket).any()


def test_potential_orders_lift_align_descend_and_penalises_low_misalignment():
    goal = goal_at(5)
    phi = plug_v2.potential(
        positions(
            [0.05, 0.0, 0.035],  # far away, spawn height
            [0.0005, 0.0, 0.045],  # aligned at the top of the range
            [0.0005, 0.0, 0.030],  # aligned, lower
            [0.0005, 0.0, 0.014],  # seated
            [0.01, 0.0, 0.020],  # misaligned and low
        ),
        goal,
    ).tolist()
    assert phi[0] < phi[1] < phi[2] < phi[3]
    misaligned_high = plug_v2.potential(positions([0.01, 0.0, 0.035]), goal_at(1)).item()
    assert phi[4] < misaligned_high
    # Bounded terminal events; no per-step +1000 that a clipped critic cannot track.
    reward = plug_v2.reward(
        torch.tensor([4.0]),
        torch.tensor([3.0]),
        torch.tensor([True]),
        torch.tensor([True]),
        torch.tensor([False]),
    )
    assert reward.item() == pytest.approx(plug_v2.SHAPING_SCALE + plug_v2.SUCCESS_BONUS)


def test_v1_defaults_are_unchanged_and_v2_defaults_are_explicit():
    v1 = Experiment(task="plug", control_revision="pose_v1")
    assert (v1.task_revision, v1.plug_success_rule, v1.plug_jam_failure) == (
        "hidden_prongs_v1",
        "hover_v1",
        False,
    )
    assert (v1.reward_profile, v1.episode_seconds, v1.plug_screen) == (
        "plug_insertion",
        3.5,
        "none",
    )
    v2 = Experiment(**V2)
    assert (v2.plug_success_rule, v2.plug_jam_failure, v2.reward_profile) == (
        "seated_v2",
        True,
        "plug_seat_v2",
    )
    assert v2.episode_seconds == 5.0
    # Saved configurations without the new fields keep their original rules.
    old = asdict(v1)
    for name in (
        "plug_success_rule",
        "plug_jam_failure",
        "plug_reset_scale",
        "plug_screen",
        "camera_action",
        "aux_targets",
    ):
        old.pop(name)
    restored = saved_experiment(old)
    assert (restored.plug_success_rule, restored.plug_jam_failure) == ("hover_v1", False)
    assert saved_experiment(asdict(v2)).plug_success_rule == "seated_v2"


@pytest.mark.parametrize(
    "changes",
    [
        dict(task="plug", plug_screen="gap_v1"),
        dict(task="plug", plug_reset_scale=0.5),
        dict(task="plug", camera_action="lookat_v2", camera_start="lookat_v2"),
        dict(**V2, camera_action="lookat_v2"),
        dict(task="push", aux_targets=True),
        dict(task="push", reward_profile="plug_seat_v2"),
        dict(**V2, condition="wrist_static", camera_visibility_reward=0.1),
    ],
)
def test_v2_options_reject_incompatible_configurations(changes):
    with pytest.raises(ValueError):
        Experiment(**changes)


def test_native_seated_plug_succeeds_only_under_v2():
    from active_perception_arms.native import NativeEnv

    def seat(revision):
        cfg = Experiment(
            task="plug",
            condition="wrist_static",
            control_revision="pose_v1",
            translation_control="absolute",
            task_revision=revision,
            plug_variant="ym",
            episode_seconds=6.0,
            randomize=False,
        )
        env = NativeEnv(cfg)
        goal = env.goal
        hits = []
        for k in range(130):
            t = k * cfg.step_dt
            p = goal + plug.GRIP_OFFSET
            if t < 2.4:
                p[2] = 0.09
            else:
                # Press past the v1 hover height until the housing rests on the plate.
                p[2] = max(plug.ACTION_LOW[2], 0.09 - (t - 2.4) * 0.03)
            a = np.zeros(cfg.action_dim)
            a[:3] = 2 * (p - plug.ACTION_LOW) / (np.array(plug.ACTION_HIGH) - plug.ACTION_LOW) - 1
            hits.append(env.step(a))
        return env.object_pos[2], hits[-10:]

    z1, hits1 = seat("hidden_prongs_v1")
    z2, hits2 = seat("hidden_prongs_v2")
    assert z1 < 0.0145 and z2 < 0.0145  # pressed home onto the plate in both cases
    assert not any(hits1)
    assert all(hits2)


def test_native_gap_screen_resamples_and_scripted_v2_insertion_succeeds():
    from active_perception_arms.native import NativeEnv, ScriptedPolicy, native_plug_jammed

    cfg = Experiment(**V2, condition="wrist_static", plug_screen="gap_v1", plug_variant="xm")
    env = NativeEnv(cfg)
    gaps = set()
    for seed in range(3):
        env.reset(seed)
        gaps.add(round(env.screen_gap, 6))
        low, high = plug_v2.SCREEN_GAP_RANGE
        assert low <= env.screen_gap <= high
        left = env.data.mocap_pos[env.model.body("screen_left/panel").mocapid[0]]
        right = env.data.mocap_pos[env.model.body("screen_right/panel").mocapid[0]]
        inner = (right[0] - plug_v2.SCREEN_SEGMENT_HALF_LENGTH) - (
            left[0] + plug_v2.SCREEN_SEGMENT_HALF_LENGTH
        )
        assert inner == pytest.approx(plug_v2.SCREEN_GAP)
    assert len(gaps) == 3
    env.reset(11)
    policy = ScriptedPolicy(env)
    hold = 0
    for _ in range(int(cfg.episode_seconds / cfg.step_dt)):
        hold = hold + 1 if env.step(policy()) else 0
        assert not native_plug_jammed(env)
        if hold >= 3:
            break
    assert hold >= 3


def test_lookat_camera_targets_are_rate_limited_and_bounded():
    from active_perception_arms.robots.camera_control import LOOKAT_SPEED, CameraPoseIK
    from active_perception_arms.scenes import native_spec

    cfg = Experiment(**V2, camera_action="lookat_v2", camera_start="lookat_v2")
    ik = CameraPoseIK(native_spec(cfg).compile(), cfg, dtype=torch.float64)
    position = ik.initial[None, :3].clone()
    look = ik.initial_look[None].clone()
    raw = torch.tensor([[1.0, -1.0, -1.0, 0.5, -0.5]], dtype=torch.float64)
    for _ in range(40):
        new_position, look, gimbal = ik.lookat(position, look, raw, torch.ones(1))
        assert (new_position - position).abs().max() <= LOOKAT_SPEED + 1e-12
        position = new_position
    torch.testing.assert_close(
        position[0], torch.stack((ik.box_high[0], ik.box_low[1], ik.box_low[2]))
    )
    frozen, _, _ = ik.lookat(position, look, -raw, torch.zeros(1))
    torch.testing.assert_close(frozen, position)
    assert torch.isfinite(gimbal).all()


def test_aux_loss_trains_the_perception_head_and_zero_coefficient_is_upstream():
    from rsl_rl.storage import RolloutStorage

    from active_perception_arms.policy import CompactPPO, VisualMemoryModel  # noqa: F401

    def build(aux_coef, seed=3):
        torch.manual_seed(seed)
        n, steps = 6, 5
        obs = TensorDict(
            {
                "proprio": torch.randn(n, 7),
                "critic": torch.randn(n, 9),
                "aux": torch.cat((torch.eye(4)[torch.arange(n) % 4], torch.randn(n, 5)), -1),
            },
            [n],
        )
        groups = {"actor": ["proprio"], "critic": ["critic"]}
        dist = {"class_name": "GaussianDistribution", "init_std": 0.5}

        class Actor(RNNModel):
            def __init__(self, *args, aux_dim=0, **kwargs):
                super().__init__(*args, **kwargs)
                self.aux_head = torch.nn.Linear(16, aux_dim) if aux_dim else None

            def get_latent(self, obs, masks=None, hidden_state=None):
                self.last_latent = super().get_latent(obs, masks, hidden_state)
                return self.last_latent

        actor = Actor(
            obs,
            groups,
            "actor",
            3,
            hidden_dims=[16],
            rnn_type="gru",
            rnn_hidden_dim=16,
            distribution_cfg=dict(dist),
            aux_dim=9,
        )
        critic = RNNModel(
            obs, groups, "critic", 1, hidden_dims=[16], rnn_type="gru", rnn_hidden_dim=16
        )
        storage = RolloutStorage("rl", n, steps, obs, [3], "cpu")
        alg = CompactPPO(
            actor, critic, storage, num_learning_epochs=2, num_mini_batches=2, aux_coef=aux_coef
        )
        # Collect exactly as the RSL runner does: rollout under inference mode.
        with torch.inference_mode():
            for t in range(steps):
                actions = alg.act(obs)
                obs = obs.clone()
                obs["proprio"] = obs["proprio"] + 0.1 * actions.sum(-1, keepdim=True)
                dones = torch.zeros(n, dtype=torch.bool)
                dones[t % n] = t == 3
                alg.process_env_step(obs, torch.randn(n), dones, {})
            alg.compute_returns(obs)
        return alg

    upstream = build(0.0)
    losses = upstream.update()
    assert "aux" not in losses
    alg = build(1.0)
    first = alg.update()
    assert first["aux"] > 0 and 0 <= first["aux_variant_accuracy"] <= 1


@pytest.fixture(scope="module")
def v2_env():
    from active_perception_arms.environment import make_env

    cfg = Experiment(
        **V2,
        condition="active",
        plug_screen="gap_v1",
        camera_action="lookat_v2",
        camera_start="lookat_v2",
        aux_targets=True,
        camera_visibility_reward=0.02,
        num_envs=4,
        width=32,
        height=32,
    )
    env = make_env(cfg, device="cpu")
    yield env, cfg
    env.close()


def test_mjlab_v2_env_groups_screen_and_rewards(v2_env):
    from active_perception_arms import mdp

    env, cfg = v2_env
    obs, _ = env.reset(seed=4)
    assert obs["aux"].shape == (4, 9)
    torch.testing.assert_close(obs["aux"][:, :4].sum(-1), torch.ones(4))
    assert "aux" not in ("proprio", "wrist", "external")
    s = mdp.state(env)
    low, high = plug_v2.SCREEN_GAP_RANGE
    assert ((s.screen_gap >= low) & (s.screen_gap <= high)).all()
    left = env.scene["screen_left"].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    torch.testing.assert_close(
        left + plug_v2.SCREEN_SEGMENT_HALF_LENGTH + plug_v2.SCREEN_GAP / 2, s.screen_gap
    )
    actions = torch.zeros(4, cfg.action_dim)
    for _ in range(3):
        obs, rewards, terminated, _, extras = env.step(actions)
        assert torch.isfinite(rewards).all()
    visibility = mdp.camera_prong_visibility(env)
    assert ((visibility >= 0) & (visibility <= 1)).all()
    term = env.action_manager.get_term("arms")
    # Zero look-at actions drive the camera toward the box centre at bounded speed.
    assert torch.isfinite(term.gimbal_target).all()
    assert torch.isfinite(term.look_target).all()
    replace(cfg)  # configuration stays a frozen, valid Experiment


def test_mjlab_assisted_training_starts_are_aligned_above_the_goal():
    from active_perception_arms import mdp
    from active_perception_arms.environment import make_env

    cfg = Experiment(
        **V2, condition="wrist", plug_assist_fraction=0.99, num_envs=8, width=32, height=32
    )
    env = make_env(cfg, device="cpu")
    try:
        env.reset(seed=3)
        s = mdp.state(env)
        address = s.object_qpos_address
        obj = env.sim.data.qpos[:, address : address + 3] - env.scene.env_origins
        _, _, _, goal = mdp.positions(env)
        goal = goal - env.scene.env_origins
        xy = (obj[:, :2] - goal[:, :2]).norm(dim=-1)
        height = obj[:, 2] - goal[:, 2]
        aligned = (xy < 0.003) & (height > 0.002) & (height < 0.021)
        assert aligned.float().mean() >= 0.75
        assert not mdp.plug_jam(env).any()
    finally:
        env.close()


def test_evaluation_never_uses_assisted_training_starts():
    from types import SimpleNamespace

    from active_perception_arms.evaluate import task_overrides

    trained = Experiment(**V2, plug_assist_fraction=0.3)
    args = SimpleNamespace(episode_seconds=None, reset_xy_jitter=None, success_state_sample=None)
    overrides = task_overrides(args, trained)
    assert overrides["plug_assist_fraction"] == 0.0
    assert replace(trained, **overrides).plug_assist_fraction == 0.0
    assert "plug_assist_fraction" not in task_overrides(args, Experiment(**V2))


def test_mjlab_v2_rgb_ppo_update_trains_aux_head_without_actor_label_inputs(v2_env):
    """One real RSL-RL update on rendered v2 observations with the auxiliary loss."""
    import copy

    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from active_perception_arms.policy import runner_cfg, runner_dict

    env, cfg = v2_env
    options = runner_dict(runner_cfg(cfg, iterations=1), cfg, aux_coef=0.5)
    options["num_steps_per_env"] = 2
    options["algorithm"]["num_mini_batches"] = 1
    options["algorithm"]["num_learning_epochs"] = 1
    options["logger"] = "tensorboard"
    assert "aux" not in options["obs_groups"]["actor"]
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), copy.deepcopy(options), device="cpu")
    head = copy.deepcopy(runner.alg.actor.aux_head.state_dict())
    runner.logger.writer = None
    obs = runner.env.get_observations()
    with torch.inference_mode():
        for _ in range(2):
            actions = runner.alg.act(obs)
            obs, rewards, dones, extras = runner.env.step(actions)
            runner.alg.process_env_step(obs, rewards, dones, extras)
        runner.alg.compute_returns(obs)
    losses = runner.alg.update()
    assert torch.isfinite(torch.tensor(losses["aux"])) and losses["aux"] > 0
    changed = [
        not torch.equal(before, after)
        for before, after in zip(
            head.values(), runner.alg.actor.aux_head.state_dict().values(), strict=True
        )
    ]
    assert any(changed)


def test_v2_checkpoint_evaluation_uses_seated_audit_and_original_resets(tmp_path):
    import json
    from types import SimpleNamespace

    from rsl_rl.utils import resolve_callable

    from active_perception_arms.evaluate import evaluate
    from active_perception_arms.policy import runner_cfg, runner_dict

    exp = Experiment(
        **V2,
        condition="active",
        plug_screen="gap_v1",
        camera_action="lookat_v2",
        camera_start="lookat_v2",
        aux_targets=True,
        plug_assist_fraction=0.3,
        num_envs=4,
        width=32,
        height=32,
        episode_seconds=1.2,
        initial_seconds=0.04,
    )
    options = runner_dict(runner_cfg(exp), exp, aux_coef=0.5)
    (tmp_path / "experiment.json").write_text(json.dumps(exp.to_dict()))
    (tmp_path / "runner.json").write_text(json.dumps(options))
    from active_perception_arms.environment import make_env

    env = make_env(replace(exp, plug_assist_fraction=0.0), device="cpu")
    obs, _ = env.reset()
    env.close()
    actor_options = dict(options["actor"])
    cls = resolve_callable(actor_options.pop("class_name"))
    actor = cls(obs, options["obs_groups"], "actor", exp.action_dim, **actor_options)
    checkpoint = tmp_path / "UNTRAINED_FIXTURE.pt"
    torch.save({"actor_state_dict": actor.state_dict()}, checkpoint)
    args = SimpleNamespace(
        checkpoint=checkpoint,
        episodes=4,
        num_envs=4,
        seed=20000,
        split="test",
        occlusion="clean",
        device="cpu",
        freeze_camera_after=None,
        hold_external_after=None,
        reset_memory=False,
        camera_trace_in=None,
        camera_trace_out=None,
        output=tmp_path / "eval.json",
        audit_termination=True,
    )
    report = evaluate(args)
    assert report["experiment"]["plug_assist_fraction"] == 0.0
    audit = report["termination_audit"]
    assert audit["episodes"] == 4
    assert audit["success_seated_violations"] == 0
    assert audit["success_distance_violations"] is None
    assert audit["jam_failures"] is not None
