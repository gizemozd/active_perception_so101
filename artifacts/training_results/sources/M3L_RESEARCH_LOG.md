# Research Log

Reverse-chronological. Newest entries on top. Each entry: one decision or one finding, with the artefact (file, commit, video) it produced.

---

## 2026-05-27 — Experiment A: Temporal separation for camera credit assignment

**Problem.** PPO acts on a single flat 8D Box space (`[a_robot (3D), a_camera (5D)]`). The task reward (`-log(distance)`) depends only on robot end-effector pose, so camera actions affect reward purely indirectly. PPO has no signal distinguishing "this camera move helped" from "this arm move helped," making the camera policy slow to learn and potentially harmful (extra noise in the action space).

**Decision.** Implement temporal separation: the camera acts every K robot steps; on in-between steps its delta is zeroed (camera pose held fixed). This forces reward changes within a K-step window to be attributable to robot actions only, giving the camera clear credit on the steps it does move.

**Implementation.**
- `utils/camera_decimation.py` — new `CameraDecimation(gym.Wrapper)`. Accepts `n_cam_action` (5) and `freq` (K). Zeroes the last `n_cam_action` dims of the action on non-camera steps. No changes to observation or action space shapes.
- `envs/__init__.py` — `make_env` now accepts `camera_freq`; applies `CameraDecimation` after `FrameStack` and before `Monitor` when `camera_freq > 1` and `camera_mode == "learned"`.
- `train.py` — `--camera_freq` arg (default 1, no-op). Passed through to `make_env`.

**Experiment grid.**

| Run | `camera_mode` | `camera_freq` | Notes |
|-----|--------------|--------------|-------|
| baseline | learned | 1 | every-step camera (original) |
| exp_A_K2 | learned | 2 | camera every 2 steps |
| exp_A_K4 | learned | 4 | camera every 4 steps |
| exp_A_K8 | learned | 8 | camera every 8 steps |

**Launch command (cluster).**
```bash
python train.py --camera_mode learned --camera_freq <K> --run_name exp_A_K<K>_s<SEED>
```

**Metrics to compare.** Task success rate, camera pose trajectory variance (how much the camera moves), episode length.

**Expected result.** K=4 or K=8 should outperform baseline by giving the policy a cleaner credit signal during the robot-only steps.

**Open items.** Results pending cluster runs. Experiment B (MAE intrinsic reward) deferred.

---

## 2026-05-04 — Active-camera actuators + sweep controller + cluster training script

**Decision.** With the camera mount fixed (see entry below), wire up actuators so we can drive pan/tilt programmatically, and implement the data-collection sweep schedule (Stream A1.2 in PLAN.md). Training (Stream A1.1) does not use the sweep; the camera stays at home (`ctrl=0`).

**Files added/changed.**
- `tactile_envs/tactile_envs/assets/insertion/scene.xml` — added `<position>` actuators `active_pan_actuator` (kp=50, ctrlrange=±60°) and `active_tilt_actuator` (kp=50, ctrlrange=[-45°, +30°]) between `base_actuator_yaw` and `fingers_actuator`. Crucially placed before fingers so M3L's `mj_data.ctrl[-1]` still maps to the gripper. Keyframe `ctrl` extended from `'0 0 0 0 200'` to `'0 0 0 0 0 0 200'` (active joints default to home pose). Active-cam body is parsed before the object body so `qpos[-7:-4]` and `qpos[-4:]` (the env's object pose accessors in `insertion.py:323,429,430`) still resolve to the object's free joint.
- `tactile_envs/tactile_envs/envs/cam_active_controller.py` — `CamActiveController` with three modes per the plan (50% random_fixed / 30% random_walk / 20% home). Walk std calibrated to 0.05 rad pan and 0.04 rad tilt per step (broad coverage in 100-step episodes without saturating bounds in 2-3 steps).
- `tactile_envs/tactile_envs/envs/active_camera_wrapper.py` — `ActiveCameraSweepWrapper`. Looks up actuator ids by name on every reset (env recompiles model each reset), writes pan/tilt commands to `mj_data.ctrl[active_pan_id]` / `[active_tilt_id]` *before* delegating to the underlying `env.step()` so the values persist across all sub-steps. Renders `cam_active` and reads finger site xpos every step into `info` dict for downstream logging.
- `M3L/scripts/test_active_camera_sweep.py` — local smoke test; wrote `M3L/scripts/outputs/active_camera_sweep_smoke.mp4` (cam_home and cam_active side-by-side, 4 episodes × 60 steps, mode label per frame).
- `M3L/scripts/train_expert.slurm` — SLURM job script for Stream A1.1 (single M3L PPO expert, `MUJOCO_GL=egl`, 1 GPU, 16 CPUs, 48h, 3M timesteps default). Cluster repo path locked to `/n/holylabs/LABS/pgozdil_lab/active-perception/M3L-active-perception`. All outputs land under `$REPO/runs/`: SLURM stdout/err in `runs/slurm-<jobid>.{out,err}`, per-job artefacts (wandb run dir + tensorboard + checkpoints) in `runs/job-<jobid>/wandb/`. Run `mkdir -p $REPO/runs` once on the login node before the first submission.

**Verification.**
- `nq=21`, `nu=7` confirmed via direct `InsertionEnv` instantiation.
- Actuator order: `base_x, base_y, base_z, base_yaw, active_pan, active_tilt, fingers` — verified `ctrl[-1].name == "fingers_actuator"`.
- Random-action smoke test (5 steps, no policy): pan/tilt qpos hold at 0 with default ctrl=0, object position settles to a sensible value during `generate_initial_pose`.
- Sweep wrapper smoke test: 4 episodes, controller picked `random_walk / random_fixed / random_fixed / random_walk` (stochastic, mode probabilities will average out over many episodes).

**Open items.** Pre-A1 placement sanity check (PLAN.md Stream B1) — render a (pan, tilt) grid over ~200 random initial states and confirm the home pose covers the slot ~25–40% of the time but a reachable (pan, tilt) fixes the rest. Not blocking training; can run during the cluster job.

---

## 2026-05-04 — Active camera mount fixed at `(x=0, y=0.2, z=0.4)`

**Decision.** `cam_active` is mounted at `(0, 0.2, 0.4)` in the insertion scene. Pan range ±60°, tilt range [−45°, +30°], FOV 45°. At pan=0 / tilt=0 the camera points at the workspace target `(0, 0, 0.05)`.

**How we picked it.** Two grid videos rendered in `M3L/scripts/active_camera_grid_video.py`:
- `M3L/scripts/outputs/active_camera_grid.mp4` — 4×3 sweep over front/side/back × heights at the workspace radius. The middle row (the side-mount, `x = 0`) was visibly the best; front/back mounts were partially blocked by the gripper or the holder walls.
- `M3L/scripts/outputs/active_camera_grid_yz.mp4` — 6×3 sweep at `x=0`, varying `y ∈ {±0.20, ±0.30, ±0.45}` and `z ∈ {0.25, 0.40, 0.55}`. `(y=0.2, z=0.4)` cleanly shows the slot through the pan-tilt sweep, doesn't clip into the gripper, and has comfortable margin to the table.

**Files changed.**
- `tactile_envs/tactile_envs/assets/insertion/scene.xml` — added `active_cam_mount` body chain (mount → pan hinge `axis="0 0 1"` → tilt hinge `axis="0 1 0"` → `<camera name="active_cam">`) with `xyaxes="-1 0 0  0 -0.868243 0.496139"` so home pose looks at `(0, 0, 0.05)` from `(0, 0.2, 0.4)`. Extended the keyframe `qpos` by 2 zeros to cover the new joints (`nq` is now 21).

**Verification.** Compiled the modified XML and rendered a single frame from `active_cam`: `M3L/scripts/outputs/active_cam_smoke.png`.

**What this unlocks.** Stream A (data collection with active-camera sweeps) and Stream B2 (offline best-view characterization). The 2-DoF range and FOV are still tunable; we can revisit if the pre-A1 placement sanity check (PLAN.md Stream B1) shows the home pose covers the slot too well or too poorly.

---
