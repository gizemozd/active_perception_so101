# SO-101 active perception experiments

Three two-arm manipulation environments in **MjLab / MuJoCo Warp**, with a shared
visual actor controlling the manipulation arm and (where enabled) the camera arm.
The SO-101 models are self-contained copies of the supplied legacy assets, with
white arm shells. [Watch the full-HD outside-view videos](artifacts/README.md).
Four matched plug-insertion pilots have completed 18,432,000 transitions each.
Their seed-0 validation remains unreliable across variants; optimization repair is
in progress before a larger sensing study. See [PROGRESS.md](PROGRESS.md) and the
[continuation results](artifacts/cluster_continuation/README.md).

The new [task-screening report](docs/TASK_SCREENING.md) evaluates six visibility
prototypes against 522 fixed viewpoints, initial-only sensing, memory, scans,
waiting and hand retreat. It recommends connector/seating variants for further
mechanical validation and retains pushing as a negative control. These are
pre-training diagnostics, not learned-policy success results.

**The original hidden-prong plug task is restored.** Four identical bodies hide
four different two-prong offsets, with the original socket, grasp, equalized
inertia, Cartesian control and camera-arm layout. See the
[restoration report](docs/PLUG_RESTORATION.md) and
[all four variants on video](artifacts/plug_restoration/README.md).
The earlier centered-pin and marker-prototype results remain historical controls;
they do not describe this restored task.

## Tasks and controls

| Task | Physical task | Information question |
|---|---|---|
| `plug` | Identify a hidden ±15 mm prong offset and insert both prongs into a two-hole socket | Does initial or continued external sensing improve offset identification and alignment? |
| `transfer` | Grasp a cube in a shallow tray and place it inside an open cubby | Does useful viewing direction change between grasping, transport, and placement? |
| `push` | Push a block with a rigid tip to a visible target and withdraw | Does fresh feedback help after occlusion or an optional object disturbance? |

Plug and pushing success require three consecutive control samples; new transfer
runs require five. New plug runs check the
current physical body position within 2 mm of the offset-corrected goal. Saved
older runs retain their lagged derived-state criterion; the
[analysis report](artifacts/plug_analysis/index.html) labels historical scores and
[corrected physical-state scores](artifacts/plug_analysis/corrected_performance_metrics.csv)
separately. Transfer additionally requires a physical bilateral grasp/lift,
placement supported by the cubby floor, release from the arm, low linear and
angular speed, and hand withdrawal. Pushing requires placement, low object
speed, and withdrawal. Plug acquisition
is excluded with a pregrasp weld; transfer uses contact grasping. No teleportation
or attachment assists the transfer/push diagnostic controllers.

All modes receive joint positions, joint velocities, servo targets, fixed camera
calibration, and elapsed time. Object/goal state and occlusion timers are restricted
to the training critic and reward calculation. Policy images are RGB at 25 Hz
with identical 48.46° vertical FOV: 128×96 for plug, 96×72 for transfer/push.
Legacy plug checkpoints use three absolute TCP actions with fixed grasp orientation plus
five camera-gimbal deltas when enabled; commanded TCP/gimbal state is also observable.

Fresh plug/push CLI training defaults to `--control-revision pose_v1`:
three world XYZ manipulator target increments (5 mm per step) and five world
XYZ/pan/tilt camera target increments (10 mm / 0.05 rad per step), followed by IK.
Both integrate the previous commanded target and clamp it to the task workspace.
The comparison uses 25 Hz, identical RGB resolution and matched training settings.
`wrist_static` holds the same arm-mounted camera at its selected starting pose, so its initial
images match `active`; searched legacy fixed-camera placements are a different baseline.
`--translation-control absolute` provides a matched ablation with the same 5 mm slew
limit. `--camera-interval` controls how often camera targets update. Historical
configurations retain `legacy_v1`; checkpoint action meanings are never reinterpreted.
Historical study/benchmark launchers explicitly retain `legacy_v1` for reproducibility;
the common-control study explicitly selects `pose_v1`. The historical Slurm grid
also accepts `CONTROL_REVISION=pose_v1` for supported sensing conditions.
The separately tested `--action-distribution squashed` bounds physical commands,
corrects their log probabilities and entropy, and bounds latent means/exploration.
The camera follow-up uses `--camera-workspace inspection_v2`; recipes with
`--camera-start inspection_v1` start fixed and active cameras at the same calibrated
reachable inspection view. These geometry changes are separate from the initial
absolute/delta comparison. Neither changes the insertion success criterion.
Study plans, source snapshots, evaluation results and the generated report live in
`artifacts/common_control_study/`; rebuild its summary with
`.venv/bin/python scripts/summarize_common_control_study.py`.
The completed 30-run screen and causal camera tests are in
[`artifacts/common_control_study/index.html`](artifacts/common_control_study/index.html).
Verify budgets, source snapshots, selection and physical outcomes with
`.venv/bin/python scripts/audit_common_control_study.py`.
Legacy transfer/push use bounded joint target increments; transfer additionally controls
gripper aperture. Both arms retain collisions.

| Condition | Actor images | Camera control |
|---|---|---|
| `wrist` | Manipulation wrist | Second arm parked |
| `static` | Fixed external | Second arm parked |
| `wrist_static` | Wrist + fixed external | Second arm parked |
| `initial` | Wrist + camera arm | Learned movement for the first second, then target held |
| `scheduled` | Wrist + camera arm | Precomputed camera path; no observation feedback |
| `active` | Wrist + camera arm | Learned camera actions throughout |

The first second holds the manipulation arm in **every** mode. A GRU is shared
between the two arm action outputs. `--memory none` trains a separate feedforward
actor using the same visual encoders; the privileged critic remains recurrent.
These components implement established methods, not a methodological novelty claim.

## Installation and local checks

```bash
uv sync --locked
uv run pytest -q
uv run arms-train --task plug --condition active --dry-run
uv run arms-sanity --output artifacts/sanity --occlusion clean
uv run arms-sanity --output artifacts/sanity --occlusion phase
uv run python -m active_perception_arms.warp_sanity --device cpu
# Restored plug: all four variants, matched views, native and Warp videos:
uv run python -m active_perception_arms.plug_audit
```

Both sanity commands also save an **external overview video at 1920×1080, 25 fps**
(`*_overview.mp4`) and a full-resolution PNG (`*_overview.png`). The fixed `overview`
camera frames both arms and the workspace from the front left. Use `--no-overview`
to skip this additional diagnostic rendering. Policy observations keep their
task-specific policy resolution; the overview is not registered as an RL sensor.
The restored plug also has a 1080p `task_detail` observer, excluded from RL inputs.

`arms-sanity` uses native MuJoCo for fast scripted physics/video and segmentation
visibility audits. `warp_sanity` uses the actual MjLab/Warp physics and batch RGB
renderer, with the same diagnostic controller. The latter is slow on CPU; use
`--device cuda:0` on a GPU node. Warp overview videos render the actual Warp state
through native MuJoCo OpenGL; the wrist/camera-arm videos retain the Warp batch RGB
output. Neither command trains a policy. Diagnostic IK uses privileged state and
is not evidence of visual policy success.

The lock pins MjLab 1.4.0, MuJoCo 3.9.0, RSL-RL 5.2.0, Torch 2.10.0, Warp 1.13.0,
and MuJoCo Warp commit `88b55fc`. Linux uses CUDA 12.8 PyTorch wheels. The MuJoCo
pin overrides MjLab's declared 3.8 range; the local integration checks exercise
this combination. GPU execution needs a compatible NVIDIA driver. Asset notices
and the precise dependency choices are documented in [docs/ASSETS.md](docs/ASSETS.md).

## Cluster jobs

Run setup once, from the repository root, after loading your site's environment:

```bash
bash scripts/setup_cluster.sh
mkdir -p logs/slurm
# Replace these site-specific values; scripts deliberately leave them configurable.
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/validate.sbatch
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/benchmark.sbatch
```

For H100/A100/new RTX nodes, select the site's GPU resource with `--gres` or
`--constraint` as appropriate; no unverified partition, account, or RTX SKU is
hardcoded. `benchmark.sbatch` measures 64/128/256/512 environments on all three
tasks with rendering and actor inference. It does **not** optimize weights.

After GPU validation and selecting one environment count that fits all comparison
runs, these are the training submissions (provided for the user; not executed):

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/train_plug.sbatch
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/train_transfer.sbatch
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/train_push.sbatch
```

Each task array has 18 jobs: six camera conditions × three seeds. Array indices
0–2 wrist, 3–5 static, 6–8 wrist+static, 9–11 initial, 12–14 scheduled, 15–17 active.
Jobs checkpoint after a completed PPO iteration when Slurm sends USR1; `RESUME` continues to the configured total iteration count. Defaults: 256 environments, 24 rollout steps, eight minibatches, four PPO epochs,
and 18,432,000 transitions/run. `NUM_ENVS`, `TOTAL_STEPS`, `MEMORY`, `OCCLUSION`,
`LOG_ROOT`, `FIXED_VIEW`, `RESUME`, and `PROJECT_ROOT` are environment overrides.
The iteration count adjusts to preserve transition budget when NUM_ENVS changes;
use the same value within a comparison because batch size affects optimization.
`PERTURB_PUSH=1` enables a brief lateral force in pushing. Plug defaults to clean
occlusion and a 3.5-second episode (1-second inspection plus 2.5-second action
budget); the other tasks retain random occlusion and 12 seconds. Old centered-pin
checkpoints are rejected by the `hidden_prongs_v1` task revision check.

Dry-check job dispatch without Slurm or training:

```bash
DRY_RUN=1 SLURM_ARRAY_TASK_ID=17 bash scripts/slurm/train_plug.sbatch
```

## Search the fixed baseline before claiming an active-camera advantage

View 0 is a reference pose, **not an optimized baseline**. The grid has 26 views,
including overhead and views at three heights around the table. For plug these
heights are 8, 22 and 35 cm; the low ring is essential for seeing the underside. Search
`static` and `wrist_static` separately, with the same training budget and seeds:

```bash
TASK=plug CONDITION=wrist_static sbatch scripts/slurm/static_search.sbatch
# Repeat for each task and for CONDITION=static.
uv run arms-evaluate logs/RUN/model_2999.pt --split validation --output artifacts/val-RUN.json
uv run python -m active_perception_arms.select_view artifacts/val-*.json --output artifacts/selected_view.json
```

Pass reports for exactly one task/condition/distribution to the selector. It
requires all 26 views with matched training seeds and rejects test results and
duplicate checkpoint selection. Final checkpoints are the default comparison;
if checkpoint selection is desired, predeclare and match it across every mode.
The winning static policies can be evaluated directly on the test set, or retrained
with `FIXED_VIEW=<selected_view>` using the same seed protocol. Default validation
and test seed bases are 10000 and 20000; preserve evaluation num-envs across paired
runs. No viewpoint has been selected in this implementation session.

## Evaluation and causal controls

```bash
uv run arms-evaluate logs/RUN/model_2999.pt --occlusion random --output artifacts/test.json
uv run arms-evaluate logs/RUN/model_2999.pt --freeze-camera-after 1 --output artifacts/frozen-camera.json
uv run arms-evaluate logs/RUN/model_2999.pt --hold-external-after 1 --output artifacts/stale-images.json
uv run arms-evaluate logs/RUN/model_2999.pt --reset-memory --output artifacts/no-memory-at-test.json
# Record a camera trajectory on one held-out episode batch, replay on another:
uv run arms-evaluate logs/RUN/model_2999.pt --seed 20000 --camera-trace-out artifacts/camera.npz --output artifacts/recorded.json
uv run arms-evaluate logs/RUN/model_2999.pt --seed 30000 --camera-trace-in artifacts/camera.npz --output artifacts/replayed.json
```

The evaluator loads only the actor and never allocates an optimizer. It records
successes, episode outcomes, Wilson intervals, completion times, and camera joint
travel. Each environment contributes only its first episode per batch to avoid
oversampling fast successes. Replay requires matching task, control rate, batch
size, and horizon. Use a separately trained `--memory none` actor as well as memory
reset at test time; resetting a trained GRU is a distribution-shift intervention.
See [docs/STUDY.md](docs/STUDY.md) for interpretation and limitations.

## Performance and implementation

- MjLab provides Warp batched physics, camera rendering, and CUDA graph execution.
- Plug uses batched tensor IK once per control step, compiled on CUDA; the other
  tasks use joint-space control. Neither learning step path copies state to the CPU.
- Only requested views render, once per control step; no depth or policy shadows.
  Plug enables a small workbench texture; HD observers only render in diagnostics.
- Images remain uint8 in observation and rollout storage, converted inside the CNN.
  Two 96×72 views × 256 environments × 24 steps occupy 255 MB as uint8 versus
  1.02 GB as float32, excluding overhead and other state.
- Compact CNNs with spatial softmax feed a 128-unit GRU; padded recurrent images
  are skipped by the encoder. Per-environment randomization is vectorized.
- `arms-benchmark --actor` reports measured throughput after warmup. Torch memory
  excludes Warp allocations; check total VRAM separately. Benchmark numbers here
  must not be presented as training throughput or as H100/A100/RTX measurements.

Artifacts, known limitations, and commands actually executed are tracked in
[PROGRESS.md](PROGRESS.md). CPU checks do not certify CUDA graph execution or GPU
contact capacity; run the supplied GPU validation job before long training runs.

## Plug insertion reward

Fresh plug runs default to `--reward-profile plug_insertion`, ported from
`active_perception_arms/src/active_perception_arms/tasks/insertion/mdp/rewards.py`
in the neighboring repository. With position error `d` in meters, the per-step
reward is `-log(1 + 100*d)`, replaced by exactly `1000` when `d < 0.002`.
There are no lift/alignment bonuses, time costs, action costs or timestep scaling.
The implementation reads current physical qpos and the variant-corrected goal.
With optional wrist yaw, shaping also includes the source's quaternion-distance
term against identity; the reward bonus still tests position alone.

The reward's instantaneous bonus matches the source. This project's success
termination and evaluation still require three consecutive physical samples
(and the existing orientation tolerance when yaw is enabled); the bonus can
therefore be paid before held success. This ports the reward, not the source's
single-sample termination rule. Both absolute and incremental controls support it.
For example, add `--reward-profile plug_insertion` to a fresh training command.

Saved runs preserve their recorded reward profile on resume, and changing it
requires a fresh run. Historical `plug_staged`, `progress` and `legacy_log_hold`
profiles remain available for reproduction. `legacy_log_hold` is a different
objective: it adds a held-success bonus and time/action costs to log shaping.
The frozen precision-study launcher reproduces its original `plug_staged` plan.

## Plug repair: optional yaw and the recovered feedforward recipe

`--plug-wrist-yaw` adds a fourth manipulator action: absolute yaw about world Z,
bounded to ±30°, with a 3°/control-step rate limit. Camera actions follow the four
manipulator actions. The inspection pause also holds yaw. Success in this mode
requires current physical position error <2 mm and orientation error <2° for
three consecutive samples. Its `legacy_log_hold` reward includes an orientation displacement
cost using a 30 mm lever arm. Existing checkpoints retain their original action
and observation dimensions; enabling yaw requires a fresh run.

`--policy-profile legacy_ff` selects feedforward actor and critic MLPs with widths
256–256–128, std 1.0, adaptive LR 0.001, entropy 0.005, five PPO epochs and four
minibatches. It defaults to `--memory none`. Pair it explicitly with
`--reward-profile legacy_log_hold` to test the recovered log objective. The
profile retains current geometry, separate image encoders, horizon and physical
hold scoring, so it is not an exact reproduction of the old pl3 experiments.

```bash
.venv/bin/python -m active_perception_arms.train --task plug \
  --condition wrist_static --fixed-view 7 --policy-profile legacy_ff \
  --reward-profile legacy_log_hold --plug-wrist-yaw \
  --num-envs 512 --iterations 751 --dry-run
```

The [repair report](artifacts/plug_repair/index.html) records the legacy reward
comparison, the native XM contact-path experiment, implementation checks, and
the bounded training pair. The [training report](artifacts/training_results/index.html)
contains completed camera comparisons and hyperparameter-search results.

### Transfer reward and environment revision

New transfer experiments default to `task_revision=pick_place_v2` and
`reward_profile=transfer_staged`. Object position/velocity and TCP kinematics
use the current integrated state. A successful episode must lift the cube at
least 25 mm while both jaws contact it, then place it within 18 mm horizontally
and 6 mm vertically of the goal. The cube must touch the cubby floor, have no
load-bearing arm contact, move below 35 mm/s and 0.5 rad/s, and remain more than
45 mm from the TCP for five consecutive samples. The cube's existing torsional
and rolling friction coefficients are enabled for floor contact in this revision.

The potential follows reach → grasp/lift → transport → release/withdraw.
Reward is `3 * (0.99 * next_potential - previous_potential) + 25 * success`,
minus 0.005 per step and 0.0005 times mean squared manipulator action.
Terminal potential is zero, including at the finite-horizon timeout.
Pickup history is privileged critic/reward state, never actor input.

Old saved transfer configurations retain v1 geometry/contact settings, scoring,
reward, and observation schema. A revision or reward change requires a fresh
run. The [transfer audit](artifacts/transfer_repair/index.html) records physical
rollouts, failure cases, contact changes, and camera coverage. Scripted success
and two-update PPO checks are separate from learned-policy performance.

### Push targets around the object

Fresh `push_v2` experiments now default to `push_target_layout=arc_v2`.
After sampling the block position, the target is placed relative to that block:
angle uniform from **−80° to +80° about world +Y** (positive angles turn toward
+X), and independent radius uniform from **50 to 120 mm**. The block keeps its
configured XY jitter; `randomize=False` uses angle 0° and radius 85 mm.
The green marker is **20 × 20 mm**, down from 52 × 52 mm. Success requires the
**block center within 5 mm XY of the target center**, plus the existing height,
speed, contact-free withdrawal and three-consecutive-sample requirements.
The marker is a center-location cue, not a region that must contain the whole block.
Reward withdrawal credit and physical evaluation audits use the same tolerance.

Use `--task push --task-revision push_v2 --manip-control cartesian
--reward-profile push_dense --push-target-layout arc_v2 --reset-xy-jitter 0.02`
for this recipe. `square_v1` explicitly selects the historical square sampling,
52 mm marker and 15 mm tolerance. Saved experiments without the new field load
as `square_v1`; resume inherits the saved layout and rejects a requested change.
Frozen studies and their evaluations retain their original definitions.

The [nine-reset preview](artifacts/push_rgb/arc_initial_conditions/target-initial-conditions-grid.png)
hides the optical crossbar only in its diagnostic observer view so target positions
are readable. Simulation/policy cameras retain it when `crossbar_v3` is enabled.
`arc_v1` retains the earlier 50–70 mm range for saved experiments.
At 50–120 mm, a static IK screen finds some far targets outside the configured
Cartesian action bounds or unreachable with the current arm pose. This does not
certify collision-free paths or learned task feasibility. These changes alone
do not establish that wrist-only vision is insufficient. No training was launched.

For `arc_v2` with ±20 mm block jitter and the fixed crossbar, the fixed camera
saw the target marker (at least 3 segmentation pixels at 96×72) in **68/512**
sampled resets. The target center was within its field of view in all 512; scene
occlusion prevented seeing the marker in most cases. See the
[fixed-camera grid](artifacts/push_rgb/arc_initial_conditions/fixed-camera-nine-resets.png)
and [raw audit](artifacts/push_rgb/arc_initial_conditions/fixed-camera-visibility.json).
Some far side targets are outside the current manipulator action range or static IK
reach, so this wider recipe is not yet validated for a new training campaign.

## Plug v2: diagnosis and redesigned active-perception task (branch `active-perception-redesign`)

[docs/DIAGNOSIS_2026-10-13.md](docs/DIAGNOSIS_2026-10-13.md) explains why end-to-end
plug training stayed near 10% and why the movable camera showed no benefit. In short:
the ported +1000-per-step reward with a 3-step hold makes dithering more valuable than
finishing (a privileged-state policy that knows the goal learns 0% deterministic
success), the 2 mm sphere sits 2 mm above the seated pose so a fully inserted plug
never counts, unbounded Gaussian actions saturated 84-100% of camera commands, and
the original scene gives one fixed camera-arm pose 100% of the variant information.

`--task-revision hidden_prongs_v2` keeps the four hidden-prong variants and the
current-qpos physics, and changes the rules: seated success (prongs engaged >= 8 mm,
body inside the hole clearance, 3 samples), optional jam failure when a prong is
pressed onto the solid plate (blocks blind probing), the bounded `plug_seat_v2`
reward, a 4 s manipulation budget, `--plug-reset-scale`, and the training-only
`--plug-assist-fraction` curriculum (evaluation always uses the original resets).
Sensing options: `--plug-screen gap_v1` (optical screen with a per-episode gap
between the camera arm and the task), `--camera-action lookat_v2` (absolute camera
position + look-at point), `--aux-loss-coef` (auxiliary variant/goal/socket head on
the actor latent from a stored privileged group, never an actor input) and the
training-only `--camera-visibility-reward`. Saved v1 configurations are unchanged.

```bash
.venv/bin/python -m active_perception_arms.train --task plug --condition active \
  --task-revision hidden_prongs_v2 --plug-screen gap_v1 --camera-action lookat_v2 \
  --action-distribution squashed --aux-loss-coef 0.5 --camera-visibility-reward 0.02 \
  --plug-assist-fraction 0.3 --num-envs 1024 --iterations 2035 --dry-run
# Cluster array (wrist / wrist_static / initial / active x 3 seeds):
sbatch --account=ACCOUNT --partition=PARTITION scripts/slurm/plug_v2_study.sbatch
# CPU learnability diagnostic with full state (threaded native MuJoCo, no rendering):
.venv/bin/python scripts/train_native_privileged.py --revision v2 --plug-assist-fraction 0.3 \
  --output artifacts/diagnosis/privileged_v2_assist
```

## Longer delta-control and memory study

The [long delta + GRU report](artifacts/long_delta_study_20261013/index.html) tracks
fresh fixed/active policies for both tasks, with three seeds per condition. Every
policy uses 5 mm manipulator target increments, a 128-unit GRU, matched calibrated
camera starts, and camera target increments updated every four control steps.
The pushing scene removes the stale partition while retaining the arc targets,
optical crossbar, and physical success criterion.

Validation milestones are 5M, 10M, and 20M transitions, rounded up to whole PPO
updates. Separate state-informed GRU actors diagnose control/reward learnability.
After 20M, the review extends both camera conditions of a task to 50M only if
mean validation success gains at least two percentage points or terminal error
drops at least 10% between 10M and 20M. This decision never reads test scores.
The initial arrays use at most eight GPUs; any extension uses at most two while
the six curriculum branches can run. Historical studies are preserved.

The plug curriculum starts with YM, 2 mm jitter and an 8-second episode, then
adds all variants, full jitter, dynamic occlusion, and the original 3.5-second
horizon. Each stage requires 80% stage-validation success and stops after at most
4M transitions if it cannot advance. Weights and normalization transfer between
stages with a fresh optimizer/exploration std; resumes within a stage restore the
optimizer. Curriculum scores are separate from evaluations on all four variants,
8 mm jitter, the original horizon and horizon-scaled sweep timing. Test splits,
physical audits, per-variant scores and action diagnostics are recorded.

```bash
# Existing plan: inspect or refresh without submitting duplicate jobs.
.venv/bin/python scripts/run_long_delta_study.py report \
  --plan artifacts/long_delta_study_20261013/plan.json

# A new destination is required for an independent study.
.venv/bin/python scripts/run_long_delta_study.py prepare --output artifacts/NEW_STUDY
.venv/bin/python scripts/run_long_delta_study.py submit --plan artifacts/NEW_STUDY/plan.json
```

All worker code and assets are hash-pinned under each study's `source/`. Creating
`suspension.json` in its output directory prevents further training/evaluation
commands and extension submissions; cancel that study's queued/running Slurm jobs
to stop a command already in progress. Preserve `state-20M.json` and earlier test
files when extending. New evaluation flags `--plug-variant all`,
`--episode-seconds`, and `--reset-xy-jitter` explicitly test a curriculum checkpoint
on the original task without changing its action mapping.
