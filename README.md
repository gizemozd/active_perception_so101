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

Success requires three consecutive control steps. Plug success uses the original
2 mm body-position tolerance around the offset-corrected goal; transfer and
pushing require placement, low object speed, and hand withdrawal. Plug acquisition
is excluded with a pregrasp weld; transfer uses contact grasping. No teleportation
or attachment assists the transfer/push diagnostic controllers.

All modes receive joint positions, joint velocities, servo targets, fixed camera
calibration, and elapsed time. Object/goal state and occlusion timers are restricted
to the training critic and reward calculation. Policy images are RGB at 25 Hz
with identical 48.46° vertical FOV: 128×96 for plug, 96×72 for transfer/push.
Plug uses three absolute TCP actions with fixed grasp orientation plus five
camera-gimbal deltas when enabled; commanded TCP/gimbal state is also observable.
Transfer/push use bounded joint target increments; transfer additionally controls
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
