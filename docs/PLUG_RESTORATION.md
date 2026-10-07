# Original hidden-prong insertion restored — 2026-10-07

The trainable `plug` task now ports the original four-variant insertion problem
from `old/active_perception_arms`. It replaces the single centered-pin task.
The separate marker/enclosure screening harness keeps its original prototype
geometry; its results must not be attributed to the restored task.

[Videos and images](../artifacts/plug_restoration/README.md) show all four
variants in native MuJoCo and actual MjLab/MuJoCo Warp. No policies were trained
and no cluster jobs were submitted.

## What was restored

| Component | Original specification preserved |
|---|---|
| Exterior | Identical 64 × 42 × 30 mm housing and the original `holder1.stl` collar |
| Hidden variants | `xm`, `xp`, `ym`, `yp`: prong-pair center shifted −15/+15 mm along x or y |
| Prongs | Two 8 × 8 × 12 mm brass prongs; centers separated by 22 mm |
| Socket | 100 × 70 × 14 mm plate with two 12 mm square through-holes, represented by five physical boxes |
| Inertial properties | Same mass, center of mass, inertia and inertial orientation for every variant; 48.80934 g centered reference |
| Spawn | Body XY centered at (0, 25) mm with independent ±50 mm uniform variation; body bottom at z=35 mm |
| Socket reset | Independent XY uniform in [0, 50] mm on each axis; no yaw variation |
| Goal | Body XY = socket XY minus hidden offset; body bottom z=16 mm; position error <2 mm |
| Grasp | TCP offset (0, −1.5, 45) mm; fixed grasp yaw −90°; compliant weld, solref=(0.01, 1); jaw starts at 0.27 rad and squeezes toward 0.24 rad |
| Manipulator | Base (−190, 25, 50) mm; original 3D absolute TCP action box; ≤30 mm command change per axis per control step |
| Camera arm | Base (−20, 500, 50) mm, yaw −1.494 rad; original five-axis gimbal home, range and action increments |
| Contacts | Prongs, socket, jaws and both arms retain mechanical contact; no jaw/object exclusions in the restored task |

Sources in the legacy checkout: `tasks/insertion/specs.py`,
`tasks/insertion/mdp/events.py::_reset_plug`, `tasks/insertion/modes.py`,
`tasks/insertion/mdp/actions.py`, `tasks/insertion/insertion_env_cfg.py`,
and `robots/so101_constants.py`, under `src/active_perception_arms/`.
The legacy checkout was not modified and is not needed at runtime.

The hidden offset enters the critic, reward and success calculation, but not
the actor's state vector. The actor gets RGB and measurable/commanded robot state.
Variant assignment is balanced round-robin across Warp worlds, fixed across
resets, as in the legacy implementation. Use a multiple of four worlds for a
balanced evaluation; training defaults use 256. `plug_variant` selects one
variant for diagnostics, not the main study.

## Explicit differences from legacy training

This restores the task mechanics and sensing problem, not a bit-for-bit
reproduction of the legacy training run. The current study retains:

- 25 Hz control (2 ms physics, 20 substeps), rather than the legacy timing.
- Matched 48.45549° vertical FOV, now 128×96 for plug. Historical trained plug
  runs used 71° and square images. Current transfer/push remain 96×72.
- A one-second manipulation pause for **every** sensing condition, followed by
  the legacy 2.5-second manipulation budget: 3.5 seconds total. The CLI exposes
  `--initial-seconds` and `--episode-seconds` for explicit timing ablations.
- Three consecutive successful control steps; the original success predicate
  used the same 2 mm sphere without this hold requirement.
- This repository's progress-difference shaping, +10 success reward and small
  action cost, rather than legacy logarithmic distance shaping/+1000 success.
- The existing shared visual actor and optional GRU, not the original network
  or historical checkpoints. Plug actions change from the retired prototype's
  joint increments to 3 Cartesian + 5 camera actions; actor state is 51 values.

The `hidden_prongs_v1` revision is saved with each experiment. Evaluation/resume
reject old plug checkpoints without it; use their historical code revision.
Default plug occlusion is clean, because the hand and plug create the original
visibility problem. Artificial panels are optional supplementary experiments.

## Implementation and visuals

Each prong offset is baked into mesh vertices. MjLab's `VariantEntityCfg` carries
the mesh geometry separately for each world; putting offsets only in primitive
positions would silently lose the intended variation. Tests check actual Warp
prong positions, entity IDs, goals and subset resets.

MjLab 1.4's variant builder compiles a copy of the scene. Its entity indexing
still reads IDs from the original attached specs, which otherwise remain −1.
The scene callback now compiles the original once before simulation setup.
Without this fix, resets can silently address the wrong body or mocap. This
workaround is local to scene construction; installed packages are unchanged.

Plug IK uses batched tensor operations once per control step and fixed iteration
counts, with CUDA compilation enabled. It performs no learning-step host copies
or per-physics-substep Jacobian calls. The supplied GPU validation/benchmark jobs
remain necessary: this Mac cannot validate CUDA graphs or H100/A100/RTX throughput.

White arms are retained. Plastic/brass finishes, warm key and cool fill lighting,
a textured workbench, fixture fasteners and a neutral background improve the
presentation. Native 1920×1080 overview and close-up cameras are diagnostic only;
they are never actor inputs. Policy shadows stay disabled in both renderers so
the improved outside-view lighting does not introduce a prong-shadow clue.

## Checks and limits of the result

- All four deterministic variants insert successfully in both native MuJoCo and
  a single four-world MjLab/Warp CPU run, within 3.5 seconds and with the three-step
  success hold. Final Warp position errors are 0.22–0.26 mm.
- Twelve additional native runs (four variants × three paired randomized reset
  seeds) all succeed, recorded individually in `randomized_native.json`.
- Geometry checks show that the offset-corrected pose fits the two holes, while
  merely centering the body on the socket causes physical prong/plate penetration
  and fails the success predicate. The compliant grasp can jam on an abrupt
  descent; the diagnostic script descends at 1.5 mm per control step near insertion.
- Matched free-air actions yield identical joint/object trajectories across
  variants to the tested tolerance, confirming the equalized-inertia safeguard
  in that check. Contact can still reveal geometry and is a legitimate alternative.

The visibility audit deliberately preserves inconvenient findings. At the matched
reset pose, the native wrist image distinguishes `xp` by 17 RGB pixels (9 in the
Warp renderer); `xm`, `ym`, `yp` are pixel-identical. The home external camera
distinguishes `yp`, while the other three match. These are pose-specific checks,
not a proof of indistinguishability throughout an episode.

Low fixed view 7 sees prongs on all four variants at that reset. Its minimum
pairwise difference is 69 native RGB pixels. This is a useful diagnostic view,
**not** the selected best-success baseline. The 26-view training search includes
eight low views at z=8 cm, sixteen higher views, the camera-home reference and
overhead. Static and initial-inspection policies remain serious comparisons.

The scripts use privileged insertion targets and a blind camera schedule. Their
success validates mechanics and backend consistency, not active perception or
learned task success. Whether adaptive motion beats searched static sensing and
initial inspection plus memory remains unanswered.

## Reproduce without training

```bash
uv run pytest -q
uv run python -m active_perception_arms.plug_audit
# Actual batch renderer on a GPU node, without a learner:
uv run python -m active_perception_arms.plug_audit --backend warp --device cuda:0
DRY_RUN=1 SLURM_ARRAY_TASK_ID=17 bash scripts/slurm/train_plug.sbatch
```
