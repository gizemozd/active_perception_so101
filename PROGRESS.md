# Implementation progress

## 2026-09-30 — scope and inspection

- Starting from an empty repository on macOS/Apple Silicon.
- Implement three tasks: pregrasped plug insertion, bin-to-cubby transfer, and occluded planar pushing.
- Implement conventional sensing, initial repositioning, scheduled camera movement, and adaptive camera control; keep recurrent memory consistent.
- Use MuJoCo Warp through MjLab for GPU simulation and policy training entry points. Create Slurm scripts only; do not run or submit training.
- Validate the same scene assets locally with native MuJoCo, scripted policies, camera images, and tests. GPU-specific checks will be clearly distinguished from checks actually executed here.
- Investigating current MjLab APIs and available robot assets before selecting pinned dependencies.

## Original implementation checklist (completed below)

- Scene/robot assets and task definitions.
- MjLab environment and recurrent visual policy integration.
- Camera modes, randomization, rewards, and success criteria.
- Scripted sanity rollouts and visibility diagnostics.
- Tests, speed-oriented configuration, cluster scripts, and usage documentation.

## 2026-10-01 — implementation and local validation

- Vendored the user's SO-101 assets, meshes, license, and provenance. Reused the
  analytical kinematics utility; no runtime dependency on the old project.
- Built common scene definitions for plug insertion, contact grasp/transfer, and
  pushing. Integrated them into MjLab manager-based environments and native MuJoCo
  diagnostics. All six sensing/control modes share timing, camera intrinsics,
  randomization, and task criteria.
- Fixed a pregrasp weld anchor that launched the plug. Moved a tray wall and enlarged
  the cubby to clear the actual gripper/camera housing. Fixed the pushing diagnostic's
  retreat so success remains stable after withdrawal.
- Clean native scripted rollouts pass all three tasks (10 seconds each), with videos,
  images, segmentation counts, object/goal state, and finite-state checks saved under
  `artifacts/sanity`. Phase-occlusion rollouts also generated for all three tasks.
- Audited wrist images: transfer and pushing objects are visible; insertion housing
  blocks the pin. The fixed reference view can be blocked by the arms. This is not
  the searched-static baseline; selection is deliberately left to validation runs.
- Pinned a compatible stack and lockfile. Latest Warp produced an upstream CPU
  kernel compile error; Warp 1.13.0 compiles and steps this MuJoCo Warp revision.
- Executed actual MjLab/Warp CPU reset/step with two environments and RGB cameras.
  Observations are finite, image tensors are uint8, and subset resets preserve the
  other environment. CUDA tests skip explicitly on this Apple Silicon host.
- Added compact uint8 rollout storage and recurrent visual actor, plus feedforward
  ablation. Tested real RSL fragmented recurrent rollout batches and backward image
  gradients without any optimizer step. Fixed missing RSL logger config defaults.
- Added training CLI with dry-run, inference-only benchmark, held-out checkpoint
  evaluator, interventions (camera freeze, external image hold, memory reset, replay),
  and validation-only static-view selection. Added Slurm training arrays for each
  task, GPU validation, throughput sweep, static search, and evaluation.
- No learning loop, optimizer update, sbatch submission, or hardware control has run.
  Scripted controller tuning and diagnostic forward/backward checks are the only
  policy-related execution so far.

## Cluster work intentionally not executed

- GPU validation, viewpoint search, trained-policy outcomes, and GPU throughput
  remain for the cluster; they are not results of this implementation session.
- Full scripted Warp videos and final test/lint checks are completed below.

### Cross-backend check and correction

- Full phase-occlusion rollouts in **actual MjLab/Warp on CPU** succeeded for plug
  and transfer. Bare-jaw pushing succeeded in native MuJoCo but failed in Warp;
  preserved its failed report in `artifacts/debug/push_bare_jaw_warp_failure.json`.
- Diagnosed initial jaw/object overlap and large off-center pushing impulses.
  Moved the initial hand back, bounded scripted approach speed, and added a visible
  physical pushing tip below the closed gripper. Updated both scene backends and
  documentation. The corrected Warp controller reaches stable placement without
  object attachment or teleportation. Regenerating final pushing artifacts.
- The optical intervention panel now lies on the front camera sightline. Native
  segmentation audits quantify its image coverage; wrist sensing remains useful
  for grasping/pushing. Native and Warp use different renderers and independent RNG
  implementations, so their visual appearance and same-number reset seeds need not
  match pixel-for-pixel.
- Local CPU inference benchmark: 2 environments, two 96×72 RGB views, 3 warmup and
  10 timed steps, **14.87 environment-steps/s**, saved in `artifacts/benchmark_cpu.json`.
  This is a small local diagnostic, not a GPU measurement or a training-throughput
  estimate. Cluster scripts cover a GPU environment-count sweep.
- Added finite-horizon handling, equal manipulation action penalty across action
  dimensions, run metadata, resume configuration checks, and Slurm signal checkpointing.

## Final verification

- Final suite: **53 passed, 3 skipped** in 26.29 seconds. Skips are the explicitly
  marked NVIDIA CUDA checks. Upstream Torch JIT deprecation warnings are recorded.
  `ruff check`, `ruff format --check`, and `uv lock --check` all pass.
- All three tasks finish successfully in the final native clean/phase rollouts and
  the actual MjLab/Warp CPU phase rollouts. The latter use the real RGB sensor outputs
  rather than substituting native images. Videos, frame galleries, and reports are
  linked from `artifacts/README.md`.
- Saved a separate overhead fixed-camera Warp image to verify the static camera
  construction path. The reference view in the three-column native audit is not a
  searched or selected baseline.
- The tests cover physical scripted feasibility, camera timing/freeze, partial reset,
  observation shapes/dtypes, hidden-state reset, recurrent padding/backward, unchanged
  actor weights, terminal success across auto-reset, checkpoint evaluation/replay,
  validation-only viewpoint selection, all 18 task/condition dry runs, shell syntax,
  Slurm dispatch, and equal transition budgets.
- CUDA performance and correctness still need the supplied GPU validation/benchmark
  jobs. No camera-policy success comparison or best-static-view claim can be made
  until the user runs the training/search/evaluation jobs. Nothing was submitted.

- Randomized native diagnostics: all nine task × seed checks (three tasks, seeds
  0/1/2, random occlusion, randomized object/fixture XY) reached and retained success.
  This small feasibility check is saved in `artifacts/randomized_script_checks.json`;
  it is not a learned-policy or statistical generalization result.
