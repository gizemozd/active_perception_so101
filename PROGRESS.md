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

## 2026-10-01 — white arms and high-resolution external videos

- Changed both SO-101 arm shells from yellow to white in the common scene builder.
- Added a named `overview` camera framing both arms, the tool/object, and the work
  area from the front left. Overview recording is native 1920×1080 at 25 fps.
- Both sanity commands now save `*_overview.mp4` and `*_overview.png`; pass
  `--no-overview` to omit this optional diagnostic rendering.
- MjLab/Warp overview videos use the current Warp physics state rendered through
  MuJoCo OpenGL. The paired wrist/camera-arm videos remain actual Warp RGB outputs.
  Policy cameras retain their existing observations, FOV, and 96×72 resolution.
- Regenerated the clean/phase native rollouts and phase Warp rollouts for all three
  tasks, including white-arm policy-camera videos and full-HD outside views. All
  nine rollouts reach and retain success. Video encoding and final frames were
  checked at 1920×1080, 25 fps, 10 seconds; results are in
  `artifacts/overview_validation.json`.
- Existing suite after the change: **53 passed, 3 CUDA tests skipped** (29.40 s).
  No training or cluster submission was performed.

## 2026-10-02 — task shortlist and pre-training information screen

- Implemented six separate native MuJoCo visibility prototypes and a reproducible
  screening CLI. The existing three trainable MjLab/Warp environments are unchanged.
  No new prototype is advertised as a validated manipulation environment.
- Searched 418 coarse fixed views, including overhead and all 32 sampled camera-arm
  endpoints, then 104 local refinements chosen using validation data only. Final
  96×72 screen: 24 validation seeds (12000–12023), 48 test seeds (41000–41047).
- Compared wrist, static, wrist+static, oracle initial placement, a predefined phase
  route, repeated two-view scan, and per-query endpoint selection. Explicit state
  epochs distinguish valid memory from a stale observation. Fixed views are not
  restricted to camera-arm reach. Moving-arm wrist visibility is recomputed.
- Replaced a development ray-bin approximation with pixel-center ray counts and
  checked these against actual segmentation. Saved marker-only counterfactual RGB
  pairs and a matched 192×144 sensitivity screen. Development probes remain locally
  archived/ignored and are excluded from the promoted report.
- Changing side access: searched fixed+wrist covers both queries in 8/48 episodes;
  a repeated endpoint scan covers 31/48 and the sampled-endpoint oracle 34/48.
  At 192×144 the scan covers 47/48 and the oracle 48/48. This is not strong evidence
  for an advantage of feedback-driven camera control over scanning.
- Two-site seating: fixed+wrist covers 0/48 at 96×72, while a known phase route and
  the endpoint oracle each cover 40/48. At 192×144 fixed+wrist reaches 22/48 and both
  moving conditions reach 48/48. Resolution and predefined motion explain much of
  the apparent benefit. Open-slot pushing is visible to the wrist in every episode.
- Audited sampled joint paths, camera/environment contacts, hand retreat and
  removal of the housing as an optimistic waiting control. A 45 mm hand lift reveals
  all sampled states; this is a substantive loophole, not hidden or disabled.
  The shortlist therefore focuses on connector/clip states whose actual physical
  mechanics require assessment under maintained engagement, pending validation.
- Rendered six original native camera-servo demonstrations, paired sensor mosaics
  and full-HD external views. Preserved the unstowed two-site failure (five camera
  collision frames and a missed final query). A corrected demonstration stows
  before hand relocation, has no camera collision frames, and observes every phase.
  Phase/feature changes remain prescribed interventions, not manipulation success.
- Wrote `docs/TASK_SCREENING.md` with concrete task definitions, quantitative
  comparisons, competing strategies, mechanical readiness gates and reproduction
  commands. Added the media/results index at `artifacts/task_screening/README.md`.
- Added tests for fresh-state memory invalidation, validation-only view selection,
  stronger scan controls, wrist visibility under camera movement, fixture clearance,
  reachable hand poses, hidden-offset proprioception leakage, stationary two-site
  geometry, joint-limit rejection and ray/segmentation agreement.
- Full suite: **69 passed, 3 CUDA checks skipped** (22.21 s); lint and formatting pass.
  Native rendering was exercised locally. No CUDA performance claim, learned-policy
  comparison, optimizer step, training run, Slurm submission or hardware motion.
- After the camera-stow correction, all **16 screening tests** pass again. Validated
  and decoded the last frame of all 14 videos: overview streams are 1920×1080,
  sensor mosaics are 864×258, and every stream is 25 fps. Current code uses the stow
  correction; the earlier recordings and their failure metadata remain preserved.

## 2026-10-02 — requested videos for each shortlisted prototype

- Freshly rendered shrouded insertion, two-site seating and open-slot pushing into
  `artifacts/task_screening/diagnostics/shortlist/`. Each has a full-HD outside video
  and synchronized wrist/searched-fixed/camera-arm comparison, all at 25 fps.
- Applied the camera-stow sequence to all three recordings. Their durations are
  7.68 s, 9.24 s and 6.00 s respectively; all have finite states, zero detected camera
  collision frames, and visible frames at every query phase.
- Checked all six encoded videos against the recorded simulation frame counts and
  decoded their first, middle and final frames. Outside resolution is 1920×1080;
  sensor mosaics are 864×258, displaying enlarged 96×72 sensor images.
- Added a direct three-video index and reproduction command. Labels explicitly
  identify prescribed query states and scripted camera motion. The clips do not
  claim completed contact-dependent task mechanics or trained-policy success.
- No source-code changes, training or cluster submission were needed.

## 2026-10-05 — correction after inspecting the original plug task

- Reviewed the user-specified legacy checkout's actual plug geometry, reset logic,
  images, task README, later work log and layout memo. The original task has four
  visually identical plug bodies with hidden two-prong offsets of ±15 mm. Correct
  alignment depends on the unknown offset; mass/inertia are equalized to remove
  a previously identified sag cue.
- Identified a substantive mismatch: this checkout's plug is a single centered
  pin, and the later screening prototype queries an independent marker. Raising
  that hand can reveal the marker, whereas raising the original held plug keeps
  its underside hidden relative to the wrist camera. The previous generalized
  retreat verdict did not apply to the original task.
- The later legacy log reports balanced historical success of active 99.5%,
  wrist 80.7%, fixed-at-home 60.1%, and frozen-active 67.1% (3 training seeds,
  400 episodes/cell). These numbers are transcribed from the log; checkpoints and
  raw evaluation records were not present locally, and no evaluation was rerun.
- Preserved the remaining scientific distinction: the legacy memo identifies lower
  fixed views that expose all four variants. The trained best-static search and
  initial inspection plus memory remain unresolved. Freezing is a useful policy
  intervention, not proof that every possible fixed/initial policy must fail.
- Corrected the report, README, study notes and media index. Restored the original
  hidden-prong plug as a primary candidate. No legacy files or environment code
  changed, and no training ran. A faithful port of the original task is still needed.
