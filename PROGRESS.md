# Implementation progress

Latest resume guide: [Agent handoff](docs/HANDOFF_2026-10-07.md); read the October 9
entry at the end of this document before acting on its historical queue snapshots.

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

## 2026-10-06 — literature review and research positioning

- Read all four user-provided papers through their methods, experiments,
  limitations, and appendices; checked key table/diagram pages visually. Saved
  page-specific findings and a proposed experiment sequence in
  `docs/LITERATURE_POSITIONING.md`, linked from the study protocol.
- Identified direct prior coverage of spare-arm vision (EFM/BAP), learned gaze
  (EyeRobot), random-time occlusion and camera-attributable visibility (BAVO),
  and precise manipulation with fixed stereo plus fixation (EyeRobot 2.0).
- Checked additional primary sources including ActiveArena, TAVIS, SaPaVe, and
  ActiveScale. Memory, VLA integration, and task-conditional comparisons are also
  established; the memo makes no field-wide novelty claim.
- Recommended first restoring the original hidden-prong task and testing searched
  fixed sensing, initial inspection/history, prescribed scans, and adaptive motion.
  Proposed separating unchanged occluded state from genuinely stale information
  before expanding contact tasks or adding a pretrained-policy comparison.
- This was research/documentation work only. No environment changes, policy
  training, cluster submission, or hardware execution. User PDFs remain untracked.

## 2026-10-07 — original hidden-prong plug restored and rendered

- Ported the original four hidden-offset variants, two-prong/two-hole geometry,
  holder collar, equalized inertial properties, spawn distributions, offset-based
  2 mm success target, compliant grasp and jaw contacts. Restored the north camera
  base and the original 3D Cartesian / 5D gimbal controls. The marker-screening
  harness retains its historical centered-pin geometry separately.
- Added batched tensor IK once per control step, CUDA compilation hooks, balanced
  per-world mesh variants and revision checks that reject retired plug checkpoints.
  Fixed MjLab 1.4's original-spec IDs remaining -1 on the variant-loading path by
  compiling the original scene before entity initialization. Regression tests
  verify body/mocap IDs, actual per-world prong positions and subset resets.
- Kept study settings explicitly documented: 25 Hz, current 48.46-degree FOV,
  128x96 plug images, common 1-second inspection pause plus 2.5-second manipulation,
  three-step success hold and the current reward. These are not a bit-for-bit
  legacy training reproduction. Default plug occlusion is clean. Slurm dry runs
  use these defaults and the task-specific 26-view grid, including low fixed views.
- Improved white-arm presentation with plastic/brass finishes, bench texture,
  fixture fasteners, lighting and background. Added a 1080p task close-up alongside
  the outside overview. Policy shadows remain disabled in native and Warp RGB.
- Recorded all four variants using native physics and a balanced four-world
  MjLab/Warp CPU rollout. All 4/4 succeed on both backends; final Warp errors are
  0.22–0.26 mm. Twelve additional native runs over three randomized paired reset
  seeds also succeed. Scripts use privileged goals and a blind camera schedule;
  these results establish mechanical feasibility, not learned sensing advantage.
- Saved 17 videos under `artifacts/plug_restoration`, decoded their first/middle/
  last frames and verified dimensions. Inspected overview, close-up, native
  matched-view and actual Warp RGB galleries. Reports, commands and video links
  are in `docs/PLUG_RESTORATION.md` and the artifact directory's README.
- Preserved the visibility caveats: the wrist has a small direct `xp` cue at the
  matched reset (17 differing native RGB pixels / 9 Warp pixels), while the other
  three wrist images match. Low fixed view 7 exposes all variants at that pose.
  Best-static and initial-inspection-plus-memory success remain open comparisons.
- Validation: `pytest -q` **81 passed, 3 skipped** (CUDA unavailable on this Mac);
  Ruff checks and formatting pass. Test output is in `artifacts/tests.log`.
  GPU throughput/CUDA execution remain for the supplied cluster validation jobs.
- No learning, cluster submission or physical robot execution. Legacy files are
  unchanged. The supplied literature PDFs remain outside the commits.

## 2026-10-07 — authorized cluster pilot study (in progress)

- Scope: restored plug only; disposable GPU validation/benchmarks and four seed-0
  pilots authorized. No transfer/push training or larger study submission.
- Read restoration/study notes and every existing Slurm script. Verified indices
  `0,6,9,15` map to wrist, wrist_static, initial, active with seed 0; fixed view 7
  remains a diagnostic candidate, not a success-selected baseline.
- Direct cluster access is available from `holy8a24101`; login identity is
  `pgozdil`. `sacctmgr -nP show assoc user=pgozdil
  format=Cluster,Account,User,Partition,QOS,DefaultQOS` confirms account
  `kempner_pgozdil_lab`; `scontrol show partition kempner_rtx` permits that account.
  The user-supplied cluster identifier is the account, not this login username.
- Partition inventory: 24 nodes, 8 advertised RTX PRO 6000 Blackwell Server
  Edition GPUs/node, 128 CPUs and 1,547,204 MiB configured RAM/node; one node
  drained at inspection. Resource availability is dynamic. Single-GPU jobs use
  8 CPUs and 48 GiB host RAM initially.
- Submitted disposable hardware inventory job `51111068` using
  `sbatch --account=kempner_pgozdil_lab --partition=kempner_rtx --gres=gpu:1`.
  Its raw output is `logs/slurm/inventory-51111068.out` (ignored).
- No project virtualenv existed on this cluster checkout. Running
  `uv sync --locked` to install the pinned stack. Existing W&B login file exists;
  no credentials copied into source or output. Logging verification is pending.
- Inventory job completed successfully: RTX PRO 6000 Blackwell Server Edition,
  **97,887 MiB VRAM**, driver **610.57.04**, on `holygpu7c1713`. Compact inventory:
  `artifacts/cluster_pilot/hardware.json`. This is hardware validation only.
- Existing W&B credentials authenticate successfully through the API as `pgozdil`,
  entity `pgozdil-harvard-university`. No local project setting exists; use
  `active-perception-so101`. Dashboard (runs pending):
  https://wandb.ai/pgozdil-harvard-university/active-perception-so101
- A server restart interrupted dependency installation before completion. Checked
  Slurm and the filesystem before resuming; inventory was not submitted twice.
- Implementing synchronized rollout/PPO timing, device-wide sampled VRAM and
  utilization, transition-indexed W&B/local metrics, bounded checkpoints and W&B
  resume identity. Additional GPU gates cover all four conditions at 128×96,
  balanced offsets, initial-only freeze, and actor/critic observation separation.
  These additions are not yet execution-validated and no optimizer has run.
- Preparation commit `79a47ec` pushed to `origin/main`. Lint, Python compilation
  and Slurm shell syntax checks pass; execution checks await dependency setup.
- Latest availability snapshot is saved in
  `artifacts/cluster_pilot/resource_snapshot.json`. Most usable nodes have all
  eight GPUs allocated; free GPUs observed on different nodes. Same-node
  co-residency testing depends on later availability, not nominal node size.
- Installing pinned dependencies required roughly 11 minutes of CUDA downloads,
  followed by copying from the home cache to the project filesystem (hardlinks
  cannot cross these filesystems). This is setup cost, not training throughput.
- `uv sync --locked` completed (Torch 2.10.0+cu128, MjLab 1.4.0,
  MuJoCo 3.9.0, MuJoCo Warp 3.8.1/pinned Git revision, Warp 1.13.0,
  RSL-RL 5.2.0, W&B 0.30.0).
- W&B upload/readback verification **passed**: preflight metric value 1 was
  retrieved using `wandb.Api()` after finishing the run. This run performed no
  training: https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/v7c2xe0f
  Receipt: `artifacts/cluster_pilot/logging_preflight.json`.
- Submitted GPU validation job **51113803** with
  `sbatch --parsable --account=kempner_pgozdil_lab --partition=kempner_rtx scripts/slurm/validate.sbatch`.
  Running on `holygpu7c1713`; includes supplied CUDA/CPU regression tests,
  full-resolution rendered actor inference, four-variant physical rollouts and
  explicit actor-state/initial-freeze GPU checks. Training remains gated on this job.
- Validation job `51113803` stopped at the regression-test gate: **53 passed,
  1 failed** (268.42 s). The plug CUDA path's Triton extension could not compile:
  `/usr/include/python3.12/Python.h` is absent from the system interpreter selected
  by uv. Transfer/push CUDA smoke tests passed; no training or inference benchmark
  started. Remedy in progress: provision a Python 3.12 runtime with development
  headers and rebase the project environment while preserving installed packages.
- An overlapping diagnostic `srun --jobid` inherited this interactive allocation's
  `SLURM_STEPMGR`, so it could not attach; SSH to the compute node also lacks a
  trusted host-key entry. Neither was needed to retrieve the complete failure from
  shared Slurm output. No SSH trust checks were disabled.
- Fixed the missing-header failure without reinstalling the large dependency set:
  `uv python install 3.12` selected managed Python **3.12.13**;
  `uv venv --allow-existing --managed-python --python 3.12 .venv && uv sync --locked`
  preserved all 142 packages. Verified `Python.h` and Torch imports. Cluster setup
  now checks headers before selecting an interpreter. Fix committed as `3ab35f7`.
- Submitted replacement validation job **51114857**, running on `holygpu7c1934`.
  Read-only attachment works with `env -u SLURM_STEPMGR srun --jobid=51114857
  --overlap --ntasks=1 --cpus-per-task=1 --nodelist=holygpu7c1934 ...`.
  The initial CPU test phase is active; no training has started.
- Local CLI regression checks: **29 passed** in 44.07 s; warnings are upstream
  Torch JIT deprecations. Final benchmark/pilot decisions remain pending CUDA gates.
- Replacement validation **51114857 completed successfully** in 5m44s:
  **54 regression tests passed**; actual 128×96 actor-camera checks passed for all
  four conditions; initial-only camera targets freeze after one second and actor
  groups exclude critic/privileged state. All four CUDA scripted variants reach
  and retain success, with final errors 0.22–0.26 mm. This is mechanical/pipeline
  feasibility, not learned-policy evidence. Reports: `artifacts/cluster_pilot/gpu_pipeline.json`
  and `artifacts/cluster_pilot/validation-51114857/warp/warp_report.json`.
- Compute-node W&B metric upload independently read back through the API:
  https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/z8hsr5ol
  The 32-world smoke measured **493.09 transitions/s for inference only**,
  800 timed transitions / 1.622 s, plus 39.97 s warmup; it is not PPO throughput.
- Submitted first disposable benchmark **51116045**, array cell 0 (wrist, N=64):
  `sbatch --account=kempner_pgozdil_lab --partition=kempner_rtx --array=0 scripts/slurm/benchmark.sbatch`.
  It runs inference and then 12 PPO iterations (first 3 excluded as warmup).
  Remaining benchmark cells and all pilots wait on successful PPO logging checks.

## 2026-10-07 13:40 EDT — resumed from first cluster stage

Direct cluster inspection on `holy8a24101` confirms the first project submission
was **51111068**, hardware inventory only. No project training/benchmark is now
running or pending. Unrelated interactive allocations were left untouched.
The accounting reconstruction is saved in
[accounting_reconstruction.json](artifacts/cluster_pilot/accounting_reconstruction.json).
`scontrol show job` no longer retains the four completed jobs; `sacct`, stdout,
metadata, artifacts and W&B API provide their retained evidence. Accounting was
queried from September 30 in two-day windows (site rejects a week-wide query).

| Job/task | Purpose and verified progress | Submitted/start (EDT) | Elapsed / exit | Node | Next action |
|---|---|---|---|---|---|
| 51111068 | Hardware inventory; completed, no environment or optimizer | 12:13:28 / same | 4s / 0:0 | holygpu7c1713 | Reuse hardware.json |
| 51113803 | GPU regression validation; failed at missing Python.h, 53 pass/1 fail | 12:37:41 / 12:37:42 | 4m53s / 1:0 | holygpu7c1713 | Already superseded by header fix and validation below |
| 51114857 | Validation completed: 54 tests, rendered actor checks for four conditions, four scripted prong successes, initial freeze and no privileged actor state | 12:46:16 / same | 5m44s / 0:0 | holygpu7c1934 | Reuse; no physics/policy changes pending |
| 51116045_0 | Disposable wrist N=64 inference completed (6400 measured transitions / 6.856s = 933.53/s); PPO failed before first rollout/update | 12:52:56 / 12:52:57 | 2m43s / 1:0 | holygpu7c2316 | Reuse inference; repair logger and run missing 12 PPO iterations |

All allocated one RTX PRO 6000 Blackwell Server Edition GPU in `kempner_rtx`,
account `kempner_pgozdil_lab` (verified independently with `sacctmgr`). GPU has
97,887 MiB VRAM, driver 610.57.04. Jobs used 8 CPUs; validation requested 32 GiB,
inventory/benchmark 48 GiB. Partition advertises 24 × 8 GPUs; actual availability
is dynamic and most devices are occupied. No eight-GPU policy scaling assumed.

The working directory is `/n/holylabs/pgozdil_lab/Lab/active-perception/active_perception_so101`.
The supplied Mac path is not mounted and this account has no `~/.ssh/config`;
its current revision/uncommitted changes cannot be inspected from this session.
The adjacent **legacy** cluster checkout `../active_perception_arms` is a different
repository at `b6cd299`, with untracked historical Slurm logs, left unchanged.
Current restored checkout was at `9d323df` (remote HEAD `3ab35f7`). Preserved the
previous agent's uncommitted telemetry compatibility fix, test, curve exporter,
validation reports, inference report and progress notes.

Revisions by submission/reflog: inventory `bc17e5c` plus then-uncommitted setup;
failed validation `2a9bb7e`; successful validation `3ab35f7`; benchmark runtime
explicitly records `9d323df`. Historical dirty snapshots beyond retained runtime
metadata are unknown. Validation entry point is `scripts/slurm/validate.sbatch`;
benchmark entry point is `scripts/slurm/benchmark.sbatch` array 0, invoking
`active_perception_arms.benchmark --task plug --condition wrist --fixed-view 0
--num-envs 64 --actor --wandb`, then `active_perception_arms.train` with GRU,
clean, seed 0, 12 iterations × 24 × 64 = **18,432 total PPO transitions**.
Task revision `hidden_prongs_v1`, images 128×96, shared 1s pause. No PPO checkpoint,
iteration records or PPO W&B identity exists for that failed attempt; verified
progress is zero PPO transitions. Original runtime/experiment/runner files remain
under `logs/benchmarks/plug_wrist_clean_gru_v0_s0_n64_20261007T165523Z`.
Slurm outputs remain in `logs/slurm/{inventory,vision-validate,plug-benchmark}-JOB.out`
(with `_0` for benchmark); no separate stderr file for these merged-output scripts.

Environment: `.venv`, managed Python 3.12.13 (failed validation used system
3.12.14 without headers); Torch 2.10.0+cu128, CUDA 12.8, MjLab 1.4.0,
MuJoCo 3.9.0, MuJoCo Warp 3.8.1, Warp 1.13.0, RSL-RL 5.2.0, W&B 0.30.0.
The specific PPO failure is RSL's removed `wandb.Settings(start_method="thread")`.
The inherited compatibility adapter removes that setting while retaining native
RSL metrics. Its regression and CLI tests pass: **31 passed**.

W&B API confirms only three finished runs so far, no pilot or PPO training run:
[login preflight](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/v7c2xe0f),
[validation inference](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/z8hsr5ol),
[wrist N64 inference](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/my5q07yk).
All metrics reached W&B; no offline/unsynchronized runs found. Added an explicit
`SKIP_INFERENCE=1` switch to reuse successful inference when repairing only PPO.

- Missing PPO attempt **51121081_0** was held before launch with
  `user_env_retrieval_failed_requeued_held` (no log/artifact/W&B run). Local
  `man sbatch` confirms `--export=ALL,KEY=value` implicitly retrieves the login
  environment. Bash startup executes Zsh; the retrieval failure is observed,
  with that shell behavior a plausible cause. Cancelled this held, unstarted job
  and submitted using command-environment variables and plain `--export=ALL`,
  also clearing inherited `SLURM_STEPMGR`. No completed PPO work repeated.

- Replacement **51121191_0 completed, exit 0:0, 2m28s**, on holygpu7c1731.
  Command: `env -u SLURM_STEPMGR SKIP_INFERENCE=1 sbatch --parsable
  --account=kempner_pgozdil_lab --partition=kempner_rtx --array=0 --export=ALL
  scripts/slurm/benchmark.sbatch`. All **12 PPO iterations / 18,432 transitions**
  completed; final checkpoint `model_11.pt` in
  `logs/benchmarks/plug_wrist_clean_gru_v0_s0_n64_20261007T174021Z`.
  [W&B spmopn8m](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/spmopn8m)
  API readback verifies cumulative transitions, iteration 12, PPO loss and reward.
  Receipt: `artifacts/cluster_pilot/ppo_logging_verified.json`.
  Measured N64 wrist training: **809.63 transitions/s**, mean rollout 1.756s,
  PPO 0.133s, end-to-end iteration 1.897s; warmup 38.53s separately; sampled
  peak device memory 2,081 MiB. Nine steady iterations, a short noisy benchmark.
  Scientific pilots remain unstarted. Remaining 15 benchmark cells may now run
  as independent one-GPU jobs with concurrency capped at four; existing cell 0
  counts as completed and will not be resubmitted.
- Submitted remaining benchmark array **51121584**, tasks **1–15%4**, command
  `env -u SLURM_STEPMGR sbatch --parsable --account=kempner_pgozdil_lab
  --partition=kempner_rtx --array=1-15%4 --export=ALL scripts/slurm/benchmark.sbatch`.
  Code `9f6d7fe`; cell 0 excluded because its PPO result is already valid.
  Each task runs actor inference and exactly 12 disposable PPO iterations.

## 2026-10-07 13:55 EDT — benchmark sweep completed; pilot count selected

All **16 condition/count cells completed** (N64 wrist repaired job 51121191;
remaining array 51121584 tasks 1–15). Each performed 12 real PPO iterations,
first three excluded as warmup. No pilot budget has been consumed.
[Measured comparison](artifacts/cluster_pilot/BENCHMARKS.md),
[CSV](artifacts/cluster_pilot/benchmark_comparison.csv),
[full current inventory](artifacts/cluster_pilot/run_inventory.json),
[Warp compile/load totals](artifacts/cluster_pilot/compilation_timings.json).
Machine-readable per-run JSON contains environment initialization, warmup,
measured transitions/time, phase timing, device-wide memory/utilization, software,
allocation, revision and W&B URL. Warp-reported module loads are extracted;
Torch/Triton compilation and graph capture are included in initialization/warmup,
not claimed as separately isolated measurements. Per-job caches are independent;
PPO follows inference in the same job and usually benefits from its cached kernels.

Choose **512 environments for all four pilots**, giving **100 iterations × 24 ×
512 = 1,228,800 transitions** each. It fits every condition with at most 10,827 MiB
sampled VRAM (97,887 available), and is fastest among tested counts. At N512:
wrist 4512/s, wrist_static 3827/s, initial 3456/s, active 3408/s. Steady-only
extrapolated pilot times are 4.54, 5.35, 5.93 and 6.01 minutes, respectively;
full 18,432,000-transition runs are 1.135, 1.338, 1.482 and 1.502 GPU-hours.
Pilot batch estimate at four-way concurrency: about 8–10 minutes including cold
startup, excluding queues and evaluation; aggregate steady training ≈0.364 GPU-h.
Short steady measurements span only nine iterations, with roughly 10–13% iteration
CV; extrapolations are not confidence intervals or full-study measurements.

Found and tested a second logging compatibility issue: MjLab emits `log={}`
between resets, but RSL 5.2 reads metric keys only from its first buffered entry.
Thus benchmark episode success metrics were silently dropped, although reward,
length, optimization metrics and all timings were logged. Those missing histories
cannot be reconstructed and are not reported as zero. Retain these runs as valid
throughput measurements. A small adapter discards empty episode dictionaries
before RSL aggregation. The regression verifies a reset after an initially empty
step logs the actual success metric. **2 telemetry tests pass**. Installed this
adapter only after all benchmark jobs ended; no running learner was changed.

Prepared a distinct three-run same-node concurrency diagnostic, not a repeated
scientific pilot: 12 iterations of active/GRU/N512 alone, then two independent
single-GPU processes on that same two-GPU allocation. It also verifies corrected
success logging on the cluster before pilot submission. Tags distinguish reference
and paired runs, checkpoints stay under logs/benchmarks/colocation. No distributed
learner is involved. Prepared full-study budget in study_budget.json, **unsubmitted**.
- Submitted **51122956**, `scripts/slurm/benchmark_colocation.sbatch`, with
  `env -u SLURM_STEPMGR NUM_ENVS=512 sbatch --parsable
  --account=kempner_pgozdil_lab --partition=kempner_rtx --export=ALL ...`.
  This reserves two GPUs on one node; each training process still uses one GPU.
  Code `815974e`, corrected episode telemetry enabled. Pilots wait for its
  reference run's episode-success metric to reach W&B.
- Corrected episode success/termination metrics are now verified through W&B API:
  [g7dnp174](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/g7dnp174).
  The 12-iteration reference completed; paired processes continue. Receipt:
  `artifacts/cluster_pilot/episode_logging_verified.json`. Training-success value
  is measured zero in this short disposable run, not missing. Pilot gate is open.
- Fixed the inventory reporter's array association: match `JobIDRaw`, not a shared
  array-parent ID (Slurm assigns that ID to the last task). The underlying run
  metadata was correct; only the initial generated report associated the last
  task with too many rows. Regenerated the report without this ambiguity.

## 2026-10-07 14:00 EDT — four authorized scientific pilots submitted

Array **51123326**, indices **0,6,9,15%4**, code **77a9b9a**. Exact command:

```bash
env -u SLURM_STEPMGR NUM_ENVS=512 TOTAL_STEPS=1228800 FIXED_VIEW=7 \
  OCCLUSION=clean MEMORY=gru JOB_TYPE=pilot LOG_ROOT=logs/pilots \
  RESULT_ROOT=artifacts/cluster_pilot \
  sbatch --parsable --account=kempner_pgozdil_lab --partition=kempner_rtx \
  --array=0,6,9,15%4 --export=ALL scripts/slurm/train_plug.sbatch
```

Mapping verified from `common.sh`: 0=wrist, 6=wrist_static/view7, 9=initial,
15=active; all seed 0. These are the first scientific pilots, each total 100 PPO
iterations / 1,228,800 transitions, no resume or prior pilot work. Each gets one
GPU, 8 CPUs, 48 GiB RAM; no distributed learning. Same hidden_prongs_v1 task,
balanced variants, 128×96/FOV, GRU, clean occlusion, reward/control/horizon and
one-second manipulation pause. Final checkpoint is `model_99.pt`. Leave healthy
jobs running; do not resubmit these cells. Evaluate each once with existing
`evaluate.sbatch`, 512 validation episodes, N128, seeds 10000–10003, videos enabled.
- All four pilots began at **14:00:03 EDT**: wrist on holygpu7c1931,
  wrist_static on holygpu7c2110, initial and active on separate GPUs of
  holygpu7c2316. No pilot restart or configuration change.
- Co-location audit caught a Slurm resource mistake: although shell processes
  were launched concurrently, each step inherited all 96 GiB host memory.
  `sacct` proves steps .0 and .1 ran sequentially (13:56:37–13:59:20,
  13:59:20–14:00:55); .2 started afterward. The filenames `paired_a`/`paired_b`
  do **not** establish concurrency. Preserve all three as serialized reference
  measurements, not paired evidence. Added explicit `srun --mem=48G` and a
  paired-only phase to fill this specific missing check without repeating the
  reference. Existing running job 51122956 continues unchanged.
- Corrected paired-only co-location job **51123535**, dependent on successful
  completion of 51122956 and pinned to the same node holygpu7c1916. Command uses
  `COLOCATION_PHASE=paired NUM_ENVS=512`, `--dependency=afterok:51122956`,
  `--nodelist=holygpu7c1916`; two separate 48 GiB / one-GPU steps. Verify overlap
  and distinct GPU UUIDs before interpreting its results.
- Saved [pilot_manifest.json](artifacts/cluster_pilot/pilot_manifest.json) with
  unique training directories, raw Slurm IDs, W&B identities and final checkpoints.
  Submitted exactly one dependent evaluation per pilot using existing
  `evaluate.sbatch`, `--dependency=afterok:TRAIN_JOB`, `RECORD_VIDEO=1`,
  `EPISODES=512 NUM_ENVS=128 SPLIT=validation` and each `model_99.pt`:
  active **51123808**, initial **51123809**, wrist **51123810**, wrist_static
  **51123811**. Pending dependency is expected; do not duplicate these evaluations.
  Outputs `artifacts/cluster_pilot/evaluation-CONDITION-s0.json`; video directories
  use the same stem plus `-videos`. Validation W&B identity will be in each report.
- Genuine simultaneous pair **51123535.0/.1** both started 14:02:52 EDT, each
  cpu=8, mem=48G, gpu=1. Distinct device UUIDs verified from each sampler:
  GPU-0653b6dd-37a2-7ba9-9264-e4c1846feace and
  GPU-c53d2c6c-5e28-831b-45a1-df64c1e14044. CUDA ordinal 0 is process-local.
- Co-location comparison saved in
  [colocation_comparison.json](artifacts/cluster_pilot/colocation_comparison.json).
  Serialized active/N512 rates: 3349, 3364, 3374 transitions/s. Genuine simultaneous
  rates: **3447 and 3329/s**, +2.9% and −0.6% against the first reference; combined
  6776/s. No clear contention penalty in this short check, not a long-run scaling
  guarantee. Both co-location jobs completed successfully; paired step overlap and
  different GPU UUIDs are verified. Other users' background workloads uncontrolled.
- Learning-curve interpretation: native RSL `Episode_Metrics/success_rate` averages
  reset batches, not all completed episodes weighted equally. It can spike when
  only a few successful episodes finish in an iteration, and the compact exporter
  carries the last logged value between resets. Label it a training metric, not
  held-out overall success. Final evaluation counts each of 512 episodes once.

## 2026-10-07 — pilot evaluations and video-replay diagnosis

All scientific pilots completed 100 iterations / 1,228,800 transitions. Checkpoints
`model_99.pt` and complete monotonic histories verified; matched configuration and
actor observation groups audited in `matched_protocol_audit.json`. Application
wall times (before W&B finish) are wrist 333.5s, wrist_static 424.5s, initial 457.4s,
active 457.1s. No resumed or repeated training iterations.

The original matched 512-episode validation scores are wrist **3**, wrist_static
**7**, initial **3**, active **2** successes. All reports have 128 episodes per
variant. These sparse seed-0 outcomes do not rank sensing methods reliably.

Wrist evaluation/video job 51123810 succeeded. Jobs **51123808, 51123809,
51123811** completed numerical evaluation and uploaded its metrics, then failed
in separate video replay with `AssertionError: Validation replay differs`.
Preserve original JSON scores and W&B runs; evaluation itself did not fail.
Replay assertions failed for active at batch 1, initial at batch 2, wrist_static
at batch 0. The precise source of trajectory divergence is not yet isolated;
do not claim bitwise reproducibility from seeds alone or label divergent replay
videos as successful. Partial videos remain in original ignored directories.

Added optional in-evaluation representative capture: buffer one batch's actual
RGB inputs/actions/physics state, retain first observed success/failure per
variant, then render saved trajectories without a second physics rollout.
This changes diagnostics only, not trained policies or environment behavior.
Plan a **distinct recorded diagnostic repeat** of all four original 512-episode
validations, with identical seeds/batch size. Original scores will not be replaced
or cherry-picked. New reports/W&B identities identify this repeat and link the
original reports. Saved trace NPZs and videos stay out of Git. The repeat also
captures initial-policy camera targets to verify their post-inspection freeze.
- Captured diagnostic repeats submitted after the evaluation regression passed:
  **51125573 active**, **51125575 initial**, **51125576 wrist**, **51125577
  wrist_static**. Code `0c94372`; existing `evaluate.sbatch`, `RECORD_CAPTURE=1`,
  `RECORD_VIDEO=1`, same final checkpoints, 512 episodes, N128, validation seeds
  10000–10003. New `capture-evaluation-CONDITION-s0.json` reports link original
  evaluations. These perform no optimizer updates and consume no pilot training
  budget. Record jobs in pilot_manifest.json; do not rerun the pilot learners.

## 2026-10-07 — final handoff for this authorized stage

[Start here: results and resume index](artifacts/cluster_pilot/README.md).
[Every relevant job's final status](artifacts/cluster_pilot/RUNS.md),
[full inventory](artifacts/cluster_pilot/run_inventory.json),
[benchmark table](artifacts/cluster_pilot/BENCHMARKS.md),
[pilot table and all W&B links](artifacts/cluster_pilot/PILOTS.md).

- **No project jobs remain pending/running.** All four pilots completed, four
  captured diagnostic evaluation/video jobs completed, all benchmarks completed.
  Historical failures retain their real Slurm states. Interactive allocations
  and the legacy checkout were left untouched. The Mac path was not available.
- Pilots each have exactly 100 completed iterations, 1,228,800 transitions, seed 0,
  N512, GRU, clean occlusion, balanced hidden_prongs_v1, shared one-second pause.
  Final model_99.pt and checkpoints at iterations 0 and 50 are retained locally
  and in W&B artifacts. No pilot was resumed, duplicated, or given extra budget.
- Original validation scores (512 each): wrist 3, wrist_static/view7 7, initial 3,
  active 2. Per-variant results, completion time, and camera travel are in PILOTS.md.
  Application GPU-hours including W&B finish: **0.4699 total**; Slurm allocation:
  **0.4803 total**. Scientific pilot batch ran roughly eight minutes, then evaluation.
- Recorded diagnostic repeats: wrist 3, wrist_static 4, initial 0, active 3.
  Changed episode outcomes: 2, 7, 3, 5, respectively. The source of individual
  repeat divergence remains unresolved; no claims of exact seeded reproducibility,
  condition ranking, convergence, or active perception being useless are warranted.
  Original scores and W&B identities remain preserved; repeats are distinct runs.
- **40 captured videos / 20 trajectories** verified by decoding first/middle/last
  frames, frame counts and 25 fps; all outside videos 1920×1080. Actual policy RGB
  is captured directly. Reviewed the outside/policy gallery visually. Initial-only
  camera targets change by exactly **0** after the one-second inspection in every
  captured initial trajectory. Existing CUDA checks independently verify actor
  state excludes privileged offsets/timers and all four variant mechanics succeed.
- Original initial-only episode 213's successful replay completed its batch-1
  assertions before the batch-2 failure. Recovered that verified clip and uploaded
  it to the original validation run **il8drljq**, preserving its identity and score.
  Its recorded diagnostic repeat had no successes; this is explicitly stated.
- Diagnostic evidence: all PPO losses finite; actor image-encoder weights changed;
  mean reward in final 25 iterations is about 1.02–1.19 versus −0.05–0.03 in the
  first 25. Selected failures often remain 10–30 mm from the goal; raw manipulation
  command clipping is rare. Learning has not demonstrated a plateau. Native RSL
  success metrics average reset batches and are not episode-weighted success.
- Some successful clips have sampled pre-action errors above 2 mm. The installed
  MjLab termination path documents one-physics-substep lag in derived state, whereas
  saved frame errors are post-forward samples. Exact termination-time error/hold
  deserves a targeted audit; the observed score flags are retained without relabeling.
- **Next experiment recommendation:** isolate reset/replay and termination-time
  sensitivity with saved initial states/actions and success-time errors, then
  consider a larger matched continuation of these four checkpoints. Do not spend
  the full search budget before this measurement issue and learning progress are
  understood. No additional learning or full matrix was submitted.
- Larger-study scripts are ready in docs/CLUSTER_PILOTS.md and existing Slurm files.
  Pilot-throughput estimate: 87-policy core = **1,603,584,000 transitions**, **118.2
  GPU-hours**; ideal four-GPU elapsed 29.5h, staged batches ≈31.3h, plus startup,
  evaluation and queues. Optional 12-policy ablations ≈19.0 GPU-h, using proxy
  rates. Eight-GPU ideal ≈14.8h is conditional on allocation, not an availability
  claim. Full budget includes 26 fixed views × three seeds, selected on validation.
- W&B API readback verified every scientific training run finished at iteration
  100 / 1,228,800 transitions, and both original/repeat validation metrics arrived.
  No offline/unsynchronized run remains. Dashboard:
  https://wandb.ai/pgozdil-harvard-university/active-perception-so101
- Verification: inherited logger/CLI checks 31 passed; episode telemetry regression
  2 passed; evaluator regression 1 passed; actual GPU benchmark/pilot/recorded-video
  execution verified above. Final lint/format/shell checks are recorded below.
  Checkpoints, raw Slurm/W&B logs, MP4s and trajectory NPZs remain outside Git;
  compact JSON/CSV, plots, source, and handoff documentation are versioned.
- Final checks: Ruff passes; all 46 Python files formatted; every Slurm script and
  common.sh passes bash syntax; git diff whitespace check passes after standardizing
  generated comparison CSV line endings to LF. Recorded-media validation verified
  40 videos and 20 captured trajectories. No project jobs remain in squeue.


## 2026-10-07 — throughput follow-up (in progress)

User asked about 20–30k steps/s, optimization, multi-GPU, and whether success
comparisons are premature. Queue was empty before this follow-up. Completed
scientific pilots and their configurations remain unchanged; no extra pilot work.

- Current training is one GPU per policy, independent concurrent runs; no
  distributed PPO launch. The adjacent legacy checkout has a retained pl3 launch
  requesting four GPUs × 1,024 envs/GPU at 96px; that log failed with SIGBUS and
  contains no verified throughput. Its benchmark script sweeps up to 8,192 envs.
- Legacy insertion config uses timestep 0.0044 and five physics substeps/control
  step; restored plug uses 0.002 and twenty substeps (25 Hz), with 128×96 RGB.
  Thus historical steps/s needs workload, GPU count, and units checked.
- Submitted disposable PPO array **51136002**, indices 0–7%4, all four conditions
  × N=1024,2048, 12 iterations/cell, three warmup. Existing benchmark script,
  BENCH_COUNTS='1024 2048', SKIP_INFERENCE=1, WANDB_RUN_GROUP=plug-scaling-20261007.
  Code e6c80c7 at initial launch; no task/policy changes. Single GPU, 8 CPUs,
  48 GiB per cell, kempner_rtx/account kempner_pgozdil_lab.
- Submitted **51136215**, a disposable active/GRU N512 rollout phase profile,
  code cda6176. First measures 100 uninstrumented actor/env control steps after
  25 warmup, then 100 instrumented steps using CUDA-event spans and host timings.
  No learning. Separate W&B benchmark-profile run; output profile-active-512.json.
  CUDA spans include host scheduling gaps and are not pure kernel utilization.
- Relative policy success is not established: only one seed, 100 updates, rare
  successes, and recorded repeats changing episode outcomes. Current results are
  diagnostics, not a policy ranking or evidence against active perception.

- Initial N2048 wrist result: 10,491 transitions/s, peak 30,607 MiB.
  Added all-four-condition N4096 disposable sweep (12 iterations, three warmup),
  same unchanged task/PPO configuration, to locate the scaling/memory limit.

- N4096 all-condition sweep is array **51136318**, indices 0–3%4; 12 PPO
  iterations/cell = 1,179,648 disposable transitions each (884,736 measured after
  warmup), same one-GPU resources. This is not additional pilot training.
- Phase-profile job **51136215** completed successfully, W&B **4hij4szn**:
  https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/4hij4szn
  Uninstrumented actor/env: 51,200 transitions/10.503s = 4,874.8/s. Instrumented
  CUDA spans/control step: physics 63.18ms, rendering 30.63ms, forward 3.04ms,
  action/IK 2.33ms, actor 2.18ms, resets 1.11ms amortized. Physics+render dominate;
  compiling reset IK is not the main opportunity in this measurement.
- Job **51136461** measures N4096 active simulation without rendering, simulation
  with rendering (both zero-action), then actor+environment inference, sequentially
  on one GPU. Source 3e15cb1. These deliberately exclude PPO and have distinct W&B
  types/labels; actual PPO throughput comes only from the training sweeps.
- Larger N changes PPO rollout batch and update frequency at fixed transitions.
  A performance win does not establish equal learning efficiency. N2048 gives
  25 updates per 1,228,800 transitions, versus the pilots' 100 at N512; N4096
  does not divide either proposed scientific transition budget into 24-step
  integer iterations. Do not silently change existing pilot configuration.


### Throughput follow-up completed

- All **14** new Slurm allocations completed with exit **0:0**, no failed/resumed
  cells; **16** W&B runs read back as finished with expected metrics/transitions.
  No project jobs remain in squeue. [Report](artifacts/cluster_pilot/THROUGHPUT.md),
  [plot](artifacts/cluster_pilot/throughput_scaling.png),
  [machine-readable results](artifacts/cluster_pilot/throughput_followup.json),
  [W&B verification](artifacts/cluster_pilot/scaling_wandb_verified.json).
- Training transitions/s at N512 → N2048 → N4096: wrist 4512 → 10491 → 12162;
  wrist_static/view7 3827 → 7169 → 7902; initial 3456 → 6266 → 6845;
  active 3408 → 6223 → 6941. All one GPU, identical restored physics/vision/GRU.
  N2048 uses 29.9–39.4 GiB, N4096 58.6–77.7 GiB. Practical scaling point: N2048;
  doubling again adds only 9–16%. Existing pilots remain N512.
- N4096 active zero-action simulation: **32,168.7 transitions/s**; with cameras:
  **12,325.5/s**; untrained actor+env: **10,464.1/s**. Each measured 409,600
  transitions after 25 warmup steps. PPO active: **6,940.8/s**, 884,736 measured
  transitions after warmup. These are separate workloads, not additive ablations.
- Twelve new PPO benchmarks used 8,257,536 disposable transitions, 6,193,152 after
  warmup, **0.573 application GPU-hours** including W&B finish. Per-cell init,
  warmup, rollout/update times, GPU memory, variation and extrapolations retained
  in JSON/CSV. GPU utilization averages include startup; do not label them steady.
- N2048 offers 1.8–2.3× measured throughput improvement, but only 25 PPO updates
  per pilot-sized transition budget versus 100 at N512. A conditional 87-policy
  N2048 budget is 62.1 GPU-h / ideal 15.5h on four GPUs, excluding overheads;
  this has not been established as learning-equivalent and is not submitted.
- Optimization priority: investigate physics/collision/solver and camera renderer
  kernels while preserving task fidelity. Reset IK and actor inference were small
  shares. Existing CUDA graphs/compiled IK already enabled. Larger timesteps or
  fewer solver iterations require mechanical revalidation, not silent adoption.
- Scientific interpretation unchanged: no supported success ranking from one seed,
  100 updates and rare, repeat-sensitive successes. Resolve measurement issues,
  then consider longer matched training. No scientific budget was extended.
- Verified no diff in task/environment/config/policy/training source versus 4e3b8ff.
  Added configurable benchmark grids, explicit simulation/inference labels, phase
  profiler, plotting and inventory support. Actual GPU execution verified all paths;
  Ruff/format/bash syntax and whitespace checks pass. No new dependency needed.
- Follow-up accounting totals **0.719 allocated GPU-hours** including the
  phase profile and three simulator/inference diagnostics. Checked all 14 job
  exit codes, all 12 PPO budgets (12 iterations each), and 16 W&B final states.
- Final artifact check normalized four GPU CSVs to LF and made the sampler
  emit LF for future runs; all final whitespace checks pass.


## 2026-10-07 — authorized longer matched continuation

User requested longer training jobs. Continue only the same four seed-0 restored
plug pilots, using the previously planned **18,432,000 total transitions/policy**
(1,500 total PPO iterations at N512). Each already completed 1,228,800 transitions /
100 updates; **remaining: 17,203,200 transitions / 1,400 updates each**. Keep N512,
GRU, clean occlusion, fixed view 7 for wrist_static, 128×96, 25 Hz, 3.5s episodes
and shared one-second pause. Do not use the performance sweep to change this batch.
No fixed-camera search, extra seeds, transfer/push or larger matrix authorized here.

- Before submission: squeue empty, accounting has no later matching submissions;
  all four model_99.pt files verified with iteration 99, optimizer state and 100
  contiguous logged iterations. Same W&B IDs verified finished at 1,228,800.
  [Preflight/checkpoint hashes/configs](artifacts/cluster_continuation/preflight.json).
- Resume fixes: native RSL restores Adam LR but leaves PPO.learning_rate at its
  constructor value. Restore that scalar from optimizer state; resume at index 100
  (next logged iteration 101). W&B train_cfg now permits an explicitly resumed
  budget/save-interval update. New LR and budget regression tests pass.
- Preserve original per-run metadata, GPU samples and history under
  logs/pilots/RUN/resume_history/TIMESTAMP before reusing a log directory. Append
  iteration history; include segment job ID, segment wall time and cumulative wall
  time. Original compact pilot results/checkpoints stay intact; continuation results
  go to artifacts/cluster_continuation. Same W&B identities and names.
- Native checkpoint semantics: model/Adam/iteration/common_step_counter restored;
  simulator episode state, RNG and GRU hidden state are reinitialized. This is a
  checkpoint continuation, not bitwise continuation of interrupted trajectories.
- Inventory now separates per-job histories/budgets in a shared resumed directory;
  archived summaries remain associated with their original job. Evaluation captures
  are labeled repeats only when a reference report is supplied.
- Verification: telemetry/CLI 35 passed, evaluator roundtrip 1 passed; Ruff/format
  and whitespace checks pass. Dry-run dispatch confirms all four exact saved
  experiment configs and total budgets. Estimated remaining training: wrist 67.6m,
  wrist_static 75.4m, initial 88.6m, active 88.4m (~5.33 GPU-h total), excluding
  startup/queues/evaluation and possible throughput changes as policies learn.


### Longer jobs running — startup verified 2026-10-07 19:57 UTC

| Condition | Training array task | Node | Evaluation (Dependency) | W&B |
|---|---|---|---|---|
| wrist | 51152806_0 | holygpu7c1710 | 51153083 | [a7ex2ycm](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/a7ex2ycm) |
| wrist_static | 51153491_6 | holygpu7c1731 | 51153495 | [zj5uevwm](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/zj5uevwm) |
| initial | 51153482_9 | holygpu7c1731 | 51153487 | [i116qwyn](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/i116qwyn) |
| active | 51153446_15 | holygpu7c1713 | 51153468 | [u1evdury](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/u1evdury) |

- All four RUNNING; actual one-GPU Blackwell allocations verified. Wrist was
  submitted first; W&B received resumed iteration 105 / 1,290,240 transitions
  before the remaining three submissions. All four original IDs now report
  RUNNING, budget 18,432,000 and fresh post-100 metrics. Checkpoint SHA256 hashes
  unchanged; restored adaptive learning rates match saved Adam state; local
  histories are contiguous, first resumed update 101 / 1,241,088 transitions.
- Four dependency evaluations submitted with the existing evaluate.sbatch:
  512 episodes each, N128, validation seeds 10000–10003, actual-trajectory capture
  and outside/policy videos. New result paths preserve old pilot evaluations.
  Evaluation IDs are pending for Dependency, not observed running.
- [Continuation handoff](artifacts/cluster_continuation/README.md),
  [manifest/commands](artifacts/cluster_continuation/manifest.json),
  [startup readback](artifacts/cluster_continuation/startup_verified.json),
  [inventory](artifacts/cluster_continuation/run_inventory.json).
- Added explicit --jobs to inventory: date-window sacct omitted dependency-pending
  evaluations; direct job accounting correctly includes all eight submissions.
- No longer-run results are claimed yet. Continue monitoring these jobs, never
  submit duplicates. Verify final checkpoint index 1499 and total 18,432,000
  transitions before treating a training job as complete. Core source revision
  04de34f; remaining changes are inventory/handoff only.


### 2026-10-07 — optimization diagnosis and intermediate validation

User clarified earlier successful runs are outside this repository; exact historical
run/configuration remains unidentified. Adjacent legacy defaults are reference only.
[Diagnosis, candidate controlled search, validation and video links](artifacts/cluster_continuation/OPTIMIZATION.md).

- Current wrist/static adaptive LR is at 1e-5 for the latest 100 sampled updates;
  some action-axis std values have fallen from 0.4 to ~0.0016. This is evidence
  for investigating LR/exploration, not proof of a hyperparameter cause.
- Intermediate validation jobs 51162174 (wrist) and 51162175 (static/view7)
  COMPLETED 0:0, 126/129 seconds allocation elapsed. Each evaluated model_750
  at 9,228,288 transitions on the same balanced 512 episodes, N128. Success:
  14/512 and 44/512, respectively; concentrated in yp and xp variants.
- W&B API verified finished runs 1q7q9tso/qendmi3r, exact metrics and 12 videos
  each. Compact reports/manifests and captured video metadata retained.
- Training success is reset-batch averaged, not episode-weighted; do not treat
  it as interchangeable with validation success. All four long training jobs
  remain RUNNING and their four final evaluations PENDING (Dependency).
- No training config changed; no HP-search or extra-seed training launched.


### 2026-10-07 — authorized hyperparameter search launched

User requested the hyperparameter search. [Protocol and live run links](artifacts/hparam_search/README.md).
Reuse original wrist_static/view7 seed0 model_99 as baseline; no duplicate control.
Three fresh matched N512/GRU/clean runs, each100 updates /1,228,800 transitions:

| Case | LR / schedule | Entropy | Training | Dependent evaluation |
|---|---|---:|---|---|
| fixed_lr | 1e-4 fixed | .003 | 51164843_6 | 51165404 |
| entropy | 3e-4 adaptive | .01 | 51165286_6 | 51165405 |
| both | 1e-4 fixed | .01 | 51165289_6 | 51165406 |

- All three RUNNING; W&B API verified fresh metrics and diagnostics under separate
  IDs/group plug-hparam-20261007. First run's metrics verified before other submissions.
- Training source25a005e frozen in sibling worktree plug-hparam-20261007; shared
  pinned .venv, explicit PYTHONPATH. Runtime configurations exactly match baseline
  except declared optimization settings; no geometry/model/reward changes.
- One Blackwell GPU/job,8CPUs,48GiB, kempner_rtx/account kempner_pgozdil_lab.
- Added validated CLI overrides, exact-resume optimization inheritance/guards,
  KL/clip fraction/per-axis std/action saturation and episode-weighted diagnostics.
 38 CLI/telemetry tests +2 real PPO update-preservation tests passed.
- Three afterok evaluations PENDING (Dependency):512 episodes,N128,seeds10000–10003,
  balanced variants,captured policy/outside videos. W&B evaluation source14d091d
  changes only group/tag selection; existing pilot evaluation defaults preserved.
- Existing four long pilots and their four final evaluations unchanged: training
  51152806_0,51153491_6,51153482_9,51153446_15 RUNNING; evaluations
  51153083,51153495,51153487,51153468 PENDING (Dependency).
- Search manifests are exclusive submission reservations; inspect them/accounting
  before any retry. Inventory now includes explicit requested IDs regardless of
  job name and associates search evaluation manifests. No extra seeds launched.
- Next: verify model_99,100 contiguous updates,total1,228,800 and validation results.
  Early screening may be inconclusive; do not infer a winner from reward alone.
  Promising cases can continue to matched9,228,288 total transitions, preserving
  their IDs and counting already completed work. No such extensions submitted yet.


### 2026-10-07 21:36 UTC — committed handoff and completed search screen

[Next-agent handoff](docs/HANDOFF_2026-10-07.md) records access, frozen code, all active
and completed job IDs, budgets, checkpoints, W&B links, resume commands, failure
recovery and ordered next steps. Model clarification recorded: CNN/proprioception
→GRU128→MLP(128,128)→actions; no architecture change was made.

- Search training51164843_6,51165286_6,51165289_6 all COMPLETED0:0; verified
  model_99 iter99,100 contiguous updates,1,228,800 transitions and finished W&B runs.
- Evaluation51165404 failed before environment initialization with GPU uncorrectable
  ECC on holygpu7c2313. Preserved failure; retried only evaluation as51168824, same
  checkpoint/settings/reserved W&B ID, excluding that node. Retry COMPLETED0:0
  on holygpu7c1716. No training rerun, no GPU reset or changes to other users' jobs.
- Final512-episode validation: fixed_lr1/512, entropy7/512, both5/512; baseline7/512
  (capture repeat4/512). No demonstrated improvement or reliable winner at this
  short single-seed budget. W&B metrics and videos for all three verified synced.
- [Results and links](artifacts/hparam_search/README.md),
  [comparison CSV](artifacts/hparam_search/comparison.csv),
  [learning curves](artifacts/hparam_search/learning_curves.csv),
  [completion verification](artifacts/hparam_search/completion_verified.json).
- Refreshed both inventories. Four long pilots still RUNNING; four final evaluations
  still PENDING (Dependency). No additional training, seeds or matrix submitted.
- Inventory read encountered a partially appended JSONL line; fixed it to read only
  complete newline-terminated records, retaining errors for malformed interior data.
  Ruff and live inventory checks pass. No training source or configuration changed.
- Next agent: collect already-queued final pilot evaluations; inspect search failures
  and diagnostics; decide whether to continue search cases to9,228,288 TOTAL
  transitions (651 additional updates from model_99), preserving their IDs/configs.
  Plan seed replication/full study afterward; do not submit the full matrix.


### 2026-10-07 21:46 UTC — complete conversation and tuning follow-up handoff

User requested that everything be consolidated into the handoff. Expanded
[HANDOFF_2026-10-07.md](docs/HANDOFF_2026-10-07.md) with current job/progress snapshot,
complete tuning results and W&B/video links, the exact second-stage recommendation,
resume/evaluation instructions, model/observation/action specifications, sensing
semantics, GRU placement, throughput findings, scope limits and unresolved historical
run provenance. Hyperparameter tuning is explicitly UNFINISHED; first screen complete.

- [Prepared next-stage plan](artifacts/hparam_search/next_stage_plan.json): all three
  candidates to9,228,288 TOTAL transitions /751 updates, resuming model_99. Each
  needs651 additional updates /7,999,488 transitions; all three23,998,464 additional.
  Same W&B IDs, saved settings, N512 and frozen25a005e source. New summary/evaluation
  paths preserve first-stage results; compare to existing baseline model_750 (44/512).
- All three resume dry-runs passed experiment/configuration compatibility and the
 751-update total. The plan records exact environment/commands and expected checkpoints.
  Status PREPARED_NOT_SUBMITTED: no additional training or evaluations submitted.
- Snapshot: four original long pilots still RUNNING; final evaluations still PENDING
  (Dependency). Refreshed continuation inventory with latest verified progress.
- Later tuning should evaluate at matched budgets, inspect learning/trajectory evidence,
  then replicate promising settings across seeds. No clear winner from the first screen.
  Full matrix and additional seed training remain unsubmitted.


## 2026-10-09 — approved training audit, repair and HTML report

User approved the staged plan, parallel work, and committing/pushing validated
changes. Start with plug training quality; other task training is conditional on
reliable learning and evaluation. Dynamic occlusion implementation may proceed
independently. No full condition/viewpoint/seed matrix is authorized by a stale
handoff instruction alone.

- Slurm confirms all four longer training jobs and four final evaluations
  COMPLETED, exit 0:0. Each final model_1499 has iteration 1499, 1500 unique contiguous
  updates, exactly 18,432,000 transitions, and finished W&B metrics. See
  artifacts/plug_analysis/training_audit.json and refreshed continuation inventory.
- All four reward curves increased then meet the disclosed descriptive final 300
  plateau rule. Reliable task learning FAILS: wrist 3/512, fixed-view7 59/512,
  initial 185/512, active 141/512; every policy has xm 0/128. These seed 0 scores are
  exploratory validation. Initial targets freeze after 1 s while external images
  continue refreshing; this is not a snapshot-only policy. No demonstrated benefit
  of continued active motion over initial view selection/live sensing with memory.
- Existing search candidates resumed, preserving source 25a005e, W&B identities,
  checkpoint/optimizer/settings/N512 and 751 TOTAL updates/9,228,288 transitions.
  Each adds 651 updates/7,999,488 transitions. New exclusive manifests are under
  artifacts/hparam_search/continuation. fixed_lr training 51499309 / evaluation 51499310,
  entropy 51499497 / evaluation 51499498, both 51499505 / evaluation 51499506. The first candidate's
  update 102 and W&B readback were verified before the other two submissions.
  Evaluation code is frozen separately at 12795fd in ../plug-eval-20261009;
  no code under an active training run was edited.
- Read-only success/repeat audit 51499848 used frozen 12795fd plus hashed
  instrumentation (artifacts/plug_analysis/termination_submission.json), 512
  episodes×2 repeats×initial/active, no optimizer updates. Initial scores 181/187,
  active 151/140. Between repeats, initial state AND actor-image hashes match for
  all 512 episodes, yet 86 initial / 125 active episode outcomes differ. This supports
  numerical trajectory variability, not different initial resets, and prevents
  assuming exact per-episode repeatability from seeds.
- Every flagged success passes the ORIGINAL derived-state three-sample 2 mm rule.
  However terminal CURRENT qpos errors exceed 2 mm in 28/181 and 40/187 initial,
  30/151 and 31/140 active successes. Upstream substep lag therefore affects
  physical-state interpretation. Original results remain intact; a separately
  labeled current-qpos criterion is being implemented/tested for corrected runs.
- User pointed to ../active_perception_arms. Recovered EIGHT actual pl3 saved W&B
  configs/summaries and final checkpoints, not merely current repository defaults.
  Source commit c179c896 verifies -log(100d+1) reward and instantaneous 1000 bonus;
  saved architecture is feedforward 256/256/128, std 1, LR 0.001 adaptive,
  entropy 0.005, 5 epochs / 4 minibatches, timestep 0.0044 × 5, 5 s horizon, N1024/rank,
  2000 updates and world_size 4 (196.608M global target,49.152M/rank).
  Saved training success~98–100% is NOT independently reproduced held-out success
  or a matched current-task result. Compact provenance is
  artifacts/plug_analysis/legacy_run_inventory.json; legacy checkout unchanged.
- HTML report: artifacts/plug_analysis/index.html; paper-oriented CSV/LaTeX,
  raw episode/learning-curve exports, standalone plots, actual captured videos,
  provenance and limitations are generated by scripts/build_plug_report.py.
  Browser preview, link checks and video decoding passed; report will be refreshed
  after corrected evaluations and tuning completion. Large raw media/checkpoints
  stay outside Git.
- Dynamic world-space optical sweeps are opt-in world_sweep_v1. Legacy panel modes
  and archived checkpoints retain their semantics. Native/CPU tests cover task
  timing, shared trajectory equations, actor-state separation, partial-reset
  isolation and unchanged physics. CUDA gate and completed native rendered
  diagnostics are collected before any dynamic training.
- NEXT: finish/verify physical-state criterion; CUDA dynamic gate and corrected
  all 4 checkpoint evaluation; collect already queued tuning results and compare
  at 9,228,288 (baseline 44/512), then make evidence-based next training repair.
  Do not launch transfer/push or a large sensing study while plug gates fail.

### Physical criterion, CUDA gate and prepared objective repair

- The current-qpos criterion is implemented and pushed at `5badc6e`. New runs use
  current physical positions at three consecutive control samples; saved old
  experiments inherit `derived_substep`, and resuming cannot silently switch
  criteria. The full CPU suite passed: 123 tests, six GPU tests deselected.
- Corrected evaluation job `51501382` uses frozen `5badc6e` in
  `../plug-corrected-20261009`. Its six CUDA checks passed, including actual RGB,
  moving-panel trajectory parity and partial resets. The same job evaluates all
  four original final checkpoints with the corrected criterion and records their
  actual evaluated trajectories. Outputs are separately named `corrected-*-s0`.
- Dynamic diagnostics contain nine native scripted successes across three seeds
  and tasks with byte-identical clean/dynamic physical trajectories. Plug prongs
  are measurably obstructed; transfer/push wrist visibility remains useful. These
  diagnostics establish feasible optical obstruction, not an active-policy gain.
- Prepared `reward_rescue_plan.json` adds a bounded fresh progress-versus-log
  reward screen only if the running optimization trials remain unreliable.
  `legacy_log_hold` uses current physical distance, a 1,000 bonus after three valid
  samples, and the shared action/time penalties. It changes both shaping and
  bonus strength and is not an exact historical reproduction. Both proposed runs
  have identical seed, view, optimizer, initial std 0.4 and 100-update budget;
  neither has been submitted. CLI reward/std choices inherit saved settings on
  resume and reject explicit changes. All 52 targeted reward/criterion/CLI checks
  pass; lint and formatting pass.

- Corrected final validation is complete: wrist 8/512 (1.56%), fixed view 7
  23/512 (4.49%), initial 147/512 (28.71%), active 102/512 (19.92%). Each has
  xm 0/128; all current-qpos/three-sample/hold audits have zero violations.
  These are separately labeled criterion interventions on the old-trained
  checkpoints, not retraining results or exactly paired trajectories.
- Live optimization trials have finite losses, healthy KL/exploration and rising
  episode-weighted training success. They are not declared failed. Since they
  retain historical training criteria, a fresh corrected-objective screen is
  justified independently: revise the reward plan entry gate to run its two
  100-update cases sequentially on the fourth GPU once corrected evaluation
  finishes. Preserve all running trials, and review screen plus completed tuning
  results before any larger extension. Stage-one cost is 2,457,600 transitions.
- Fresh screen submitted as `51505177`, source frozen at `434c37a` in
  `../plug-reward-20261009`. It runs both profiles sequentially on one GPU,
  two corrected balanced validations each, plus actual first-repeat videos.
  Submission/runtime/profile manifests record exclusive reservations, exact
  source/script/checkpoint hashes, W&B identities and verified transition counts.
  The existing tuning jobs continue independently; the total is four GPUs.
- The completed corrected report has 100 evaluated-trajectory videos plus three
  native occlusion diagnostic videos. All 290 local links and all evaluated
  videos pass validation; the portable ZIP includes media and tables without
  checkpoints or raw NPZ captures. Firefox renders the corrected headline and
  confidence intervals correctly. Report source can ingest the fresh screen
  and completed continuation results without replacing historical measurements.
