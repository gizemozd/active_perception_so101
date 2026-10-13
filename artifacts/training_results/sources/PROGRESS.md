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

### Recurrent training correction and delayed-credit diagnostic

- The complete CPU suite passes with headless EGL: 137 tests, six CUDA tests
  deselected. The separate completed CUDA gate passed all six checks.
- A read-only RSL 5.2 probe confirms recurrent value bootstrapping consumes the
  next observation, then the next rollout processes that observation again.
  `CompactPPO.compute_returns` now preserves/restores critic memory around the
  upstream bootstrap. GRU/LSTM/uninitialized/feedforward cases, bit-identical
  returns/advantages, next-step behavior and exception restoration pass eight
  focused tests. This corrects future training bookkeeping; it is not evidence
  that this issue explains the learned-policy failures.
- All already running worktrees remain unchanged, including both fresh
  100-update reward screens at `434c37a`. Any extension must disclose its source
  revision and apply the same correction to both matched profiles.
- `recurrent_credit_audit.json` documents the 24-step (0.96 s) gradient window
  versus the 25-step (1 s) inspection period. Inference memory persists and GAE
  bootstraps, so delayed camera credit is not discarded. A longer-window
  comparison is only a conditional regimen diagnostic, explicitly
  PREPARED_NOT_EXECUTABLE until configurable rollout budgets, resume accounting,
  evaluation metadata and GPU-memory checks are implemented. No such job ran.

### Completed repair screens and matched continuation decision

- Fresh 100-update current-qpos training completed for both profiles at
  1,228,800 transitions each. Each observed one success in 13,824 training
  episodes; repeated balanced validation was progress 0/512 and 1/512,
  log-distance 0/512 twice. Losses are finite and exploration has not collapsed.
  Mean physical terminal error is about 20 mm versus 30 mm respectively; this
  early screen selects no winner and does not establish repaired learning.
- The new critic-bookkeeping source `11a591f` passed two actual CUDA PPO updates
  in job `51510145`, 1,536 transitions. This is a device integration smoke check,
  not a trained-policy result.
- All three historical-criterion optimization continuations reached 751 total
  updates. Balanced validation: fixed LR 120/512, entropy 169/512, both 76/512;
  matched old baseline 44/512. All retain xm 0/128. Their scores are explicitly
  `derived_substep` and cannot be combined with corrected physical scores.
- `reward_repair/continuation_decision.json` records extending both fresh
  profiles to the same 751-total-update / 9,228,288-transition budget, preserving
  checkpoints, optimizer/LR, profile/std and W&B identity. Each adds 651 updates
  / 7,999,488 transitions. Both receive the same critic fix after their first
  100 updates; this is a disclosed mixed-source continuation, not a clean causal
  test of that fix. Maximum two extension GPUs. Repeated corrected validations
  and actual evaluated videos follow automatically. Larger grids and transfer/
  pushing remain conditional on the learning results.

### Matched physical tuning audit and live corrected continuations

- Both corrected reward continuations are running from frozen source `11a591f`:
  progress training `51514952` / dependent validation `51514953`, log-distance
  training `51515059` / validation `51515060`. Each validation runs two balanced
  512-episode repeats after final checkpoint verification. No winner is selected.
- Historical tuning completion is independently verified in
  `artifacts/hparam_search/continuation/completion_verified.json`: all three
  checkpoints have iteration 750, 751 contiguous updates, 9,228,288 cumulative
  transitions and finished W&B runs. The original criterion evaluations stay
  preserved. Baseline comparison uses model_750 at that same budget.
- Physical audit `51517158` evaluates baseline/fixed-LR/entropy/both twice under
  `current_qpos`, 4,096 episodes total, with no optimizer updates. This is a
  criterion intervention on historically trained checkpoints. Its submission
  records the source and script hash; all outputs use new filenames.
- That audit completed 0:0 in 8m42s. Successes per 512 episodes in repeats 1/2:
  baseline 20/26, fixed LR 129/109, entropy 120/132, both 66/67. Every case still
  has xm 0/128. All physical criterion violation counts are zero. Outcomes flip
  in 22/68/108/61 episodes respectively despite identical initial physics/image
  hashes; repeats are numerical diagnostics, not new training replicates. Exact
  report/checkpoint hashes and checks are in `physical_completion_verified.json`.
- Corrected continuation startup checks pass; original model_99 and screen
  artifacts are intact. At updates 298/327, latest-100 count-weighted training
  success is 2.64%/0.195% for progress/log. Losses are finite, but per-action std
  is contracting rapidly and adaptive LR sometimes reaches its floor. This is
  not established heldout competence; see `continuation/startup_audit.json`.
- Transfer/push readiness is prepared and CLI-validated in
  `artifacts/dynamic_occlusion/follow_on_readiness.json`, explicitly not submitted.
  The physical-qpos sampling correction and log-distance profile are plug-only.
  Current sweeps leave transfer/push wrist sensing useful; retain them as negative
  controls unless a measured scenario establishes an information disadvantage.
- The refreshed portable report passes independent extraction validation:
  312 archive files, 285 local links, 107 referenced videos decoded at first,
  middle and last frames, CRC and all included-file hashes verified. The report
  includes eight separate physical optimizer rows and immutable 100-update repair
  curves while continuations run. The bundle omits its own external checksum
  audit to avoid including a stale previous-ZIP checksum.

### Randomized physical feasibility and conditional exploration repair

- The unchanged privileged native controller passes 63/64 randomized rollouts
  at the training horizon: 16 matched native reset seeds times all four variants,
  current raw qpos within 2 mm for three samples. xm is 15/16; the other variants
  are 16/16. All successes occur at 2.12 s, before the nominal 3.5 s timeout.
  The failed xm seed 10011 stalls during approach with 53 mm TCP tracking error
  in a high-x/low-y socket corner. This does not establish an IK impossibility or
  identify contact/actuator/IK causality. Full traces and source hashes are in
  `randomized_feasibility_native.json`. Native float64 results are not paired to
  Warp by seed and do not establish learned visual competence.
- `analyze_plug_terminal_errors.py` summarizes first-terminal-step physical
  errors for all scored episodes, keeping stages/repeats separate. Original xm
  median error is 3.12/3.28 mm for active/initial versus 14.39 mm at fixed view 7.
  About 65–67% of active/initial xm terminals are within 5 mm, yet none passes the
  held-2-mm criterion. Near-goal fractions are single samples, not insertion
  success or trajectory minima, and do not identify the failure cause.
- One fresh exploration configuration is prepared, explicitly NOT SUBMITTED:
  the selected objective must be provided after both current final validations
  finish. It retains plug/fixed-view7/GRU/N512/seed0/current-qpos and caps at
  751 updates / 9,228,288 transitions; initial std 1 and entropy .01 change
  together with adaptive LR .0003. Frozen source `11a591f` applies the critic
  correction from the first update. One global reservation prevents a new grid.
  Read-only diagnostics at 100/376 updates do not alter settings or select an
  early checkpoint. Two physical balanced repeats follow, with actual videos.
- The operational competence gate for proceeding is at least 103/128 physical
  successes for EACH variant in EACH repeat, zero physical audit violations.
  This single-seed gate is not paper-level replication. If this bounded fresh
  repair fails, investigate learnability/control/representation rather than
  automatically extending budgets or opening another optimization grid.

### Completed corrected continuations and one fresh exploration repair

- Both 751-update corrected continuations and paired physical evaluations completed
  0:0. Each has 9,228,288 cumulative transitions; the continuation adds 7,999,488
  transitions to its immutable 100-update screen. Progress successes are 21/512
  and 18/512 (xm 0/0, xp 17/16, ym 0/0, yp 4/2); log-distance successes are 11/512
  and 18/512 (only xp 11/18). All physical audit violations are zero. Outcome flips
  are 27/15 despite equal initial physics/images. These repeats measure numerical
  variation, not independent trained-policy replicates.
- Progress reward rises and meets the descriptive tail-300 plateau heuristic;
  log-distance still rises and does not meet it. Reward convergence has not yielded
  insertion competence. Neither passes the all-variant physical gate.
- Submitted exactly one fresh progress exploration repair: training `51531364`,
  dependent validation `51531365`, W&B `4d4de324264d`. Initial std 1 and entropy .01
  change together, adaptive LR .0003, N512/seed0/GRU/fixed7/current-qpos/clean, capped
  at 751 updates / 9,228,288 new transitions. Frozen `11a591f` applies the critic fix
  from the first update. No resume, midrun settings changes or automatic extension.
  The choice retains the default objective; 39/1024 vs 29/1024 earlier successes
  does not establish a statistically superior reward. Exact decision, source and
  launcher hashes are in `exploration_repair/`.
- Submitted a separate 10-minute, one-GPU privileged control feasibility audit,
  `51531753`: N128 × four reset seeds = 512 balanced first episodes, normal Warp RGB,
  actual Warp IK/physics and enabled physical terminations. Native proxies perform
  only copied-state FK for the unchanged scripted controller. No native stepping,
  learner or visual policy. Initial state/sensor hashes compare against the fixed-view
  physical validation. Results are pending and cannot yet establish feasibility.
- Transfer/push training remains conditional on at least 103/128 successes for
  every variant in both physical repeats, with zero physical criterion violations.
  If the bounded fresh repair fails, no further optimization grid is automatic.

### Actual randomized Warp physical-feasibility result

- GPU audit `51531753` completed 0:0 in 1m46s; 39 independent checks pass in
  `randomized_feasibility_gpu_verified.json`. The unchanged privileged controller
  succeeds in 503/512 first episodes (98.24%): xm 119/128, xp/ym/yp 128/128 each.
  Every success is at 2.12 s; all current-qpos three-sample holds are valid, all
  qpos remain finite, and no normalized controller action clips.
- All 512 initial combined/physics/wrist/external hashes exactly match the corrected
  fixed-view policy validation. Native FK proxies never step physics; actual Warp
  IK/physics and normal RGB/terminations execute. This establishes sampled control
  feasibility, including most xm worlds, not learned visual competence or universal
  solvability. All nine failures are xm stage-0 approaches at high goal-x, with
  roughly 63 mm body error and 53–55 mm derived TCP tracking error. The cause is
  not identified; derived TCP can lag one physics substep.
- Both corrected continuation W&B runs independently report `finished` via API;
  model/history/source/budget/media checks are in `continuation/completion_verified.json`.
- Fresh exploration read-only update-100 diagnostics show finite metrics and std
  [0.739, 0.536, 0.799], with 4/13,824 completed training episodes successful.
  Settings are unchanged and the full capped run continues; no early competence
  claim or checkpoint selection is made.

### Completed within-checkpoint camera and memory diagnostics

- Job `51535182` completed 0:0 in 10m36 on one GPU: five original-model_1499
  cases, two physical 512-episode repeats each (5,120 total), no learner updates.
  Exact source/checkpoint/helper hashes and independent checks are in
  `camera_controls/completion_verified.json`. All physical scoring violations
  are zero; raw terminal XYZ norms agree with scalar errors within 4.13e-9 m.
- Active nominal succeeds 96/102 times; freeze camera targets after 1 s succeeds
  17/18 times, a drop of 15.43/16.41 percentage points. RGB remains live, actual
  joints may settle, and camera embodiment/contact changes too. This active
  checkpoint depends on continued camera commands; it is not a pure visual
  causal effect or proof that continuous control is preferable to initial-only.
- Initial nominal succeeds 152/149 times; holding external RGB after 1 s succeeds
  152/148 times (0/-0.20 percentage points). Wrist RGB/proprio/GRU remain live.
  This shows no aggregate late-external-feedback advantage in these executions,
  not universal equivalence. Resetting the entire GRU every step yields 6/10
  successes, a drop of 28.52/27.15 points, but also alters inspection and view
  selection: it does not isolate camera-specific memory or a retrained FF baseline.
- Initial physics and both RGB hashes match exactly within each checkpoint and
  intervention/repeat. Nominal outcomes nevertheless flip 86 active / 73 initial
  episodes. Diagnostic flips and Wilson episode intervals are not causal-effect
  or training-seed uncertainty. Every case remains xm 0/128; no reliable
  all-variant or generally optimal active-perception claim follows.
- The next diagnosis is prepared, not executed: separate visual target/variant
  accessibility from phase-conditioned command and tracking precision using one
  passively instrumented, fixed-checkpoint capture. See `diagnosis_plan.json`;
  no further PPO grid or automatic budget extension follows from this plan.

### Final bounded exploration repair and frozen-policy diagnosis — October 9

- Fresh training `51531364` and evaluation `51531365` completed 0:0. All 31
  independent integrity checks pass, all three W&B runs are API-verified finished,
  history contains 751 contiguous updates / 9,228,288 transitions, model_750 and
  optimizer state are finite, and frozen source11a applies both confirmed repairs
  from the first update. This run jointly changes initial std to1 and entropy to.01;
  it does not isolate either setting's causal contribution.
- Reward rises from0.3167 to6.4986 (actual first/last100-update windows), but the
  descriptive tail300 plateau fails: fitted change1.0245 exceeds tolerance0.6174.
  Latest100 completed training episodes succeed4,900/15,362 (31.90%); this is
  training-only, episode-weighted telemetry, not balanced physical validation.
- Balanced physical repeats score167/512 (32.62%) and157/512 (30.66%): xm0/0,
  xp31/29, ym72/74, yp64/54. Physical hold/distance violations are all zero. Initial
  state and both camera streams match exactly across repeats and earlier fixed-view
  validation, but128/512 outcomes change. Repeats remain numerical executions of
  one trained seed. The >=103/128-each-variant-in-each-repeat competence gate fails.
  See `exploration_repair/completion_verified.json` and separate fresh paper table.
- Conditional target diagnostic `51542399` completed0:0 in1m28s on one GPU, with
  no learner updates. The final frozen actor scores179/512 in this further execution
  (xm0,xp38,ym75,yp66); 512 initial physics/RGB hashes match the reference. Passive
  hooks observe exactly352 real forwards and unchanged actor/normalization buffers.
  All40,159 live first-episode samples and512 inspection snapshots are retained in
  ignored NPZs; tracked JSON/CSV retain provenance and compact calculations.
- Target-diagnostic independent verification passes35/35 checks, including physical
  holds, command processing, all phase summaries and all16 fixed-ridge recalculations.
  Seven xm episodes briefly sample body error below2mm, but none sustain3 samples;
  median closest sampled xm body error is2.248mm. This distinguishes near-goal
  approach from successful holding without establishing a single failure cause.
- Sixteen predeclared fixed-ridge accessibility fits use train-only normalization
  and train3resetseeds/test1, without holdout tuning. CNN variant accuracy64.84–72.66%
  exceeds proprio25.00–30.47%; median socket-XY error6.26–6.56mm is not physical
  body-goal error. GRU accuracy57.81–62.50%, median6.68–7.56mm. Proprio and combined
  seed10003 RMSE reach1,332.38 and532.17mm despite ordinary medians; retain all folds
  and report instability. This does not establish absent information, policy use,
  architectural necessity or a causal explanation for insertion failures.
- Late live-first-episode processed-target and derived-tracking residuals are much
  larger than body errors, but final TCP requests can compensate controller/contact
  effects and legitimately differ by phase. These are command measurements, not a
  direct visual-goal estimate. No universal action-box/goal-sign bug was identified.
- Dynamic occlusions remain implemented and physically/sensor validated. Transfer
  and push pilots remain prepared and unsubmitted because plug competence is unmet.
  No additional PPO grid or budget extension was started. The report separates
  original18.432M camera comparisons, physical rescoring, matched9.228M repairs and
  passive diagnostics; these stages are not interchangeable paper comparisons.
- All five actual paper-table LaTeX fragments compile in standalone articles;
  `paper_tables_validation.json` records their exact hashes. The HTML includes
  separate fresh-repair metrics, physical closest-approach counts, phase residuals
  and all16 probe folds. `follow_on_decision.json` records the conditional deferral
  of transfer/push without duplicate submissions.

### Legacy recipe audit and optional wrist yaw — October 9, follow-up repair

- Read the requested sibling `active_perception_arms` insertion reward, task
  README, and saved pl3 configurations. Legacy reward is `-log(1+100*distance)`,
  replaced by +1000 at instantaneous derived-position error <2 mm. The current
  default uses potential differences and +10 after a physical three-sample hold.
  Existing `legacy_log_hold` retains the strict hold, current-qpos sampling and
  small action/time costs; it is not exactly the historical reward.
- Saved pl3 used feedforward 256–256–128 actor/critic, initial std1, adaptive
  LR0.001, entropy0.005, five epochs/four minibatches and 196.608M global
  transitions across four GPUs. Its horizon, control rate, images, encoders,
  physics solver and timeout bootstrap also differ. The legacy task README
  explicitly marks its wrist94.0%/static86.2%/active100% seed-0 scores superseded
  because variant-dependent inertials leaked identity through plug sag. These
  historical numbers are not a matched post-fix visual baseline.
- Reproduced a native hard XM start at seed10011: original approach jams with
  62.96 mm final body error despite a 1.43e-17 m commanded IK position residual.
  Prong/socket normal contact forces reach7.49 N and6.50 N. Holding XY while
  lifting the TCP to90 mm first makes the same translation-only controller
  succeed after57 steps/2.28 s, with0.437 mm final error. This is one privileged
  native diagnostic, not a learned-policy or aggregate Warp result. The existing
  production scripted controller and training resets remain unchanged.
- Added opt-in `plug_wrist_yaw`: fourth manipulator action, ±30° absolute world-Z
  yaw, 3°/step rate limit, inspection pause, native/Warp control, reset/proprio
  and camera action slicing. Yaw runs require physical orientation error <2°
  alongside <2 mm for three samples; the evaluator records/audits all three
  orientation samples. Yaw log shaping uses a30 mm lever-arm angle cost.
  Old saved checkpoints default to yaw disabled and preserve their schemas.
- Added `policy_profile=legacy_ff` with both actor and critic feedforward and
  the recovered widths/PPO settings. Its default memory is none; existing
  compact profiles stay unchanged. Resume checks reject action/architecture/
  objective changes. This is a legacy-style recipe, retaining current geometry,
  image encoders, horizon and physical scoring, not an exact pl3 reproduction.
- Validation:166 non-GPU tests passed, followed by15 targeted checks after the
  orientation-audit addition. Native physical yaw/realignment and insertion for
  all four variants pass; CPU Warp tests cover camera slicing and partial reset;
  a real PPO update changes both image encoders with finite losses. The first
  GPU harness job51621445 failed before learning because it bypassed the normal
  RSL observation wrapper; its source/manifest/failure are preserved under
  `artifacts/plug_repair/attempts/`. Job51622754 completed two rendered PPO
  updates before the harness's checkpoint reload exposed nested runner-config
  mutation. Model construction now deep-copies its options;71 targeted tests,
  including a real CPU save/load regression, pass after that fix. Final GPU
  job51623922 completed0:0 in2m45s: yaw-off wrist+static and yaw-on active each
  execute two finite PPO updates, update both RGB encoders, and pass checkpoint
  roundtrip checks; compiled yaw IK and camera-channel routing pass too. These
  tiny smoke updates are implementation checks, not competence results.
- Prepared a fresh, sequential one-GPU pair: fixed-view7 wrist+static, seed0,
  `legacy_ff` + `legacy_log_hold`, yaw off/on,751 updates/9,228,288 transitions
  each,18,456,576 total. Each final checkpoint gets two512-episode balanced
  physical validations and representative media. Yaw changes scoring/shaping
  as well as action dimension, so this is not a pure DOF causal ablation.
  Gate:>=103/128 on every variant in both repeats. No automatic extension or
  active/wrist/transfer/push sweep. The source/asset snapshot has59 hashed files
  plus its working-tree patch; base Git revision alone does not identify it.
- Submitted the exact bounded pair as job51624849, one GPU with a4-hour wall
  cap. The launcher verifies every frozen file before each stage and performs
  both repeats plus trajectory/video recording after each final checkpoint.
  `train-submission.json` pins the plan and passing GPU validation hashes.
- The first run entered the real learner with finite initial losses, roughly
  3.5 s/update after warmup, W&B run `674x90se`. CPU-only report job51626017
  depends on completion of51624849 (success or failure) and will rebuild both
  HTML pages and independently validate the portable ZIP. Its three report
  scripts are copied and hash-pinned separately from the training source.
- New evidence/report: `artifacts/plug_repair/index.html`,
  `xm-native-diagnosis.json`, and `recipe-plan.json`. The full camera comparison
  and hyperparameter digest remains `artifacts/training_results/index.html`.
  New learned-policy success rates are pending; the ~30% baseline is not fixed
  merely by implementing and validating the new control/recipe options.

## 2026-10-09 — Transfer reward/environment repair (one task at a time)

- User requested starting the two remaining tasks sequentially. Transfer is
  first; pushing has no reward/environment redesign or training submission.
  Existing frozen plug jobs continue independently.
- The old transfer task ignored `current_qpos` for object/TCP success sampling,
  did not require pickup history or actual floor support/release, and rewarded
  keeping the TCP near the cube even when the correct final action is withdrawal.
  Along the canonical native rollout its old potential fell from about 1.60 to
  1.41 during successful withdrawal.
- Added versioned `pick_place_v2` / `transfer_staged` defaults for fresh transfer
  runs. Old saved transfer experiments retain v1 behavior and schemas; resume
  inherits/rejects changes to the task revision and reward. Current object
  qpos/qvel and analytical TCP FK drive v2 scoring. Contacts are sampled from
  the last solved physics substep; the audit records state before auto-reset.
- Success now requires bilateral contact while lifting at least 25 mm, then
  floor-supported placement within 18 mm XY / 6 mm Z, no arm contact, linear
  speed <35 mm/s, angular speed <0.5 rad/s, and TCP withdrawal >45 mm for five
  consecutive samples. The pickup history is critic/reward-only; actor inputs
  remain RGB and measurable robot state.
- Staged potential: reach [0,1], grasp/lift [1,2], transport [2,3],
  release/withdraw [3,4]. Reward is `3*(0.99*next_phi-prev_phi)+25*success`,
  minus 0.005 and 0.0005*mean squared manipulator action. Terminal potential is
  zero at every terminal condition. Camera modes have the same manipulator
  effort cost. Transfer already exposes five arm joints (including wrist flex
  and roll) plus gripper control; no additional DOF was needed for the oracle.
- Fixed evaluation/capture grouping for non-plug tasks to `nominal`; previous
  code incorrectly assigned plug XM/XP/YM/YP labels and passed them into native
  transfer replay. Termination audits now retain physical transfer criteria.
- Native baseline and v2 checks each pass 17/17 (one canonical +16 randomized).
  CPU Warp v2 passes 4/4 with normal termination/reset, zero criterion/hold
  violations. Initial 180 non-GPU tests passed; 44 targeted tests passed after
  the contact change below. New tests reject fake goal-only success, continuing
  arm contact, missing support, motion, insufficient withdrawal and timeouts;
  they cover raw-state FK, critic dtype, actor information isolation and reset.
- First GPU job51642162 completed 121/128 physical transfers. All seven
  timeouts were lifted/released/placed but exhibited residual rocking at the
  floor. Its PPO smoke performed two finite updates then hit a harness-only
  error resetting a GRU inference tensor outside inference mode. Both findings
  and the original frozen source/results are retained.
- Cube-floor contact previously used condim=3, so existing spinning/rolling
  friction coefficients were inactive. V2 now uses condim=6 (v1 unchanged).
  GPU job51643152 completed0:0 in2m00s: 126/128 scripted transfers, zero false
  success/hold violations, both RGB encoders updated in two recurrent PPO
  updates, and checkpoint reload matched. Two settling timeouts remain and
  are recorded as failures. Native contact-enabled audit still passes17/17.
- Camera screen: actual96x72 segmentation,27 states from three clean native
  rollouts ×28 cameras (26 fixed+wrist+camera-arm home). Fixed home and camera
  arm home see ≥3 cube pixels in27/27 states; wrist24/27, with target in frame
  12/27. Fixed04 has strongest coverage in this small screen, but is not a
  learned-success-selected view. This is not evidence of an active advantage.
- `artifacts/transfer_repair/index.html` contains rewards, criteria, mechanical
  results, failure rows, camera gallery, actual Warp videos and raw evidence.
  No transfer competence training has been launched yet. Scripted feasibility
  and384 PPO smoke transitions are not trained static/active/wrist results.
- Follow-up job51644216 isolates the remaining settling failures using a
  captured placement, zero initial velocity, and held arm targets. It compares
  current settings,100 solver iterations,1 ms timestep, and5× rolling friction.
  Results and source provenance: `settling-probe.json` / `settling-submission.json`.
- Settling probe completed: all four settings satisfy linear/angular settling
  for all final50 samples in eight cloned worlds. Current and100-iteration
  settings give identical results. Zeroing initial velocity does not reproduce
  the dynamic-release failures; retain both failures and existing thresholds.
  No timestep/iteration/friction-coefficient sweep was adopted from this probe.
- Submitted one bounded transfer learning pilot as job51644983: clean
  wrist+static home view0, seed0,256 worlds,501 PPO updates/3,078,144 transitions,
  compact GRU actor/critic with initial_std1.0 for gripper/joint exploration.
  Final checkpoint gets512 clean plus512 dynamic-occlusion validation episodes
  at seed20000, physical audits, stage failure counts and captured media. There
  is no automatic budget extension or camera/task sweep. The source snapshot
  matches the GPU-validated environment/reward/policy/assets exactly; only the
  CLI help text differs. Plan and runtime live under `artifacts/transfer_repair`.
  W&B run: `ew70u0sv`. Real training is running; learned success is pending.
- Pilot entered the learner with finite losses, about1.4 s/update after warmup
  (46 updates /282,624 transitions observed). CPU report job51645257 runs after
  the pilot exits, including failure/time-limit exits, using a separately
  hash-pinned builder. The report's20 local evidence/media links validate.

## 2026-10-09 — Completed plug runs and XM failure diagnosis

- The legacy-recipe pair completed: job51624849, 751 updates /9,228,288
  transitions each. Both 512-episode repeats score XYZ77/512 (all77 YP;
  XM/XP/YM0) and XYZ+yaw0/512. Adding yaw also enabled orientation shaping
  and a2-degree success criterion, so this is not a pure DOF ablation.
- XM means prongs shifted -15mm in local X relative to the common housing;
  XP means +15mm. Correct aligned housing position is socket minus offset.
  Equivalent definitions apply to YM/YP. No variant sign/indexing error was
  found; mass/inertia remain equalized and housing geometry is identical.
- Added frozen-policy diagnostic scripts `diagnose_xm_controls.py` and
  `analyze_xm_geometry.py`, and the verified HTML report
  `artifacts/xm_diagnosis/index.html`. No production reward/control change or
  learner update was made during this investigation.
- Three checkpoints each received five128-episode control interventions
  (seed10000,32/variant). Baseline XM success was0/32 for all three: earlier
  progress-GRU, new legacy-feedforward XYZ, and new XYZ+yaw. Their XM median
  closest physical position errors were2.25,10.80,2.54mm respectively. Late
  processed target jumps averaged38.9,41.9,45.7mm per40ms control step. These
  are command displacements, not object displacements; evaluation uses the
  deterministic actor mean and does not sample training noise.
- A3mm target slew cap and holding measured TCP at2s generally made outcomes
  worse. Replacing the command with the privileged true final target at2s
  rescued18/32,3/32,25/32 XM episodes respectively. This distinguishes
  recoverable final-command failures from approaches already stuck by2s.
- A privileged raise-and-traverse approach solved29/32 XM and32/32 each
  XP/YM/YP for all three cases. The three remaining XM starts require housing
  X near+58 to+64mm and jam about63mm away. Lifting in place before horizontal
  translation solved512/512 across seeds10000–10003, including128/128 XM,
  with the original3XYZ actions,3.5s horizon and2mm/three-sample physical hold.
  This is scripted feasibility with true target information, not learned success.
- Saved GRU Gaussian std corresponds to17.10,13.49,4.87mm XYZ request noise
  before clipping. Adding it to the successful lift-first oracle gives0/512.
  At10% noise,447/512 succeed (XM111/128). Applying the same normalized noise
  to5mm increments gives449/512 (XM106/128). Both retain failures. These tests
  isolate exploration-scale sensitivity but do not measure its causal share
  of trained-policy failure. Reducing eval std cannot fix deterministic means.
- Feedforward XYZ std is27.58,0.19,4.58mm (strong Y exploration collapse).
  XYZ+yaw std is7.76,12.37,1.31mm and22.59degrees before clipping/rate limits.
  A captured yaw XM frame reaches1.54mm position error but~3.5degrees angular
  error; later commands lose alignment. Captured XYZ/GRU failures show prong
  edge contacts and tilt. Native reconstruction uses actual Warp poses without
  resimulation; reconstructed contact geometry is not a GPU force measurement.
- Jobs51654119/51655476 produced the15 control cases. The initial job failed
  after four saved cases due to a helper-only Python-bool/torch.where error;
  the corrected continuation verified and preserved them. Jobs51656582 and
  51656972 completed all four512-episode approach/noise settings. Report
  validation checks19 result files/3,968 episodes, checkpoint/source hashes,
  matched initial state+image hashes, raw counts and physical holds: zero
  success violations. Noise paths may diverge after different episode resets.
- Next justified learning experiment: lift/align/insert curriculum or staged
  reward, precision-scale control and reduced insertion exploration. Plain
  distance reward, more DOFs or extra steps alone are not validated fixes.
  Perception/memory limits remain unresolved. Camera rankings and80–100%
  learned targets remain hypotheses; old high scores used an inertia shortcut.
- Transfer pilot51644983 and its report51645257 also completed0:0. The bounded
  3,078,144-transition pilot scores0/512 clean and0/512 dynamic. The transfer
  report contains its measured stage failures; further transfer diagnosis is
  deferred while addressing the user's XM request. Pushing has not been trained.

## 2026-10-09 — Transfer/push billion-transition study (current authorization)

- User assigned this worker transfer and push while another worker continues plug.
  User explicitly confirmed **at least 1B environment transitions per policy run**
  and permitted multiple seeds. This supersedes earlier transfer/push deferrals
  and the earlier one-pilot-only scope; plug experiment ownership remains separate.
- Rechecked completed transfer job51644983: 501 updates /3,078,144 transitions,
  **0/512 clean and0/512 dynamic**; all1024 failed before lifting. Its source,
  checkpoints, physical scoring and results remain preserved.
- Added opt-in Cartesian manipulation (`manip_control=cartesian`): XYZ+gripper
  for transfer, XYZ for push, fixed grasp orientation and bounded TCP slew.
  Camera-arm controls retain five joint increments. Old joint-control and plug
  configurations retain their default behavior and saved checkpoint schemas.
- Added opt-in `transfer_dense` and `push_dense` stage shaping,0.2*density per
  step,+100 held success,-0.005 time cost and equal manipulator-only effort cost.
  Transfer retains physical bilateral pickup/lift, support, release, settling and
  five-sample withdrawal criteria. Density rewards pickup/transport/release;
  lifted history alone cannot earn transport credit for a dropped object.
- Introduced `push_v2`: current-qpos object position/velocity and current-joint TCP
  FK. Old push withdrawal used the TCP above the rigid tip and could complete
  during the pushing approach. V2 requires the **actual pushing tip** to clear
  the object by25mm, no load-bearing arm/object contact, XY<15mm,Z<4mm and
  speed<35mm/s for three samples. Termination audit retains all criterion samples
  before auto-reset. Physical contacts remain sampled from the solved substep.
- Increased common study reset XY jitter from±8mm to±20mm for both object and
  goal, with explicit saved configuration and resume checks. Wider goals make
  fixed nominal action sequences less sufficient; this is not an active-vision
  performance result. No actor observations contain privileged object/goal state.
- Initial native Cartesian audit passed8/8 each under±8mm and the earlier push
  criterion. GPU job51659449 passed125/128 transfer and128/128 push under that
  intermediate configuration; preserve it separately. **Final GPU job51660328**
  passed127/128 transfer and128/128 push under±20mm and actual-tip push scoring;
  both tasks passed two finite rendered PPO updates, both RGB encoder changes,
  and checkpoint roundtrip. These are scripted feasibility/smoke checks only.
- Validation:186 non-GPU/non-render tests passed, plus two launcher tests. The
  earlier full non-GPU invocation aborted in the native OpenGL renderer on this
  CPU host; its log is preserved. GPU-rendered audits passed on allocated GPUs.
  New tests cover camera-channel slicing, bounded Cartesian movement, partial
  reset, actor information isolation, current-qpos/FK, physical tip withdrawal,
  historical schemas, exact billion-step rounding and idempotent Slurm submission.
- Immutable plan: `artifacts/transfer_push/study-plan.json`; frozen source/assets
  `.cache/transfer-push-study-20261009T195224Z`, including per-file SHA256 and the
  working-tree patch. All18 configuration dry runs passed. Audit source exactly
  matches the frozen training source; base Git revision alone is insufficient.
- Plan: two tasks × wrist/wrist_static/active × seeds0/1/2 =18 policies,
  **20,346 updates ×2,048 worlds ×24 steps =1,000,046,592 transitions each**.
  All use dynamic optical occlusion,96×72 RGB,12s horizon,compact GRU,initial
  std1,adaptive LR0.0003,entropy0.01,four PPO epochs/eight minibatches. Transfer
  static view4 comes from the earlier segmentation screen; push uses home view0.
  These are specified static baselines, not learned-success-selected best views.
- Submitted seed0 wrist+static runs: **transfer51661708**, **push51661732**.
  Both have the full1B target. Intermediate validation begins near12M transitions,
  then50M,100M and approximately100M intervals. A task scoring>=80% in both
  clean/dynamic512-episode validation sets unlocks its other eight runs. Every
  started policy retains the full1B target; an unsuccessful competence check
  does not truncate its training. Final1024-episode tests use reset seed80000;
  competence decisions use validation seed30000, never final test results.
- The worker handles Slurm24h signals by checkpointing and dependency-linked
  continuation (maximum eight allocations/run), preserving optimizer/W&B state.
  Simulation/RNG/recurrent episode state resets at resume and is disclosed.
  Final active policies also get freeze-after1s and hold-external-after1s tests.
- Results destination: `artifacts/training_results/transfer_push.html`, linked
  near the top of `artifacts/training_results/index.html`; machine-readable
  status `transfer_push_summary.json`. Workers rebuild after evaluations.
  Reports distinguish target/completed budgets, validation/test, scripted
  feasibility and learned success. No new learned camera ranking is established.
- Launch verification at19:56 UTC: transfer24 updates/1,179,648 transitions,
  push11 updates/540,672 transitions; all logged PPO losses finite. W&B runs
  `ba1poiu4` (transfer) and `g52ki3vz` (push). Early measured throughput is about
  11.9k and10.5k transitions/s; these are startup-adjacent measurements, not
  final runtime estimates. All145 frozen source files still match their hashes.
- The full worker interruption/resume/final-budget path also passes a simulated
  launcher regression: seven completed updates are preserved across a signal,
  resume reaches exactly1,000,046,592 accounted transitions, and only validation
  results enter competence decisions. This regression performs no actual PPO.
  Three launcher tests and eight combined final targeted tests pass.
- Main portable report validation passes:204 files hash-verified,70 local main
  links,14 repair links,30 XM-diagnosis links,16 embedded images and20 decoded
  videos. Transfer/push page links and unique IDs also pass. The ZIP is a dated
  snapshot; live local HTML is rebuilt by study workers as evaluations finish.

## 2026-10-09 — Plug precision policies: authorized 18M per camera condition

- User requested starting the plug policies and explicitly selected **18 million
  transitions per policy**, rather than the neighboring transfer/push 1B budget.
  Three fresh seed0 RGB policies are submitted: wrist-only **51665970**,
  static+wrist **51666125**, active+wrist **51666132**. Each receives1465 updates
  ×512 worlds ×24 steps = **18,001,920 transitions**,54,005,760 total. All three
  jobs were running on allocated RTX PRO6000 GPUs at20:13 UTC.
- Added opt-in `plug_control=delta5mm` and `reward_profile=plug_staged`.
  XYZ commands now increment the requested TCP by up to5mm/axis/40ms; zero
  holds the target. Old absolute controls and old checkpoint schemas remain
  the default. New resume checks reject a control-mode change.
- New physical stage reward first credits lift above87mm, then alignment within
  3mm while TCP stays above84mm, then insertion. Current-qpos object positions
  and current-joint FK drive shaping; stale command targets cannot advance a
  stage. Density[0,3], reward0.2*density+100*held_success-0.005-0.001*mean(action²)
  over manipulator channels only. Alignment credit falls if XY error increases.
  Stage history is critic/reward-only, not an actor observation.
- Common actor: compact GRU with RGB+measurable robot state. Initial normalized
  Gaussian std0.25 gives1.25mm per XYZ axis; std remains learned, not capped.
  Adaptive initial LR0.0003, entropy0.001. Three translation controls, no yaw;
  unchanged2mm/three-sample physical success and3.5s episode including1s pause.
  Training scenes are clean. Static uses the previous grazing view7.
- GPU audit **51663836** completed0:0 in3m03s. Privileged lift-first controller
  solves512/512 new randomized episodes,128/128 each variant, under actual
  incremental controls and normal termination/reset. All three camera modes
  pass two real rendered PPO updates, image-encoder updates and checkpoint
  roundtrip. These are feasibility/smoke results, not learned success rates.
- Targeted final tests:66 passed (including control, actor isolation, old config
  compatibility, resume and simulated interruption/budget accounting). Broader
  shared-checkout recheck:197 passed, two contemporaneous transfer/push
  privileged-actor tests failed with TensorDict type errors, outside plug RGB.
  Both logs are retained in `artifacts/plug_precision`; no claim of an entirely
  passing shared checkout. Ruff and diff whitespace checks pass.
- Immutable `artifacts/plug_precision/plan.json` pins65 source/helper/asset files
  in `.cache/plug-precision-train-20261009T201219Z`, derived from the GPU-tested
  snapshot. The only production differences from that audit are explicit
  defaulted reward-profile attribute reads for old lightweight diagnostic stubs;
  their exact hashes and equivalence for Experiment objects are recorded.
  This isolates these jobs from concurrent transfer/push source changes.
- Midpoint:750 updates /9,216,000 transitions;512 clean+512 dynamic validation
  episodes at seed30000, followed by an exact optimizer/count resume to1465.
  Final:1024 clean+1024 dynamic test episodes at seed80000 (256 per variant),
  physical audits, stage failure counts and captured videos. Dynamic evaluation
  is robustness to an unseen obstruction, not in-distribution dynamic training.
  Simulator/RNG/GRU episode state resets on midpoint/wall-time resume; weights,
  Adam, adaptive LR, W&B identity and transition accounting are preserved.
- Each job has a6h allocation cap with signal-triggered checkpoint/continuation,
  up to three allocations, always retaining the same18,001,920-transition target.
  Learned performance is pending. Live HTML: `artifacts/plug_precision/index.html`;
  workers rebuild it after evaluations. No independent training-seed comparison
  or general active/static/wrist performance ranking is claimed yet.
- Launch verification: all three entered real PPO with finite logged losses.
  Observed wrist23 updates/282,624 transitions, static+wrist15/184,320,
  active16/196,608. W&B runs are `hstja9b1`, `odi1z7rw`, `oapzq7fk` respectively.
  All65 frozen files still match. Exact observation time and counts are saved in
  `artifacts/plug_precision/launch-verification.json`. Learned evaluation remains
  pending; startup throughput is not a completion-time guarantee.


## Transfer/push: privileged-first diagnostic (2026-10-09 20:18 UTC)

- User corrected the study order: validate reward/action/task learnability with full state before RGB. Explicit diagnostic budget: **10M per task first; extend only if useful**. The eventual matched RGB runs still require **at least 1B transitions each**.
- Added opt-in `--actor-observation privileged`: actor and critic get robot/object/goal state, velocities and physical contact/stage features, with a 256×256 MLP and no rendering. Same Cartesian controls, rewards, success tests, ±20mm resets and 12s horizon as the paused RGB study. RGB remains the default and cannot silently resume into privileged mode.
- CPU validation: 5 privileged tests pass, including real PPO weight updates/checkpoint reload and bitwise-equal rewards/physics against fixed-action RGB-configured environments. Related regression suite: 72 passed. Launcher cancellation/suspension tests: 5 passed; null-capture/resume test: 1 passed.
- Diagnostic plan `artifacts/transfer_push/privileged/plan.json`; frozen simulation/policy source `.cache/privileged-diagnostic-20261009T201024Z`. Checkpoints at 2,015,232, 5,013,504 and 10,027,008 transitions, each evaluated on two 512-episode validation seed sets (30000, 31000). Provisional competence requires ≥90% on both final sets, plus physical audit/trajectory inspection. No automatic RGB launch.
- Initial jobs 51665235/51665237 reached 2,015,232 transitions; first held-out evaluations both **0/512**: transfer no lift, push outside goal. Drivers then failed because the optional capture key existed with a null value. Fixed the driver, tested resume without replaying learning, and resumed from the same checkpoints as jobs **51666937 (transfer), 51666942 (push)**. Driver repair copy/hash and resumption commands are recorded; frozen physics/reward/policy code is unchanged.
- RGB shutdown correction: old SIGTERM handlers submitted continuations despite cancellation. Cancelled 51663565/51663566/51666215/51666235 and directly killed the final transfer driver 51666277. Queue verified clear of transfer/push RGB jobs. Total logged transitions reached **11,059,200 transfer / 9,388,032 push**; preserved checkpoints remain **5,947,392 / 4,718,592** with verified hashes. Full record `artifacts/transfer_push/suspension.json`. The new launcher checks suspension on both submission and worker startup, and SIGTERM no longer resubmits (USR1 wall-time warning only).
- Live HTML: `artifacts/training_results/privileged.html`, linked from the main report and transfer/push appendix. Report reward, physical success, goal distance, lift stages, raw held-out outcomes and final trajectory videos. Reward increases alone do not establish learnability or active-perception advantage.


### Privileged 10M results and reviewed 20M extensions

- Both initial diagnostics completed **10,027,008 transitions**. Push: **361/512 (70.5%) / 345/512 (67.4%)** at validation seeds30000/31000, up from0/512 twice at5M. Transfer: **0/512 / 0/512**, all lacking a valid25mm lift history. Neither meets the provisional90%-on-both-sets criterion.
- Transfer training return rose from~21 at2M to~69 at10M without any complete transfer; do not interpret reward improvement as competence. Terminal training bilateral grasps rose9.3%→84.8% between5M and10M; normalized lift0.0125→0.5214 (~13mm mean partial lift), with2.8% lifted histories under stochastic training actions. Deterministic held-out rollouts still never completed the required lift.
- Actual10M evaluated trajectories are preserved under `artifacts/transfer_push/privileged/{task}/capture` and `videos`; full state/actions/rewards stored inNPZ, rendered observer videos linked inHTML. Two push representatives (success/failure) visually inspected; all published scores checked against raw terminal flags and physical hold criteria.
- Reward audit job**51667536** completed. Matched initial states,64 resets/case: transfer scripted completion63/64, discounted return37.99, versus reach-only12.16/grasp-hold13.15/lift-hold23.01; push completion64/64, discounted41.79 versus behind-only11.41/controller-without-explicit-withdrawal18.65 (9/64 successes). Completion also wins undiscounted return in all tested comparisons. This checks specific obvious traps, not PPO learnability. Evidence `privileged/reward-audit.json`.
- User authorized extensions when useful. Extend both to **20,004,864 cumulative transitions** (407updates), with unchanged frozen physics/reward/policy/optimizer settings. Push has clear held-out learning; transfer has developing grasp/lift behavior worth one bounded continuation. Extension decision, plan, copied driver hash and commands recorded. Jobs **51668904 push**, **51669740 transfer**. Initial `runtime-10M.json` and evaluations/checkpoints/videos preserved; extension videos use separate milestone directories. No RGB submission follows automatically.
- Live HTML `artifacts/training_results/privileged.html` includes reward/success/distance/lift curves, every checkpoint evaluation, automatic latest-score interpretation, reward ranking, and trajectory videos. Main and transfer/push HTML link the diagnostic. The prior portableZIP is a dated snapshot, not the live status.

- Post-report validation: all12 initial checkpoint evaluations reproduce published counts with zero physical/hold violations; three captured videos decode; main/transfer-push/privileged HTML have unique IDs and all local references resolve. Final launcher regression6passed. The original RGB campaign is retired: any future RGB launch must use a new reviewed plan and frozen snapshot with the cancellation guards.

## 2026-10-09 — Fixed-FOV push visibility and RGB policy comparison

- The earlier push occlusion screen did not satisfy the camera hypothesis: the wrist
  saw both the source and goal on every sampled reset. Kept that prototype as a
  rejected design and reran the audit with wrist and external-camera segmentation
  measured separately.
- Added opt-in `push_visibility_revision=crossbar_v3`. This fixed optical crossbar
  is collisionless, spans the source/goal sightline, and leaves the cube/tool push
  lane below it. Reset coordinates remain independent ±20 mm source/goal jitter;
  the crossbar is present for every layout and every camera condition. Policy FOV
  remains 48.4554906359° in all three sensors.
- Rendered64 held-out reset seeds. Wrist alone sees both landmarks in49/64 (76.6%);
  static home sees source64/64 and goal0/64, hence the wrist+static pair rate is
  49/64 (76.6%). The fixed camera alone never sees both. A set of11 IK-reachable
  active-camera endpoints with sampled collision-free joint paths covers source
  and goal together for64/64 resets (100%). This is a geometric coverage oracle,
  not evidence that a policy has learned to reposition the camera. Audit and gallery:
  `artifacts/push_rgb/visibility-audit/visibility-audit.json` and
  `visibility-gallery.png`; discarded `crossbar_v1/v2` measurements are preserved.
- The actual Warp/MjLab scene includes the crossbar, preserves its disabled contact
  masks, and produces uint8 wrist/external images. GPU audit51676182 completed:
  scripted randomized push success128/128, two rendered PPO updates finite, both
  RGB encoders updated, checkpoint roundtrip passed. The separate privileged
  push_v2 task/reward diagnostic reached501/512 and510/512 at20,004,864 transitions
  on two held-out sets, establishing task/reward learnability before RGB.
- Froze145 source/helper/dependency files at
  `.cache/push-rgb-visibility-20261009T210456Z`; plan and hashes are in
  `artifacts/push_rgb/study-plan.json`. Three matched seed0 RGB runs each target
  **1,000,046,592 transitions (at least1B)**: wrist51677171,
  wrist+static51677156, wrist+active51677159. All target the same reset/task/reward
  and crossbar. Each policy has its own 24h resumable allocation chain; only the
  explicit wall-time signal can checkpoint and continue. No competence gate truncates
  the budget. Early startup is in progress; learned success is pending.
- Live result page: `artifacts/training_results/push_rgb.html`, with the 64-reset
  visibility comparison, FOV check, Warp/scripted feasibility, privileged learnability
  evidence, per-run status and held-out results as they arrive. Main results index
  links the page. No active-vs-static learned advantage is claimed yet.

### First push RGB checkpoint and transfer reward repair update

- At12,042,240 transitions, wrist-only validation reached406/512 clean and418/512
  dynamic successes (79.3%/81.6%). Wrist+active reached162/512 and166/512
  (31.6%/32.4%); wrist+static reached0/512 clean at its first checkpoint.
  These early matched outcomes do not support an active-policy advantage yet;
  all three runs continue to the full1B target. Reward and episodic success curves
  plus raw evaluations are in the live push RGB page.
- The new transfer_transport privileged diagnostic completed its first2,015,232
  transitions:0/512 success on each validation seed, with106/512 and120/512
  reaching the lift stage (the previous dense reward had0/512 lifts at2M).
  Placement/transport has not yet improved at this checkpoint; the planned5M and10M
  evaluations will decide whether this reward revision is learnable.

### Live follow-up (2026-10-09 21:34 UTC)

- Added measured-throughput remaining-time estimates and active training curves to
  `training_results/push_rgb.html`. Early measured rates are ~20.8k transitions/s
  wrist, ~10.3k wrist+static, and ~13.1k active; estimated GPU time for each 1B
  run is ~13h, ~27h, and ~21h respectively before evaluation/restarts/queueing.
- Push workers remain active and have resumed from their 12,042,240-transition
  checkpoints. Iteration logs show progress to 27.6M wrist, 17.4M active, and
  13.6M wrist+static at the latest inspection. The first held-out results remain
  406/512 and418/512 for wrist, 0/512 and0/512 for wrist+static, and162/512
  and166/512 for active (clean/dynamic). Do not infer final ranking from this
  early checkpoint.
- Transfer transport-reward diagnostic reached5,013,504 transitions. Seed30000
  has19/512 held-out successes (3.7%), compared with0/512 at2M; its training
  episodic success metric was31.8%, so generalization remains poor. Seed31000 is
  still evaluating. The diagnostic must complete its paired5M and10M results
  before any RGB transfer run is planned.

## 2026-10-09 — Completion check and plug precision analysis (22:46 UTC)

- Checked Slurm accounting and saved checkpoints: plug jobs51665970/51666125/
  51666132 all completed0:0, each at1,465 updates /18,001,920 transitions.
  Their planned midpoint validation, final tests and trajectory captures finished.
- Ran `scripts/analyze_plug_precision.py` on the saved evaluations and training
  records. Final scores are **0/1,024 clean and0/1,024 dynamic for all three
  camera conditions**, including0/256 in each variant. Midpoint scores were also
  zero. All6,144 final episodes timed out; every episode reached lift history.
- Final alignment histories (clean/dynamic): wrist222/243, static175/185,
  active468/281 out of1,024 each. Remaining episodes lifted but never aligned.
  Alignment is historical and does not imply maintained alignment or insertion.
  Mean terminal physical errors: wrist34.27/34.22mm, static34.47/34.47mm,
  active32.65/35.24mm. No final episode ended within5mm of the goal.
- Returns plateaued without success. Final XYZ exploration request std(mm):
  wrist[0.0670,0.0225,0.0058], static[0.0075,0.1436,0.2845],
  active[0.0728,0.0545,0.5014], versus initial1.25mm/axis. These are Gaussian
  request scales before clipping/control limits, not actual TCP displacement.
  Shrinkage is descriptive, not causal proof. One training seed gives no general
  camera ranking. A full-state learnability diagnostic for this exact plug
  recipe is the next justified experiment; no new training was launched.
- Analysis verifies all65 frozen source hashes, three final checkpoint hashes,
  full1,465-update logs, and12 evaluation files /9,216 raw episode records.
  Published counts, variant counts, evaluation seeds/splits and physical held
  success checks agree. Zero physical success violations. All48 recorded videos
  exist;24 representative trace files summarized. These are selected examples,
  not population trajectory minima. Checked92 report/image/media local links.
- Detailed report: `artifacts/plug_precision/analysis.html`, linked from the
  refreshed plug report; machine-readable `analysis.json` and `analysis.csv`,
  learning/failure-stage plots, and rerun output `analysis-run.log` are saved.
- Push RGB jobs51677171/51677156/51677159 remain **RUNNING**, with observed
  wrist112,656,384, static55,541,760, active71,319,552
  transitions toward1,000,046,592 each. At the latest shared50,036,736 checkpoint,
  clean/dynamic validation is wrist511/510, static507/508, active506/510 out
  of512. Wrist's100,024,320 checkpoint is511/510. These are intermediate
  validation results, not final tests or evidence of active-camera advantage.
- Refreshed `artifacts/training_results/push_rgb.html`; timestamped Slurm evidence
  and push counts saved in `artifacts/plug_precision/completion-check.json`.

## 2026-10-09 23:00 UTC — Port original insertion reward

- User requested the insertion reward from the neighboring
  `/n/holylabs/pgozdil_lab/Lab/active-perception/active_perception_arms` repository.
  Read `src/active_perception_arms/tasks/insertion/mdp/rewards.py::insertion_reward`
  and its configuration (weight1, no dt scaling, rotation disabled by default).
  Source Git HEAD: `b6cd299d7754a03879bfd4b77f46729f1b340d47`; exact reward-file
  SHA256: `1f0274bfc74520bc134a1b29cd18da0a5d385745b58f1efbcb96f3e13ebbdc94`.
- Added `plug_insertion`, the default for fresh current-qpos plug runs:
  `-log(1 + 100*d)` for position error d in meters, REPLACED by exactly1000
  when d<0.002. No lift/alignment shaping, time cost, action cost or dt scaling.
  Optional yaw folds the source's Euclidean quaternion error against identity
  into shaping; its reward bonus remains position-only, exactly as in source.
- Adapter uses current physical qpos and the existing variant-corrected goal;
  critic positions use current qpos/FK as well. Both absolute/delta5mm controls
  work. Existing three-sample physical success termination/evaluation remains;
  unlike source's immediate termination, the instantaneous bonus can be paid
  before held success. This is an exact reward port, not a termination-rule port.
- Historical profiles and frozen studies remain reproducible. Old serialized
  configs missing reward_profile still deserialize to progress; resume inherits
  the saved objective and rejects a requested reward change. The historical
  precision-study driver explicitly retains plug_staged. No training submitted.
- Validation: exact equality to the actual upstream function (extracted from its
  source AST) on10,000 sampled states each with rotation off/on.98 targeted tests
  passed, including real CPU Warp environment integration, reward boundary,
  no-cost behavior, current-state/variant targeting, unchanged hold scoring,
  control/CLI and resume compatibility. Ruff and whitespace checks passed.
  Evidence: `artifacts/plug_precision/insertion-reward-port.json` and
  `insertion-reward-tests.log`. GPU rollout performance is not measured by these
  checks. README and CLI help document the new default and reward/hold distinction.


## Push RGB study stopped by user (2026-10-09T23:06:07.983144+00:00)

- User requested stopping all three push RGB runs after the latest matched 50,036,736-transition validation showed near-ceiling success without a clear active-camera advantage. This conclusion is specific to this setup and one training seed.
- Set `artifacts/push_rgb/suspension.json` before signaling the frozen drivers; submissions, worker starts and automatic continuations are disabled. Jobs 51677171/51677156/51677159 exited normally; Slurm shows COMPLETED because shutdown was graceful, while study states correctly say stopped. The planned 1B budgets were not completed.
- Preserved interrupted checkpoints: wrist 136,642,560; wrist+static 67,190,784; wrist+active 86,114,304 transitions. These new stopping checkpoints have not been evaluated. Prior checkpoint evaluations remain preserved. Verified checkpoint/evaluation hashes and no queued/running jobs from this push study.

## 2026-10-09 23:15 UTC — Short original-insertion-reward pilot completed

- User requested a short policy run to see whether the new reward rises.
  Job51706235 completed0:0 in9m06s, including two evaluations:100 updates ×512
  worlds ×24 steps = **1,228,800 transitions**, one fresh seed0 policy.
- Configuration: wrist+static grazing view7, clean scenes, RGB+proprio actor and
  privileged recurrent critic, compact GRU, original absolute XYZ control,
  no yaw,3.5s episode, initial std1.0, entropy0.01, adaptive LR0.0003.
  Control/optimizer choices follow the earlier successful exploration baseline;
  objective is the new `plug_insertion` exact source reward. This short run is
  not a matched-budget comparison against the previous18M precision policies.
- **Reward increased:** first10 available logged updates(4–13) mean episode return
  **-143.21**, final10(91–100) **-49.77**. Terminal-distance training metric fell
  **55.35→26.48mm**. First logged held successes appeared at update69.
  Native distance/success metrics average reset batches unevenly; do not report
  their averages as an aggregate per-episode training success rate. Reward
  includes instantaneous1000 bonuses; reward gain alone is not competence.
- Deterministic128-episode clean validation at seed30000: checkpoint after one
  PPO update **0/128**, mean terminal physical error**38.80mm**; final after100
  updates **2/128 (1.56%)**, mean terminal error**23.76mm**. Final per-variant
  counts:XM0/32,XP0/32,YM1/32,YP1/32. First checkpoint is not untrained.
  All initial physics+image hashes match across these two evaluations. Both
  successes meet <2mm for three samples; zero physical audit violations.
- Frozen source `.cache/plug-reward-pilot-20261009T230424Z`; all62 files hash
  verified. Added terminal goal-distance telemetry for plug_insertion runs.
  Training/evaluation worker: `scripts/run_plug_reward_pilot.py`; analysis:
  `scripts/summarize_plug_reward_pilot.py`. Evidence/curve/summary under
  `artifacts/plug_reward_pilot/20261009T230424Z/`. W&B run`5hqef3tu`.
- Initial allocation51705919 was cancelled during compilation before any logged
  PPO update after finding a launcher-only evaluator CLI mismatch. Corrected
  helper/source/plan were frozen separately before51706235; cancelled startup
  evidence is preserved under`artifacts/plug_reward_pilot/20261009T230338Z`.
- This pilot shows an upward learning trend and initial held insertions. It has
  not established reliable insertion. No automatic extension or other policies
  were submitted; the separate push workers continue their existing campaign.

## Push object-relative arc and tighter targets (2026-10-09 23:19 UTC)

- User requested target directions sampled around the block over −80° to +80°,
  and a smaller green area to require precise placement. Fresh push_v2 now uses
  versioned arc_v1: angle uniform about +Y, radius independently uniform 50–70 mm,
  relative to the sampled object. Source jitter remains configurable (preview
  ±20 mm); deterministic reset is 0°/60 mm.
- Marker shrank 52×52→20×20 mm; XY center tolerance 15→5 mm. Native and Warp
  scoring, reward withdrawal gate, at-goal metric and physical evaluation audit
  agree. Existing three-sample, speed, Z and contact/withdrawal checks remain.
- Serialized old configurations load square_v1; CLI resume inherits the saved
  layout and rejects changes. Frozen stopped RGB runs are untouched. No training
  submitted. This is a harder manipulation task, not proof of active-vision need.
- Native nine-reset image/coordinates/scripts in artifacts/push_rgb/arc_initial_conditions.
  The crossbar is hidden only in the clearly labelled observer illustration.
  All 729 static IK endpoint cases pass (<2 mm, within action/joint bounds), with
  a lower 35 mm diagnostic withdrawal replacing 65 mm for arc_v1 only. Static
  reachability is not path/physical feasibility. Existing scripted controller
  completes 3/9 preview episodes; wider-angle approach/control remains unvalidated
  for a new training campaign. Raw results in scripted-rollouts.json.
- Validation: 47 targeted tests pass, including native/Warp CPU sampling, subset resets, tighter physical scoring/audits, contact rejection, saved-config/resume compatibility and related reward/transfer regressions. Ruff and diff whitespace checks pass.

### Wider arc and fixed-camera audit (2026-10-09)

- User extended radius to 50–120 mm while keeping ±80° angle and 20×20 mm
  marker/5 mm center tolerance. This is now `arc_v2`; `arc_v1` retains 50–70 mm
  so serialized earlier experiments are not silently reinterpreted. Fresh push_v2
  defaults to arc_v2; older square_v1 configurations remain unchanged.
- Re-rendered nine-reset overhead preview and a separate grid of actual 96×72
  fixed policy-camera images with the crossbar present. Over 512 native randomized
  resets (seeds 40000–40511), fixed-camera green marker has >=3 segmentation pixels
  in **68/512 (13.3%)**, versus 61/512 wrist and 96/512 in at least one of the
  two. The target center is geometrically in the fixed-camera frustum in 512/512;
  visibility is blocked by scene geometry. All 512 fixed views see the object.
  Raw rows/config: `artifacts/push_rgb/arc_initial_conditions/fixed-camera-visibility.json`.
- Widened arc exposes a manipulation feasibility limit. Of 729 sampled endpoint
  poses over source jitter, angles, radii and approach/contact/withdrawal stages,
  only 663 lie within current Cartesian action bounds and only 696 have <2 mm
  static IK error. Some 120 mm side targets cannot be reached with this setup.
  This is a static screen, not a path or trained-policy test; do not launch a
  1B-transition RGB comparison on this distribution without repairing/validating
  reachability and active-camera coverage. Existing stopped jobs remain stopped.
- Ten targeted tests pass, including arc_v1/arc_v2 separation, native/CPU Warp
  sampling and scoring, and resume rejection. Ruff and diff checks pass.
## 2026-10-09 23:35 UTC — Plug insertion three-seed camera study submitted

- Submitted nine independent 18,001,920-transition policies: wrist, wrist +
  static view 7, and wrist + active view; training seeds 0, 1, and 2 for each.
  Each Slurm job requests **one GPU** and runs 512 parallel simulation worlds;
  all nine allocations were RUNNING at the 23:35 UTC check. Some allocations
  share a node but each receives its own GPU. Total scheduled training budget
  is 162,017,280 transitions.
- Job IDs by seed and camera (wrist / wrist+static / wrist+active):
  seed 0: 51715352 / 51715355 / 51715366;
  seed 1: 51716169 / 51716174 / 51716189;
  seed 2: 51716198 / 51716201 / 51716211.
- All use the hash-frozen source from the successful 100-update pilot, the
  `plug_insertion` reward, absolute XYZ control, no wrist yaw, compact GRU,
  initial action std 1.0, entropy coefficient 0.01, adaptive LR 0.0003, and
  clean training scenes. Each uses 1,465 PPO updates, with 512-episode clean
  and dynamic validation at update 750 and 1,024-episode clean and dynamic
  final tests at update 1,465. Per-seed plans and live reports are in
  `artifacts/plug_insertion_study_v2/`, `artifacts/plug_insertion_study_seed1/`,
  and `artifacts/plug_insertion_study_seed2/`; the nine-job manifest and links
  are in `artifacts/plug_insertion_study_3seeds/`.
- The first seed-0 submission attempt (51714960/51714972/51714997) failed on
  a launcher filename before PPO began. Corrected seed-0 jobs above reached
  finite PPO updates; the failed attempt remains documented under
  `artifacts/plug_insertion_study/`. Seeds 1 and 2 passed all three dry-run
  configuration checks and source-hash verification before submission. A
  post-launch check found all nine jobs RUNNING with at least one logged PPO
  update each; no startup errors were found.

## Push RGB arc120 nine-run parallel campaign (2026-10-09 23:38 UTC)

- User explicitly requested fresh wrist, wrist+fixed and active training with the
  arc_v2 initialization, three training seeds per condition, submitted in parallel
  across GPUs. New campaign is separate from the retired seed0 push study. Frozen
  154-file source: `.cache/push-rgb-arc120-20261009T233352Z`; exact hashes:
  `artifacts/push_rgb_arc120/source-manifest.json`. Nine independent one-GPU
  Slurm jobs, no inter-job dependencies, each targeting 1,000,046,592 transitions
  with 2,048 environments, 24 rollout steps, seed 0/1/2, crossbar_v3, ±20 mm
  object jitter, arc_v2 targets 50–120 mm and ±80°, 5 mm held physical success.
  Plan: `artifacts/push_rgb_arc120/study-plan.json`. Initial validation at 12M,
  50M and 100M, then ~100M intervals; final clean/dynamic tests at 1B.
- GPU audit job 51716133 completed successfully on the frozen source. Two real
  rendered PPO updates had finite metrics, both RGB encoders updated and the
  checkpoint round-tripped. Scripted physical controller completed only 15/64
  widened-arc starts, with zero hold/push audit violations. This, plus prior
  729-point static IK screen (66 outside action bounds; 33 >2 mm IK error), is a
  material learnability/ceiling risk; the user requested training anyway. The
  plan records this risk and sets no scripted-success competence gate.
- Jobs: wrist s0/s1/s2 51716337/51716339/51716341; wrist+fixed s0/s1/s2
  51716344/51716345/51716349; active s0/s1/s2 51716350/51716352/51716355.
  At 23:38 UTC seven were RUNNING on distinct GPU allocations; active s1/s2
  were PENDING due QOSMaxGRESPerUser, with no dependency and automatic queue
  start when the per-user GPU cap frees. All seven running jobs had logged
  multiple PPO updates with finite losses. Frozen hashes verified after launch.
  Exact Slurm/status snapshot: `artifacts/push_rgb_arc120/launch-verification.json`.
- Live status/report: `artifacts/push_rgb_arc120/index.html` and `summary.json`.
  Original stopped push jobs remain stopped. No scientific comparison claim is
  made until matched held-out evaluations across completed seeds.

## 2026-10-11 — Completed plug/push model analysis and refreshed report

- Reviewed this log and the neighboring M3L research log, then audited the saved
  completion evidence. All nine original-reward plug policies completed 18,001,920
  transitions each; all nine arc120 pushing policies completed 1,000,046,592 each.
  Each task has wrist, wrist+fixed and wrist+active policies for seeds 0/1/2.
- Final clean success, averaged across three seeds: plug **6.51% / 6.41% / 7.91%**
  (wrist/fixed/active); push **58.43% / 64.49% / 55.47%**. Dynamic results and
  per-seed counts are in the report. Each final setting uses 1,024 episodes per
  seed. Plug insertion remains unreliable, especially XM; pushing favors fixed
  sensing on average. Three seeds do not establish a general active-camera benefit.
- Verified all 18 final checkpoint hashes, 343 frozen source-file entries, 252
  evaluation hashes, and 147,456 raw episode outcomes/physical success records
  (36,864 final test episodes). No physical success-audit violations. Push scoring
  uses XY distance and separate height/speed/contact/withdrawal checks, not a 3D
  distance threshold. No policy inference or training was rerun.
- Rebuilt `artifacts/training_results/index.html` with white backgrounds, final
  success and learning plots, plug variant heatmaps, per-seed terminal errors,
  72 selected final-test success/failure examples and 144 outside/policy videos.
  Preserved the prior index as `historical.html`. Added machine-readable
  `latest_analysis.json`, `latest_final_results.csv`, plot SVG/PDF/PNG files and
  a refreshed portable ZIP. Training reward and validation-to-test split changes
  are explicitly distinguished from final held-out competence.
- Rebuild through `scripts/build_training_results_report.py`; implementation is
  `scripts/build_latest_training_results.py`. Independent bundle/link/hash/video
  checks run through `scripts/validate_training_results_report.py` and are saved
  to `artifacts/training_results/portable_validation.json`.

## 2026-10-12 — User-authorized competence repair in progress

- User requested continued work until plug reaches 80–90% and push improves,
  authorizing reward/environment changes and easier plug variants. Original
  studies/checkpoints remain preserved; new evidence is under
  `artifacts/competence_repair/`.
- Fixed report portability: embedded all five plots and 144 video streams.
  Chromium could not decode the original H.264 MP4 on this host, so added
  embedded VP9/WebM copies while preserving downloadable original MP4s.
  An isolated HTML playback test now decodes a 960×540 video successfully.
- Found a stale physical push partition at X=.036, Y=.003, with half-extents
  (.004,.020,.025), intersecting the expanded arc target distribution. Among
  1,024 saved clean test resets, 17 goal centers are inside its footprint and
  121 within 12 mm. Original Cartesian bounds also exclude some arc endpoints.
  New opt-in `push_workspace_revision=arc_clear_v2` removes the old partition;
  `push_control_revision=wide_v2` expands action bounds without changing the arc
  distribution, optical crossbar or 5 mm physical success rule. Old configs keep
  the partition and original action mapping. Added native/config/resume tests.
- Frozen-policy matched validation at seed31000: original best fixed seed2
  gets346/512 (67.58%); removing only the partition gives383/512 (74.80%).
  Longer horizons do not help:20s342/512,30s333/512. These are interventions,
  not retrained or final test outcomes. Geometry is a demonstrated contributor;
  it does not explain all failures.
- Started bounded fresh-optimizer PPO repairs from the original fixed seed2
  checkpoint, with smaller exploration std and zero entropy coefficient.
  `--initialize-from` preserves actor/critic weights, resets optimizer/std and
  records source hashes. Widened action mappings analytically remap the actor's
  output head to preserve original unclipped physical requests at initialization.
- Plug direct RGB DAgger was ineffective at4,000 steps (all1/256, YM2/256),
  so stopped those trials with checkpoints and failure evidence retained.
  Structured RGB target estimation plus a prescribed lift-align-descend motion
  is more promising: YM improved18/256 at2,000 steps to158/256 at4,000 on
  independent validation. It has NOT yet met the requested threshold. All-four
  variants and an encoder retaining image appearance are also being trained.
  The structured method uses privileged target labels during training only;
  deployment uses RGB/proprioception and a calibrated time-based motion plan.
  It is explicitly distinct from the earlier end-to-end PPO camera benchmark.

### Competence repair: plug threshold reached; push controls

- Frozen dense RGB target-estimator checkpoint `plug_dense_ym/model_6000.pt`
  selected with218/256 validation successes before test job52306061. Fresh test
  seed90000: **890/1024 (86.91%; Wilson95%84.71–88.84%)**, zero physical
  success violations. Teacher/privileged target labels absent at evaluation.
  This explicitly easier setting uses YM only,8s horizon,original±8mm fixture
  jitter and2mm/three-sample hold. It is a learned visual estimator with a
  prescribed insertion motion, not the original end-to-end PPO policy.
- Later dense-estimator training regressed; preserve6000 as selected rather than
  claiming monotonic learning. Unsuccessful direct DAgger/target-estimator trials
  and checkpoints remain available. Factorized all-variant visual estimation is
  a separate ongoing diagnostic and is not covered by the86.9% result.
- Push small-noise controls at250updates:336/512(std.3),345/512(std1) versus
  original346/512 on identical validation seeds. Stopped these and the weak
  fresh privileged-state diagnostic, retaining logs. Initial partition-free
  fine-tunes get364/512(originalbounds),360/512(widebounds), both below the
  frozen partition-removal control383/512. Thus retraining has not yet shown
  additional benefit; do not attribute the geometry benefit to PPO updates.
- Fresh matched push test and action-mean-regularized validation are ongoing.
  HTML now embeds browser-compatible WebM, plots, and video poster frames;
  final report refresh will include repaired-model evidence and scope.

### Repair audit and bounded control outcomes

- Clean push fresh matched test(seed90000): original694/1024(67.77%);
  partition removal737/1024(71.97%). All1,024 initial physics hashes match.
  The corrected environment gains122 successes and loses79 relative to the
  original. This is an environment correction with unchanged weights, not a
  learned-policy improvement. All-nine-policy paired tests are in progress.
- At500 updates, PPO controls regress: original-bounds clear317/512; wide-bounds
  clear322/512. Mean-regularized wide-bounds250 gets368/512, below frozen
  partition-removal383/512. These bounded trials are stopped and preserved.
  Smaller-noise fine-tunes also change optimizer, learning rate and entropy;
  they should not be described as pure noise-only causal interventions.
- Initial plug model robustness:420/1024(41.02%) under dynamic occlusion.
  A separate dynamic-occlusion warm start improves validation from137/256 at
 2k to159/256 at4k and186/256 at6k; clean validation211,225,218 respectively.
  Final checkpoint selection and a new test seed will follow the8k endpoint.
- Freezing the initial model's visual estimate earlier does not solve robustness:
  inspection_end .72s gives114/256 clean and114/256 dynamic;1.2s144/128;
 1.6s153/117. All-four-variant factorized estimation reaches61/256 at6k;
  stopped without claiming the YM result generalizes to other variants.
-72 targeted tests pass. Portable report validation checks8 embedded plots,
 156 videos decoded at first/middle/last frames, links, hashes and ZIP CRC.
  Real Chromium/CDP verification decodes and seeks14 sampled embedded WebM
  streams and renders all8 plots on white; screenshot visually inspected.
- Selected deployable checkpoint/config copies are under
  `artifacts/competence_repair/selected/`; `selected_models.json` records hashes
  and exact evaluation argument lists. They currently reflect the initial clean
  YM model and the unchanged push model in the corrected environment.

## 2026-10-12 — Competence repairs evaluated and selected

- Selected plug checkpoint: `plug_dense_ym_dynamic/model_8000.pt`, SHA256
  `c544e0c1a90100f65b6db40877ccda119f58dbeb21e8f1f87b9a6ded745969ca`.
  Validation-only selection maximized dynamic success subject to clean success
  at least80%. Selected validation:225/256 clean,197/256 dynamic. Fresh tests
  at reserved seed91000: **920/1024 clean(89.84%;95% Wilson87.84–91.55%)** and
  **814/1024 dynamic(79.49%;76.91–81.85%)**. All physical-success audits pass.
  The requested80–90% range is achieved on clean scenes; dynamic performance
  is explicitly reported below80%, not rounded up into a success claim.
- Scope: YM only, wrist/fixed RGB estimator trained with simulator target labels,
  prescribed lift-align-descend motion,8s horizon. Inference uses RGB/proprio;
  no teacher or privileged target state. Original2mm/three-sample success and
  ±8mm fixture jitter remain. It does not solve the original four-variant PPO
  benchmark. Total supervised transitions:1,536,000 initial+2,048,000 warm-start.
- Push partition correction improves **all nine** existing policies on matched
 512-episode clean tests: aggregate2809/4608(60.96%) ->2955/4608(64.13%). Means
  across three seeds: wrist60.74 ->63.15%; fixed65.49 ->68.16%; active56.64 ->
 61.07%. Initial physical-state hashes match within every pair. Original arc,
  visual crossbar, action bounds,5mm tolerance and settling/withdrawal/contact
  success conditions remain unchanged.
- Selected fixed-camera push policy's separate1024-episode comparison:
 694/1024(67.77%) original ->737/1024(71.97%) corrected. This is a modest,
  consistent environment benefit. No tested PPO fine-tune surpassed the frozen
  corrected-policy validation result383/512. Frozen bounds-only checks also
  fell short: direct wider mapping366/512; affine-remapped head370/512.
- Remaining push failure evidence: all287 selected-policy misses were outside
 5mm; median failed XY error19.86mm.126 had goal X outside the original command
  range. Those facts identify placement/control limitations; they do not prove
  a widened command range alone solves them. A privileged contact-controller
  diagnostic also remained weak(136/512), and is not a learned-policy result.
- `selected_models.json`, `selected/`, and `README.md` provide deployable copies,
  hashes, exact test arguments and training recipes. Historical checkpoints,
  failed trials and original scientific defaults are preserved. All repair
  training jobs are complete or explicitly stopped with retained checkpoints;
  the user's interactive allocation was left untouched.
- Report now includes the selected plug tests, all-nine-policy push comparison,
  spatial failure map, validation curves, unsuccessful controls and recorded
  success/failure cases. Figures and videos are embedded, with white HTML
  backgrounds and video thumbnails. Final bundle/browser checks follow rebuild.
