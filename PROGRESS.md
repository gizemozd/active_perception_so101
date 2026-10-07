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
