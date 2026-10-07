# Restored-plug cluster pilots

Status: resumed from verified cluster inventory; benchmarks in progress. See `PROGRESS.md`
for executed jobs and `artifacts/cluster_pilot` for compact measurements.

## Verified allocation

Login user `pgozdil`; Slurm account `kempner_pgozdil_lab`; partition `kempner_rtx`.
Each node has eight RTX PRO 6000 Blackwell Server Edition GPUs (97,887 MiB/device),
128 CPUs and 1,547,204 MiB configured host RAM. Inventory job 51111068 verified
one allocated GPU, driver 610.57.04. Availability and scheduling are dynamic.
Use independent one-GPU jobs. No distributed-policy scaling is assumed.

## Logging and measurement

W&B entity: `pgozdil-harvard-university`, project: `active-perception-so101`.
Use existing login credentials; never place keys in commands or Git.
Cluster setup checks Python development headers required by Triton. The site system
Python lacked them; the project environment uses uv-managed Python 3.12.13.
[Dashboard](https://wandb.ai/pgozdil-harvard-university/active-perception-so101).
Groups/job types separate inference, disposable PPO benchmarks, pilots and validation.
The native RSL-RL W&B logger retains reward, episode and PPO metrics; the adapter
adds synchronized rollout and optimization timings and cumulative transitions.
Local `iterations.jsonl`, `summary.json`, and sampled `gpu.csv` accompany each run.
W&B checkpoint artifacts contain at most three scheduled checkpoints per fresh
pilot; the local run also accepts a signal-triggered interruption checkpoint.
Resume uses the same directory and W&B ID, with the next unfinished PPO iteration.

One vectorized control step advances **N environments**, giving N transitions.
There are 20 physics substeps/control step and 24 control steps/PPO iteration.
Inference throughput excludes PPO and must never be labeled training throughput.
PPO timing includes return computation; steady-state totals include previous
iteration logging/checkpoint overhead. The first three iterations are separately
reported as warmup (including lazy compilation). Environment construction time is
separate. Total job wall time also includes setup and W&B initialization.
Device-wide VRAM/utilization is sampled once per second using `nvidia-smi`, so short
peaks may be missed. Torch memory is recorded separately and excludes Warp.
Short benchmark extrapolations exclude queue and evaluation time and are uncertain.

## Commands (gated; do not submit the larger matrix)

From the repository root:

```bash
bash scripts/setup_cluster.sh
mkdir -p logs/slurm artifacts/cluster_pilot
sbatch --account=kempner_pgozdil_lab --partition=kempner_rtx scripts/slurm/validate.sbatch
# After validation and W&B metric receipt:
sbatch --account=kempner_pgozdil_lab --partition=kempner_rtx scripts/slurm/benchmark.sbatch
```

Benchmark mapping: four conditions × N=64,128,256,512, indices 0–15; each cell
runs inference followed by 12 actual PPO iterations, the first three warmup.
The default concurrency is one for initial uncontended comparisons. Queue/node
co-residency must be reported; partition jobs can share a node with other users.
Choose a common environment count only after every condition fits and completes.

Selected common count: 512. The command below is the pilot protocol; consult
PROGRESS.md for existing submissions before invoking it:

```bash
env -u SLURM_STEPMGR NUM_ENVS=512 TOTAL_STEPS=1228800 FIXED_VIEW=7 \
  OCCLUSION=clean MEMORY=gru JOB_TYPE=pilot LOG_ROOT=logs/pilots \
  RESULT_ROOT=artifacts/cluster_pilot \
  sbatch --account=kempner_pgozdil_lab --partition=kempner_rtx \
  --array=0,6,9,15%4 --export=ALL scripts/slurm/train_plug.sbatch
```

These indices are wrist, wrist_static, initial, active, each seed 0. Budgets remain
1,228,800 transitions/policy; iterations = budget/(24N). All conditions retain the
same one-second pause, clean occlusion, balanced variants, 128×96 images, FOV,
reward, 25 Hz control and 3.5-second horizon. View 7 is a diagnostic candidate.

Each final checkpoint gets 512 validation episodes, batch 128, reset seeds
10000–10003, and 128 episodes per variant:

```bash
CHECKPOINT=logs/pilots/RUN/model_FINAL.pt RECORD_VIDEO=1 \
  sbatch --account=kempner_pgozdil_lab --partition=kempner_rtx scripts/slurm/evaluate.sbatch
```

The recorder replays selected episodes with that same batch size/seed. It saves
first successes/failures per variant when present, with outside views and actual
policy inputs. It verifies reproduced outcomes. These are exploratory seed-0
pilots, not statistically supported comparisons or evidence against active sensing.

## Larger study (unsubmitted)

The existing static-search script maps 26 views × seeds 0/1/2 to indices 0–77.
Plan a validation-selected wrist+static search at the matched 18,432,000-transition
budget, followed by wrist, initial and active over three seeds each. This is 87
policies total, including the search. An external-only static search is a separately
budgeted additional 78 policies, not silently included.

Scheduled motion (3 policies), feedforward initial/active (6), and a trained
initial-snapshot control (3; requires an explicit training intervention) are
conditional additions. Frozen-camera/live-image and frozen-camera/stale-image
checkpoint evaluations require no extra training, but impose distribution shift.
Use `scripts/plan_plug_study.py` after benchmarks to compute measured-rate budgets.
Do not submit this larger matrix until pilot results justify the next comparison.

Submission caveat: pass variables in the command environment with plain
`--export=ALL`. Site Slurm treats `--export=ALL,KEY=value` as a request to
retrieve the login environment; this failed before Python in job 51121081.
Clear inherited `SLURM_STEPMGR` when submitting from an interactive allocation.
Use `scripts/inventory_plug_runs.py` for a fresh read-only accounting/artifact
snapshot; consult PROGRESS.md before any submission to avoid duplicates.

## Prepared larger-study commands (NOT submitted)

The measured N512 comparison and extrapolation are linked in
[the benchmark table](../artifacts/cluster_pilot/BENCHMARKS.md) and
[study_budget.json](../artifacts/cluster_pilot/study_budget.json).
The following commands are a reviewable plan only. Pilots must first establish
whether to keep this task/reward/optimization setup. At N512 the full budget is
1,500 iterations per policy; the exploratory pilot is 100 iterations.

```bash
# 26 views × three seeds; choose by mean validation success, never test success.
env -u SLURM_STEPMGR TASK=plug CONDITION=wrist_static NUM_ENVS=512 \
  TOTAL_STEPS=18432000 MEMORY=gru OCCLUSION=clean JOB_TYPE=study \
  sbatch --account=kempner_pgozdil_lab --partition=kempner_rtx \
  --array=0-77%4 --export=ALL scripts/slurm/static_search.sbatch

# Matched wrist, initial-only and active, seeds 0/1/2 (nine policies).
# The selected wrist+static policies already exist in the search above.
env -u SLURM_STEPMGR NUM_ENVS=512 TOTAL_STEPS=18432000 MEMORY=gru \
  OCCLUSION=clean JOB_TYPE=study LOG_ROOT=logs/study \
  sbatch --account=kempner_pgozdil_lab --partition=kempner_rtx \
  --array=0-2,9-11,15-17%4 --export=ALL scripts/slurm/train_plug.sbatch

# Conditional additions, only if pilot/study evidence justifies them:
# Scheduled camera motion: train_plug array 12-14, otherwise identical settings.
# Initial/active feedforward: array 9-11,15-17 with MEMORY=none.
# Test-time freeze: evaluate --freeze-camera-after 1.0
# Freeze and stale external frame: additionally --hold-external-after 1.0
# These interventions do not substitute for a separately trained snapshot policy.
```

Core training is 87 policies, 1,603,584,000 transitions, approximately 116.7 GPU-h
from the short benchmark. Four-GPU ideal packing is at least 29.2 hours;
conservative sequential scheduling of the search and three condition batches
is about 30.9 hours, plus setup, evaluation, queue delays and contention.
Eight-GPU ideal packing is about 14.6 hours, **conditional on actual allocation**;
eight devices per node is not evidence that eight devices will be available.
Optional scheduled, feedforward and snapshot training add 12 policies and
221,184,000 transitions, about 18.0 GPU-h using proxy rates. The trained snapshot
intervention is deliberately unimplemented/unsubmitted until evidence justifies
its explicit protocol. Frozen-camera/stale-image checkpoint evaluations add no
training transitions and retain their distribution-shift caveat.
