# Longer matched plug training — submitted 2026-10-07

At the 19:57 UTC startup check, all four training jobs were RUNNING and sending
new metrics to their original W&B runs. Four evaluations are PENDING (Dependency).
This stage was explicitly requested after the throughput investigation. Do not
submit duplicate training or evaluation jobs. No broader experiment matrix launched.

| Condition | Training job | Evaluation job | Latest verified iteration | W&B |
|---|---|---|---:|---|
| wrist | 51152806_0 | 51153083 | 138 | [same run](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/a7ex2ycm) |
| wrist_static | 51153491_6 | 51153495 | 112 | [same run](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/zj5uevwm) |
| initial | 51153482_9 | 51153487 | 109 | [same run](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/i116qwyn) |
| active | 51153446_15 | 51153468 | 104 | [same run](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/u1evdury) |

## Budget and protocol

Each resumes **model_99.pt** after 100 completed updates / 1,228,800 transitions.
Target: **18,432,000 total transitions = 1,500 total updates**, so **17,203,200
additional transitions = 1,400 additional updates** per policy. Expected final
checkpoint: model_1499.pt in the original run directory. No repeated updates:
first resumed log entry is iteration 101 / 1,241,088 transitions. Histories remain
contiguous through startup. Original model_0.pt, model_50.pt and model_99.pt retained;
new regular saves use the native extended-budget interval (model_750.pt, final).

All retain seed 0, N512, GRU, clean occlusion, balanced hidden-prong variants,
128×96 policy images, identical FOV/reward/control settings, 25 Hz, 3.5s episodes,
and the shared one-second manipulation pause. wrist_static uses fixed view 7.
N2048 throughput experiments did not alter this continuation’s optimization batch.

Each training allocation: one RTX PRO 6000 Blackwell Server Edition, 8 CPUs,
48 GiB host RAM, kempner_rtx, account kempner_pgozdil_lab, six-hour safety limit.
Four independent single-GPU policies, not distributed PPO. Expected training
wall time from prior measured rates: wrist 68m, static 75m, initial/active 89m;
about 5.33 remaining training GPU-hours total, plus startup/evaluation/queue delays.
Rates may change as episodes become successful and reset more frequently.

## Continuation integrity

Source revision: 04de34f. Restored actor/critic, Adam state, adaptive PPO learning
rate, update index and environment common_step_counter. Checkpoint hashes and
initial learning rates are retained in [preflight.json](preflight.json). Native
checkpoints do not contain simulator episode state, RNG or GRU hidden state;
fresh episodes begin after loading. This is not bitwise trajectory continuation.

Resume fixes passed 35 telemetry/CLI checks and one evaluator roundtrip. W&B
now accepts an explicit continuation’s longer budget; the adaptive learning-rate
scalar is restored from Adam rather than reset to its constructor value. W&B API
readback confirmed the same four IDs, target 18,432,000 and new metrics past
iteration 100. See [startup verification](startup_verified.json). Original pilot
scores remain historical measurements of model_99.pt, not longer-run outcomes.

Per-run metadata, original summary, iteration history and GPU samples were archived
under RUN/resume_history/TIMESTAMP. The main iterations.jsonl appends new entries,
with Slurm job ID and both cumulative and segment wall time. Compact original
pilot artifacts are unchanged; continuation summaries will appear in this directory.

## Queued evaluation and next check

Each evaluation has afterok dependency on its training job and loads model_1499.pt.
It uses the same 512 validation episodes, N128, reset seeds 10000–10003, balanced
four-variant coverage. Numerical metrics and captured successes/failures, outside
views and actual policy inputs are uploaded to a separate linked validation W&B run.
Captured states are rendered without physics replay. Evaluation results do not yet
exist. Existing rare-success/repeatability caveats still apply; these remain seed-0
exploratory continuations, not statistically supported condition comparisons.

Inspect status and progress without resubmitting:

```bash
squeue -u pgozdil -p kempner_rtx
.venv/bin/python scripts/inventory_plug_runs.py \
  --jobs 51152806,51153083,51153446,51153468,51153482,51153487,51153491,51153495 \
  --output artifacts/cluster_continuation/run_inventory.json
```

Explicit --jobs is needed to include dependency-pending jobs that the date-filtered
sacct query can omit. [Manifest](manifest.json) retains all commands, environment
variables, exact paths, checkpoint hashes, node/GPU UUIDs and W&B identities.
[Inventory](run_inventory.json) contains Slurm accounting/controller snapshots.

On completion, verify exit codes, model_1499.pt iteration=1499, exactly 1,500
contiguous logged iterations and 18,432,000 cumulative transitions, then collect
evaluations. If interrupted, inspect logs/interrupted.pt and the total budget;
do not add another full budget or change W&B IDs. A signal checkpoint can exit
zero before the final model exists, in which case the dependent evaluation may fail
and must wait for the successfully completed resumed training. Do not infer
completion from exit code alone. No training or evaluation result is yet claimed.

## Intermediate optimization check

See [2026-10-07 diagnosis](OPTIMIZATION.md) for the newer status, two completed
model_750 evaluations, W&B videos, and the proposed controlled optimization search.
The earlier table on this page is the startup snapshot, not current progress.
