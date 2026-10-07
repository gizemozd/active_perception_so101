# Plug hyperparameter search — 2026-10-07

Authorized by the user after the optimization diagnosis. This is a small exploratory
search on **wrist_static/view7, GRU, seed0, clean hidden-prong plug**, not a comparison
of sensing conditions. Existing four long pilots are preserved. Earlier successful
runs outside this repository remain unidentified.

| Case | Initial learning rate | Schedule | Entropy coefficient | Execution |
|---|---:|---|---:|---|
| baseline | 0.0003 | adaptive KL=0.01 | 0.003 | Reuse original pilot's first 100 updates/model_99 |
| fixed_lr | 0.0001 | fixed | 0.003 | New independent run |
| entropy | 0.0003 | adaptive KL=0.01 | 0.01 | New independent run |
| both | 0.0001 | fixed | 0.01 | New independent run |

All new cases start from scratch, with **1,228,800 total transitions = 100 updates**,
N512 and 24 rollout steps. Same geometry, balanced variants, 128×96 RGB/FOV, two
independent CNN encoders, GRU128, actor/critic widths, initial std 0.4, PPO epochs4,
minibatches8, reward, 25Hz control, 3.5s episodes and one-second manipulation pause.
Only the table's optimization settings differ. This tests a learning-rate policy
(initial value plus schedule), not the effect of schedule alone.

Each job gets one RTX PRO 6000 Blackwell GPU, 8 CPU cores and 48 GiB host RAM,
account kempner_pgozdil_lab / partition kempner_rtx. Expected new training cost:
roughly 0.3–0.6 GPU-hours for three cases, plus compilation and evaluation; extrapolated
from earlier N512 measurements, not a new timing measurement. Concurrent jobs can
share a node but receive distinct GPUs. This is not multi-GPU PPO.

W&B project: active-perception-so101, group **plug-hparam-20261007**, job type search.
Run names contain task, condition, view, memory, seed and case. Submission JSONs retain
run IDs, full commands, environment, code revision and dry-run configs. The baseline's
existing W&B identity is unchanged and linked in [baseline.json](baseline.json).

## Diagnostics and validation

Search-only instrumentation logs native PPO losses plus minibatch mean analytic KL,
clip fraction, per-axis effective std, raw action saturation, completed-episode count,
success count and episode-weighted success. These additional metrics appear under
train/Loss/diagnostic_*; episode weighting is over each rollout's resets, and should
be aggregated using counts, not unweighted iteration means. A stale displayed success
scalar is not a new observation when diagnostic_completed_episodes is zero.

CPU tests confirm identical actor and critic parameter updates and native losses
with/without instrumentation for both adaptive and fixed schedules, including
fragmented recurrent minibatches. CLI/telemetry tests: 38 passed; new update-preservation
tests: 2 passed. Default pilot settings and task code are unchanged.

Each completed model_99 receives independent validation: same 512 episodes, N128,
seeds10000–10003, balanced128/variant, captured outside and policy-input videos.
Use evaluation success overall/per variant as the primary screen; inspect completion
time, failed trajectories, reward and diagnostics. Baseline original score7/512,
capture repeat4/512; rare-success GPU repeatability limits precision. No test split
is used for tuning. Seed0 alone cannot establish a robust best configuration.

Initial screening is deliberately short. Do not reject a setting for failing to solve
the task by100 updates. Prefer configurations showing both validation improvement and
healthy optimization; if early results are inconclusive, report that rather than
selecting by reward alone. Any continuation must preserve its W&B ID and count the
first1,228,800 toward its total budget. Planned next comparison point is9,228,288
transitions (751 updates), matching existing baseline model_750, before assessing
late stagnation. Additional seeds follow a promising configuration, not this screen.

## Reproduction and recovery

Source is frozen at commit25a005e in sibling worktree plug-hparam-20261007, with its
own PYTHONPATH and a link to the existing pinned environment. Raw logs/checkpoints
live under the main checkout's logs/hparam_search/CASE; compact results live here.
The existing train_plug.sbatch is reused with verified array index6 (wrist_static,
seed0). Each submission manifest is reserved exclusively before sbatch, preventing
accidental repeated submission. Inspect a reserved/failed manifest and Slurm accounting
before retrying; never delete it merely to resubmit. The original pilot source and
artifacts remain preserved in Git and their existing log directories.

```bash
.venv/bin/python scripts/submit_plug_search.py fixed_lr \
  --source-root /n/holylabs/pgozdil_lab/Lab/active-perception/plug-hparam-20261007 \
  --data-root /n/holylabs/pgozdil_lab/Lab/active-perception/active_perception_so101
```

Without --submit this only prints the plan and validates dispatch. Existing manifest
files mean that case has already been submitted; inspect it instead of re-running.
A successful training exit must also have model_99.pt and100 contiguous updates at
1,228,800 transitions. Dependent evaluations should not be treated as complete until
Slurm exit0:0, JSON metrics, and W&B synchronization are verified.

## Verified launch

At 2026-10-07T21:16:47.193475+00:00, all three new runs were RUNNING and W&B API readback
confirmed fresh transitions and diagnostic metrics in group plug-hparam-20261007.
Exact experiment configs match the reused baseline. Hyperparameter settings and
frozen source revision were verified from each runtime/runner file.

| Case | Training job | Validation job (Dependency) | Verified local updates | W&B |
|---|---|---|---:|---|
| both | 51165289_6 | 51165406 | 10 | [fda983c1](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/fda983c1) |
| entropy | 51165286_6 | 51165405 | 10 | [9d338c65](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/9d338c65) |
| fixed_lr | 51164843_6 | 51165404 | 49 | [c59f06be](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/c59f06be) |

[Startup readback](startup_verified.json), [Slurm/artifact inventory](run_inventory.json).
Training source25a005e; evaluation source14d091d adds W&B group/tag selection only.
No final search results or preferred setting are claimed at this snapshot.
