# Restored-plug pilot handoff — 2026-10-07

**All four seed-0 pilots and captured diagnostic evaluations are finished. No project
jobs remain running/pending. Do not submit these pilots again.** Historical failures
and their replacements remain explicitly recorded in the inventory.

- [Every job, status, allocation and verified progress](RUNS.md), [full JSON](run_inventory.json).
- [Sixteen measured environment-count benchmarks](BENCHMARKS.md), [CSV](benchmark_comparison.csv), [same-node concurrency check](colocation_comparison.json).
- [Pilot scores, per-variant results and W&B links](PILOTS.md), [CSV](pilot_comparison.csv).
- [Learning curves](learning_curves.png), [PDF](learning_curves.pdf), [CSV](learning_curves.csv).
- [Representative outside/policy-input frames](recorded_representative_frames.png), [video decoding checks](media_validation.json).
- [Matched configuration/checkpoint audit](matched_protocol_audit.json), [learning diagnostics](learning_diagnostics.json), [selected trajectory diagnostics](trajectory_diagnostics.json).
- [Original vs recorded-repeat outcomes](evaluation_repeat_audit.json).
- [Unsubmitted study budget](study_budget.json), [prepared Slurm commands](../../docs/CLUSTER_PILOTS.md).
- [Resume manifest: checkpoints, W&B identities, evaluation job IDs](pilot_manifest.json).
- [Full chronology, fixes, limitations and next action](../../PROGRESS.md).

## Result and interpretation

Original validation success: wrist **3/512**, wrist+static view 7 **7/512**,
initial **3/512**, active **2/512**. Each policy completed exactly **1,228,800
transitions**, 100 PPO iterations at N512. Slurm allocated **0.4803 GPU-hours** for
these four learners; application wall time including W&B finish was about 0.470
GPU-hours. View 7 remains a diagnostic candidate, not a success-selected optimum.

The recorded repeats scored 3, 4, 0 and 3, respectively. Individual outcomes changed
in 2, 7, 3 and 5 episodes. These are same-checkpoint diagnostic repeats, **not extra
training seeds or a replacement score**. The exact cause of evaluation divergence
has not been isolated. No credible ranking or active-perception conclusion follows
from these sparse exploratory successes.

All four reward curves improved; losses remained finite and camera-encoder weights
changed. Selected failed trajectories usually remain 10–30 mm from the target,
with little action saturation. Initial-only captured camera targets remain exactly
constant after one second. Actor groups contain proprioception/RGB, not critic
state; earlier CUDA mutation checks exclude hidden offsets from actor state.

A temporal caveat also needs follow-up: some successful episodes have pre-action,
post-forward sampled distances above 2 mm. MjLab's termination path uses derived
state that can lag integration by one physics substep; sampled video-frame errors
are not the precise termination-time error. Preserve the original flags and audit
this timing before treating millimetric success as a robust physical hold.

**Next experiment:** isolate repeatability and termination timing with saved initial
states/actions and success-time errors, then consider extending the same four
checkpoints at a larger matched total budget. Continue to separate exploration,
mechanical feasibility, and statistically supported comparisons. Do not launch the
26-view/three-seed study yet. Its 87-policy core costs about 118.2 GPU-hours at the
measured pilot rates (ideal 29.5 hours at four GPUs; about 31.3 hours in staged
batches, before startup/evaluation/queues).

## Videos

Videos below were rendered from actual state/RGB captured during the recorded
validation repeat, without a second physics rollout. Each selects the first
observed success/failure per variant when present. Every captured video decoded at
its first, middle and final frame; outside views are 1920×1080, all streams 25 fps.

| Condition | W&B videos | Local metadata |
|---|---|---|
| wrist | [recorded evaluation](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/fe0cjkje) | [5 trajectories](capture-evaluation-wrist-s0-videos/representatives.json) |
| wrist+static view 7 | [recorded evaluation](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/s79bb2ko) | [6 trajectories](capture-evaluation-wrist_static-s0-videos/representatives.json) |
| initial | [recorded evaluation](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/86jeiyec) | [4 failures](capture-evaluation-initial-s0-videos/representatives.json) |
| active | [recorded evaluation](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/5l27k1n4) | [5 trajectories](capture-evaluation-active-s0-videos/representatives.json) |

The initial-only recorded repeat had no successes. Its original episode-213 success
clip passed batch-1 replay assertions before batch 2 failed; it was recovered and
uploaded to the [original validation run](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/il8drljq).
[Verification metadata](evaluation-initial-s0-videos/verified_partial_success.json).

Large checkpoints, MP4s, and captured NPZ trajectories stay on the cluster, outside
Git; W&B holds checkpoint artifacts and videos. Compact JSON/CSV, plots, and code
are committed. W&B API readback verified all four training runs, original validation
scores, and recorded-repeat metrics; all are finished, with no unsynchronized run.
The supplied Mac checkout was unavailable from this cluster session; its state was
not assumed or modified.
