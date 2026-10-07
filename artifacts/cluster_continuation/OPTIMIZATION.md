# Optimization diagnosis — 2026-10-07

The user's earlier successful runs are outside this repository and have not been
identified. The adjacent active_perception_arms checkout is reference source only;
its defaults are not evidence of the configuration used by those successful runs.
Do not attribute its reward, architecture, or historical bugs to the user's runs.

## Current evidence

At 20:59 UTC, wrist and wrist_static had spent their latest 100 logged updates at
the adaptive learning-rate floor, 1e-5 (initial LR 3e-4). Saved model_750 action std:
wrist [0.07362, 0.001738, 0.05040], static [0.001566, 0.001653, 0.08655], from
initial 0.4. The configured Gaussian clamp minimum is 1e-6, so these are effective
standard deviations in normalized action space. This suggests testing exploration
and LR adaptation, but does not establish the cause of poor success. Initial and
active reward curves are still improving; mean std across their eight actions
cannot be compared directly with the three-action conditions.

Native training success averages reset-batch means, not individual episodes;
telemetry also forward-fills when no fresh metric arrives. Use balanced independent
validation to assess actual success. Fix/add episode-weighted telemetry in a future
run without silently changing the meaning of historical metrics.

## Matched intermediate validation

Both model_750 checkpoints represent **751 completed updates / 9,228,288 transitions**.
Same 512 episodes, N128, seeds 10000–10003, 128 episodes per variant, deterministic
policy evaluation. Both Slurm jobs completed with exit 0:0. W&B API verified finished
runs, matching success metrics and 12 uploaded outside/policy-input videos each.

| Condition | Job | Success | xm / xp / ym / yp successes (each /128) | W&B metrics and videos |
|---|---|---:|---|---|
| wrist | 51162174 | 14/512 (2.73%) | 0 / 0 / 2 / 12 | [1q7q9tso](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/1q7q9tso) |
| wrist_static/view7 | 51162175 | 44/512 (8.59%) | 0 / 43 / 0 / 1 | [qendmi3r](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/qendmi3r) |

Original model_99 evaluations were 3/512 and 7/512, respectively. There is some
improvement, but success remains low and strongly variant-specific. Single training
seed, numerical replay sensitivity, and incomplete four-condition evaluation preclude
conclusions about the relative value of active perception. No matched intermediate
initial/active evaluation is claimed. Captured trajectories avoid physics replay
when generating videos; metadata is committed, large NPZ/MP4 files remain on cluster
and videos are available in W&B.

## Proposed next experiment (not submitted)

Use wrist_static/view7 as a diagnostic condition, holding task, seed, model, N512,
reward and transition budget fixed. Compare the current configuration with (1) a
constant 1e-4 LR, (2) increased entropy coefficient 0.01 versus 0.003, and (3) both.
These are candidate settings, not established optima. Train each from scratch as a
distinct W&B diagnostic run; preserve the ongoing scientific continuations. Evaluate
at equal transition counts, initially 1,228,800 and extend promising candidates to
9,228,288 before judging late stagnation. Log KL, clipping fraction, per-action std,
action saturation and episode-weighted training success. Verify observations,
action responses, resets and reward components from actual failed trajectories
before attributing everything to optimization. Retrieve the user's exact historical
run config when available, then compare it on the current task as a separate control.
Confirm a promising setting on additional training seeds before expanding conditions.

No HP-search training or extra seeds were submitted during this diagnosis. At the
21:02 UTC check all four existing longer training jobs remained RUNNING:
51152806_0, 51153491_6, 51153482_9, 51153446_15. Final evaluations 51153083,
51153495, 51153487, 51153468 remained PENDING (Dependency). Their original total
18,432,000-transition budgets, configuration and W&B identities remain unchanged.

[Machine-readable audit](optimization_audit.json),
[diagnostic submission manifest](optimization_diagnostic_manifest.json),
[W&B readback](interim-wandb-readback.json),
[wrist report](interim-evaluation-wrist-s0-i750.json),
[static report](interim-evaluation-wrist_static-s0-i750.json).
