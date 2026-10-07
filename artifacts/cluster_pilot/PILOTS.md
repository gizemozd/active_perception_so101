# Exploratory seed-0 plug pilots

512 matched validation episodes per policy, 128 per variant, batch 128, reset seeds 10000–10003.
Each trained for 1,228,800 transitions. View 7 is a diagnostic candidate, not a searched optimum.
One training seed supports feasibility and diagnosis, not a statistically supported condition ranking.

| Condition | Success /512 | xm | xp | ym | yp | Success completion s | Camera travel rad | Training GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wrist | 3 (0.6%) | 0.0% | 0.0% | 0.0% | 2.3% | 2.15 | 0.00 | 0.095 |
| wrist_static | 7 (1.4%) | 0.0% | 4.7% | 0.0% | 0.8% | 2.74 | 0.00 | 0.119 |
| initial | 3 (0.6%) | 0.8% | 1.6% | 0.0% | 0.0% | 1.99 | 1.32 | 0.128 |
| active | 2 (0.4%) | 1.6% | 0.0% | 0.0% | 0.0% | 2.30 | 4.79 | 0.128 |

Camera travel is summed absolute camera-arm joint motion; terminal reset jumps are excluded.
Training GPU-hours are measured application wall time on one GPU, including W&B finish; Slurm allocation time is recorded separately.

- wrist: [training](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/a7ex2ycm), [original validation](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/o23icgdv), [JSON](evaluation-wrist-s0.json).
- wrist_static: [training](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/zj5uevwm), [original validation](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/exx2hu2j), [JSON](evaluation-wrist_static-s0.json).
- initial: [training](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/i116qwyn), [original validation](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/il8drljq), [JSON](evaluation-initial-s0.json).
- active: [training](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/u1evdury), [original validation](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/7z9jr312), [JSON](evaluation-active-s0.json).

## Recorded diagnostic repeats and media

Separate replay did not reproduce every rare success. Original scores above are preserved.
A separately labeled repeat used identical seeds/batch size and captured actual state/RGB during evaluation; videos render these saved trajectories without resimulation.

| Condition | Original successes | Recorded-repeat successes | Changed episode outcomes | Videos |
|---|---:|---:|---:|---|
| wrist | 3 | 3 | 2 | [recorded videos](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/fe0cjkje) |
| wrist_static | 7 | 4 | 7 | [recorded videos](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/s79bb2ko) |
| initial | 3 | 0 | 3 | [recorded videos](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/86jeiyec) |
| active | 2 | 3 | 5 | [recorded videos](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/5l27k1n4) |

Repeatability of individual outcomes is unresolved; do not use these small score differences to rank sensing conditions.
Initial-only has no successes in its recorded repeat. Its verified original episode-213 success clip is retained in the original validation W&B run; batch 1 passed replay assertions before the later batch-2 failure.

[Learning curves](learning_curves.png) / [CSV](learning_curves.csv), [representative frames](recorded_representative_frames.png), [media checks](media_validation.json), [trajectory diagnostics](trajectory_diagnostics.json).
Native training success curves average reset batches and carry the last value between resets; they are not overall episode-weighted success rates.
