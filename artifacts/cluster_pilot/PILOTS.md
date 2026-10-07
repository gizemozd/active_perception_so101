# Exploratory seed-0 plug pilots

512 matched validation episodes per policy, 128 per variant, batch 128, reset seeds 10000–10003.
Each trained for 1,228,800 transitions. View 7 is a diagnostic candidate, not a searched optimum.
One training seed supports feasibility and diagnosis, not a statistically supported condition ranking.

| Condition | Success /512 | xm | xp | ym | yp | Success completion s | Camera travel rad | Training GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active | 2 (0.4%) | 1.6% | 0.0% | 0.0% | 0.0% | 2.30 | 4.79 | 0.128 |
| initial | 3 (0.6%) | 0.8% | 1.6% | 0.0% | 0.0% | 1.99 | 1.32 | 0.128 |
| wrist | 3 (0.6%) | 0.0% | 0.0% | 0.0% | 2.3% | 2.15 | 0.00 | 0.095 |
| wrist_static | 7 (1.4%) | 0.0% | 4.7% | 0.0% | 0.8% | 2.74 | 0.00 | 0.119 |

Camera travel is summed absolute camera-arm joint motion; terminal reset jumps are excluded.
Training GPU-hours are measured application wall time on one GPU, including W&B finish; Slurm allocation time is recorded separately.

- active: [training](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/u1evdury), [validation/videos](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/7z9jr312), [JSON](evaluation-active-s0.json).
- initial: [training](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/i116qwyn), [validation/videos](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/il8drljq), [JSON](evaluation-initial-s0.json).
- wrist: [training](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/a7ex2ycm), [validation/videos](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/o23icgdv), [JSON](evaluation-wrist-s0.json).
- wrist_static: [training](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/zj5uevwm), [validation/videos](https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/exx2hu2j), [JSON](evaluation-wrist_static-s0.json).
