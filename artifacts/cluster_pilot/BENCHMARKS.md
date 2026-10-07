# Measured PPO comparison

Disposable runs; first three iterations excluded as warmup.
One vectorized control step gives N environment transitions; 24 steps/iteration.
Pilot/full GPU-hours are extrapolations, excluding setup, evaluation and queues.

| Condition | N | transitions/s | Rollout s | PPO s | Iteration s | Peak MiB | Pilot GPU-h | Full GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wrist | 64 | 809.6 | 1.756 | 0.133 | 1.897 | 2081.0 | 0.422 | 6.324 |
| wrist | 128 | 1504.7 | 1.875 | 0.156 | 2.042 | 2671.0 | 0.227 | 3.403 |
| wrist | 256 | 2679.3 | 2.052 | 0.233 | 2.293 | 4727.0 | 0.127 | 1.911 |
| wrist | 512 | 4512.1 | 2.305 | 0.409 | 2.723 | 8397.0 | 0.076 | 1.135 |
| wrist_static | 64 | 777.3 | 1.811 | 0.157 | 1.976 | 2211.0 | 0.439 | 6.587 |
| wrist_static | 128 | 1395.6 | 1.976 | 0.217 | 2.201 | 3491.0 | 0.245 | 3.669 |
| wrist_static | 256 | 2428.0 | 2.150 | 0.371 | 2.531 | 5937.0 | 0.141 | 2.109 |
| wrist_static | 512 | 3827.0 | 2.486 | 0.715 | 3.211 | 10825.0 | 0.089 | 1.338 |
| initial | 64 | 740.8 | 1.898 | 0.165 | 2.073 | 2211.0 | 0.461 | 6.911 |
| initial | 128 | 1303.4 | 2.118 | 0.230 | 2.357 | 3493.0 | 0.262 | 3.928 |
| initial | 256 | 2262.5 | 2.326 | 0.377 | 2.716 | 5939.0 | 0.151 | 2.263 |
| initial | 512 | 3455.5 | 2.824 | 0.722 | 3.556 | 10827.0 | 0.099 | 1.482 |
| active | 64 | 741.5 | 1.907 | 0.157 | 2.071 | 2225.0 | 0.460 | 6.905 |
| active | 128 | 1302.0 | 2.133 | 0.217 | 2.360 | 3493.0 | 0.262 | 3.933 |
| active | 256 | 2244.6 | 2.353 | 0.374 | 2.737 | 5939.0 | 0.152 | 2.281 |
| active | 512 | 3408.1 | 2.874 | 0.720 | 3.606 | 10827.0 | 0.100 | 1.502 |
