# Cluster run inventory

Checked 2026-10-07T14:30:02.343427-04:00. Login pgozdil; account kempner_pgozdil_lab; partition kempner_rtx.
All allocated GPUs are RTX PRO 6000 Blackwell Server Edition. Each job used one GPU except the two-GPU co-location allocations; each learner still used one GPU.
Full submission commands, working directories, allocations, revisions, budgets, checkpoint paths and W&B identities are in [run_inventory.json](run_inventory.json). Historical first-stage reconstruction is in [PROGRESS.md](../../PROGRESS.md).

| Job/task | Purpose | State | Start (EDT) | Elapsed | Exit | Node | Verified progress |
|---|---|---|---|---|---|---|---|
| 51111068 | hardware inventory | COMPLETED | 12:13:28 | 00:00:04 | 0:0 | holygpu7c1713 |  |
| 51113803 | GPU validation | FAILED | 12:37:42 | 00:04:53 | 1:0 | holygpu7c1713 |  |
| 51114857 | GPU validation | COMPLETED | 12:46:16 | 00:05:44 | 0:0 | holygpu7c1934 |  |
| 51116045_0 | disposable inference/PPO benchmark | FAILED | 12:52:57 | 00:02:43 | 1:0 | holygpu7c2316 | wrist N64: 0/18,432 transitions |
| 51121081_0 | disposable inference/PPO benchmark | CANCELLED by 68400 | None | 00:00:00 | 0:0 | None assigned |  |
| 51121191_0 | disposable inference/PPO benchmark | COMPLETED | 13:39:50 | 00:02:28 | 0:0 | holygpu7c1731 | wrist N64: 18,432/18,432 transitions |
| 51121584_1 | disposable inference/PPO benchmark | COMPLETED | 13:43:42 | 00:02:16 | 0:0 | holygpu7c1731 | wrist N128: 36,864/36,864 transitions |
| 51121584_2 | disposable inference/PPO benchmark | COMPLETED | 13:43:42 | 00:02:18 | 0:0 | holygpu7c1731 | wrist N256: 73,728/73,728 transitions |
| 51121584_3 | disposable inference/PPO benchmark | COMPLETED | 13:43:42 | 00:03:10 | 0:0 | holygpu7c1931 | wrist N512: 147,456/147,456 transitions |
| 51121584_4 | disposable inference/PPO benchmark | COMPLETED | 13:44:22 | 00:02:18 | 0:0 | holygpu7c1934 | wrist_static N64: 18,432/18,432 transitions |
| 51121584_5 | disposable inference/PPO benchmark | COMPLETED | 13:46:18 | 00:02:56 | 0:0 | holygpu7c2331 | wrist_static N128: 36,864/36,864 transitions |
| 51121584_6 | disposable inference/PPO benchmark | COMPLETED | 13:46:24 | 00:02:06 | 0:0 | holygpu7c1731 | wrist_static N256: 73,728/73,728 transitions |
| 51121584_7 | disposable inference/PPO benchmark | COMPLETED | 13:46:45 | 00:02:21 | 0:0 | holygpu7c1934 | wrist_static N512: 147,456/147,456 transitions |
| 51121584_8 | disposable inference/PPO benchmark | COMPLETED | 13:46:59 | 00:02:19 | 0:0 | holygpu7c1931 | initial N64: 18,432/18,432 transitions |
| 51121584_9 | disposable inference/PPO benchmark | COMPLETED | 13:48:30 | 00:03:17 | 0:0 | holygpu7c1728 | initial N128: 36,864/36,864 transitions |
| 51121584_10 | disposable inference/PPO benchmark | COMPLETED | 13:49:06 | 00:02:03 | 0:0 | holygpu7c1731 | initial N256: 73,728/73,728 transitions |
| 51121584_11 | disposable inference/PPO benchmark | COMPLETED | 13:49:14 | 00:02:32 | 0:0 | holygpu7c1731 | initial N512: 147,456/147,456 transitions |
| 51121584_12 | disposable inference/PPO benchmark | COMPLETED | 13:49:18 | 00:01:58 | 0:0 | holygpu7c1934 | active N64: 18,432/18,432 transitions |
| 51121584_13 | disposable inference/PPO benchmark | COMPLETED | 13:51:09 | 00:02:42 | 0:0 | holygpu7c1713 | active N128: 36,864/36,864 transitions |
| 51121584_14 | disposable inference/PPO benchmark | COMPLETED | 13:51:16 | 00:01:43 | 0:0 | holygpu7c1731 | active N256: 73,728/73,728 transitions |
| 51121584_15 | disposable inference/PPO benchmark | COMPLETED | 13:51:46 | 00:02:16 | 0:0 | holygpu7c1931 | active N512: 147,456/147,456 transitions |
| 51122956 | disposable concurrency diagnostic | COMPLETED | 13:56:37 | 00:05:53 | 0:0 | holygpu7c1916 | active N512: 147,456/147,456 transitions; active N512: 147,456/147,456 transitions; active N512: 147,456/147,456 transitions |
| 51123326_0 | scientific pilot | COMPLETED | 14:00:03 | 00:05:46 | 0:0 | holygpu7c1931 | wrist N512: 1,228,800/1,228,800 transitions |
| 51123326_6 | scientific pilot | COMPLETED | 14:00:03 | 00:07:21 | 0:0 | holygpu7c2110 | wrist_static N512: 1,228,800/1,228,800 transitions |
| 51123326_9 | scientific pilot | COMPLETED | 14:00:03 | 00:07:51 | 0:0 | holygpu7c2316 | initial N512: 1,228,800/1,228,800 transitions |
| 51123326_15 | scientific pilot | COMPLETED | 14:00:03 | 00:07:51 | 0:0 | holygpu7c2316 | active N512: 1,228,800/1,228,800 transitions |
| 51123535 | disposable concurrency diagnostic | COMPLETED | 14:02:51 | 00:01:40 | 0:0 | holygpu7c1916 | active N512: 147,456/147,456 transitions; active N512: 147,456/147,456 transitions |
| 51123808 | validation + video | FAILED | 14:08:03 | 00:02:36 | 1:0 | holygpu7c2316 | active: 512/512 episodes; 2 successes |
| 51123809 | validation + video | FAILED | 14:08:03 | 00:02:48 | 1:0 | holygpu7c2316 | initial: 512/512 episodes; 3 successes |
| 51123810 | validation + video | COMPLETED | 14:05:57 | 00:02:32 | 0:0 | holygpu7c1731 | wrist: 512/512 episodes; 3 successes |
| 51123811 | validation + video | FAILED | 14:07:35 | 00:02:29 | 1:0 | holygpu7c2110 | wrist_static: 512/512 episodes; 7 successes |
| 51125573 | recorded diagnostic repeat | COMPLETED | 14:19:14 | 00:02:49 | 0:0 | holygpu7c2116 | active: 512/512 episodes; 3 successes |
| 51125575 | recorded diagnostic repeat | COMPLETED | 14:19:20 | 00:02:23 | 0:0 | holygpu7c1916 | initial: 512/512 episodes; 0 successes |
| 51125576 | recorded diagnostic repeat | COMPLETED | 14:19:24 | 00:02:07 | 0:0 | holygpu7c1713 | wrist: 512/512 episodes; 3 successes |
| 51125577 | recorded diagnostic repeat | COMPLETED | 14:19:26 | 00:02:43 | 0:0 | holygpu7c1710 | wrist_static: 512/512 episodes; 4 successes |

No project jobs remain pending/running.

Known terminal failures: 51113803 missing Python headers; 51116045_0 W&B constructor failure before PPO; 51121081_0 held before launch then cancelled; 51123808/09/11 numerical validation succeeded but separate replay failed. Later fixes and recorded diagnostic repeats are retained, not substituted into historical statuses.
The first job 51111068 was hardware inventory, not training. The first attempted PPO job was 51116045_0; the first completed PPO benchmark was 51121191_0. The scientific pilots are 51123326_0/6/9/15.
