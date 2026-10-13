# Competence repair

Fresh plug tests: **920/1,024 clean (89.84%)** and **814/1,024 moving occlusion (79.49%)**. Both use YM only and unchanged physical success criteria. Push improves across all nine original policies, from **2,809/4,608 (60.96%)** to **2,955/4,608 (64.13%)**, after partition removal. The selected fixed-camera policy achieves **737/1,024 (71.97%)**, compared with **694/1,024 (67.77%)** on matched original-scene resets.

The selected models are in `selected/plug/` and `selected/push/`. Keep each
`model.pt` next to its `experiment.json` and `runner.json`. Exact evaluation
argument lists and checkpoint hashes are in `selected_models.json`.

The plug model learns a visual target estimator from simulator labels. At
inference it receives wrist/fixed RGB and robot proprioception, then uses a
prescribed lift-align-descend motion. It uses YM only, an 8-second episode,
8 mm fixture jitter, and the original 2 mm/three-sample success criterion.
It is distinct from the original end-to-end, four-variant PPO benchmark.

Reproduce the training recipe on a GPU from the repository root:

```bash
.venv/bin/python scripts/train_visual_target_repair.py --dense --variant ym --steps 6000 --output artifacts/reproduce_plug_initial
.venv/bin/python scripts/train_visual_target_repair.py --dense --variant ym --occlusion dynamic --eval-dynamic --learning-rate .00003 --initialize-actor artifacts/reproduce_plug_initial/model_6000.pt --steps 8000 --output artifacts/reproduce_plug_robust
```

The robust checkpoint selection rule was fixed before the new tests: maximize
moving-occlusion validation success among checkpoints with at least 80% clean
validation success; break ties by clean success, then earlier checkpoint.
Validation seed is 31000. The initial model's test at90000 exposed its occlusion
weakness; the updated model's final tests use previously unused seed91000.
Do not select a checkpoint using the final-test outcomes.

The selected push model retains the original learned weights and action bounds.
Its saved configuration sets `push_workspace_revision=arc_clear_v2`, removing an
obsolete colliding partition. Targets, visual crossbar and the full physical
5 mm success criteria remain unchanged. The three-seed, three-camera paired
comparison is under `push_population/`, with512 matched episodes per model and
intervention. It is an environment correction, not a claim that fine-tuning
improved the learned weights. PPO controls did not beat the frozen corrected
model on validation; their logs/checkpoints are retained.

For fresh push training with the corrected scene, explicitly pass
`--push-workspace-revision arc_clear_v2`. Historical saved configurations and
legacy defaults retain their original scene for reproducibility. The optional
`wide_v2` control mapping and action-mean regularization are research controls;
they are not used in the selected push model.

`source_snapshot/` stores compatible final source and dependency versions.
The report at `../training_results/index.html` contains results, scope, learning
curves, spatial failure maps, recorded actor inputs and outside views.
Rebuild and validate it with:

```bash
.venv/bin/python scripts/build_training_results_report.py
.venv/bin/python scripts/validate_training_results_report.py
.venv/bin/python scripts/validate_report_browser.py
```
