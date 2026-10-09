# Plug insertion training and active-perception report

Open [index.html](index.html) locally. It works from `file://` with local plots,
paired overview/actor-input videos, and downloadable paper-table data. The report
is a scientific diagnostic of the existing runs, not a hosted website.

The original captured final scores use **historical `derived_substep` scoring** at
matched 18,432,000-transition budgets: wrist **3/512**, fixed view 7 **59/512**,
initial-only **185/512**, and active **141/512**. These are preserved historical
scores, not corrected physical insertion measurements. Termination audits found
flagged successes outside the current physical threshold and substantial episode
disagreement across seeded repeats despite identical initial-state/image hashes.

The separately evaluated **corrected `current_qpos` criterion** checks three
consecutive physical-position samples at 25 Hz. It scores wrist **8/512**, fixed
view 7 **23/512**, initial-only **147/512 (28.71%)**, and active **102/512 (19.92%)**.
All four conditions remain **0/128** on `xm`. Exact termination audits find zero
hold, distance, three-sample or raw-qpos violations in these corrected evaluations.
See the [corrected performance CSV](corrected_performance_metrics.csv) and
[corrected LaTeX table](corrected_performance_metrics.tex). Three discrete samples
do not establish continuous dwell between samples.

These are exploratory **single-training-seed validation** results. Reward increased
and flattened, but reliable balanced insertion remains unestablished. Initial
camera positioning outperforms continued camera motion in this seed; fixed-view
selection, training-seed replication and untouched test evaluation remain necessary
before a general active-perception claim. Initial-only freezes camera targets after
1 s while external/wrist images continue updating; it is not a snapshot-only condition.

Dynamic occlusions passed **six CUDA tests**, covering rendered RGB obstruction,
moving-panel parity, partial resets and existing pipeline regressions; see
[GPU validation evidence](../dynamic_occlusion/gpu_validation.json). The separate
[native scenario diagnostics](../dynamic_occlusion/summary.json) demonstrate optical
obstruction and scripted feasibility. They do not establish learned-policy benefits;
the tested pushing/transfer wrist views remain strong controls.

Rebuild from the repository root:

```bash
.venv/bin/python scripts/build_plug_report.py --validate-media
```

Add `--bundle` to build `plug-report.zip`. Extract it and open its root `index.html`.
It includes HTML, plots, tables, compact scored JSONs and referenced videos, with
the relative directory layout preserved. It omits checkpoints and raw NPZ traces,
and replaces their HTML links with plain labels. The archive stays outside Git.

The builder reads local histories, final/intermediate evaluations, representative
captures, `training_audit.json`, recovered historical-run provenance, completed
`artifacts/hparam_search/continuation/*/evaluation.json` results, corrected final
evaluations, termination repeats and bounded reward-repair evaluations. It never
trains, evaluates a policy, requests network access,
or advances simulator physics. Native forward kinematics, contact geometry and ray
tests are reconstructed from stored states and clearly labeled as diagnostics.

Outputs:

- `performance_metrics.csv` and `performance_metrics.tex`: original historical
  derived-substep scores, Wilson intervals and performance/provenance fields.
  `corrected_performance_metrics.csv` / `.tex` contain separate current-qpos scores.
  CSV rates are
  fractions; displayed HTML/LaTeX rates are percentages.
- `evaluation_episodes.csv`: all 2,048 final-evaluation episodes, variant, elapsed
  time and camera-joint travel.
- `learning_curves.csv`: all 6,000 logged updates and training metrics.
- `checkpoint_metrics.csv`: available early/intermediate/final checkpoint scores.
- `representative_trajectories.csv`: selected-trace motion, alignment, action
  saturation, geometric visibility and contact diagnostics.
- `analysis.json`: compact derived results, convergence heuristics and limitations.
- `assets/`: standalone SVG/PDF scientific figures and video-preview JPEGs.
- `report_validation.json`: local-link checks and decoded video/frame-rate checks.
- `corrected_hparam_metrics.csv` / `.tex`: two separately reported physical-state
  evaluations of each matched-budget optimizer checkpoint; numerical repeats
  are not pooled as independent episodes or training seeds.
- `repair_learning_curves.csv` / `repair_learning_summary.json`: actual corrected
  reward-screen and completed continuation telemetry, with each reward scale and
  source stage retained. Unfinished continuations are not shown as final results.
- `physical_terminal_errors.csv` / `.json`: all evaluated terminal errors by
  variant, with single-sample near-goal fractions kept separate from held success.
- `randomized_feasibility_native.json`: 64 privileged-controller native rollouts
  at the training horizon, including the failed approach case and full traces.
  This is a physical diagnostic, not GPU or learned visual-policy performance.

For a standalone copy with videos, generate and independently verify the bundle:

```bash
.venv/bin/python scripts/analyze_plug_terminal_errors.py
.venv/bin/python scripts/build_plug_report.py --validate-media --bundle
.venv/bin/python scripts/validate_report_bundle.py --decode-media
```

Open `index.html` after extracting `plug-report.zip`. The ZIP excludes checkpoints
and raw NPZ captures; its manifest records included-file hashes. The separate
`portable_validation.json` verifies extracted links, hashes, CRC and video frames.

Videos are the first observed success/failure per variant in the scored evaluation,
not a second physics rollout. Overview videos are rendered from captured states;
policy videos show the exact recorded actor inputs. The original evaluation contains
25 representative trajectories and 50 paired videos; corrected evaluations add
separately labeled captures. No `xm` success occurred.

Large checkpoints, NPZ captures and MP4 videos remain outside Git in their original
local directories. Clone the repository alone to access compact report evidence;
the complete local media report also needs those original artifacts. Independent
repeats and exact termination-time audits are separate diagnostic results and do
not silently replace the original primary scores.
