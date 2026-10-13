"""Build the cross-task results digest from saved evaluations, without running policies.

Usage: .venv/bin/python scripts/build_training_results_report.py
The current builder links vector figures and includes a portable ZIP with media.
Historical report-generation functions are retained below for provenance.
"""

import base64
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get("TRAINING_REPORT_ROOT", Path(__file__).resolve().parents[1])).resolve()
BASE = ROOT / "artifacts/plug_analysis"
OUT = ROOT / "artifacts/training_results"
ORDER = ("wrist", "wrist_static", "active", "initial")
LABELS = {
    "wrist": "Wrist only",
    "wrist_static": "Static + wrist (view 7)",
    "active": "Active + wrist",
    "initial": "Initial positioning + wrist",
}
COLORS = ["#64748b", "#2563eb", "#ea580c", "#059669"]
VARIANTS = ("xm", "xp", "ym", "yp")
SOURCES = set()


def read(path):
    path = Path(path).resolve()
    SOURCES.add(path)
    return json.loads(path.read_text())


def rows(name):
    path = BASE / name
    SOURCES.add(path)
    return list(csv.DictReader(path.open()))


def esc(value):
    return html.escape(str(value))


def href(path, label):
    path = Path(path).resolve()
    SOURCES.add(path)
    return f'<a href="{esc(os.path.relpath(path, OUT))}">{esc(label)}</a>'


def pct(value):
    return f"{100 * float(value):.2f}%"


def score(row):
    return f"<strong>{pct(row['success_rate'])}</strong> <small>({row['successes']}/{row['episodes']})</small>"


def ci(row):
    return f"{100 * float(row['wilson95_low']):.2f}–{100 * float(row['wilson95_high']):.2f}%"


def table(headers, records):
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f'<th scope="col">{h}</th>' for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in records)
        + "</tbody></table></div>"
    )


def figure(path, caption):
    path = Path(path)
    SOURCES.add(path.resolve())
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img alt="{esc(caption)}" src="data:image/svg+xml;base64,{data}"><figcaption>{esc(caption)}</figcaption></figure>'


def savefig(fig, name):
    path = OUT / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    return path


def validate_rows(data, report_key="report"):
    """Check published tables against actual episode flags and variant counts."""
    for row in data:
        raw = read(ROOT / row[report_key])
        assert int(row["episodes"]) == raw["episodes"] == len(raw["outcomes"])
        assert int(row["successes"]) == raw["successes"] == sum(raw["outcomes"])
        assert math.isclose(float(row["success_rate"]), raw["successes"] / raw["episodes"])
        assert raw["experiment"]["success_state_sample"] == "current_qpos"
        for variant in VARIANTS:
            selected = [
                s for s, v in zip(raw["outcomes"], raw["variants"], strict=True) if v == variant
            ]
            assert len(selected) == int(row[f"{variant}_episodes"]) == 128
            assert sum(selected) == int(row[f"{variant}_successes"])
        for key in (
            "success_hold_violations",
            "success_distance_violations",
            "success_three_sample_violations",
            "success_qpos_above_distance_threshold",
        ):
            assert raw["termination_audit"][key] == 0


def variant_table(data, label):
    return table(
        ["Configuration", "Overall", "95% Wilson interval"]
        + [f"{v.upper()} /128" for v in VARIANTS],
        [
            [label(r), score(r), ci(r)]
            + [
                f"{r[v + '_successes']} <small>({pct(r[v + '_success_rate'])})</small>"
                for v in VARIANTS
            ]
            for r in data
        ],
    )


def media_card(record, title, condition, poster=None):
    paths = {k: Path(record[k]).resolve() for k in ("outside_video", "policy_video")}
    SOURCES.update(paths.values())
    poster_attr = ""
    if poster and poster.exists():
        poster_attr = (
            ' poster="data:image/jpeg;base64,'
            + base64.b64encode(poster.read_bytes()).decode()
            + '"'
        )
    result = (
        "success"
        if record.get("expected_success", str(record.get("success")) == "True")
        else "failure"
    )
    return f'''<article class="media-card" data-condition="{condition}" data-result="{result}">
<h3>{esc(title)}</h3><p class="small">{esc(record["variant"].upper())} · episode {record["episode"]} · {result}</p>
<video controls preload="none"{poster_attr} aria-label="{esc(title)}" src="{esc(os.path.relpath(paths["outside_video"], OUT))}"></video>
<p>{href(paths["outside_video"], "Outside video")} · {href(paths["policy_video"], "Exact policy inputs")}</p></article>'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sources").mkdir(exist_ok=True)
    source_snapshots = []
    for original, name in [
        (ROOT / "PROGRESS.md", "PROGRESS.md"),
        (ROOT.parent / "M3L-active-perception/RESEARCH_LOG.md", "M3L_RESEARCH_LOG.md"),
    ]:
        target = OUT / "sources" / name
        shutil.copyfile(original, target)
        source_snapshots.append(
            {
                "original": str(original),
                "snapshot": str(target.relative_to(OUT)),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )

    matched = {r["condition"]: r for r in rows("corrected_performance_metrics.csv")}
    matched_rows = [matched[c] for c in ORDER]
    latest = rows("exploration_repair_metrics.csv")
    tuning = rows("corrected_hparam_metrics.csv")
    repair_screen = rows("reward_repair_metrics.csv")
    repair = rows("reward_repair_continuation_metrics.csv")
    controls = rows("camera_control_metrics.csv")
    for data, key in [
        (matched_rows, "evaluation_report"),
        (latest, "report"),
        (tuning, "report"),
        (repair_screen, "report"),
        (repair, "report"),
        (controls, "report"),
    ]:
        validate_rows(data, key)
    audit = read(BASE / "exploration_repair/completion_verified.json")
    manifest = read(BASE / "exploration_repair/progress/run_manifest.json")
    analysis = read(BASE / "analysis.json")
    diagnostic = rows("target_diagnostic_performance_metrics.csv")
    probes = rows("target_feature_probe_metrics.csv")
    terminal = rows("physical_terminal_errors.csv")
    historical = {r["condition"]: r for r in rows("performance_metrics.csv")}
    hp_early_path = ROOT / "artifacts/hparam_search/comparison.csv"
    SOURCES.add(hp_early_path)
    hp_early = {r["case"]: r for r in csv.DictReader(hp_early_path.open())}
    hp_final = read(ROOT / "artifacts/hparam_search/continuation/completion_verified.json")
    historical_tuning = {r["case"]: r["historical_validation_successes"] for r in hp_final["runs"]}
    historical_tuning["baseline"] = 44
    assert audit["passed_checks"] == audit["total_checks"] == 31
    assert manifest["training_updates"] == 751 and manifest["training_transitions"] == 9228288
    assert Path(audit["checkpoint"]).is_file()
    with Path(audit["checkpoint"]).open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    assert checkpoint_hash == audit["checkpoint_sha256"]

    failure_rows = []
    for condition in ORDER:
        raw = read(ROOT / matched[condition]["evaluation_report"])
        failed = [r for r in raw["termination_audit"]["records"] if not r["success"]]
        distances = [r["qpos_position_error_m"] * 1000 for r in failed]
        bins = [
            sum(d <= 2 for d in distances),
            sum(2 < d <= 5 for d in distances),
            sum(5 < d <= 10 for d in distances),
            sum(d > 10 for d in distances),
        ]
        assert sum(bins) == len(failed) == 512 - raw["successes"]
        failure_rows.append(
            {
                "condition": condition,
                "failures": len(failed),
                "failed_timeout": sum(r["timeout"] for r in failed),
                "explicit_failure": sum(r["failure"] for r in failed),
                "terminal_le2mm": bins[0],
                "terminal_2to5mm": bins[1],
                "terminal_5to10mm": bins[2],
                "terminal_gt10mm": bins[3],
            }
        )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "training-results",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 3.8))
    values = np.array([float(r["success_rate"]) * 100 for r in matched_rows])
    intervals = np.array(
        [[float(r[k]) * 100 for r in matched_rows] for k in ("wilson95_low", "wilson95_high")]
    )
    ax.barh(
        range(4),
        values,
        color=COLORS,
        height=0.56,
        xerr=[values - intervals[0], intervals[1] - values],
        capsize=4,
    )
    ax.set_yticks(range(4), [LABELS[c] for c in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Physical success (%) · 512 episodes per condition")
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    for i, r in enumerate(matched_rows):
        ax.text(intervals[1, i] + 1.5, i, f"{values[i]:.2f}% ({r['successes']}/512)", va="center")
    comparison_plot = savefig(fig, "matched_success")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    values = np.array(
        [[100 * float(r[v + "_success_rate"]) for v in VARIANTS] for r in matched_rows]
    )
    im = ax.imshow(values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(4), [v.upper() for v in VARIANTS])
    ax.set_yticks(range(4), [LABELS[c] for c in ORDER])
    for i, r in enumerate(matched_rows):
        for j, v in enumerate(VARIANTS):
            ax.text(
                j,
                i,
                f"{values[i, j]:.1f}%\n{r[v + '_successes']}/128",
                ha="center",
                va="center",
                color="white" if values[i, j] > 55 else "#102a43",
            )
    fig.colorbar(im, ax=ax, label="Physical success (%)", fraction=0.04)
    variant_plot = savefig(fig, "variant_success")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(4)
    cases = ("baseline", "fixed_lr", "entropy", "both")
    for repeat, offset, color in [("1", -0.17, "#2563eb"), ("2", 0.17, "#0d9488")]:
        selected = [
            next(r for r in tuning if r["case"] == c and r["repeat"] == repeat) for c in cases
        ]
        vals = np.array([100 * float(r["success_rate"]) for r in selected])
        low = [vals[i] - 100 * float(r["wilson95_low"]) for i, r in enumerate(selected)]
        high = [100 * float(r["wilson95_high"]) - vals[i] for i, r in enumerate(selected)]
        ax.bar(
            x + offset,
            vals,
            0.32,
            yerr=[low, high],
            capsize=3,
            label=f"Repeat {repeat}",
            color=color,
        )
    ax.set_xticks(x, ["Baseline", "Fixed LR 1e-4", "Entropy .01", "Fixed LR + entropy"])
    ax.set_ylim(0, 40)
    ax.set_ylabel("Physical success (%)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    tuning_plot = savefig(fig, "hyperparameter_success")

    histories = {}
    history_export = []
    training_curves = (
        ("train/Train/mean_reward", "Episode reward (30-update mean)"),
        ("train/Loss/learning_rate", "Learning rate"),
        ("train/Policy/mean_std", "Mean action standard deviation"),
    )
    for case in cases:
        checkpoint = next(r["checkpoint"] for r in tuning if r["case"] == case)
        history_path = Path(checkpoint).parent / "iterations.jsonl"
        histories[case] = [json.loads(line) for line in history_path.read_text().splitlines()][:751]
        assert [r["iteration"] for r in histories[case]] == list(range(1, 752))
        assert histories[case][-1]["transitions"] == 9228288
        for row in histories[case]:
            history_export.append(
                {
                    "case": case,
                    "iteration": row["iteration"],
                    "transitions": row["transitions"],
                    **{key: row.get(key, "") for key, _ in training_curves},
                }
            )
        historical_path = ROOT / (
            "artifacts/cluster_continuation/interim-evaluation-wrist_static-s0-i750.json"
            if case == "baseline"
            else f"artifacts/hparam_search/continuation/{case}/evaluation.json"
        )
        historical_raw = read(historical_path)
        assert (
            historical_raw["successes"]
            == historical_tuning[case]
            == sum(historical_raw["outcomes"])
        )
    history_csv = OUT / "hyperparameter_learning_curves.csv"
    with history_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history_export[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history_export)
    SOURCES.add(history_csv)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for case, color in zip(cases, COLORS, strict=True):
        for axis, (key, title) in zip(axes, training_curves, strict=True):
            selected = [row for row in histories[case] if key in row]
            xs = np.array([row["transitions"] / 1e6 for row in selected])
            ys = np.array([row[key] for row in selected])
            if key == "train/Train/mean_reward":
                xs, ys = xs[29:], np.convolve(ys, np.ones(30) / 30, mode="valid")
            axis.plot(xs, ys, color=color, label=case, linewidth=1.4)
            axis.set_ylabel(title)
    for axis in axes:
        axis.axvline(1.2288, color="#94a3b8", linestyle="--", linewidth=1)
        axis.grid(alpha=0.15)
    axes[0].legend(ncol=4, frameon=False)
    axes[1].set_yscale("log")
    axes[-1].set_xlabel("Cumulative training transitions (millions)")
    hp_learning_plot = savefig(fig, "hyperparameter_learning")

    sections = []
    privileged_section = OUT / "privileged_section.html"
    if privileged_section.exists():
        sections.append(
            "<!-- privileged-live-start -->"
            + privileged_section.read_text()
            + "<!-- privileged-live-end -->"
        )
        SOURCES.update({OUT / "privileged.html", OUT / "privileged_summary.json"})
    manipulation_section = OUT / "transfer_push_section.html"
    if manipulation_section.exists():
        sections.append(
            "<!-- transfer-push-live-start -->"
            + manipulation_section.read_text()
            + "<!-- transfer-push-live-end -->"
        )
        SOURCES.update({OUT / "transfer_push.html", OUT / "transfer_push_summary.json"})

    def section(anchor, title, body):
        sections.append(f'<section id="{anchor}"><h2>{title}</h2>{body}</section>')

    repair_root = ROOT / "artifacts/plug_repair"
    repair_status_path = repair_root / "train-runtime.json"
    if not repair_status_path.exists():
        repair_status_path = repair_root / "train-submission.json"
    if repair_status_path.exists():
        repair_status = read(repair_status_path)
        section(
            "repair",
            "New repair experiment",
            f"<p><strong>Job {esc(repair_status.get('job_id', 'pending'))}: "
            f"{esc(repair_status['status'])}.</strong> Two equal-budget feedforward/log-reward "
            "runs compare translation-only control with optional wrist yaw. Each receives "
            "9.23M transitions and two balanced physical evaluations. The historical tables "
            "below describe the earlier completed experiments; "
            f"{href(repair_root / 'index.html', 'open the repair report for new results and the legacy reward audit')}.</p>",
        )

    transfer_rows = []
    transfer_root = ROOT / "artifacts/transfer_repair"
    for setting in ("clean", "dynamic"):
        path = transfer_root / f"pilot-evaluation-{setting}.json"
        if path.exists():
            result = read(path)
            records = result["termination_audit"]["records"]
            assert len(records) == result["episodes"] == len(result["outcomes"])
            assert (
                sum(r["success"] for r in records) == result["successes"] == sum(result["outcomes"])
            )
            transfer_rows.append(
                [
                    setting,
                    f"{result['successes']}/{result['episodes']} ({result['success_rate']:.1%})",
                    str(sum(r["timeout"] for r in records)),
                    str(sum(r["transfer_lifted"] for r in records)),
                    href(path, "Physical episode audit"),
                ]
            )
    if transfer_rows:
        section(
            "transfer-pilot",
            "Completed transfer pilot",
            "<p>Job 51644983 completed the first transfer learning pilot: wrist+static view0, "
            "seed0, compact GRU, pick_place_v2 / transfer_staged, 501 updates /3,078,144 "
            "transitions. This is a single pilot; active and wrist-only transfer runs do not exist. "
            "Clean and dynamic validation each use 512 episodes at seed20000.</p>"
            + table(
                [
                    "Evaluation",
                    "Success",
                    "Timeouts",
                    "Episodes with valid pickup history",
                    "Evidence",
                ],
                transfer_rows,
            )
            + "<p>Valid pickup history requires bilateral contact while lifting at least25 mm. "
            "The model has not demonstrated task competence; scripted feasibility is separate evidence.</p>",
        )

    section(
        "latest",
        "Completed exploration-repair baseline",
        f"""
<div class="callout"><strong>Completed baseline: static camera + wrist, fresh exploration repair.</strong>
Training finished October 9, 2026 at 04:05:23 EDT; its two evaluation repeats finished at 04:08:45 EDT.
The model achieved {score(latest[0])} and {
            score(latest[1])
        }. Neither repeat passed the all-variant competence gate.</div>
<div class="cards"><div class="card"><span>Completed updates</span><b>751</b><small>9,228,288 transitions</small></div>
<div class="card"><span>Validation repeat 1</span><b>32.62%</b><small>167 / 512 episodes</small></div>
<div class="card"><span>Validation repeat 2</span><b>30.66%</b><small>157 / 512 episodes</small></div>
<div class="card"><span>XM success</span><b>0 / 128</b><small>In each repeat</small></div></div>
{
            table(
                ["Field", "Recorded value"],
                [
                    [
                        "Task / setting",
                        "Hidden-prong plug insertion · clean occlusion · fixed view 7 · wrist RGB + fixed RGB · GRU · training seed 0",
                    ],
                    [
                        "Optimization",
                        "Progress reward; initial action std 1.0; entropy coefficient 0.01; initial LR 0.0003, adaptive; N512 × 24 rollout steps",
                    ],
                    [
                        "Corrections",
                        "Current-qpos success and terminal bonus; potential shaping retains derived positions. Recurrent critic bootstrap-memory correction applies from the first update.",
                    ],
                    ["Jobs", "Training 51531364; evaluation 51531365; both completed, exit 0:0"],
                    ["Source", "<code>11a591f7a2129484d555caf115fcdbf1a709a0c9</code>"],
                    [
                        "Checkpoint",
                        "<code>"
                        + esc(str(Path(audit["checkpoint"]).relative_to(ROOT)))
                        + "</code>",
                    ],
                    ["Checkpoint SHA-256", "<code>" + checkpoint_hash + "</code>"],
                    [
                        "Training run",
                        '<a href="https://wandb.ai/pgozdil-harvard-university/active-perception-so101/runs/4d4de324264d">W&amp;B 4d4de324264d</a>',
                    ],
                    [
                        "Completion evidence",
                        href(
                            BASE / "exploration_repair/completion_verified.json",
                            "31/31 saved integrity checks",
                        )
                        + " · "
                        + href(
                            BASE / "exploration_repair/progress/run_manifest.json", "Run manifest"
                        ),
                    ],
                ],
            )
        }
<p>A subsequent <strong>frozen-policy diagnostic</strong> (51542399), with no optimizer updates, scored 179/512 (34.96%) in another numerical execution of this same model. It does not replace either predeclared validation repeat. Later recipe/yaw learning runs and the transfer pilot are reported above.</p>
<p>There is no active or wrist-only counterpart trained under this fresh repair configuration. Comparing its 32.62% directly with the older active policy would mix training budgets, source revisions, success criteria and optimization settings.</p>""",
    )

    section(
        "coverage",
        "Task coverage and the two logs",
        f"""
<p>Plug insertion has the only completed three-way sensing comparison in the SO-101 project. Transfer now has a completed wrist+static pilot, with no active or wrist-only counterpart. A missing trained result is shown as unavailable, never as zero success.</p>
{
            table(
                ["Task / project", "Training evidence", "Static", "Active", "Wrist only"],
                [
                    [
                        "Hidden-prong plug · SO-101",
                        "Completed matched runs and static-only tuning/repairs",
                        "Completed",
                        "Completed",
                        "Completed",
                    ],
                    [
                        "Bin-to-cubby transfer · SO-101",
                        "Completed wrist+static pilot,501 updates"
                        if transfer_rows
                        else "Pilot pending",
                        "0/512 clean;0/512 dynamic (pilot)" if transfer_rows else "Unavailable",
                        "Unavailable",
                        "Unavailable",
                    ],
                    [
                        "Occluded planar pushing · SO-101",
                        "Pilot prepared; deferred after plug competence gate failed",
                        "Unavailable",
                        "Unavailable",
                        "Unavailable",
                    ],
                    [
                        "Insertion / camera decimation · neighboring M3L project",
                        "May 27 RESEARCH_LOG records K=1/2/4/8 experiment with results pending",
                        "No completed comparison in this log",
                        "Results pending in this log",
                        "No wrist-only result in this log",
                    ],
                ],
            )
        }
<p>The requested PROGRESS is this repository’s {
            href(OUT / "sources/PROGRESS.md", "PROGRESS.md snapshot")
        }, through its final October 9 entry.
No RESEARCH-LOG file exists in this checkout; the neighboring {
            href(
                OUT / "sources/M3L_RESEARCH_LOG.md",
                "M3L-active-perception/RESEARCH_LOG.md snapshot",
            )
        } is the matching file located and read. Its latest experiment entry is May 27 and contains plans, not measured success rates. This report does not claim a full audit of unrelated M3L run directories.</p>
<p>{
            href(ROOT / "artifacts/dynamic_occlusion/follow_on_decision.json", "Follow-on decision")
        } records the earlier deferral of transfer/push. Transfer was subsequently authorized and its first pilot is reported above; pushing remains untrained. Scripted native/Warp demonstrations, visibility screens and CUDA smoke tests demonstrate feasibility or pipeline behavior, not trained-policy performance.</p>""",
    )

    section(
        "comparison",
        "Matched static vs active vs wrist-only comparison",
        f"""
<p><strong>Matched training budget:</strong> 18,432,000 transitions / 1,500 PPO updates per policy; final model_1499; seed 0; N512; recurrent GRU; clean occlusion. Each checkpoint was then evaluated using the corrected physical criterion on 512 balanced validation episodes (128 per variant), N128, reset seeds 10000–10003.</p>
{
            figure(
                comparison_plot,
                "Matched original checkpoints, corrected physical evaluation. Error bars are 95% Wilson episode intervals, not training-seed uncertainty.",
            )
        }
{variant_table(matched_rows, lambda r: LABELS[r["condition"]])}
<p>In this validation execution, active exceeds static by <strong>15.43 percentage points</strong> and wrist-only by <strong>18.36 points</strong>. Initial camera positioning exceeds continued active control by <strong>8.79 points</strong>. These are observed differences for one trained seed; they do not establish a general ranking.</p>
{
            table(
                ["Sensing condition", "What the policy receives / controls"],
                [
                    ["Wrist only", "Wrist RGB and proprioception, with recurrent memory."],
                    [
                        "Static + wrist",
                        "Wrist RGB plus a fixed external view (view 7). The fixed view is a diagnostic candidate; a trained best-static search was not completed.",
                    ],
                    [
                        "Active + wrist",
                        "Wrist RGB plus the camera-arm image; policy commands the camera throughout the episode.",
                    ],
                    [
                        "Initial positioning + wrist",
                        "Policy selects camera position during the first 1 s, then camera targets freeze. External images, wrist images, proprioception and GRU continue updating. This is not a single-snapshot condition.",
                    ],
                ],
            )
        }
<p>All share a 1 s inspection pause and nominal 3.5 s horizon at 25 Hz, 128×96 policy images, and four hidden prong offsets (XM/XP/YM/YP, ±15 mm along the indicated axis). The actor has no privileged offset/goal state. <strong>Physical success requires body-position error within 2 mm at three consecutive control samples.</strong> This checks sampled holding, not continuous dwell between samples.</p>
{
            figure(
                variant_plot,
                "Per-variant physical success at the matched 18.432M-transition budget. Every condition fails all 128 XM episodes.",
            )
        }
<p>{href(BASE / "corrected_performance_metrics.csv", "Full comparison CSV")} · {
            href(BASE / "corrected_evaluation_episodes.csv", "Episode-level export")
        } · {href(BASE / "training_audit.json", "Training completion audit")}</p>""",
    )

    section(
        "failures",
        "Failure cases and terminal accuracy",
        f"""
<p><strong>The shared failure is XM.</strong> Every learned configuration in the reported matched comparisons, hyperparameter continuations and repaired runs scores 0/128 on XM in every physical repeat. Other failure patterns depend strongly on the policy: wrist-only succeeds only on YP; static view 7 primarily succeeds on XP; initial/active cover XP, YM and YP but incompletely.</p>
{
            table(
                [
                    "Condition",
                    "Unsuccessful",
                    "Failure rate",
                    "Mean terminal error, all episodes",
                    "Mean successful completion",
                    "Mean camera joint travel",
                ],
                [
                    [
                        LABELS[r["condition"]],
                        str(512 - int(r["successes"])) + "/512",
                        pct(r["unsuccessful_fraction"]),
                        f"{1000 * float(r['mean_terminal_position_error_m']):.2f} mm",
                        f"{float(r['mean_success_completion_seconds']):.2f} s",
                        f"{float(r['mean_camera_joint_travel_rad']):.3f} rad",
                    ]
                    for r in matched_rows
                ],
            )
        }
<h3>Where unsuccessful episodes end</h3>
<p>The following bins are computed from raw physical qpos at the first terminal sample of every unsuccessful episode. Counts are mutually exclusive; these are endpoint descriptions, not causal diagnoses or closest-approach measurements.</p>
{
            table(
                [
                    "Condition",
                    "Failed episodes",
                    "≤2 mm, hold unmet",
                    ">2–5 mm",
                    ">5–10 mm",
                    ">10 mm",
                    "Timeout among failures",
                    "Explicit failure flag",
                ],
                [
                    [
                        LABELS[r["condition"]],
                        r["failures"],
                        r["terminal_le2mm"],
                        r["terminal_2to5mm"],
                        r["terminal_5to10mm"],
                        r["terminal_gt10mm"],
                        r["failed_timeout"],
                        r["explicit_failure"],
                    ]
                    for r in failure_rows
                ],
            )
        }
<p>All unsuccessful matched episodes time out; none uses the explicit failure termination. A timeout flag can coexist with success at the horizon, so raw timeout totals must not be substituted for unsuccessful episode counts.</p>
{
            table(
                [
                    "XM condition",
                    "Median physical terminal error",
                    "Terminal error &lt;5 mm",
                    "Held success",
                ],
                [
                    [
                        LABELS[r["case"]],
                        f"{float(r['median_terminal_error_mm']):.2f} mm",
                        pct(r["terminal_below_5mm_fraction"]),
                        "0/128",
                    ]
                    for r in terminal
                    if r["stage"] == "original_checkpoints" and r["variant"] == "xm"
                ],
            )
        }
<p>Active and initial policies often approach XM closely yet miss the strict hold requirement. The latest static repair’s passive diagnostic similarly finds <strong>7/128 XM episodes sample below 2 mm at least once, but none sustain three samples</strong>; median closest sampled XM error is 2.248 mm. This distinguishes near-goal approach from valid insertion.</p>
<p>A privileged scripted controller in actual Warp succeeds on <strong>503/512 (98.24%)</strong> matched first episodes: XM 119/128, all other variants 128/128. Its nine failures are XM approaches in a high-goal-X corner, with roughly 63 mm body error and 53–55 mm derived TCP tracking error. This establishes sampled control feasibility, including most XM worlds; it does not identify why the learned policies fail. {
            href(BASE / "randomized_feasibility_gpu_verified.json", "GPU feasibility audit")
        }.</p>
<p>{href(OUT / "failure_breakdown.csv", "Computed failure-bin CSV")} · {
            href(BASE / "physical_terminal_errors.csv", "All physical terminal-error summaries")
        }</p>""",
    )

    settings = {
        "baseline": ("Baseline", "3e-4 adaptive", ".003"),
        "fixed_lr": ("Fixed LR", "1e-4 fixed", ".003"),
        "entropy": ("Higher entropy", "3e-4 adaptive", ".01"),
        "both": ("Fixed LR + higher entropy", "1e-4 fixed", ".01"),
    }
    hp_rows = []
    for case in cases:
        r1, r2 = [
            next(r for r in tuning if r["case"] == case and r["repeat"] == str(i)) for i in (1, 2)
        ]
        hp_rows.append(
            [
                settings[case][0],
                settings[case][1],
                settings[case][2],
                score(hp_early[case]),
                f"{historical_tuning[case]}/512 ({historical_tuning[case] / 512 * 100:.2f}%)",
                score(r1),
                score(r2),
            ]
        )
    section(
        "hyperparameters",
        "Hyperparameter search: every completed stage",
        f"""
<div class="callout"><strong>LR/entropy tuning improved measured validation performance over its matched baseline, but no configuration solved all variants.</strong> At 751 updates, corrected physical scores are 25.20% / 21.29% for fixed LR and 23.44% / 25.78% for higher entropy, versus 3.91% / 5.08% baseline. Their ordering reverses across numerical repeats; neither is a robust winner.</div>
<p>All four cases use <strong>static view 7 + wrist, training seed 0, N512, GRU, clean plug insertion</strong>. Initial action std is 0.4, with 4 PPO epochs and 8 minibatches. The fixed-LR case changes both the initial LR and its schedule; it does not isolate schedule alone. The first screen uses 100 updates / 1,228,800 transitions. Each longer run has <strong>751 total updates / 9,228,288 transitions</strong>, including the initial 100 (651 additional updates / 7,999,488 transitions).</p>
{table(["Case", "Initial LR / schedule", "Entropy", "100 updates: historical score", "751 updates: historical score", "751 updates: physical repeat 1", "751 updates: physical repeat 2"], hp_rows)}
<p><strong>Criteria are separate:</strong> historical columns use the original derived-substep rule. Physical columns are independent current-qpos evaluations of those same historically trained checkpoints. They do not represent retraining with the corrected rule, and subtracting historical from physical scores does not isolate scoring effects because trajectories also vary numerically.</p>
{figure(tuning_plot, "LR/entropy search at a matched 9.228M-transition budget. Each repeat has 512 balanced episodes; error bars are Wilson episode intervals.")}
<h3>Physical validation by variant</h3>
{figure(hp_learning_plot, "Actual LR/entropy search histories through 751 updates. Dashed line marks continuation after 100 updates; reward uses a 30-update moving mean. These runs trained with the historical criterion.")}
<p>{href(history_csv, "Download all 3,004 matched-budget hyperparameter history rows")}.</p>
{variant_table(tuning, lambda r: settings[r["case"]][0] + " · repeat " + r["repeat"])}
<p>The short screen (1–7 successes per 512) was inconclusive. At 751 updates, fixed LR succeeds only on YM/YP; the higher-entropy run has XP/YM/YP successes; the combined change has no XP or XM physical successes. None meets the competence target of at least <strong>103/128 in each variant in each repeat</strong>.</p>
<p>All three historical continuations have verified model_750, contiguous 751-update histories, 9,228,288 cumulative transitions and finished W&amp;B runs. The initial fixed-LR evaluation encountered a GPU ECC error; its evaluation-only retry completed with the same checkpoint. No learner was rerun for that hardware failure.</p>
<p>{href(hp_early_path, "100-update comparison CSV")} · {href(ROOT / "artifacts/hparam_search/continuation/completion_verified.json", "Continuation completion")} · {href(BASE / "corrected_hparam_metrics.csv", "Physical tuning CSV")} · {href(ROOT / "artifacts/hparam_search/continuation/physical_completion_verified.json", "Physical-repeat verification")}</p>
<h3>Corrected reward screen and continuations</h3>
<p>Two fresh objective profiles then trained with current-qpos physical sampling: progress shaping and log-distance plus a held-success bonus. The log profile changes both shaping and bonus magnitude; its reward scale is not comparable to progress reward. Both use initial std 0.4 and matched budgets.</p>
{variant_table(repair_screen, lambda r: r["profile"] + " · 100 updates · repeat " + r["repeat"])}
{variant_table(repair, lambda r: r["profile"] + " · 751 updates · repeat " + r["repeat"])}
<p>Their first 100 updates use source <code>434c37a</code>; both receive the recurrent critic-memory repair for the subsequent 651 updates at <code>11a591f</code>. This shared mixed-source continuation does not isolate the effect of that correction. Progress reaches 4.10% / 3.52%; log-distance reaches 2.15% / 3.52%. Neither establishes competent insertion.</p>
<p>{href(BASE / "reward_repair_metrics.csv", "100-update physical reward screen")} · {href(BASE / "reward_repair_continuation_metrics.csv", "751-update physical reward continuations")} · {href(BASE / "reward_repair/continuation/completion_verified.json", "Completion audit")}</p>
<h3>Latest fresh exploration repair</h3>
<p>One fresh progress run changes initial std to 1.0 and entropy to .01 together, retaining adaptive LR .0003. It applies both confirmed code corrections from update one and trains for 751 updates. This is a new configuration, not another matched sensing condition or an isolated std/entropy ablation.</p>
{variant_table(latest, lambda r: "Fresh progress · repeat " + r["repeat"])}
<p>It improves observed aggregate success over the corrected objective continuations, but still fails XM entirely. Repeats disagree on 128/512 episode outcomes despite identical initial physics and both image streams. These are two numerical executions of one trained seed, not two independent training replicates; do not pool them as a 1,024-episode independent trial.</p>
<p>{href(BASE / "exploration_repair_metrics.csv", "Latest repair CSV")} · {href(BASE / "exploration_repair/decision.json", "Configuration decision")} · {href(BASE / "exploration_repair/completion_verified.json", "Completion and competence audit")}</p>""",
    )

    section(
        "learning",
        "Learning curves and training quality",
        f"""
{figure(BASE / "assets/learning_curves.svg", "Original matched policies: reward, native reset-batch training success, episode length and task reward. Training success is not held-out balanced validation.")}
{table(["Original condition", "Reward: first 100 updates", "Reward: last 100 updates", "Descriptive plateau", "Measured training GPU-hours"], [[LABELS[c], f"{analysis['convergence'][c]['first_100_reward_mean']:.3f}", f"{analysis['convergence'][c]['last_100_reward_mean']:.3f}", "Yes" if analysis["convergence"][c]["descriptive_reward_plateau"] else "No", f"{float(matched[c]['training_gpu_hours']):.2f}"] for c in ORDER])}
<p>All four original reward curves rose and passed the recorded descriptive tail-300 plateau heuristic. Their low physical validation success shows that a reward plateau is not a competence criterion. All spend the last 100 updates at the adaptive LR floor; narrow action distributions motivate exploration diagnostics without proving a failure cause.</p>
{figure(BASE / "assets/exploration_learning_progress.svg", "Latest fresh exploration repair: recorded 751-update training history. Reward and training metrics remain distinct from the two physical validation repeats.")}
<p>Latest repair reward rises from <strong>{audit["learning"]["first100_reward_mean"]:.4f} to {audit["learning"]["last100_reward_mean"]:.4f}</strong> across the first/last 100-update windows. The final 100 updates contain <strong>4,900 successes / 15,362 completed training episodes (31.90%)</strong>. Its tail-300 fitted reward change ({audit["learning"]["last_300_reward_fitted_change"]:.4f}) exceeds the plateau tolerance ({audit["learning"]["descriptive_plateau_tolerance"]:.4f}), so the recorded plateau heuristic fails. That does not authorize a budget extension.</p>
<details><summary>Corrected objective training curves</summary>{figure(BASE / "assets/repair_learning_progress.svg", "Corrected progress objective: initial screen and continuation.")}{figure(BASE / "assets/repair_learning_legacy_log_hold.svg", "Corrected log-distance objective: initial screen and continuation. Reward values use a different scale.")}</details>
<p>The plateau rule requires both fitted tail-300 change and final-150 versus prior-150 mean change to be within 10% of max(0.1, absolute tail reward mean). It is descriptive, not a statistical convergence test. {href(BASE / "learning_curves.csv", "Original histories")} · {href(BASE / "repair_learning_curves.csv", "Objective repair histories")} · {href(BASE / "exploration_learning_curves.csv", "Fresh repair history")}.</p>""",
    )

    control_groups = list(dict.fromkeys(r["case"] for r in controls))
    section(
        "interventions",
        "Camera and memory interventions",
        f"""
<p>These evaluations keep the original trained model_1499 weights frozen. Each row has two independent numerical executions of the same validation initial states; all use physical scoring.</p>
{table(["Frozen-checkpoint intervention", "Repeat 1", "Repeat 2", "XM"], [[next(r["case_label"] for r in controls if r["case"] == case)] + [score(next(r for r in controls if r["case"] == case and r["repeat"] == str(i))) for i in (1, 2)] + ["0/128 in both"] for case in control_groups])}
<p>Freezing the active policy’s camera targets after 1 s reduces success by 15.43 / 16.41 percentage points. It also changes camera embodiment/contact and allows joints to settle, so this is not a pure visual causal effect. Holding the initial policy’s external image after 1 s changes success by 0 / −0.20 points; wrist sensing, proprioception and GRU remain live. Resetting its whole GRU every step drops success by 28.52 / 27.15 points, but also changes inspection and view selection.</p>
<p>These checks show dependence of specific checkpoints on camera commands or memory. They do not establish that continuous active control beats retrained initial-only or feedforward alternatives. {href(BASE / "camera_control_metrics.csv", "All intervention metrics")} · {href(BASE / "camera_controls/completion_verified.json", "Intervention audit")}.</p>""",
    )

    section(
        "diagnostic",
        "Latest frozen-policy diagnosis",
        f"""
<p>The final static repair was instrumented passively, without learner updates: 512 episodes, 40,159 live first-episode samples, 512 inspection snapshots, and 352 real actor forwards. Initial physics/RGB matched the validation reference, and actor/normalization buffers remained unchanged. Its 179/512 score is a diagnostic execution of the same checkpoint.</p>
{table(["Variant", "Physical success", "Ever sampled body error &lt;2 mm", "Median closest sampled body error"], [[r["variant"].upper(), score(r), r["episodes_sampled_body_error_below2mm"] + "/" + r["episodes"], f"{float(r['median_minimum_sampled_body_error_mm']):.3f} mm"] for r in diagnostic])}
<p>Fixed-ridge probes fit on three reset seeds and evaluate on the fourth, using train-only normalization and no holdout tuning. CNN variant accuracy is 64.84–72.66%, versus 25.00–30.47% from proprioception. CNN median socket-XY error is 6.26–6.56 mm; GRU accuracy is 57.81–62.50% with median 6.68–7.56 mm. These probe errors are not physical body-goal errors.</p>
<details><summary>All 16 accessibility-probe folds</summary>{table(["Features", "Held-out reset seed", "Variant accuracy", "Median socket XY error", "Socket XY RMSE"], [[r["features"], r["heldout_reset_seed"], pct(r["variant_accuracy"]), f"{float(r['socket_xy_median_error_mm']):.2f} mm", f"{float(r['socket_xy_rmse_euclidean_mm']):.2f} mm"] for r in probes])}</details>
<p>Proprioception and combined-feature seed-10003 RMSE reach 1,332.38 and 532.17 mm despite ordinary medians: those folds are unstable and are retained. Accessibility probes cannot establish policy use, missing information, architectural necessity or a causal failure explanation. Large late command/tracking residuals also may reflect legitimate contact compensation. No universal action-box/goal-sign bug was identified.</p>
<p>{href(BASE / "target_diagnostic/report.json", "Full diagnostic and phase summaries")} · {href(BASE / "target_diagnostic/completion_verified.json", "35/35 saved verification checks")} · {href(BASE / "target_feature_probe_metrics.csv", "Probe CSV")}.</p>""",
    )

    representatives = rows("corrected_representative_trajectories.csv")
    cards = []
    for condition in ORDER:
        subset = [r for r in representatives if r["condition"] == condition]
        for r in [
            next(r for r in subset if r["variant"] == "xm"),
            next(r for r in subset if r["success"] == "True"),
        ]:
            poster = BASE / f"assets/corrected-{condition}_{r['episode']}_outside_video.jpg"
            cards.append(
                media_card(r, LABELS[condition] + " · matched checkpoint", condition, poster)
            )
    latest_media = read(
        BASE / "exploration_repair/progress/evaluation-repeat1-videos/representatives.json"
    )
    for r in [
        next(r for r in latest_media["records"] if r["variant"] == "xm"),
        next(r for r in latest_media["records"] if r["variant"] == "ym" and r["expected_success"]),
    ]:
        cards.append(media_card(r, "Latest static exploration repair · repeat 1", "latest"))
    section(
        "videos",
        "Recorded success and failure examples",
        """
<p>These are actual captured evaluated trajectories. Outside videos are rendered from saved physics states; the policy-input links show the exact recorded images seen by the actor. Selection is the first observed success/failure per variant in the source captures. This gallery displays each matched policy’s first XM failure and first captured success, plus XM failure/YM success for the fresh repair. It illustrates behavior, not failure prevalence.</p>
<div class="filters"><label>Condition <select id="condition-filter"><option value="all">All conditions</option><option value="wrist">Wrist only</option><option value="wrist_static">Static + wrist</option><option value="active">Active + wrist</option><option value="initial">Initial positioning</option><option value="latest">Latest repair</option></select></label>
<label>Outcome <select id="outcome-filter"><option value="all">All outcomes</option><option value="success">Success</option><option value="failure">Failure</option></select></label><span id="media-count" aria-live="polite"></span></div>
<div class="media-grid">"""
        + "".join(cards)
        + "</div>",
    )

    section(
        "criteria",
        "Historical scores, limitations and provenance",
        f"""
<h3>Why older success rates differ</h3>
{table(["Original model_1499", "Historical derived-substep score", "Separate corrected physical execution"], [[LABELS[c], score(historical[c]), score(matched[c])] for c in ORDER])}
<p>An audit found that the historical derived-state rule could flag success while current physical qpos lay outside 2 mm. The physical rule fixes sampling, but re-evaluation also changes trajectories numerically. Historical scores remain archived and are not pooled with current-qpos scores. Every physical success reported in this digest has zero recorded distance, hold, three-sample and raw-qpos violations.</p>
<p>All SO-101 scientific comparisons here use <strong>one training seed</strong>, exploratory validation, and clean occlusion. Repeated execution can change episode outcomes despite identical initial states and images. Wilson intervals describe episode-count uncertainty within an execution; they do not capture training-seed, simulator or selection uncertainty. A searched static baseline, independent training seeds and untouched test evaluation remain missing.</p>
<p>Legacy pl3 summaries report roughly 98–100% training success under different architecture, reward, horizon and budget. Those saved training values are not independently reproduced held-out results and are excluded from the matched tables. {href(BASE / "legacy_run_inventory.json", "Recovered legacy-run inventory")}.</p>
<p><strong>Evidence-based status:</strong> camera access improves observed performance in the original matched seed, with initial positioning above continuous active control. Optimization repair improves static aggregate performance, while universal XM failure and numerical variability prevent a claim of reliable insertion. The first transfer pilot has also failed to demonstrate competence; pushing remains untrained.</p>
<h3>Downloads and reproducibility</h3>
<p>{href(OUT / "summary.json", "Report data and source hashes")} · {href(OUT / "matched_comparison.csv", "Matched comparison CSV")} · {href(OUT / "failure_breakdown.csv", "Failure bins CSV")} · {href(BASE / "index.html", "Existing full technical report")}.</p>
<p>Rebuild from the repository root with <code>.venv/bin/python scripts/build_training_results_report.py</code>. The builder reads saved evidence and verifies published physical counts against raw episode flags and variant assignments. It hashes the latest checkpoint without loading it or running inference. Charts are embedded in this HTML; video/source links use local relative paths. Extract <code>training-results.zip</code> and open its root <code>index.html</code> for a portable copy with the selected media and evidence.</p>
<p class="small">Snapshot built {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}. Completion times and job states are from saved verified records; this build does not query the live scheduler or W&amp;B.</p>""",
    )

    style = """
:root{color-scheme:light;--ink:#142d40;--muted:#536879;--line:#dce5eb;--blue:#1d4ed8}
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:16px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--blue);text-underline-offset:3px}a:hover{color:#0f766e}
header{background:#102d40;color:#fff;padding:54px max(24px,calc((100vw - 1180px)/2)) 40px}header p{max-width:900px;color:#d3e5ee}h1{font-size:clamp(30px,4vw,48px);line-height:1.12;margin:10px 0 20px;letter-spacing:-1.4px}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:12px;color:#8de0cf}
nav{background:white;border-bottom:1px solid var(--line);padding:16px 24px;display:flex;justify-content:center;gap:12px 22px;flex-wrap:wrap}nav a{font-size:14px;font-weight:650;text-decoration:none}main{max-width:1240px;margin:auto;padding:20px 24px 64px}section{background:#fff;border:1px solid var(--line);border-radius:12px;padding:28px 32px;margin:24px 0;scroll-margin-top:20px}h2{font-size:26px;line-height:1.25;margin:0 0 20px;letter-spacing:-.5px}h3{font-size:19px;margin:26px 0 10px}p{margin:14px 0}.callout{background:#edf7f5;border-left:4px solid #059669;padding:18px 22px;border-radius:5px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.card{background:#f5f8fb;border:1px solid var(--line);padding:20px;border-radius:8px}.card span,.card small{display:block;color:var(--muted)}.card b{display:block;font-size:32px;line-height:1.5}.table-wrap{overflow-x:auto;margin:20px 0}table{width:100%;border-collapse:collapse;font-size:14px}th{background:#edf3f7;font-weight:650;text-align:left}td,th{padding:11px 13px;border-bottom:1px solid var(--line);vertical-align:top}tbody tr:nth-child(even){background:#fafcfd}td:first-child{font-weight:550}small,.small,figcaption{font-size:13px;color:var(--muted)}td small{white-space:nowrap}figure{margin:25px 0;border:1px solid var(--line);border-radius:8px;padding:16px}figure img{width:100%;display:block}figcaption{margin-top:12px}code{overflow-wrap:anywhere;font-size:12px;background:#f0f4f7;padding:2px 5px;border-radius:4px}details{border:1px solid var(--line);padding:14px 18px;border-radius:8px;margin:20px 0}summary{cursor:pointer;font-weight:650}.media-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.media-card{border:1px solid var(--line);border-radius:10px;padding:16px}.media-card h3{margin:0;font-size:17px}video{display:block;width:100%;aspect-ratio:16/9;background:#102d40;border-radius:6px}.filters{display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin:24px 0}select{padding:8px;background:white;border:1px solid #afbfcc;border-radius:5px;font:inherit;max-width:100%}[hidden]{display:none!important}footer{color:var(--muted);text-align:center;padding:20px}button{font:inherit;cursor:pointer;border:1px solid #aabcca;background:white;color:#123045;padding:7px 12px;border-radius:6px}
@media(max-width:700px){header{padding:32px 20px}main{padding:10px 12px 30px}section{padding:22px 16px}.cards{grid-template-columns:repeat(2,1fr)}.card{padding:14px}.card b{font-size:26px}.media-grid{grid-template-columns:1fr}th,td{padding:9px}h2{font-size:23px}}
@media print{body{background:white;font-size:11px}header{background:white;color:#142d40;padding:10px}header p{color:#536879}nav,.filters,button{display:none}main{padding:0;max-width:none}section{border:0;margin:0;padding:16px 0;break-inside:auto}.card,figure,tr,.media-card{break-inside:avoid}figure img{max-height:500px}a{color:#142d40}video{max-height:220px}h2{break-after:avoid}}
"""
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Training results · SO-101 active perception · October 9, 2026</title><style>{style}</style></head><body>
<header><div class="eyebrow">Research results · evidence snapshot · October 9, 2026</div><h1>What is trained, what works,<br>and where it fails.</h1><p>Static, active and wrist-only sensing on the restored hidden-prong plug task, with the complete hyperparameter search and latest physical validation. Transfer and pushing have no completed trained-policy comparison.</p></header>
<nav aria-label="Report sections"><a href="#latest">Latest run</a><a href="#coverage">Task coverage</a><a href="#comparison">Camera comparison</a><a href="#hyperparameters">Hyperparameter search</a><a href="#failures">Failures</a><a href="#learning">Learning</a><a href="#videos">Videos</a><a href="#criteria">Sources</a><a href="../plug_repair/index.html">Repair investigation</a><button onclick="window.print()">Print / save PDF</button></nav>
<main>{"".join(sections)}</main><footer>Built from saved research evidence. One training seed; validation results remain exploratory.</footer>
<script>const condition=document.getElementById('condition-filter'),outcome=document.getElementById('outcome-filter');function filterMedia(){{let n=0;document.querySelectorAll('.media-card').forEach(card=>{{card.hidden=!((condition.value==='all'||card.dataset.condition===condition.value)&&(outcome.value==='all'||card.dataset.result===outcome.value));if(!card.hidden)n++;}});document.getElementById('media-count').textContent=n+' examples';}}condition.addEventListener('change',filterMedia);outcome.addEventListener('change',filterMedia);filterMedia();</script></body></html>"""
    (OUT / "index.html").write_text(document)
    for name, data in [
        ("matched_comparison.csv", matched_rows),
        ("failure_breakdown.csv", failure_rows),
    ]:
        with (OUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "SO-101 completed scientific training; neighboring M3L research-log status only",
        "source_snapshots": source_snapshots,
        "latest_checkpoint": str(Path(audit["checkpoint"]).relative_to(ROOT)),
        "latest_checkpoint_sha256": checkpoint_hash,
        "latest": latest,
        "matched": matched_rows,
        "hyperparameter_physical": tuning,
        "reward_screen": repair_screen,
        "reward_continuation": repair,
        "failure_bins": failure_rows,
        "raw_physical_evaluations_verified": sum(
            map(len, [matched_rows, latest, tuning, repair_screen, repair, controls])
        ),
        "sources": [],
    }
    for path in sorted(SOURCES):
        if path.is_file() and path != OUT / "summary.json":
            summary["sources"].append(
                {
                    "path": os.path.relpath(path, ROOT),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Preserve repository-relative layout. Include directly linked evidence/media;
    # the existing giant technical report remains a workspace-only link in this ZIP.
    targets = []
    for target in re.findall(r'(?:href|src)="([^"]+)"', document):
        if target.startswith(("#", "https:", "data:")):
            continue
        path = (OUT / html.unescape(target)).resolve()
        assert path.is_file(), f"Missing report link: {target}"
        targets.append(path)
    bundle_document = document.replace(
        href(BASE / "index.html", "Existing full technical report"),
        "Existing full technical report (workspace only)",
    )
    bundle_files = set(targets) - {BASE / "index.html"}
    bundle_files.update(
        p
        for p in SOURCES
        if p.is_relative_to(ROOT) and p.suffix in {".json", ".csv", ".md", ".svg"}
    )
    bundle_files.update(OUT.glob("*.pdf"))
    # Include the follow-up diagnosis and its local evidence, excluding model binaries.
    bundle_files.update(
        p
        for p in (ROOT / "artifacts/plug_repair").rglob("*")
        if p.is_file() and p.suffix in {".html", ".json", ".md", ".py"}
    )
    bundle_files.update(
        p
        for p in (ROOT / "artifacts/xm_diagnosis").rglob("*")
        if p.is_file() and p.suffix in {".html", ".json", ".svg", ".pdf", ".png"}
    )
    bundle_files.add(BASE / "randomized_feasibility_gpu_verified.json")
    # Preserve the independently maintained transfer/push appendix and its evidence.
    bundle_files.update(
        p
        for p in (ROOT / "artifacts/transfer_push").rglob("*")
        if p.is_file() and p.suffix in {".json", ".html", ".svg", ".pdf", ".mp4"}
    )
    bundle_manifest = []
    with zipfile.ZipFile(OUT / "training-results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        report_path = (OUT / "index.html").relative_to(ROOT).as_posix()
        archive.writestr(
            "index.html",
            f'<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={report_path}"><title>Training report</title><a href="{report_path}">Open report</a></html>',
        )
        archive.writestr(report_path, bundle_document)
        for path in sorted(bundle_files):
            assert path.is_relative_to(ROOT)
            data = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            archive.writestr(
                name,
                data,
                compress_type=zipfile.ZIP_STORED if path.suffix == ".mp4" else zipfile.ZIP_DEFLATED,
            )
            bundle_manifest.append(
                {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
        archive.writestr("manifest.json", json.dumps(bundle_manifest, indent=2))
        assert archive.testzip() is None
    (OUT / "validation.json").write_text(
        json.dumps(
            {
                "built_at": datetime.now(UTC).isoformat(),
                "raw_evaluations_checked": summary["raw_physical_evaluations_verified"],
                "local_links_checked": len(targets),
                "latest_checkpoint_hash_matches": True,
                "failure_bins_sum_correctly": True,
                "bundle_crc_passed": True,
                "bundle_files": len(bundle_files) + 3,
                "media_examples": len(cards),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "html": str(OUT / "index.html"),
                "zip_bytes": (OUT / "training-results.zip").stat().st_size,
                "validation": json.loads((OUT / "validation.json").read_text()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    from build_latest_training_results import main as build_latest

    build_latest()
