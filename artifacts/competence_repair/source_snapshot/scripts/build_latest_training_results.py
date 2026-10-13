"""Audited three-seed plug/push report from saved training and evaluation evidence."""

import base64
import csv
import hashlib
import html
import json
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/training_results"
CONDITIONS = ["wrist", "wrist_static", "active"]
LABELS = ["Wrist", "Wrist + fixed", "Wrist + active"]
COLORS = ["#2563eb", "#0f766e", "#c2410c"]
STYLE = """<style>:root{color-scheme:light}*{box-sizing:border-box}body{background:#fff;color:#182b3a;font:16px/1.6 system-ui;margin:0}header,main,footer{max-width:1200px;margin:auto;padding:28px}header{padding-top:48px}h1{font-size:42px;line-height:1.15}h2{margin-top:42px}a{color:#1763ae}nav{display:flex;gap:22px;flex-wrap:wrap;border-bottom:1px solid #ddd;padding-bottom:20px}section{scroll-margin-top:20px}img{width:100%;height:auto}figure{margin:24px 0}figcaption,small{color:#53616b;font-size:14px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}th{background:#f7f9fb}.note{border-left:4px solid #1763ae;padding:12px 20px;background:#f7fafc}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.clip{border:1px solid #ddd;border-radius:8px;padding:16px}video{width:100%;background:#fff}select{padding:8px;margin:8px;font:inherit;background:#fff}code{overflow-wrap:anywhere}[hidden]{display:none!important}@media(max-width:700px){.grid{grid-template-columns:1fr}header,main,footer{padding:18px}h1{font-size:30px}}@media print{nav,select,video{display:none}}</style>"""
SOURCES = set()


def read(path):
    path = Path(path).resolve()
    SOURCES.add(path)
    return json.loads(path.read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rel(path):
    return os.path.relpath(Path(path).resolve(), OUT)


def link(path, label):
    return f'<a href="{html.escape(rel(path))}">{html.escape(label)}</a>'


def table(headers, rows):
    return (
        '<div class="table"><table><thead><tr>'
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def figure(fig, name, caption):
    fig.tight_layout()
    for ext in ["svg", "pdf", "png"]:
        fig.savefig(OUT / "assets" / f"{name}.{ext}", facecolor="white", dpi=150)
    plt.close(fig)
    return f'<figure><img src="assets/{name}.svg" alt="{html.escape(caption)}"><figcaption>{caption} · <a href="assets/{name}.pdf">PDF</a></figcaption></figure>'


def audit_evaluation(path, entry, task):
    d = read(path)
    assert digest(path) == entry["sha256"], path
    assert len(d["outcomes"]) == d["episodes"] == entry["episodes"]
    assert sum(d["outcomes"]) == d["successes"] == entry["successes"]
    a = d["termination_audit"]
    assert len(a["records"]) == d["episodes"]
    for i, record in enumerate(a["records"]):
        assert record["episode"] == i and record["success"] == d["outcomes"][i]
        if record["success"]:
            assert record["success_hold_steps"] >= 3
            if task == "plug":
                assert max(record["last_three_manager_errors_m"]) < 0.002
            else:
                assert record["push_xy_error"] < 0.005
            if task == "push":
                assert all(record["push_criterion_history"])
                assert record["push_withdrawn"] and record["push_contact_free"]
    for flag in ["success", "failure", "timeout"]:
        assert sum(r[flag] for r in a["records"]) == a["terminal_flags"][flag]
    for key, value in a.items():
        if key.endswith("_violations") and value is not None:
            assert value == 0, (path, key, value)
    for variant, counts in d["per_variant"].items():
        indices = [i for i, v in enumerate(d["variants"]) if v == variant]
        assert len(indices) == counts["episodes"]
        assert sum(d["outcomes"][i] for i in indices) == counts["successes"]
    return d


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "assets").mkdir(exist_ok=True)
    if not (OUT / "historical.html").exists():
        shutil.copyfile(OUT / "index.html", OUT / "historical.html")
    runs, evaluations, media, source_checks = [], [], [], []
    folders = [
        ("plug", seed, c, ROOT / "artifacts" / folder / c)
        for seed, folder in enumerate(
            ["plug_insertion_study_v2", "plug_insertion_study_seed1", "plug_insertion_study_seed2"]
        )
        for c in CONDITIONS
    ]
    folders += [
        ("push", seed, c, ROOT / "artifacts/push_rgb_arc120/runs" / f"push_{c}_s{seed}")
        for seed in range(3)
        for c in CONDITIONS
    ]
    for plan_path in [
        ROOT / "artifacts" / p / "plan.json"
        for p in [
            "plug_insertion_study_v2",
            "plug_insertion_study_seed1",
            "plug_insertion_study_seed2",
        ]
    ] + [ROOT / "artifacts/push_rgb_arc120/source-manifest.json"]:
        plan = read(plan_path)
        for name, expected in plan["source_sha256"].items():
            p = Path(plan["source"]) / name
            assert digest(p) == expected, p
        source_checks.append(
            {
                "manifest": str(plan_path.relative_to(ROOT)),
                "files_verified": len(plan["source_sha256"]),
            }
        )
    for task, seed, condition, folder in folders:
        state = read(folder / "state.json")
        assert state["status"] == "completed", folder
        budget = 18001920 if task == "plug" else 1000046592
        assert state["transitions"] == budget
        assert digest(state["checkpoint"]) == state["checkpoint_sha256"]
        history_path = Path(state["run_directory"]) / "iterations.jsonl"
        SOURCES.add(history_path.resolve())
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        assert history[-1]["transitions"] == budget
        assert history[-1]["iteration"] == state["updates"]
        run = dict(
            task=task,
            seed=seed,
            condition=condition,
            transitions=budget,
            checkpoint=str(Path(state["checkpoint"]).resolve().relative_to(ROOT)),
            checkpoint_sha256=state["checkpoint_sha256"],
            updates=state["updates"],
            status=state["status"],
            history=history,
        )
        runs.append(run)
        for entry in state["evaluations"]:
            d = audit_evaluation(entry["path"], entry, task)
            assert d["training_seed"] == seed and d["split"] == entry["split"]
            assert d["experiment"]["condition"] == condition
            assert d["experiment"]["occlusion"] == entry["occlusion"]
            if entry["split"] == "test":
                assert (
                    entry["transitions"] == budget and d["seed"] == 80000 and d["episodes"] == 1024
                )
            evaluations.append(
                dict(
                    task=task,
                    seed=seed,
                    condition=condition,
                    split=d["split"],
                    occlusion=entry["occlusion"],
                    transitions=entry["transitions"],
                    successes=d["successes"],
                    episodes=d["episodes"],
                    rate=100 * d["success_rate"],
                    error_mm=1000 * d["termination_audit"]["mean_terminal_position_error_m"],
                    timeouts=d["termination_audit"]["terminal_flags"]["timeout"],
                    per_variant=d["per_variant"],
                    path=str(Path(entry["path"]).resolve().relative_to(ROOT)),
                )
            )
        for occlusion in ["clean", "dynamic"]:
            path = folder / f"videos-{occlusion}/representatives.json"
            records = read(path)["records"]
            for result in [True, False]:
                record = next((r for r in records if r["expected_success"] == result), None)
                if record:
                    assert record["reproduced_success"] == result
                    final = next(
                        e
                        for e in evaluations
                        if e["task"] == task
                        and e["seed"] == seed
                        and e["condition"] == condition
                        and e["split"] == "test"
                        and e["occlusion"] == occlusion
                    )
                    raw = read(ROOT / final["path"])
                    assert raw["outcomes"][record["episode"]] == result
                    for key in ["outside_video", "policy_video"]:
                        assert Path(record[key]).is_file()
                    media.append(
                        dict(
                            task=task,
                            seed=seed,
                            condition=condition,
                            occlusion=occlusion,
                            outcome="success" if result else "failure",
                            variant=record["variant"],
                            episode=record["episode"],
                            outside=rel(record["outside_video"]),
                            policy=rel(record["policy_video"]),
                        )
                    )
    finals = [e for e in evaluations if e["split"] == "test"]
    assert len(finals) == 36
    aggregates = []
    for task in ["plug", "push"]:
        for c in CONDITIONS:
            for occlusion in ["clean", "dynamic"]:
                es = sorted(
                    [
                        e
                        for e in finals
                        if (e["task"], e["condition"], e["occlusion"]) == (task, c, occlusion)
                    ],
                    key=lambda e: e["seed"],
                )
                rates = [e["rate"] for e in es]
                aggregates.append(
                    dict(
                        task=task,
                        condition=c,
                        occlusion=occlusion,
                        mean=float(np.mean(rates)),
                        std=float(np.std(rates, ddof=1)),
                        minimum=min(rates),
                        maximum=max(rates),
                        successes=sum(e["successes"] for e in es),
                        episodes=sum(e["episodes"] for e in es),
                    )
                )
    sections = []
    for task, title in [("plug", "Plug insertion"), ("push", "Precision pushing · arc120")]:
        subset = [e for e in finals if e["task"] == task]
        if task == "plug":
            narrative = "All nine policies completed 18,001,920 transitions (1,465 PPO updates) using the original insertion reward, absolute XYZ control and a recurrent RGB actor. Held success requires &lt;2 mm error for three samples. Success remains low; active has the highest mean, but this is not reliable insertion. Fixed-camera performance varies substantially between training seeds. XM is almost never solved (clean: 3/768 wrist, 1/768 fixed, 1/768 active); YM accounts for most successes. The earlier staged-reward experiment with zero success is a different recipe."
        else:
            narrative = "All nine policies completed 1,000,046,592 transitions (20,346 PPO updates). Targets lie 50–120 mm from the object over ±80°. Held success requires &lt;5 mm placement plus speed, height, withdrawal and contact criteria. Fixed-camera sensing has the highest mean in both settings; active does not beat wrist-only on average. The older near-ceiling square-target study was stopped and is not comparable. The arc reachability audit identified targets outside the current action range; aggregate failures alone cannot attribute the cause. Clean validation at 900M averaged 66.80% wrist, 68.88% fixed and 61.26% active. Final test means are lower, but the reset split also changes, so this does not by itself establish late-training regression."
        rows = []
        for c, label in zip(CONDITIONS, LABELS, strict=False):
            ag = [
                next(
                    a
                    for a in aggregates
                    if (a["task"], a["condition"], a["occlusion"]) == (task, c, o)
                )
                for o in ["clean", "dynamic"]
            ]
            rows.append(
                [label]
                + [
                    f"{a['mean']:.2f}% ± {a['std']:.2f} pp<br><small>{a['successes']}/{a['episodes']}; seed range {a['minimum']:.2f}–{a['maximum']:.2f}%</small>"
                    for a in ag
                ]
            )
        body = f'<section id="{task}"><h2>{title}</h2><p>{narrative}</p>' + table(
            ["Camera", "Clean test · mean ± seed SD", "Dynamic test · mean ± seed SD"], rows
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, o in zip(axes, ["clean", "dynamic"], strict=False):
            for i, (c, color) in enumerate(zip(CONDITIONS, COLORS, strict=False)):
                ys = [
                    next(
                        e["rate"]
                        for e in subset
                        if e["condition"] == c and e["seed"] == s and e["occlusion"] == o
                    )
                    for s in range(3)
                ]
                ax.errorbar(
                    i,
                    np.mean(ys),
                    yerr=np.std(ys, ddof=1),
                    fmt="_",
                    ms=24,
                    c=color,
                    capsize=7,
                    lw=2,
                )
                ax.scatter(np.array([i - 0.09, i, i + 0.09]), ys, c=color, s=40)
            ax.set(
                xticks=range(3),
                xticklabels=LABELS,
                ylabel="Held test success (%)",
                title=o.capitalize(),
                ylim=(0, max(e["rate"] for e in subset) * 1.35),
            )
            ax.grid(axis="y", alpha=0.2)
        body += figure(
            fig,
            f"{task}_final_success",
            "Dots are individual training seeds; bars show mean ± sample standard deviation across three seeds, not a confidence interval. Each dot uses 1,024 test episodes.",
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for c, color, label in zip(CONDITIONS, COLORS, LABELS, strict=False):
            for s in range(3):
                hist = next(
                    r["history"]
                    for r in runs
                    if (r["task"], r["condition"], r["seed"]) == (task, c, s)
                )
                valid = [h for h in hist if "train/Train/mean_reward" in h]
                window = 25 if task == "plug" else 100
                y = np.array([h["train/Train/mean_reward"] for h in valid])
                x = np.array([h["transitions"] / 1e6 for h in valid])
                axes[0].plot(
                    x[window - 1 :],
                    np.convolve(y, np.ones(window) / window, mode="valid"),
                    color=color,
                    alpha=0.65,
                    label=label if s == 0 else None,
                )
                es = sorted(
                    [
                        e
                        for e in evaluations
                        if (e["task"], e["condition"], e["seed"], e["occlusion"])
                        == (task, c, s, "clean")
                    ],
                    key=lambda e: e["transitions"],
                )
                axes[1].plot(
                    [e["transitions"] / 1e6 for e in es],
                    [e["rate"] for e in es],
                    color=color,
                    alpha=0.65,
                    marker="o",
                    ms=3,
                )
        axes[0].set(title="Training return", ylabel="Mean episode return (smoothed)")
        axes[1].set(title="Clean held-out checkpoints", ylabel="Held success (%)")
        for ax in axes:
            ax.set_xlabel("Training transitions (millions)")
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=9)
        body += figure(
            fig,
            f"{task}_learning",
            f"Each line is one training seed. Returns use trailing {window}-logged-update means. Checkpoints before the endpoint use validation resets; the endpoint uses the separate final test split. Returns are not success rates, and the split change prevents treating the final segment as a pure learning effect.",
        )
        if task == "plug":
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, o in zip(axes, ["clean", "dynamic"], strict=False):
                matrix = np.array(
                    [
                        [
                            100
                            * sum(
                                e["per_variant"][v]["successes"]
                                for e in subset
                                if e["condition"] == c and e["occlusion"] == o
                            )
                            / 768
                            for v in ["xm", "xp", "ym", "yp"]
                        ]
                        for c in CONDITIONS
                    ]
                )
                ax.imshow(matrix, cmap="Blues", vmin=0, vmax=25)
                for i in range(3):
                    for j in range(4):
                        ax.text(
                            j,
                            i,
                            f"{matrix[i, j]:.1f}%",
                            ha="center",
                            va="center",
                            color="white" if matrix[i, j] > 12.5 else "black",
                        )
                ax.set(
                    xticks=range(4),
                    xticklabels=["XM", "XP", "YM", "YP"],
                    yticks=range(3),
                    yticklabels=LABELS,
                    title=o.capitalize(),
                )
            body += figure(
                fig,
                "plug_variants",
                "Variant success pooled across three seeds (768 episodes per cell). Both panels share a 0–25% color scale. XM is almost never solved; YM contributes most successes.",
            )
        rows = []
        for e in sorted(
            subset, key=lambda e: (CONDITIONS.index(e["condition"]), e["seed"], e["occlusion"])
        ):
            rows.append(
                [
                    LABELS[CONDITIONS.index(e["condition"])],
                    e["seed"],
                    e["occlusion"],
                    f"{e['successes']}/{e['episodes']} ({e['rate']:.2f}%)",
                    e["timeouts"],
                    f"{e['error_mm']:.2f}",
                    link(ROOT / e["path"], "Raw evaluation"),
                ]
            )
        body += (
            "<details><summary>All final tests, failure counts and terminal error</summary>"
            + table(
                [
                    "Camera",
                    "Seed",
                    "Setting",
                    "Success",
                    "Timeouts",
                    "Mean terminal error (mm)",
                    "Evidence",
                ],
                rows,
            )
            + "</details></section>"
        )
        sections.append(body)
    from build_competence_repair_report import build as build_repairs

    repair_section, repair_media, repair_summary = build_repairs(
        ROOT, OUT, read, link, table, figure, audit_evaluation
    )
    media.extend(repair_media)
    cards = []
    for m in media:
        caption = f"{m['task'].capitalize()} · {LABELS[CONDITIONS.index(m['condition'])]} · seed {m['seed']} · {m['occlusion']} · {m['outcome']}"
        caption = html.escape(m.get("caption", caption))
        cards.append(
            f'''<article class="clip" data-task="{m["task"]}" data-seed="{m["seed"]}" data-condition="{m["condition"]}" data-occlusion="{m["occlusion"]}" data-outcome="{m["outcome"]}"><h3>{caption}</h3><p>Variant {m["variant"].upper()}, episode {m["episode"]}. Captured final-test trajectory.</p><video controls preload="none" src="{m["outside"]}"></video><p><a href="{m["policy"]}">Watch actual policy observations</a> · <a href="{m["outside"]}">Outside view</a></p></article>'''
        )
    filters = "".join(
        f'<label>{label}<select id="{key}"><option value="all">All</option>'
        + "".join(f'<option value="{v}">{v}</option>' for v in values)
        + "</select></label>"
        for key, label, values in [
            ("task", "Task", ["plug", "push"]),
            ("condition", "Camera", CONDITIONS),
            ("seed", "Seed", ["0", "1", "2"]),
            ("occlusion", "Setting", ["clean", "dynamic"]),
            ("outcome", "Outcome", ["success", "failure"]),
        ]
    )
    sources = []
    for source, name in [
        (ROOT / "PROGRESS.md", "PROGRESS.md"),
        (ROOT.parent / "M3L-active-perception/RESEARCH_LOG.md", "M3L_RESEARCH_LOG.md"),
    ]:
        dest = OUT / "sources" / name
        shutil.copyfile(source, dest)
        SOURCES.add(dest)
        sources.append(link(dest, name))
    timestamp = datetime.now(UTC).isoformat()
    summary = dict(
        generated_at=timestamp,
        repair=repair_summary,
        runs=[{k: v for k, v in r.items() if k != "history"} for r in runs],
        aggregates=aggregates,
        evaluations=evaluations,
        media=media,
        source_checks=source_checks,
        checks=dict(
            completed_runs=len(runs),
            checkpoint_hashes_verified=len(runs),
            evaluations_verified=len(evaluations),
            raw_episodes_verified=sum(e["episodes"] for e in evaluations),
            final_test_episodes=sum(e["episodes"] for e in finals),
            physical_success_violations=0,
        ),
        sources=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for p in sorted(SOURCES)],
    )
    (OUT / "latest_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "latest_final_results.csv").open("w") as f:
        keys = [k for k in finals[0] if k != "per_variant"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: e[k] for k in keys} for e in finals)
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SO-101 · latest trained models</title>{STYLE}</head><body><header><p>SO-101 ACTIVE PERCEPTION · OCTOBER 2026</p><h1>Plug and push training is complete.</h1><p>18 policies · 3 camera conditions · 3 training seeds per task · 36 final tests.</p><nav><a href="#plug">Plug insertion</a><a href="#push">Precision pushing</a><a href="#videos">Videos</a><a href="#sources">Evidence</a><a href="historical.html">Historical report</a></nav></header><main><p class="note">Plug insertion remains unreliable. Pushing reaches moderate success, with fixed-camera sensing ahead on average. These three-seed results do not establish an advantage for continuous active-camera control.</p>{"".join(sections)}<section id="videos"><h2>Final-policy videos</h2><p>Selected first recorded success and failure per run and setting, where available. These examples illustrate behavior; the tables above use every test episode. Outside views render saved physics states; policy-view videos show the captured actor inputs. Use the filters to compare tasks, seeds and outcomes.</p>{filters}<p id="media-count"></p><div class="grid">{"".join(cards)}</div></section><section id="sources"><h2>Evidence and interpretation</h2><p>Both studies trained on clean scenes. Dynamic results test robustness. Three independently trained seeds quantify optimization variability; 3,072 pooled episodes per cell are not 3,072 independent training runs. The reported SD describes variation between seeds, not statistical significance. No checkpoint was selected using final-test performance.</p><p>The earlier progress-log entries recorded campaign launch; a new completion-analysis entry records these results. Saved completion states, full-budget training histories, checkpoint hashes, and final-test evidence now supersede those pending statuses. The neighboring M3L research log contains older camera-credit experiments, not outcomes for these SO-101 campaigns. Historical reward recipes and the stopped square-target pushing study remain separate.</p><p>Verified {len(runs)} final checkpoint hashes, {len(evaluations)} evaluation hashes and {sum(e["episodes"] for e in evaluations):,} episode outcome/physical hold records, with zero success-audit violations. Frozen source manifests verified: {sum(s["files_verified"] for s in source_checks)} file entries. This report analyzes saved evidence; it does not rerun training or policy inference.</p><p>{" · ".join(sources)} · <a href="latest_analysis.json">Analysis and source hashes</a> · <a href="latest_final_results.csv">Final tests CSV</a> · <a href="training-results.zip">Portable report, plots and videos</a></p><p>Rebuild: <code>.venv/bin/python scripts/build_training_results_report.py</code></p></section></main><footer>Built {timestamp}. White-background figures include downloadable PDFs.</footer><script>const keys=['task','condition','seed','occlusion','outcome'];function filter(){{let n=0;document.querySelectorAll('.clip').forEach(c=>{{c.hidden=!keys.every(k=>document.getElementById(k).value==='all'||document.getElementById(k).value===c.dataset[k]);if(!c.hidden)n++;else c.querySelector('video').pause();}});document.getElementById('media-count').textContent=n+' examples';}}keys.forEach(k=>document.getElementById(k).addEventListener('change',filter));filter();</script></body></html>"""

    if repair_section:
        document = document.replace(
            "<h1>Plug and push training is complete.</h1>",
            "<h1>Plug insertion and pushing: repairs and evidence.</h1>",
        )
        document = document.replace(
            "18 policies · 3 camera conditions · 3 training seeds per task · 36 final tests.",
            "New competence repairs, followed by the original 18-policy, three-seed benchmark.",
        )
        document = document.replace("<nav>", '<nav><a href="#repair">Latest repairs</a>', 1)
        document = document.replace(
            "<main>", "<main>" + repair_section + "<h2>Original three-seed benchmark</h2>", 1
        )
        document = document.replace(
            "Plug insertion remains unreliable.",
            "In the original four-variant benchmark, plug insertion remained unreliable.",
        )
    from encode_report_webm import main as encode_media

    encode_media()

    # A directly opened/downloaded HTML must not depend on adjacent media files.
    # Embed displayed plots and videos; retain relative download links for the ZIP.
    def embedded(path, mime):
        if mime == "video/mp4":
            path = "media/" + hashlib.sha256(path.encode()).hexdigest()[:16] + ".webm"
            mime = "video/webm"
        return "data:" + mime + ";base64," + base64.b64encode((OUT / path).read_bytes()).decode()

    for item in media:
        poster = "media/" + hashlib.sha256(item["outside"].encode()).hexdigest()[:16] + ".jpg"
        original = f'<video controls preload="none" src="{item["outside"]}">'
        document = document.replace(
            original,
            f'<video controls preload="none" poster="{embedded(poster, "image/jpeg")}" src="{item["outside"]}">',
        )

    document = re.sub(
        r'src="([^"\n]+\.(?:svg|mp4))"',
        lambda match: (
            'src="'
            + embedded(match[1], "image/svg+xml" if match[1].endswith(".svg") else "video/mp4")
            + '"'
        ),
        document,
    )
    for item in media:
        policy_link = f'<a href="{item["policy"]}">Watch actual policy observations</a>'
        toggle = (
            f'<a href="{item["policy"]}" data-policy="{embedded(item["policy"], "video/mp4")}" '
            "onclick=\"event.preventDefault();const v=this.closest('.clip').querySelector('video');"
            "if(!v.dataset.outside)v.dataset.outside=v.src;"
            "this.showPolicy=!this.showPolicy;v.src=this.showPolicy?this.dataset.policy:v.dataset.outside;"
            "this.textContent=this.showPolicy?'Watch outside view':'Watch actual policy observations';v.play();\">"
            "Watch actual policy observations</a>"
        )
        document = document.replace(policy_link, toggle)
    document = document.replace(
        "White-background figures include downloadable PDFs.",
        "Plots and playable videos are embedded in this HTML. White-background figures include downloadable PDFs.",
    )
    (OUT / "index.html").write_text(document)
    # Explicit light background for every existing report/appendix in this report directory.
    for path in OUT.glob("*.html"):
        text = path.read_text()
        override = "<style data-report-light>:root{color-scheme:light}html,body,header{background:#fff!important;color:#182b3a}header p,header .eyebrow{color:#53616b!important}</style>"
        if "data-report-light" not in text:
            text = (
                text.replace("</head>", override + "</head>")
                if "</head>" in text
                else text + override
            )
            path.write_text(text)
    # Portable bundle includes all local dependencies reachable from the current report.
    from html.parser import HTMLParser
    from urllib.parse import unquote, urlsplit

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = []

        def handle_starttag(self, tag, attrs):
            for k, v in attrs:
                if k in ("href", "src", "poster") and v:
                    self.paths.append(v)

    bundled = set()
    pending = [OUT / "index.html"]
    documents = {}
    while pending:
        path = pending.pop().resolve()
        if path in bundled:
            continue
        assert path.is_relative_to(ROOT) and path.is_file(), path
        bundled.add(path)
        if path.suffix != ".html":
            continue
        text = path.read_text()
        # Historical report remains accessible in the workspace but is outside this latest-results bundle.
        if path == OUT / "index.html":
            text = text.replace(
                '<a href="historical.html">Historical report</a>',
                "<span>Historical report (workspace only)</span>",
            )
        text = text.replace(
            '<a href="training-results.zip">Portable report, plots and videos</a>',
            "Portable report: this extracted bundle",
        )
        documents[path] = text
        parser = Links()
        parser.feed(text)
        for target in parser.paths:
            parts = urlsplit(target)
            if parts.scheme or not parts.path:
                continue
            dest = (path.parent / unquote(parts.path)).resolve()
            if dest == OUT / "training-results.zip":
                continue
            pending.append(dest)
    bundled.update(
        OUT / "assets" / f"{task}_{name}.png"
        for task, name in [
            ("plug", "final_success"),
            ("push", "final_success"),
            ("plug", "learning"),
            ("push", "learning"),
            ("plug", "variants"),
        ]
    )
    bundled.update((OUT / "media").glob("*.webm"))
    bundled.add(OUT / "media/manifest.json")
    manifest = []
    with zipfile.ZipFile(OUT / "training-results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "index.html",
            '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=artifacts/training_results/index.html"><a href="artifacts/training_results/index.html">Open report</a>',
        )
        for path in sorted(bundled):
            data = documents[path].encode() if path in documents else path.read_bytes()
            name = str(path.relative_to(ROOT))
            archive.writestr(
                name,
                data,
                compress_type=zipfile.ZIP_STORED if path.suffix == ".mp4" else zipfile.ZIP_DEFLATED,
            )
            manifest.append(
                dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            )
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    print(
        json.dumps(
            dict(
                checks=summary["checks"],
                aggregates=aggregates,
                video_examples=len(media),
                bundle_files=len(bundled),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
