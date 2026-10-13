"""Build the live push visibility and RGB-training report from saved evidence."""

import argparse
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "artifacts/training_results"
    data = root / "artifacts/push_rgb"
    plan = read(data / "study-plan.json", {})
    audit = read(data / "visibility-audit/visibility-audit.json", {})
    warp = read(data / "warp-audit/summary.json", {})
    result = audit.get("summary", {})
    visibility_rows_raw = audit.get("rows", [])

    def visibility_rate(camera, key):
        if not visibility_rows_raw:
            return "Pending"
        return f"{100 * sum(r[camera][key] >= 3 for r in visibility_rows_raw) / len(visibility_rows_raw):.1f}%"

    runs, html_rows, curves = [], [], []
    throughput_rows = []
    names = {"wrist": "Wrist only", "wrist_static": "Wrist + static", "active": "Wrist + active"}
    for item in plan.get("runs", []):
        folder = data / "runs" / item["id"]
        state = read(folder / "state.json", {})
        submissions = [read(p) for p in sorted(folder.glob("submission-*.json"))]
        training = read(folder / "training.json", {})
        status = state.get("status", "submitted" if submissions else "not submitted")
        observed = int(state.get("transitions", 0))
        history = []
        log_dir = state.get("run_directory")
        if log_dir:
            path = Path(log_dir) / "iterations.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
                if history:
                    observed = max(observed, int(history[-1].get("transitions", 0)))
                    curves.append((item, history))
        # Estimate remaining accelerator time from observed steady-state throughput.
        rate = training.get("steady_transitions_per_second")
        if not rate and history:
            rates = [r.get("transitions_per_second") for r in history[-100:] if r.get("transitions_per_second")]
            rate = sum(rates) / len(rates) if rates else None
        remaining_hours = (max(0, plan.get("rounded_transitions", 1_000_046_592) - observed) / rate / 3600) if rate else None
        throughput_rows.append(
            f"<tr><td>{html.escape(names[item['condition']])}</td>"
            f"<td>{'Pending' if not rate else f'{rate:,.0f}'}</td>"
            f"<td>{'Pending' if remaining_hours is None else f'{remaining_hours:.1f} h'}</td></tr>"
        )
        validations = state.get("evaluations", [])
        eval_text = (
            "; ".join(
                f"{ev['occlusion']} {ev['successes']}/{ev['episodes']} at {ev['transitions']:,}"
                for ev in validations
            )
            or "Pending"
        )
        html_rows.append(
            "<tr><td>"
            + html.escape(names[item["condition"]])
            + "</td><td>"
            + html.escape(status)
            + "</td><td>"
            + f"{observed:,} / {plan.get('rounded_transitions', 1_000_046_592):,}"
            + "</td><td>"
            + html.escape(eval_text)
            + "</td><td>"
            + html.escape(
                ", ".join(str(s.get("job_id", "submission failed")) for s in submissions) or "—"
            )
            + "</td></tr>"
        )
        runs.append(
            {
                "run": item,
                "status": status,
                "transitions": observed,
                "target": plan.get("rounded_transitions", 1_000_046_592),
                "evaluations": validations,
                "submissions": submissions,
                "training": training,
            }
        )

    visibility_rows = []
    for key, label in (
        ("wrist_pair_rate", "Wrist alone"),
        ("static_pair_rate", "Fixed home camera alone"),
        ("wrist_plus_static_pair_rate", "Wrist + fixed home camera"),
        ("active_single_view_pair_rate", "Best of reachable active views, one view"),
        ("active_scan_pair_rate", "Reachable active-view scan"),
    ):
        val = result.get(key)
        visibility_rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{'Pending' if val is None else f'{100 * val:.1f}%'}</td></tr>"
        )
    fovy = result.get("fovy", {})
    physics = warp.get("push", {})
    smoke = physics.get("smoke", {})
    figure = ""
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for item, history in curves:
            history = history[:: max(1, len(history) // 2500)]
            xs = [r.get("transitions", r.get("iteration", 0)) / 1e6 for r in history]
            for axis, key in zip(
                axes, ("train/Train/mean_reward", "train/Episode_Metrics/success_rate"), strict=True
            ):
                ys = [r.get(key, float("nan")) for r in history]
                axis.plot(xs, ys, label=names[item["condition"]], linewidth=1)
        axes[0].set_ylabel("Training episode return")
        axes[1].set_ylabel("Training success metric")
        for axis in axes:
            axis.set_xlabel("Million environment transitions")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        svg = io.BytesIO()
        fig.savefig(svg, format="svg", bbox_inches="tight")
        plt.close(fig)
        (out / "push_rgb_learning.svg").write_bytes(svg.getvalue())
        figure = '<figure><img src="push_rgb_learning.svg" alt="RGB training reward and success curves"><figcaption>Training metrics are diagnostics; held-out physical evaluations determine task success.</figcaption></figure>'
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Push RGB active-perception study</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1050px;margin:36px auto;padding:0 20px;color:#183244}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #ccd7df}}th{{background:#edf3f7}}.callout{{background:#edf7f5;padding:16px;border-left:4px solid #059669}}img{{max-width:100%;border:1px solid #ccd7df}}small{{color:#536879}}</style></head><body>
<p><a href="index.html">Training results index</a></p><h1>Push: active camera under fixed-FOV occlusion</h1>
<p>Live report. Updated {html.escape(datetime.now(timezone.utc).isoformat())}. Three matched RGB policies use the same push task, reset distribution, reward, and static crossbar. The only planned difference is whether the external camera can move.</p>
<div class="callout"><b>Visibility audit:</b> At 64 held-out randomized resets, the fixed home view sees both landmarks in {100 * result.get("static_pair_rate", 0):.1f}% of resets; wrist + fixed sees both in {100 * result.get("wrist_plus_static_pair_rate", 0):.1f}%; the reachable active-view scan sees both over its reachable view set in {100 * result.get("active_scan_pair_rate", 0):.1f}%. Candidate-view coverage is a camera-geometry oracle, not a learned-policy result. All policy camera FOVs are {fovy.get("configured", 48.4554906359):.11f}°.</div>
<h2>Reset-state visibility</h2><p>Landmark visibility requires at least three segmentation pixels for both the source object and goal marker. Source and goal reset independently within ±20 mm. The fixed wall is non-colliding, and the push lane remains below its lower edge.</p>
<table><thead><tr><th>Sensor access</th><th>Both landmarks visible</th></tr></thead><tbody>{"".join(visibility_rows)}</tbody></table>
<p><b>Landmarks seen individually:</b> wrist source {visibility_rate("wrist", "object_pixels")}, wrist goal {visibility_rate("wrist", "fixture_pixels")}; static source {visibility_rate("static", "object_pixels")}, static goal {visibility_rate("static", "fixture_pixels")}. <b>FOV check:</b> wrist {fovy.get("wrist", "pending")}°; static {fovy.get("static", "pending")}°; active {fovy.get("active", "pending")}°; configured {fovy.get("configured", "pending")}°. Camera-arm endpoints with clear sampled paths: {result.get("active_path_clear_endpoint_count", "pending")}.</p>
<figure><img src="../push_rgb/visibility-audit/visibility-gallery.png" alt="Rendered wrist, static and reachable active camera views"><figcaption>Rendered reset views from a held-out reset. The active candidates use IK-reachable camera-arm poses and sampled collision-free paths.</figcaption></figure>
<h2>Training-simulator and task validation</h2><p>Warp physical feasibility: {physics.get("successes", "pending")}/{physics.get("episodes", "pending")} scripted randomized pushes. Rendered RGB PPO smoke: {"passed" if smoke.get("passed") else "pending"}; actual source and screen are included in the MjLab scene. The crossbar collision masks are disabled.</p>
<p>The same push_v2 task, ±20 mm reset jitter, Cartesian control and dense reward were first checked with privileged state: at 20,004,864 transitions it reached 501/512 and 510/512 held-out successes on two validation seeds. This supports task/reward learnability before the RGB comparison; it is not an RGB result. <a href="../transfer_push/privileged/push/validation-407-s30000.json">Validation seed 30000</a> · <a href="../transfer_push/privileged/push/validation-407-s31000.json">validation seed 31000</a>.</p>
<h2>1B-transition RGB policies</h2><p>Each run targets {plan.get("rounded_transitions", 1_000_046_592):,} transitions (at least 1B) with seed 0. Training curves and held-out evaluation results appear here as workers save checkpoints. Latest observed throughput gives a rough remaining compute estimate; it excludes evaluation, restarts, queue time, and future throughput changes.</p>
<table><thead><tr><th>Condition</th><th>Observed transitions/s</th><th>Estimated remaining GPU time</th></tr></thead><tbody>{"".join(throughput_rows)}</tbody></table>
{figure}
<table><thead><tr><th>Condition</th><th>Status</th><th>Transitions / target</th><th>Latest evaluations</th><th>Slurm jobs</th></tr></thead><tbody>{"".join(html_rows) if html_rows else '<tr><td colspan="5">Plan pending</td></tr>'}</tbody></table>
<h2>Interpretation</h2><p>Perception coverage establishes that reachable active views can recover reset landmarks missed by the fixed sensors. It does not establish that PPO learns to move the camera or improves push success; those claims depend on matched completed 1B runs and held-out evaluations. Results from other tasks or privileged policies are not mixed into this comparison.</p>
<p><small>Machine-readable state: <a href="../push_rgb/study-plan.json">study plan</a> · <a href="../push_rgb/visibility-audit/visibility-audit.json">rendered visibility audit</a> · <a href="../push_rgb/warp-audit/summary.json">Warp task validation</a>.</small></p></body></html>"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "push_rgb.html").write_text(body)
    summary = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "visibility": result,
        "warp_audit": physics,
        "runs": runs,
    }
    (out / "push_rgb_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    index = out / "index.html"
    if index.exists():
        text = index.read_text()
        link = '<a href="push_rgb.html">Push RGB camera study</a>'
        if link not in text:
            text = text.replace("<nav>", "<nav>" + link, 1)
            index.write_text(text)


if __name__ == "__main__":
    main()
