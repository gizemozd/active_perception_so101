"""Summarize paired environment corrections over all nine trained push policies."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build(root, read, link, table, figure, audit):
    folder = root / "artifacts/competence_repair/push_population"
    if len(list(folder.glob("*/*/final-test.json"))) != 18:
        return "", {}
    details, rows, aggregates = [], [], []
    conditions = [
        ("wrist", "Wrist"),
        ("wrist_static", "Wrist + fixed"),
        ("active", "Wrist + active"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for index, (condition, label) in enumerate(conditions):
        old, new = [], []
        for seed in range(3):
            pair = []
            paths = []
            for intervention in ["original", "clear"]:
                path = folder / f"push_{condition}_s{seed}" / intervention / "final-test.json"
                data = read(path)
                entry = dict(
                    episodes=data["episodes"],
                    successes=data["successes"],
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                data = audit(path, entry, "push")
                assert data["episodes"] == 512 and data["seed"] == 90000 and data["split"] == "test"
                provenance = read(path.with_name("provenance.json"))
                assert (
                    hashlib.sha256(Path(data["checkpoint"]).read_bytes()).hexdigest()
                    == provenance["sha256"]
                )
                pair.append(data)
                paths.append(path)
            a, b = pair
            assert a["initial_physics_state_sha256"] == b["initial_physics_state_sha256"]
            gain = sum(not x and y for x, y in zip(a["outcomes"], b["outcomes"], strict=True))
            loss = sum(x and not y for x, y in zip(a["outcomes"], b["outcomes"], strict=True))
            old.append(100 * a["successes"] / 512)
            new.append(100 * b["successes"] / 512)
            details.append(
                dict(
                    condition=condition,
                    seed=seed,
                    episodes=512,
                    original_successes=a["successes"],
                    clear_successes=b["successes"],
                    gained=gain,
                    lost=loss,
                    original_report=str(paths[0].relative_to(root)),
                    clear_report=str(paths[1].relative_to(root)),
                )
            )
        aggregates.append(
            dict(
                condition=condition,
                original_mean=float(np.mean(old)),
                clear_mean=float(np.mean(new)),
                original_sd=float(np.std(old, ddof=1)),
                clear_sd=float(np.std(new, ddof=1)),
                gain_pp=float(np.mean(new) - np.mean(old)),
            )
        )
        r = aggregates[-1]
        rows.append(
            [
                label,
                f"{r['original_mean']:.1f} ± {r['original_sd']:.1f}%",
                f"{r['clear_mean']:.1f} ± {r['clear_sd']:.1f}%",
                f"{r['gain_pp']:+.1f} pp",
            ]
        )
        for offset, values, color, name in [
            (-0.18, old, "#64748b", "Original"),
            (0.18, new, "#0f766e", "Partition removed"),
        ]:
            ax.bar(
                index + offset,
                np.mean(values),
                width=0.32,
                color=color,
                alpha=0.8,
                label=name if index == 0 else None,
            )
            ax.scatter(
                index + offset + np.array([-0.055, 0, 0.055]), values, c="#182b3a", s=22, zorder=3
            )
    ax.set(
        xticks=range(3),
        xticklabels=[label for _, label in conditions],
        ylim=(0, 100),
        ylabel="Held success (%)",
        title="Same learned policies, corrected push environment",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    body = "<h3>Push: all three camera conditions and all three seeds</h3><p>Matched clean tests use 512 episodes per model and environment, with identical initial physics within each pair. Means and SDs below are across three trained policies; every point in the plot is one training seed. No weights were updated for this comparison.</p>"
    body += table(
        ["Camera", "Original · mean ± SD", "Partition removed · mean ± SD", "Change"], rows
    )
    body += figure(
        fig,
        "repair_push_population",
        "Bars average three training seeds; dots show individual seeds. The physical partition is the only environment intervention. The optical crossbar, arc distribution, original action bounds and physical success criteria remain.",
    )
    body += (
        "<details><summary>All paired push results</summary>"
        + table(
            [
                "Camera",
                "Seed",
                "Original successes",
                "Corrected successes",
                "New / lost successes",
                "Evidence",
            ],
            [
                [
                    r["condition"],
                    r["seed"],
                    f"{r['original_successes']}/512",
                    f"{r['clear_successes']}/512",
                    f"{r['gained']} / {r['lost']}",
                    link(root / r["original_report"], "Original")
                    + " · "
                    + link(root / r["clear_report"], "Corrected"),
                ]
                for r in details
            ],
        )
        + "</details>"
    )
    return body, dict(
        aggregates=aggregates,
        pairs=details,
        episodes_per_environment=4608,
        overall_original_rate=100 * sum(r["original_successes"] for r in details) / 4608,
        overall_clear_rate=100 * sum(r["clear_successes"] for r in details) / 4608,
    )
