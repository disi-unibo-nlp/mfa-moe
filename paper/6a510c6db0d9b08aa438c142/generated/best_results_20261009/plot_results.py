"""Replot saved estimates; no model fitting, resampling or new inference.

Run from any directory with Python, NumPy and Matplotlib. Input CSVs are
bundled beside this script. All original confidence limits are copied.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = ["gpt", "nemotron", "qwen35", "qwen36", "gemma", "glm", "qwen330b"]
NAMES = ["GPT-OSS", "Nemotron", "Qwen3.5", "Qwen3.6", "Gemma", "GLM", "Qwen3-30B"]
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#777777"


def rows(filename):
    with (HERE / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def finish(fig, name):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=3)
    fig.savefig(HERE / (name + ".pdf"), bbox_inches="tight",
                metadata={"Creator": "plot_results.py; saved estimates only", "CreationDate": None, "ModDate": None})
    fig.savefig(HERE / (name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def forest(ax, values, low, high, positions, color=BLUE, marker="o"):
    values, low, high = map(np.asarray, (values, low, high))
    assert np.isfinite([values, low, high]).all()
    assert (low <= values).all() and (values <= high).all()
    ax.errorbar(values, positions, xerr=np.array([values - low, high - values]),
                fmt=marker, color=color, markersize=5, capsize=3, elinewidth=1.3, linestyle="none")
    ax.axvline(0, color="0.35", linewidth=0.9, linestyle="--")


def routing():
    data = rows("routing_readout.csv")
    assert len(data) == 7 and {r["model"] for r in data} == set(MODELS)
    lookup = {r["model"]: r for r in data}
    assert all(r["status"] == "CLEAN" and int(r["questions"]) == 522 and int(r["families"]) == 509 for r in data)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.35), sharey=True, layout="constrained")
    for ax, prefix, title in zip(axes, ["routing", "routing_excess_over_lexical_null"],
                               ["(a) Beyond judge-context text", "(b) Excess over lexical null"]):
        forest(ax, [float(lookup[m][prefix + "_delta_logloss"]) for m in MODELS],
               [float(lookup[m][prefix + "_simultaneous_ci7_low"]) for m in MODELS],
               [float(lookup[m][prefix + "_simultaneous_ci7_high"]) for m in MODELS], np.arange(7))
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(-0.18, 0.009)
        ax.set_xticks([-0.16, -0.12, -0.08, -0.04, 0])
        ax.set_xlabel("Loss change (nats/sentence)")
    axes[0].set_yticks(np.arange(7), NAMES)
    axes[0].invert_yaxis()
    finish(fig, "routing_readout")


def retention():
    data = rows("selection_coverage.csv")
    arms = ["state_balanced", "labelled_frequency", "random"]
    assert len(data) == 7 * 5 * 3
    lookup = {(r["model"], int(r["fold"]), r["arm"]): r for r in data}
    assert len(lookup) == len(data)
    for m in MODELS:
        assert sum(int(lookup[m, f, arms[0]]["test_families"]) for f in range(5)) == 1486
        for f in range(5):
            assert len({lookup[m, f, a]["test_families"] for a in arms}) == 1
            for a in arms:
                assert lookup[m, f, a]["minimum_complete"] == "True"
                assert 0 <= float(lookup[m, f, a]["minimum_class_coverage"]) <= 1
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.75), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1]}, layout="constrained")
    summary = []
    for y, (m, name) in enumerate(zip(MODELS, NAMES)):
        measures = {a: np.array([100 * float(lookup[m, f, a]["minimum_class_coverage"])
                                for f in range(5)]) for a in arms}
        for a, color, marker, offset in zip(arms, [BLUE, ORANGE, GREY], ["o", "s", "^"], [-0.19, 0, 0.19]):
            vals = measures[a]
            axes[0].plot([vals.min(), vals.max()], [y + offset] * 2, color=color, linewidth=1.1)
            axes[0].scatter(vals, np.full(5, y + offset), s=17, color=color, marker=marker,
                            edgecolors="white", linewidths=0.3, zorder=3)
        difference = measures[arms[0]] - measures[arms[1]]
        assert (difference > 0).all() if m not in {"qwen35", "qwen36"} else (difference < 0).all()
        color = BLUE if difference.mean() > 0 else ORANGE
        axes[1].plot([0, difference.mean()], [y, y], color=color, linewidth=2.8, alpha=0.5)
        axes[1].scatter(difference, y + np.linspace(-0.14, 0.14, 5), s=20,
                        marker="o", color=color, edgecolors="white", linewidths=0.3, zorder=3)
        summary.append({"model": m, "balanced_mean_pct": float(measures[arms[0]].mean()),
                        "frequency_mean_pct": float(measures[arms[1]].mean()),
                        "random_mean_pct": float(measures[arms[2]].mean()),
                        "balanced_minus_frequency_mean_pp": float(difference.mean()),
                        "difference_min_pp": float(difference.min()), "difference_max_pp": float(difference.max()),
                        "folds": 5, "independent_families": 1486})
    axes[0].set_yticks(np.arange(7), NAMES)
    axes[0].invert_yaxis()
    axes[0].set_title("(a) Minimum class coverage", loc="left", fontsize=10)
    axes[0].set_xlabel("Held-out selected-slot coverage (%)")
    axes[0].set_xlim(70, 101)
    axes[0].set_xticks([70, 80, 90, 100])
    axes[1].axvline(0, color="0.35", linewidth=0.9, linestyle="--")
    axes[1].set_title("(b) Paired fold differences", loc="left", fontsize=10)
    axes[1].set_xlabel("State balanced − frequency (pp)")
    handles = [Line2D([], [], marker=marker, color=color, linestyle="none", markersize=4, label=label)
               for marker, color, label in zip(["o", "s", "^"], [BLUE, ORANGE, GREY],
                   ["State balanced", "Labelled frequency", "Fixed random"])]
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.13),
                   ncol=3, frameon=False, fontsize=8, columnspacing=1.1, handletextpad=0.4)
    finish(fig, "heldout_retention")
    with (HERE / "retention_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def anticipation():
    data = rows("anticipation.csv")
    selected = [r for r in data if r["feature"] == "composition16" and r["mode"] in {"next", "next-contained"}]
    lookup = {(r["model"], r["mode"]): r for r in selected}
    assert len(lookup) == len(selected) == 4
    fig, ax = plt.subplots(figsize=(7.3, 2.45), layout="constrained")
    labels = []
    for y, (m, mode) in enumerate([(m, mode) for m in ["gpt", "qwen36"] for mode in ["next", "next-contained"]]):
        r = lookup[m, mode]
        assert r["status"] == "CLEAN"
        forest(ax, [float(r["delta_logloss"])], [float(r["simultaneous_ci6_low"])],
               [float(r["simultaneous_ci6_high"])], [y], color=BLUE if mode == "next" else ORANGE,
               marker="o" if mode == "next" else "s")
        name = NAMES[MODELS.index(m)]
        labels.append(name + (" / adjacent" if mode == "next" else " / contained"))
        ax.text(0.002, y, f"{int(r['questions']):,} questions; {int(r['pairs']):,} pairs", va="center", fontsize=8)
    ax.set_yticks(np.arange(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.077, 0.047)
    ax.set_xticks([-0.06, -0.04, -0.02, 0])
    ax.set_xlabel("Next-class loss change (nats/pair)")
    finish(fig, "next_class")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    routing()
    retention()
    anticipation()
    files = ["routing_readout.csv", "selection_coverage.csv", "anticipation.csv", "plot_results.py",
             "retention_summary.csv", "routing_readout.pdf", "routing_readout.png", "heldout_retention.pdf",
             "heldout_retention.png", "next_class.pdf", "next_class.png"]
    manifest = {"schema": "saved-paper-figures-v1", "analysis": "Saved estimates only; no refits or new resampling",
                "files": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in files},
                "uncertainty": {"routing": "Original model-specific simultaneous_ci7; not simultaneous across models or panels",
                                "anticipation": "Original model-specific simultaneous_ci6; two measured models only",
                                "retention": "All five paired family folds; spread and mean are descriptive, not confidence intervals"}}
    (HERE / "FIGURES.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"figures": 3, "routing_models": 7, "retention_models": 7, "retention_cells": 105,
                      "anticipation_models": 2, "new_inference": False}))


if __name__ == "__main__":
    main()
