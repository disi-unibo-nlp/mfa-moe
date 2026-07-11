"""Generate the preliminary paper figures from the local experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODEL = "allenai--OLMoE-1B-7B-0924-Instruct"
OUT = Path(__file__).resolve().parent


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def geometry_plot() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharey=True)
    for ax, dataset, title in zip(
        axes,
        ("gsm8k", "processbench"),
        ("GSM8K", "ProcessBench"),
    ):
        rows = load_json(
            RESULTS / "exp3" / MODEL / dataset / "geometry_correlation.json"
        )
        layers = [row["layer"] for row in rows]
        for key, label, style in (
            ("overall_correlation", "overall", "-"),
            ("correct_correlation", "correct", "--"),
            ("failed_correlation", "failed", ":"),
            ("backtracking_correlation", "backtracking (old labels)", "-."),
        ):
            ax.plot(layers, [row[key] for row in rows], style, marker="o", ms=3,
                    linewidth=1.4, label=label)
        ax.set_title(title)
        ax.set_xlabel("MoE layer")
        ax.set_xticks(range(0, 16, 2))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Pearson/Mantel correlation")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Hidden-state similarity predicts routing similarity")
    fig.tight_layout()
    fig.savefig(OUT / "geometry_correlation.pdf", bbox_inches="tight")
    plt.close(fig)


def processbench_event_plot() -> None:
    data = load_json(
        RESULTS / "exp2" / MODEL / "processbench" / "event_routing_relabelled.json"
    )["summary"]
    phases = ("before", "at", "after")
    metrics = (
        ("entropy", "Routing entropy"),
        ("switch_rate", "Expert-switch rate"),
        ("topk_overlap", "Top-k overlap"),
        ("margin", "Router margin"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0))
    x = np.arange(3)
    for ax, (metric, title) in zip(axes.flat, metrics):
        for event, label, style in (
            ("normal", "normal-step centers", "--"),
            ("first_error", "first error", "-"),
        ):
            y = [data[event][f"{metric}_{phase}"] for phase in phases]
            ax.plot(x, y, style, marker="o", linewidth=1.6, label=label)
        ax.set_title(title)
        ax.set_xticks(x, phases)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("ProcessBench routing around the gold first-error step (n=207)")
    fig.tight_layout()
    fig.savefig(OUT / "processbench_first_error_routing.pdf", bbox_inches="tight")
    plt.close(fig)


def expert_phase_plot() -> None:
    datasets = ("gsm8k", "processbench")
    phases = ("backtracking", "contradiction", "self_correction", "first_error", "final_answer")
    labels = ("backtrack", "contradiction", "self-correct", "first error", "final answer")
    values: dict[str, list[float]] = {}
    for dataset in datasets:
        data = load_json(
            RESULTS / "exp5" / MODEL / dataset / "expert_events_relabelled.json"
        )
        normal = np.asarray(data["phases"]["normal"]["activation_frequency"])
        values[dataset] = []
        for phase in phases:
            row = data["phases"][phase]
            if row["n_tokens"] == 0:
                values[dataset].append(np.nan)
            else:
                phase_frequency = np.asarray(row["activation_frequency"])
                values[dataset].append(float(np.mean(np.abs(phase_frequency - normal))))
    x = np.arange(len(phases))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(x - width / 2, values["gsm8k"], width, label="GSM8K")
    ax.bar(x + width / 2, values["processbench"], width, label="ProcessBench")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Mean |activation-frequency delta|\nvs. normal tokens")
    ax.set_title("Expert-use shifts by reasoning phase (corrected labels)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "expert_phase_deltas.pdf", bbox_inches="tight")
    plt.close(fig)


def probe_plot() -> None:
    datasets = ("gsm8k", "prm800k", "processbench")
    titles = ("GSM8K", "PRM800K", "ProcessBench")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), sharey=True)
    for ax, dataset, title in zip(axes, datasets, titles):
        probe = load_json(
            RESULTS / "probes" / MODEL / dataset / "router_trace_probe.json"
        )
        baseline = load_json(
            RESULTS / "probes" / MODEL / dataset / "structure_trace_probe.json"
        )
        rows = probe["targets"]["correctness"]["layers"]
        ax.plot(
            [row["layer"] for row in rows],
            [row["metrics"]["auroc"] for row in rows],
            marker="o",
            ms=3,
            label="router probe",
        )
        ax.axhline(
            baseline["targets"]["correctness"]["metrics"]["auroc"],
            color="tab:orange",
            linestyle="--",
            label="length/structure baseline",
        )
        ax.axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("MoE layer")
        ax.set_xticks(range(0, 16, 3))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Out-of-fold AUROC")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle("Full-trace router decoding of final correctness")
    fig.tight_layout()
    fig.savefig(OUT / "router_correctness_probes.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    geometry_plot()
    processbench_event_plot()
    expert_phase_plot()
    probe_plot()
