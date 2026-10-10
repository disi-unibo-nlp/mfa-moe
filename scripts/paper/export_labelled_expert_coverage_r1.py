"""Describe saved labelled-route concentration, without fitting a pruning policy.

The source is the existing cohort-A expert enrichment summary, not raw model
activations. Ranking and evaluation use the same observed labelled episodes.
This intentionally provides no holdout interval, accuracy or resident-byte claim.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time


STATES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")
LABELS = {
    "gpt": "GPT-OSS", "nemotron": "Nemotron", "qwen35": "Qwen3.5",
    "qwen36": "Qwen3.6", "gemma": "Gemma", "glm": "GLM",
    "qwen330b": "Qwen3-30B",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def describe(path: Path, layers: int, experts: int, native_k: int) -> dict:
    counts: dict[int, dict[int, dict[str, float]]] = {}
    support: dict[str, int] = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            state = row["cls"]
            if state not in STATES:
                raise ValueError(f"Unexpected state {state}")
            layer, expert = int(row["layer"]), int(row["expert"])
            token_value = float(row["class_tokens"])
            if not math.isfinite(token_value) or not token_value.is_integer():
                raise ValueError("Nonintegral state token support")
            tokens = int(token_value)
            if tokens <= 0 or support.setdefault(state, tokens) != tokens:
                raise ValueError("Inconsistent or missing state support")
            rate = float(row["rate_in"])
            if not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError("Invalid selection rate")
            record = counts.setdefault(layer, {}).setdefault(expert, {})
            if state in record:
                raise ValueError("Duplicate state/layer/expert")
            # Archived analyses.py: rate_in = counts / (class_tokens * top_k).
            # This is a selected-slot share, not token membership probability.
            record[state] = rate * tokens * native_k
    if len(counts) != layers or set(support) != set(STATES):
        raise ValueError("Incomplete layer/state frame")
    totals = {state: 0.0 for state in STATES}
    ranked = {}
    for layer, records in sorted(counts.items()):
        if set(records) != set(range(experts)):
            raise ValueError("Incomplete expert frame")
        for values in records.values():
            if set(values) != set(STATES):
                raise ValueError("Incomplete expert/state frame")
        for state in STATES:
            mass = sum(values[state] for values in records.values())
            expected = native_k * support[state]
            if not math.isclose(mass, expected, rel_tol=1e-8, abs_tol=1e-4):
                raise ValueError(f"Native-k mass mismatch {layer}/{state}: {mass}/{expected}")
            totals[state] += mass
        ranked[layer] = sorted(records, key=lambda e: (-sum(records[e].values()), e))
    cumulative = {state: 0.0 for state in STATES}
    curves = []
    previous = {state: 0.0 for state in STATES}
    for keep in range(1, experts + 1):
        for layer, order in ranked.items():
            record = counts[layer][order[keep - 1]]
            for state in STATES:
                cumulative[state] += record[state]
        coverage = {state: cumulative[state] / totals[state] for state in STATES}
        for state in STATES:
            if coverage[state] < previous[state] - 1e-12 or coverage[state] > 1 + 1e-10:
                raise ValueError("Invalid nonmonotone coverage")
        previous = coverage
        if keep >= native_k:
            curves.append({
                "retained_per_layer": keep,
                "retained_expert_fraction": keep / experts,
                "labelled_selection_coverage": sum(cumulative.values()) / sum(totals.values()),
                "minimum_state_coverage": min(coverage.values()),
                "minimum_state": min(STATES, key=lambda s: coverage[s]),
                **{f"coverage_{state}": coverage[state] for state in STATES},
            })
    if any(not math.isclose(previous[s], 1.0, abs_tol=1e-10) for s in STATES):
        raise ValueError("All-expert coverage is not one")
    half = next(row for row in curves if row["retained_per_layer"] == experts // 2)
    return {
        "layers": layers, "experts_per_layer": experts, "native_k": native_k,
        "class_tokens": support, "labelled_tokens": sum(support.values()),
        "half_expert_summary": half, "curves": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    with args.inventory.open(newline="") as handle:
        frame = list(csv.DictReader(handle))
    if {row["model"] for row in frame} != set(LABELS) or len(frame) != 7:
        raise ValueError("Exactly the seven-model roster is required")
    args.output.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    models, sources, flat = {}, [], []
    for row in frame:
        model = row["model"]
        source = Path(row["source"]).with_name("E_expert_enrichment.csv.gz")
        before = digest(source)
        result = describe(source, int(row["routed_layers"]),
                          int(row["native_experts_per_layer"]), int(row["native_k"]))
        if digest(source) != before:
            raise ValueError("Source changed during read")
        models[model] = result
        sources.append({"model": model, "path": str(source), "sha256": before,
                        "bytes": source.stat().st_size})
        flat.extend({"model": model, **point} for point in result["curves"])
    table = args.output / "labelled_expert_coverage.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True, sharey=True)
    for index, (model, result) in enumerate(models.items()):
        points = result["curves"]
        xs = [p["retained_expert_fraction"] for p in points]
        for ax, metric in zip(axes, ("labelled_selection_coverage", "minimum_state_coverage")):
            ax.plot(xs, [p[metric] for p in points], label=LABELS[model],
                    color=plt.get_cmap("tab10")(index), linewidth=1.35)
    for ax, title in zip(axes, ("All labelled episodes", "Least-covered episode class")):
        ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1.01), title=title,
               xlabel="Retained expert identities per layer")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Observed selection coverage")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4,
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    figure_paths = []
    for suffix in ("pdf", "png"):
        path = args.figure_dir / f"labelled_expert_coverage.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        figure_paths.append(path)
    plt.close(fig)
    receipt = {
        "schema": "labelled-expert-coverage-v1", "source_cohort": "A",
        "question_support_per_model": 1547,
        "selection_rule": "Within each layer, rank by summed selected counts across the seven labelled classes; break ties by expert ID.",
        "estimand": "In-sample fraction of original labelled token-layer-expert selections included in the retained identity set; raw token/slot weighting, no equal-family or equal-benchmark reweighting.",
        "reconstruction": "Archived v3an/analyses.py uses rate_in=counts/(class_tokens*native_k); reconstruct selected-slot counts by rate_in*class_tokens*native_k without smoothing.",
        "scope": [
            "Labels cover only a subset of reasoning tokens; no all-reasoning coverage claim.",
            "Retrospective descriptive concentration only; no held-out stability or inferential interval.",
            "No model is pruned, masked, loaded or rerun. No accuracy or memory savings are measured.",
            "Expert counts are not bytes; these sets are not validated deployment masks, including group-routing feasibility.",
            "Half-expert summaries are a common plotting landmark, not an adopted deployment budget.",
        ],
        "validation": "Complete 7-state x layer x expert frames; native-k mass in each layer/state; monotonic coverage; all-expert coverage=1; sources unchanged during read.",
        "sources": sources,
        "inventory": {"path": str(args.inventory), "sha256": digest(args.inventory)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "models": models,
        "outputs": [{"path": str(p), "sha256": digest(p)} for p in [table, *figure_paths]],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (args.output / "LABELLED_EXPERT_COVERAGE-r1.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"elapsed_seconds": receipt["elapsed_seconds"],
                      "half_expert_summaries": {m: r["half_expert_summary"] for m, r in models.items()}}, indent=2))


if __name__ == "__main__":
    main()
