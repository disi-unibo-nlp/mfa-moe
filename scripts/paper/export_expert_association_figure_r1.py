"""Export all seven-model saved adjusted mean-use correlations; do not refit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

MODELS = ("gpt", "nemotron", "qwen35", "qwen36", "gemma", "glm", "qwen330b")
NAMES = ("GPT-OSS", "Nemotron", "Qwen3.5", "Qwen3.6", "Gemma", "GLM", "Qwen3-30B")
STATES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    source_hash = sha(args.source)
    with args.source.open(newline="") as handle:
        selected = [r for r in csv.DictReader(handle) if r["stat"] == "usage_all"]
    cells = {}
    for row in selected:
        assert row["layer"] == "mean" and row["adjustment"] == "lendiff_frac"
        assert row["family"] == "cls_expert" and row["scope"] == "pooled"
        key = row["model"], row["window"]
        assert key not in cells and -1 <= float(row["r"]) <= 1
        assert 0 <= float(row["q"]) <= 1
        cells[key] = row
    assert set(cells) == {(model, state) for model in MODELS for state in STATES}
    assert sha(args.source) == source_hash
    args.output.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output / "expert_usage_correctness_mean_all.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(cells[(m, s)] for m in MODELS for s in STATES)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    values = [[float(cells[(m, s)]["r"]) for s in STATES] for m in MODELS]
    bound = max(abs(v) for row in values for v in row)
    fig, ax = plt.subplots(figsize=(7.0, 3.45))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(7), STATES)
    ax.set_yticks(range(7), NAMES)
    ax.set_xlabel("Episode class used to nominate the expert profile")
    for i, model in enumerate(MODELS):
        for j, state in enumerate(STATES):
            value = values[i][j]
            mark = "*" if float(cells[(model, state)]["q"]) < 0.05 else ""
            ax.text(j, i, f"{value:+.3f}{mark}", ha="center", va="center",
                    color="white" if abs(value) > 0.62 * bound else "black", fontsize=8)
    fig.colorbar(im, ax=ax, pad=0.025, label="Saved adjusted correlation (r)")
    fig.tight_layout()
    paths = [output_csv]
    for suffix in ("pdf", "png"):
        path = args.figure_dir / f"expert_usage_correctness.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    receipt = {
        "schema": "saved-expert-association-figure-v1",
        "source": {"path": str(args.source), "sha256": source_hash},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "selection": "All seven models x seven states; pooled cls_expert / usage_all / mean / lendiff_frac; no result-sign or significance selection.",
        "n_cells": len(cells),
        "new_fits": 0, "new_inference": 0,
        "values": [{"model": m, "state": s, "r": float(cells[(m, s)]["r"]),
                    "q": float(cells[(m, s)]["q"]), "n_questions": int(cells[(m, s)]["n_questions"])}
                   for m in MODELS for s in STATES],
        "interpretation": "Saved same-cohort exploratory nuisance-adjusted association, not held-out keep-set validity or a deletion effect. Stars retain the source q-values; no new multiplicity calculation. Existing intervals and full counts remain in the CSV.",
        "outputs": [{"path": str(p), "sha256": sha(p)} for p in paths],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (args.output / "EXPERT_ASSOCIATION_FIGURE-r1.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"n_cells": len(cells), "elapsed_seconds": receipt["elapsed_seconds"],
                      "saved_q_below_005": [v for v in receipt["values"] if v["q"] < 0.05]}, indent=2))


if __name__ == "__main__":
    main()
