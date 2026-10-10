"""Export observed exhibits; absent model cells remain explicit."""
from __future__ import annotations

import csv
from pathlib import Path

from .contracts import file_sha, load, private_path, require, save
from .protocol import ARMS, MODELS

LABELS = {"gpt": "GPT-OSS", "nemotron": "Nemotron", "qwen35": "Qwen3.5", "qwen36": "Qwen3.6",
          "gemma": "Gemma", "glm": "GLM", "qwen330b": "Qwen3-30B"}


def export(campaign, root, output):
    root, output = Path(root), private_path(output)
    require(not output.exists(), "exhibit export must use a fresh versioned destination")
    output.mkdir(parents=True)
    sources = {}
    capabilities = []
    memory_rows = []
    for model in MODELS:
        p = root / model / "qualification/QUALIFICATION.json"
        row = {"model": model, "qualification": "UNMEASURED", "native_expert_bytes": None,
               "compact_expert_bytes": None, "accuracy": "UNMEASURED"}
        if p.exists():
            q = load(p)
            require(q["campaign_sha256"] == campaign["sha256"], "foreign capability receipt")
            row["qualification"] = q["status"]
            sources[str(p)] = file_sha(p)
            if q["status"] == "PASS":
                native = next(c for c in q["cases"] if c["arm"] == "native")
                compact = next(c for c in q["cases"] if c["arm"] == "state_balanced")
                row["native_expert_bytes"] = sum(s["total_expert_bytes"] for s in native["loader"])
                row["compact_expert_bytes"] = sum(s["total_expert_bytes"] for s in compact["loader"])
                for case in q["cases"]:
                    observed = next(c for c in q["cases"] if c["arm"] == case.get("alias_of")) if "alias_of" in case else case
                    for rank, status in enumerate(observed["loader"]):
                        fixture = observed["context_fixture"]
                        memory_rows.append({"model": model, "arm": case["arm"], "rank": rank,
                            "expert_bytes": status["total_expert_bytes"],
                            "model_tensor_bytes": status["total_model_storage"]["unique_bytes"],
                            "cache_sha256": status["cache_sha256"],
                            "load_peak_allocated": status["load_memory"]["peak_allocated"],
                            "prefill_peak_allocated": fixture["prefill_memory"][rank]["peak_allocated"],
                            "decode_peak_allocated": fixture["memory_after"][rank]["memory"]["peak_allocated"],
                            "allocated": fixture["memory_after"][rank]["memory"]["allocated"],
                            "reserved": fixture["memory_after"][rank]["memory"]["reserved"],
                            "nvml_process_bytes": sum(r["bytes"] for r in fixture["memory_after"][rank]["memory"]["nvml_process_residence"])})
        capabilities.append(row)
    with (output / "capability.csv").open("x") as f:
        writer = csv.DictWriter(f, fieldnames=list(capabilities[0]))
        writer.writeheader()
        writer.writerows(capabilities)
    text = [r"% Generated from observed receipts; UNMEASURED is not a failed result.",
            r"\begin{table}[t]\centering\small", r"\caption{Physical expert-removal qualification. Memory and accuracy require observed receipts.}",
            r"\label{tab:expert-memory-capability}", r"\begin{tabular}{llrr}\toprule",
            r"Model & Qualification & Native expert GiB & Compact expert GiB \\", r"\midrule"]
    for row in capabilities:
        numbers = ["--" if row[k] is None else f"{row[k]/1024**3:.2f}"
                   for k in ("native_expert_bytes", "compact_expert_bytes")]
        text.append(" & ".join([LABELS[row["model"]], row["qualification"], *numbers]) + r" \\")
    text += [r"\bottomrule\end{tabular}\end{table}", ""]
    (output / "capability.tex").write_text("\n".join(text))
    transition_path = root / "TRANSITIONS.json"
    if transition_path.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        transitions = load(transition_path)
        require(set(transitions["models"]) == set(MODELS) and
                transitions["campaign_sha256"] == campaign["sha256"], "transition roster differs")
        sources[str(transition_path)] = file_sha(transition_path)
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
        for ax, model in zip(axes.flat, MODELS):
            result = transitions["models"][model]
            matrix = np.asarray(result["family_balanced_probability"])
            unsupported = np.asarray(result["source_family_support"]) == 0
            matrix[unsupported] = np.nan
            graphic = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            ax.set_title(LABELS[model])
            ax.set_xticks(range(7), result["classes"], rotation=90, fontsize=7)
            ax.set_yticks(range(7), result["classes"], fontsize=7)
        axes.flat[-1].axis("off")
        fig.colorbar(graphic, ax=list(axes.flat[:7]), shrink=.6, label="Family-balanced transition probability")
        fig.savefig(output / "transitions.pdf")
        plt.close(fig)
    selections, class_rows, stability, fitted_models = [], [], [], []
    for model in MODELS:
        p = root / model / "SELECTION.json"
        if p.exists():
            s = load(p)
            require(s["campaign_sha256"] == campaign["sha256"], "foreign selection receipt")
            sources[str(p)] = file_sha(p)
            fitted_models.append(model)
            for fold in s["folds"]:
                for arm, result in fold["held_out"].items():
                    selections.append({"model": model, "fold": fold["fold"], "arm": arm,
                                       "minimum_class_coverage": result["minimum_class_coverage"],
                                       "test_families": fold["test_families"],
                                       "minimum_complete": result["minimum_complete"]})
                    for c, label in enumerate(s["classes"]):
                        for l, layer in enumerate(s["layers"]):
                            class_rows.append({"model": model, "fold": fold["fold"], "arm": arm,
                                "class": label, "layer": layer, "coverage": result["coverage"][c][l],
                                "family_support": result["family_support"][c][l]})
                for layer, jaccard in zip(s["layers"], fold["balanced_frequency_jaccard"]):
                    stability.append({"model": model, "comparison": "balanced_frequency_overlap", "arm": None,
                        "fold_i": fold["fold"], "fold_j": None, "layer": layer, "jaccard": jaccard})
            for arm, comparisons in s["stability"].items():
                for pair in comparisons:
                    for layer, jaccard in zip(s["layers"], pair["layer_jaccard"]):
                        stability.append({"model": model, "comparison": "fold_stability", "arm": arm,
                            "fold_i": pair["folds"][0], "fold_j": pair["folds"][1], "layer": layer, "jaccard": jaccard})
    if selections:
        with (output / "selection_coverage.csv").open("x") as f:
            writer = csv.DictWriter(f, fieldnames=list(selections[0]))
            writer.writeheader()
            writer.writerows(selections)
        for name, rows in (("class_layer_coverage.csv", class_rows), ("mask_overlap_stability.csv", stability)):
            with (output / name).open("x") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
        for ax, model in zip(axes.flat, MODELS):
            ax.set_title(LABELS[model])
            if model not in fitted_models:
                ax.text(.5, .5, "UNMEASURED", transform=ax.transAxes, ha="center")
            for pos, arm in enumerate(ARMS[1:]):
                values = [r["minimum_class_coverage"] for r in selections if r["model"] == model and
                          r["arm"] == arm and r["minimum_complete"] and r["minimum_class_coverage"] is not None]
                if values:
                    ax.scatter(np.full(len(values), pos), values, alpha=.6, s=20)
            ax.set_xticks(range(3), ["Balanced", "Labelled freq.", "Random"], rotation=20, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Minimum class coverage")
        axes.flat[-1].axis("off")
        fig.savefig(output / "held_out_coverage.pdf")
        plt.close(fig)
    if memory_rows:
        with (output / "realized_memory.csv").open("x") as f:
            writer = csv.DictWriter(f, fieldnames=list(memory_rows[0]))
            writer.writeheader()
            writer.writerows(memory_rows)
    analysis_path = root / "ACCURACY_ANALYSIS.json"
    if analysis_path.exists():
        a = load(analysis_path)
        require(a["campaign_sha256"] == campaign["sha256"], "foreign accuracy analysis")
        sources[str(analysis_path)] = file_sha(analysis_path)
        with (output / "accuracy_contrasts.csv").open("x") as f:
            writer = csv.DictWriter(f, fieldnames=list(a["strict"]["contrasts"][0]))
            writer.writeheader()
            writer.writerows(a["strict"]["contrasts"])
        joint = []
        for row in a["arms"]:
            metrics = row["memory"]
            joint.append({"model": row["model"], "arm": row["arm"], "families": row["assigned_families"],
                          "strict_accuracy": row["strict_accuracy"], "semantic_lower": row["semantic_bounds"][0],
                          "semantic_upper": row["semantic_bounds"][1], "memory_observed_families": row["memory_observed_families"],
                          **{name: metrics[name]["mean"] if metrics[name] is not None else None for name in metrics},
                          "assigned_failures": row["assigned_failures"], "caps": row["caps"]})
        with (output / "accuracy_memory.csv").open("x") as f:
            writer = csv.DictWriter(f, fieldnames=list(joint[0]))
            writer.writeheader()
            writer.writerows(joint)
        table = [r"% Source-bound all-assignment scores; memory has its own observed denominator.",
                 r"\begin{table}[t]\centering\small",
                 r"\caption{Accuracy and realized memory at the frozen expert budget. Semantic brackets are identification bounds, not confidence intervals. Paired simultaneous uncertainty is reported separately.}",
                 r"\label{tab:accuracy-memory}\begin{tabular}{llrrrr}\toprule",
                 r"Model & Arm & Strict (\%) & Semantic (\%) & Expert GiB & GPU GiB \\", r"\midrule"]
        names = {"native": "Native", "state_balanced": "Balanced", "labelled_frequency": "Labelled freq.", "random": "Random"}
        for row in joint:
            memory = ["--" if row[k] is None else f"{row[k]/1024**3:.2f}" for k in ("expert_bytes", "nvml_process_bytes")]
            table.append(" & ".join([LABELS[row["model"]], names[row["arm"]], f"{100*row['strict_accuracy']:.1f}",
                f"[{100*row['semantic_lower']:.1f},{100*row['semantic_upper']:.1f}]", *memory]) + r" \\")
        table += [r"\bottomrule\end{tabular}\end{table}", ""]
        (output / "accuracy_memory.tex").write_text("\n".join(table))
    return save(output / "EXPORT.json", {"schema": "expert-memory-paper-export-v1", "source_files": sources,
                "campaign_sha256": campaign["sha256"], "all_seven_explicit": True,
                "outputs": {str(p): file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
