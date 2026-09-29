"""Standalone research report/plots; unsupported hypotheses remain explicit."""
from pathlib import Path
import json
from .common import CLASSES


def render_report(output, summary, tables, evaluation, *, plots=False):
    output = Path(output)
    text = ["# Routing and reasoning dynamics", "", "Status: descriptive.", "",
            f"{summary['traces']} traces; {summary['questions']} questions; "
            f"{summary['labeled_sentences']}/{summary['sentences']} labeled sentences; "
            f"{summary['adjacent_labeled_pairs']} adjacent labeled pairs; "
            f"{summary['routing_windows']} routing windows.", "",
            f"Primary outcome denominator: {summary['scored_attempts']} scored attempts, "
            f"{summary['correct_attempts']} correct. All {summary['capped_attempts']} capped attempts are retained.",
            "", "Missing labels break sequences. Run durations are usable only when entry is observed; "
            "gaps and generation caps censor exits. Natural termination is a separate event.",
            "", "No compensation intervention is justified by this analysis alone.",
            "", "## Predictive evaluation", "",]
    for task, result in evaluation.items():
        if isinstance(result, dict):
            text.append(f"- {task}: {result.get('status')}; {result.get('reason', '')}")
    text += ["", "See evaluation.json for folds, class support, calibration, OOF predictions and bootstrap contrasts.",
             "BH families include all searched layers/experts for each association target and all prefix landmarks "
             "for each model/dataset correctness target. Small-support rows are not ranked.",
             "", "## Metric definitions", "",
             "Overlap is intersection/top-k; uniform correction uses k/E. The usage-preserving shuffle expectation "
             "is exact for a random permutation without replacement of the available token sets. JSD compares "
             "the two halves of each trailing window, with local order shuffles as control. Entropies use nats.",
             "Full probabilities and normalized native executed-mixture weights are separate distributions. "
             "Weak-boundary thresholds are training-fold tenth percentiles; descriptive windows retain gaps "
             "without fitting a corpus-wide threshold.",
             "", "## Sources and implementation", "",
             "Original implementation of mathematical definitions; no research framework code copied.",
             "- [ThinkARM](https://arxiv.org/html/2512.19995v3): adapted transition/motif summaries to the existing seven classes.",
             "- [SteerMoE](https://arxiv.org/html/2509.09660v2): layer-qualified selection-rate differences.",
             "- [Semantic routing](https://arxiv.org/html/2502.10928v2): adapted expert-set overlap to token lags.",
             "- [LIMoE](https://arxiv.org/html/2206.02770): local and marginal entropy.",
             "", "See report/ROUTING_DYNAMICS.md for provenance, license review and execution status."]
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        for model, dataset in sorted({(r["model"], r["dataset"]) for r in tables["transition_rates"]}):
            rows = [r for r in tables["transition_rates"] if (r["model"], r["dataset"]) == (model, dataset)]
            matrix = np.full((7, 7), np.nan)
            for row in rows:
                if row["rate"] is not None:
                    matrix[CLASSES.index(row["source"]), CLASSES.index(row["destination"])] = row["rate"]
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1)
            ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES,
                   xlabel="Next class", ylabel="Current class", title="Observed adjacent labels (descriptive)")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.colorbar(im, ax=ax, label="Transition rate")
            fig.tight_layout()
            name = model.replace("/", "--") + "-" + dataset + "-transitions"
            fig.savefig(output / (name + ".pdf"))
            fig.savefig(output / (name + ".png"), dpi=160)
            plt.close(fig)
            text += ["", f"![Transition rates]({name}.png)"]
        if tables["windows"]:
            for model in sorted({r["model"] for r in tables["windows"]}):
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                for width in sorted({r["window"] for r in tables["windows"]}):
                    layers = sorted({r["layer"] for r in tables["windows"] if r["model"] == model})
                    for ax, metric, title in zip(axes, ("overlap_lag1", "full_jsd"),
                                                ("Lag-1 top-k overlap", "Full-router half-window JSD")):
                        means = []
                        for layer in layers:
                            values = [r[metric] for r in tables["windows"]
                                      if r["model"] == model and r["layer"] == layer
                                      and r["window"] == width and r.get(metric) is not None]
                            means.append(float(np.mean(values)) if values else np.nan)
                        ax.plot(layers, means, label=f"{width} tokens")
                        ax.set(xlabel="Decoder layer", ylabel=title, title=title)
                        ax.legend()
                fig.suptitle("Descriptive window-weighted means; no causal interpretation")
                fig.tight_layout()
                name = model.replace("/", "--") + "-routing"
                fig.savefig(output / (name + ".pdf"))
                fig.savefig(output / (name + ".png"), dpi=160)
                plt.close(fig)
                text += ["", f"![Routing summaries]({name}.png)"]
    (output / "report.md").write_text("\n".join(text) + "\n")
