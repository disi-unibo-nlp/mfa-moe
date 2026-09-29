"""Bounded question-balanced outcome/cost tables from a frozen metadata inventory."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np

from .contracts import sampling_weight, seal, validate
from .contrasts import joint_question_draws


def summarize(inventory, *, bootstrap=1000, seed=42):
    validate(inventory, "inventory")
    attempts = inventory["payload"]["attempts"]
    names, indices = joint_question_draws([r["question_id"] for r in attempts], bootstrap=bootstrap, seed=seed)
    rows = []
    for model, cohort in sorted({(r["model"], m["cohort"]) for r in attempts for m in r["memberships"]}):
        eligible = [r for r in attempts if r["model"] == model and any(m["cohort"] == cohort for m in r["memberships"])]
        for scope in ("fixed_budget", "natural_completion_sensitivity"):
            group = [r for r in eligible if scope == "fixed_budget" or r["termination"] == "stop"]
            for metric in ("is_correct", "completion_tokens", "capped"):
                grouped = defaultdict(list)
                for r in group:
                    if r[metric] is not None:
                        grouped[r["question_id"]].append((float(r[metric]), sampling_weight(r, "B") if cohort == "B" else 1.))
                per_question = {q: sum(v * w for v, w in pairs) / sum(w for _, w in pairs)
                                for q, pairs in grouped.items()}
                vector = np.array([per_question.get(q, np.nan) for q in names])
                if not per_question:
                    continue
                samples = vector[indices]
                n = np.isfinite(samples).sum(1)
                values = np.nansum(samples, axis=1)[n > 0] / n[n > 0]
                rows.append(dict(model=model, cohort=cohort, scope=scope, metric=metric,
                    estimate=float(np.mean(list(per_question.values()))),
                    ci=np.quantile(values, [.025, .975]).tolist(), questions=len(per_question),
                    attempts=sum(len(p) for p in grouped.values()),
                    unresolved=sum(r["is_correct"] is None for r in group),
                    population="question_balanced_A" if cohort == "A" else "question_balanced_B_eligible_only",
                    claim="descriptive", uncertainty="joint_question_bootstrap_across_models"))
    return seal("analysis", dict(tables=rows, routing_results="pending_verified_feature_extraction"),
        inputs={"inventory": inventory["binding"]}, config=dict(bootstrap=bootstrap, seed=seed),
        population=inventory["population"])


def render(artifact, path):
    validate(artifact, "analysis")
    lines = ["# Question-balanced fixed-budget baselines", "",
        "Descriptive estimates from the frozen original-budget inventory. Correctness and termination are separate.",
        "The 95% intervals jointly resample question identities across models; they do not treat sibling attempts as independent questions.",
        "Unresolved correctness is omitted only from the correctness denominator. Cost retains all attempts.", "",
        "| Model | Cohort | Questions scored | Accuracy (95% CI) | Mean completion tokens | Cap fraction |",
        "|---|---|---:|---|---:|---:|"]
    groups = defaultdict(dict)
    for row in artifact["payload"]["tables"]:
        if row["scope"] == "fixed_budget":
            groups[(row["model"], row["cohort"])][row["metric"]] = row
    for (model, cohort), values in sorted(groups.items()):
        correct, cost, cap = (values[k] for k in ("is_correct", "completion_tokens", "capped"))
        lo, hi = correct["ci"]
        lines.append(f"| {model} | {cohort} | {correct['questions']} | {correct['estimate']:.3f} ({lo:.3f}–{hi:.3f}) | {cost['estimate']:.0f} | {cap['estimate']:.3f} |")
    lines.extend(["", "B results use arm inclusion probabilities within each question and apply only to that model's historically eligible questions. They are not benchmark accuracy estimates.",
        "", "Natural-completion sensitivity tables, denominators, unresolved counts, configuration and input/code bindings are retained in the adjacent JSON artifact.",
        "", "No routing candidate, prospective gain, or causal improvement is established by these tables.", ""])
    Path(path).write_text("\n".join(lines))
