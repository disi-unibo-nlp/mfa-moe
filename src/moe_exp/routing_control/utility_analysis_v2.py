"""Prespecified joint accuracy/cost contrasts with family-paired intervals."""
from __future__ import annotations

import numpy as np

from .study_v1 import require, utility_frontier as historical_frontier
from .study_v2 import CAP, seal


def utility_frontier(rows, families, arms, pairs, *, draws=50000, seed=20261006):
    require(arms and arms[0] == "native" and len(arms) == len(set(arms)) and pairs and
            len({tuple(p) for p in pairs}) == len(pairs) and
            all(len(p) == 2 and p[0] in arms and p[1] in arms and p[0] != p[1] for p in pairs), "invalid prespecified utility contrasts")
    for row in rows:
        require(row.get("continuation_cap") == CAP and (row.get("generated_tokens") is None or
                type(row["generated_tokens"]) is int and 0 <= row["generated_tokens"] <= CAP),
                "historical or over-cap utility row")
    base = historical_frontier(rows, families, arms[1:], draws=100, seed=seed)
    indexed = {(r["family"], r["seed"], r["arm"]): r for r in rows}
    complete = base["complete_families"]
    endpoints = [(a, b, endpoint) for a, b in pairs for endpoint in ("accuracy_pp", "total_tokens")]
    data = np.zeros((len(complete), len(endpoints)))
    def value(row, endpoint):
        return 100 * row["correct"] if endpoint == "accuracy_pp" else row["generated_tokens"] + row["controller_tokens"]
    for j, (a, b, endpoint) in enumerate(endpoints):
        for i, family in enumerate(complete):
            data[i, j] = np.mean([value(indexed[family, s, a], endpoint) - value(indexed[family, s, b], endpoint) for s in (0, 1)])
    intervals = [None] * len(endpoints)
    estimate = data.mean(axis=0) if len(data) else None
    require(type(draws) is int and draws >= 100, "insufficient bootstrap draws")
    if len(data) >= 2:
        rng = np.random.default_rng(seed)
        boot = np.concatenate([data[rng.integers(0, len(data), (min(1000, draws - i), len(data)))].mean(axis=1)
                               for i in range(0, draws, 1000)])
        scale = boot.std(axis=0, ddof=1)
        active = scale > 0
        q = float(np.quantile(np.max(abs((boot[:, active] - estimate[active]) / scale[active]), axis=1), .95)) if active.any() else 0.
        intervals = np.column_stack([estimate - q * scale, estimate + q * scale]).tolist()
    contrasts = []
    for j, (a, b, endpoint) in enumerate(endpoints):
        bounds = []
        if endpoint == "accuracy_pp":
            for family in families:
                limits = []
                for s in (0, 1):
                    left, right = indexed.get((family, s, a), {}).get("correct"), indexed.get((family, s, b), {}).get("correct")
                    limits.append([(0 if left is None else left) - (1 if right is None else right),
                                   (1 if left is None else left) - (0 if right is None else right)])
                bounds.append(np.mean(limits, axis=0) * 100)
        contrasts.append({"left": a, "right": b, "endpoint": endpoint, "estimate": float(estimate[j]) if estimate is not None else None,
            "simultaneous_ci95": intervals[j], "missing_accuracy_bounds": np.mean(bounds, axis=0).tolist() if bounds else None,
            "missing_cost_upper_bound": "unbounded" if endpoint == "total_tokens" and len(complete) < len(families) else None,
            "degenerate": bool(len(data) and np.ptp(data[:, j]) == 0)})
    return seal({**{k: v for k, v in base.items() if k not in ("schema", "contrasts", "draws", "seed")},
        "schema": "routing-study-utility-frontier-v2", "contrasts": contrasts, "prespecified_pairs": pairs,
        "draws": draws, "seed": seed, "inference": "Paired family bootstrap, simultaneous across all prespecified accuracy and cost contrasts"})
