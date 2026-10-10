"""Family-matched nominations; score support is distinct from causal evidence."""
from __future__ import annotations

from collections import defaultdict
import math

from .contracts import digest, require, seal, verify


def nominate(model, rows, boundary):
    verify(model); verify(boundary)
    groups = defaultdict(lambda: defaultdict(lambda: {"positive": [], "comparison": []}))
    for row in rows:
        require(row["family"] in boundary["discovery"] and row["population"] == "discovery" and
                row["information_boundary"] == "past_only" and row["matched"] is True,
                "foreign/unmatched/future-visible nomination row")
        require(row["condition"] in ("positive", "comparison") and
                math.isfinite(row["inclusion_rate"]) and 0 <= row["inclusion_rate"] <= 1,
                "invalid nomination rate")
        layer, expert = row["layer"], row["expert"]
        require(type(layer) is int and layer >= 0 and type(expert) is int and
                0 <= expert < model["experts"], "invalid nomination coordinates")
        groups[layer, expert][row["family"]][row["condition"]].append(row["inclusion_rate"])
    ranked = []
    mean = lambda x: sum(x) / len(x)
    for (layer, expert), families in groups.items():
        differences = {f: mean(r["positive"]) - mean(r["comparison"])
                       for f, r in families.items() if r["positive"] and r["comparison"]}
        fold = {f: int(digest(["verification-fold", f])[:8], 16) % 3 for f in differences}
        effects = [mean([v for f, v in differences.items() if fold[f] == k])
                   if any(fold[f] == k for f in differences) else None for k in range(3)]
        supported = len(differences) >= 10 and all(v is not None and v > 0 for v in effects)
        ranked.append({"layer": layer, "expert": expert, "matched_families": len(differences),
            "difference": mean(list(differences.values())) if differences else None,
            "fold_effects": effects, "eligible": supported})
    eligible = sorted((r for r in ranked if r["eligible"]), key=lambda r: (-r["difference"], r["layer"], r["expert"]))
    return seal({"schema": "verification-nomination-v3", "model_sha256": model["sha256"],
        "boundary_sha256": boundary["sha256"], "status": "NOMINATED" if eligible else "UNQUALIFIED_SUPPORT",
        "selected": eligible[0] if eligible else None, "candidates": ranked,
        "source_rows_sha256": digest(rows), "causal_outcomes_used": False,
        "target_label_scope": "declared offline destination label; not online semantic truth"})


def calibration(evaluate, scale, *, target=.25, tolerance=.025):
    """Evaluate returns family-balanced target-inclusion INCREMENT on native rows."""
    require(math.isfinite(scale) and scale > 0, "positive native score scale required")
    points = {0.: float(evaluate(0.))}
    require(abs(points[0.]) <= 1e-6, "zero bias is not an inclusion no-op")
    high = scale
    for _ in range(9):
        points[high] = float(evaluate(high))
        if points[high] >= target - tolerance:
            break
        if len(points) < 10:
            high *= 2
    low = 0.
    for _ in range(12):
        middle = (low + high) / 2
        points[middle] = float(evaluate(middle))
        if points[middle] >= target:
            high = middle
        else:
            low = middle
    ordered = sorted(points.items())
    monotone = all(math.isfinite(y) and -1e-6 <= y <= 1 for _, y in ordered) and all(
        b[1] >= a[1] - 1e-6 for a, b in zip(ordered, ordered[1:]))
    valid = [(bias, dose) for bias, dose in ordered if bias > 0 and abs(dose - target) <= tolerance]
    return seal({"schema": "verification-calibration-v3", "status": "PASS" if monotone and valid else "UNQUALIFIED_DOSE",
        "selected_bias": min(valid)[0] if monotone and valid else None, "target_increment": target,
        "tolerance": tolerance, "monotone": monotone, "points": ordered,
        "causal_outcomes_used": False})


def random_expert(pool, *, family, model, target):
    require(pool and target not in pool and len(pool) == len(set(pool)), "invalid matched-random pool")
    return sorted(pool)[int(digest(["verification-random-v3", model, family, target])[:16], 16) % len(pool)]
