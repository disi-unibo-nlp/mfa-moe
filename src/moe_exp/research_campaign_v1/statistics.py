"""Paired whole-family inference and explicit unestimable estimates."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from .contracts import SEED, digest, require


def bootstrap_panel(records, columns, *, draws=10000, panel="primary", alpha=.05, transform=None):
    """One multiplicity per family across benchmarks, arms and model copies.

    Each cell first averages repeats within a benchmark/family occurrence. Missing
    cells stay missing. The bootstrap conditions on frozen masks and nominations.
    Simultaneous intervals use the maximum centered standardized deviation.
    """
    require(draws == 10000 or panel.startswith("test:"), "reported inference requires 10000 resamples")
    grouped = defaultdict(list)
    for r in records:
        grouped[r["benchmark"], r["family"]].append(r)
    families = sorted({f for _, f in grouped})
    benchmarks = sorted({b for b, _ in grouped})
    B, F, C = len(benchmarks), len(families), len(columns)
    cells = np.full((B, F, C), np.nan)
    fi, bi = {f: i for i, f in enumerate(families)}, {b: i for i, b in enumerate(benchmarks)}
    for (b, f), rows in grouped.items():
        for j, col in enumerate(columns):
            values = [r.get(col) for r in rows if r.get(col) is not None]
            if values:
                cells[bi[b], fi[f], j] = float(np.mean(values))
    valid = np.isfinite(cells)
    if not valid.any() and transform is None:
        return {"panel":panel,"results":{col:{"estimate":None,"family_support":0,
            "benchmark_family_support":dict.fromkeys(benchmarks,0),"interval":None,
            "status":"insufficient_support_or_degenerate","bootstrap_valid_resamples":0} for col in columns},
            "resamples":draws,"simultaneous":True,"critical_value":None,"paired_family_multiplicities":True,
            "conditional_on_frozen_selection":True,"run_to_run_sd":None}
    nums = np.where(valid, cells, 0.)
    den = valid.sum(1)
    bm = np.divide(nums.sum(1), den, out=np.full((B, C), np.nan), where=den > 0)
    supported = (den > 0).all(0) if B else np.zeros(C, bool)
    point = np.mean(bm, axis=0) if B else np.full(C, np.nan)
    samples = np.full((draws, C), np.nan)
    rng = np.random.default_rng(int(digest([SEED, "bootstrap", panel])[:16], 16))
    if F > 1:
        for start in range(0, draws, 256):
            n = min(256, draws - start)
            multiplicity = rng.multinomial(F, np.full(F, 1 / F), size=n)
            numerator = (multiplicity @ nums.transpose(1,0,2).reshape(F,B*C)).reshape(n,B,C)
            denominator = (multiplicity @ valid.transpose(1,0,2).reshape(F,B*C).astype(float)).reshape(n,B,C)
            avg = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            samples[start:start+n] = avg.mean(1)
    if transform is not None:
        output_columns,operation=transform
        point=operation(point[None,:])[0]
        samples=operation(samples)
        support_input=valid.any(0).sum(0)
        # All sufficient-statistic columns in a declared nonlinear panel must
        # have support. Missing denominators cannot create certain log odds.
        supported=np.full(len(output_columns),bool(supported.all()))
        minimum=int(support_input.min()) if len(support_input) else 0
        den=np.repeat(den.min(1)[:,None],len(output_columns),axis=1)
        valid=np.repeat(valid.all(2)[:,:,None],len(output_columns),axis=2)
        columns=output_columns
        C=len(columns)
    scale=np.full(C,np.nan)
    usable=np.isfinite(samples).sum(0)>1
    if F>1 and usable.any():scale[usable]=np.nanstd(samples[:,usable],axis=0,ddof=1)
    eligible = supported & (valid.any(0).sum(0) > 1) & np.isfinite(scale) & (scale > 1e-14)
    critical = None
    if eligible.any():
        z = np.abs((samples[:, eligible] - point[eligible]) / scale[eligible])
        complete = np.isfinite(z).all(1)
        if complete.sum() >= .95 * draws:
            critical = float(np.quantile(z[complete].max(1), 1 - alpha))
    results = {}
    for j, col in enumerate(columns):
        ok = bool(eligible[j] and critical is not None)
        results[col] = {"estimate": float(point[j]) if supported[j] and np.isfinite(point[j]) else None,
                        "family_support": int(valid[:, :, j].any(0).sum()),
                        "benchmark_family_support": dict(zip(benchmarks, den[:, j].tolist())),
                        "interval": [float(point[j] - critical * scale[j]), float(point[j] + critical * scale[j])] if ok else None,
                        "status": "estimable" if ok else "insufficient_support_or_degenerate",
                        "bootstrap_valid_resamples": int(np.isfinite(samples[:, j]).sum())}
    return {"panel": panel, "results": results, "resamples": draws,
            "simultaneous": True, "critical_value": critical, "paired_family_multiplicities": True,
            "conditional_on_frozen_selection": True, "run_to_run_sd": None}


def joint_tables(ids_a, ids_b, experts_a, experts_b):
    """Full native top-k joints at the same token, without top-1 projection."""
    a, b = np.asarray(ids_a), np.asarray(ids_b)
    require(a.ndim == b.ndim == 2 and len(a) == len(b), "top-k token alignment differs")
    require(len(a) > 0 and (a >= 0).all() and (a < experts_a).all() and
            (b >= 0).all() and (b < experts_b).all(), "invalid native expert IDs")
    require((np.diff(np.sort(a, axis=1), axis=1) != 0).all() and
            (np.diff(np.sort(b, axis=1), axis=1) != 0).all(), "duplicate native selected IDs")
    x = np.zeros((len(a), experts_a), np.int64)
    y = np.zeros((len(b), experts_b), np.int64)
    np.put_along_axis(x, a, 1, axis=1)
    np.put_along_axis(y, b, 1, axis=1)
    joint = x.T @ y / len(a)
    p, q = x.mean(0), y.mean(0)
    return joint, p, q


def log_odds_ratio(joint, p, q):
    a, b, c, d = joint, p[:, None] - joint, q[None, :] - joint, 1 - p[:, None] - q[None, :] + joint
    good = (a > 0) & (b > 0) & (c > 0) & (d > 0)
    result = np.full_like(a, np.nan)
    result[good] = np.log(a[good]) + np.log(d[good]) - np.log(b[good]) - np.log(c[good])
    return result


def trajectories(rows):
    """Original sentence index and segment must both establish adjacency."""
    ordered = sorted(rows, key=lambda r: (r["attempt_id"], r["sentence_index"]))
    transitions, runs, reentries = [], [], []
    previous, run, seen = None, None, set()
    for r in ordered:
        adjacent = previous is not None and r["attempt_id"] == previous["attempt_id"] and r["segment"] == previous["segment"] and r["sentence_index"] == previous["sentence_index"] + 1
        if not adjacent:
            if run:
                runs.append({**run, "right_censored": True})
            seen = set()
            run = {"attempt_id": r["attempt_id"], "class": r["class"], "sentences": 1,
                   "tokens": r["n_tokens"], "left_censored": True}
        else:
            transitions.append({"source": previous["class"], "destination": r["class"],
                                "attempt_id": r["attempt_id"], "boundary": r["sentence_index"]})
            if r["class"] == previous["class"]:
                run["sentences"] += 1
                run["tokens"] += r["n_tokens"]
            else:
                runs.append({**run, "right_censored": False})
                if r["class"] in seen:
                    reentries.append({"attempt_id": r["attempt_id"], "class": r["class"], "boundary": r["sentence_index"]})
                seen.add(previous["class"])
                run = {"attempt_id": r["attempt_id"], "class": r["class"], "sentences": 1,
                       "tokens": r["n_tokens"], "left_censored": False}
        previous = r
    if run:
        runs.append({**run, "right_censored": True})
    return {"transitions": transitions, "dwell": runs, "reentry": reentries,
            "gaps_are_censored": True}
