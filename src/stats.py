"""Family-clustered bootstrap panel and the unknown-outcome bounds.

`bootstrap_panel` resamples question families (never single attempts) and gives simultaneous 95%
intervals; its seeds come from `store.digest`, so the numbers are fixed by the inputs.
`unknown_bounds` turns a possibly unknown correctness into a lower and an upper value.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numbers
import numpy as np

from store import SEED, clean, digest, require


def finite(value):
    return isinstance(value, numbers.Real) and math.isfinite(value)


def clean_rows(rows):
    """Clean every row; a missing or non-finite value is listed under the row's "unavailable_reasons"."""
    output = []
    for row in rows:
        reasons = {}
        row = clean(row, reasons)
        if reasons:
            row["unavailable_reasons"] = {**row.get("unavailable_reasons", {}), **reasons}
        output.append(row)
    return output


def bootstrap_panel(records, columns, *, draws=10000, panel="primary", resample_key=None):
    """One multiplicity per family across benchmarks, arms and model copies.

    Each cell first averages repeats within a benchmark/family occurrence. Missing
    cells stay missing. The bootstrap conditions on frozen masks and nominations.
    Simultaneous intervals use the maximum centered standardized deviation.
    """
    alpha = .05
    require(draws == 10000, "reported inference requires 10000 resamples")
    records = clean_rows(records)
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
            values = [r.get(col) for r in rows if finite(r.get(col))]
            if values:
                cells[bi[b], fi[f], j] = float(np.mean(values))
    valid = np.isfinite(cells)
    nums = np.where(valid, cells, 0.)
    den = valid.sum(1)
    benchmark_means = np.divide(nums.sum(1), den, out=np.full((B, C), np.nan), where=den > 0)
    point = benchmark_means.mean(0) if B else np.full(C, np.nan)
    # A singleton family has no bootstrap loop and all draws remain undefined.
    # One NaN sentinel yields the identical zero-valid-draw result without a
    # large unused allocation for each terminal missingness panel.
    samples = np.full((draws if F > 1 else 1, C), np.nan)
    if F > 1:
        rng = np.random.default_rng(int(digest([SEED, 'bootstrap', resample_key or panel])[:16], 16))
        multiplicities = rng.multinomial(F, np.full(F, 1/F), size=draws)
        for start in range(0, draws, 256):  # blocks only bound memory
            multiplicity = multiplicities[start:start+256]
            numerator = np.einsum('nf,bfc->nbc', multiplicity, nums, optimize=False)
            denominator = np.einsum('nf,bfc->nbc', multiplicity, valid, optimize=False)
            avg = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            samples[start:start+len(multiplicity)] = avg.mean(1)
    supported = (den > 0).all(0) if B else np.zeros(C, bool)
    # Undefined resamples remain part of the experiment. A nonlinear outcome
    # with any undefined draw has no finite interval; it cannot suppress an
    # independently supported linear outcome or produce conditional intervals.
    finite_draws = np.isfinite(samples).all(0)
    scale = np.full(C, np.nan)
    if F > 1 and finite_draws.any():
        scale[finite_draws] = np.std(samples[:, finite_draws], axis=0, ddof=1)
    eligible = supported & (valid.any(0).sum(0) > 1) & finite_draws & np.isfinite(point) & np.isfinite(scale) & (scale > 1e-14)
    critical = None
    if eligible.any():
        z = np.abs((samples[:, eligible] - point[eligible]) / scale[eligible])
        require(np.isfinite(z).all(), "undefined draws in declared simultaneous panel")
        critical = float(np.quantile(z.max(1), 1 - alpha))
    results = {}
    for j, col in enumerate(columns):
        ok = bool(eligible[j] and critical is not None)
        results[col] = {"estimate": float(point[j]) if supported[j] and np.isfinite(point[j]) else None,
                        "family_support": int(valid[:, :, j].any(0).sum()),
                        "benchmark_family_support": dict(zip(benchmarks, den[:, j].tolist())),
                        "interval": [float(point[j] - critical * scale[j]), float(point[j] + critical * scale[j])] if ok else None,
                        "status": "estimable" if ok else "insufficient_support_or_degenerate",
                        "bootstrap_valid_resamples": int(np.isfinite(samples[:, j]).sum()),
                        "unavailable_reason":None if ok else "missing_required_benchmark_or_undefined_draw_or_degenerate_variance"}
    return {"panel": panel, "results": results, "resamples": draws,
            "simultaneous": True, "critical_value": critical, "paired_family_multiplicities": True,
            "conditional_on_frozen_selection": True, "run_to_run_sd": None, "alpha": alpha, "undefined_draws_discarded": 0,
            "resample_key": resample_key or panel, "sufficient_statistics_sha256": digest(records),
            "family_universe_sha256":digest(families),"resampling_identity":digest([SEED,resample_key or panel,families,draws]),"benchmark_universe":benchmarks,
            "dependencies":{c:[c] for c in columns},
            "coverage_scope":"declared panel conditional on frozen selections; not global across panels"}


def unknown_bounds(correct,finish_reason,answer_present):
    uncertain=correct is None or (correct is False and finish_reason=="stop" and answer_present)
    return float(correct) if correct is not None else 0.,1. if uncertain else float(correct)
