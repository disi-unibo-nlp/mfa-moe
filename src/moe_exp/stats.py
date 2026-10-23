"""Family-clustered bootstrap panel and the unknown-outcome bounds.

Vendored verbatim from `research_campaign_v3_r6.statistics` and `research_campaign_v3_r6.scoring`.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from moe_exp.receipts import SEED, digest, require
from moe_exp.receipts import records as normalize_records, finite


def bootstrap_panel(records, columns, *, draws=10000, panel="primary", alpha=.05, transform=None, resample_key=None, chunk_size=256, benchmarks=None, family_universe=None):
    """One multiplicity per family across benchmarks, arms and model copies.

    Each cell first averages repeats within a benchmark/family occurrence. Missing
    cells stay missing. The bootstrap conditions on frozen masks and nominations.
    Simultaneous intervals use the maximum centered standardized deviation.
    """
    require(draws == 10000 or panel.startswith("test:"), "reported inference requires 10000 resamples")
    records = normalize_records(records)
    grouped = defaultdict(list)
    for r in records:
        grouped[r["benchmark"], r["family"]].append(r)
    families = sorted(family_universe if family_universe is not None else {f for _, f in grouped})
    benchmarks = sorted(benchmarks if benchmarks is not None else {b for b, _ in grouped})
    require(chunk_size>0,"bootstrap chunk must be positive")
    B, F, C = len(benchmarks), len(families), len(columns)
    cells = np.full((B, F, C), np.nan)
    fi, bi = {f: i for i, f in enumerate(families)}, {b: i for i, b in enumerate(benchmarks)}
    for (b, f), rows in grouped.items():
        for j, col in enumerate(columns):
            values = [r.get(col) for r in rows if finite(r.get(col))]
            if values:
                cells[bi[b], fi[f], j] = float(np.mean(values))
    valid = np.isfinite(cells)
    if transform is not None:
        output_columns,operation,*declared=transform
        dependencies=declared[0] if declared else [list(range(C)) for _ in output_columns]
        require(len(dependencies)==len(output_columns) and all(d and all(0<=i<C for i in d) for d in dependencies),
                'transformed statistic support declaration differs')
        output_valid=np.stack([valid[:,:,d].all(2) for d in dependencies],axis=2)
        groups={}
        for j in range(len(output_columns)):
            mask=output_valid[:,:,j];key=np.packbits(mask).tobytes()
            groups.setdefault(key,{'mask':mask,'outputs':[]})['outputs'].append(j)
        specifications=list(groups.values());out_count=len(output_columns)
    else:
        output_columns=columns;out_count=C
        specifications=[{'mask':None,'outputs':list(range(C))}]
    # A singleton family has no bootstrap loop and all draws remain undefined.
    # One NaN sentinel yields the identical zero-valid-draw result without a
    # large unused allocation for each terminal missingness panel.
    point=np.full(out_count,np.nan);samples=np.full((draws if F>1 else 1,out_count),np.nan)
    rng=np.random.default_rng(int(digest([SEED,'bootstrap',resample_key or panel])[:16],16))
    # Save precisely the same multinomial draws for every operation/subpanel.
    multiplicities=rng.multinomial(F,np.full(F,1/F),size=draws) if F>1 else None
    for specification in specifications:
        mask=specification['mask'];outputs=specification['outputs']
        use=valid if mask is None else valid & mask[:,:,None]
        nums=np.where(use,cells,0.)
        den=use.sum(1)
        bm=np.divide(nums.sum(1),den,out=np.full((B,C),np.nan),where=den>0)
        rawpoint=bm.mean(0) if B else np.full(C,np.nan)
        point[outputs]=(operation(rawpoint[None,:])[0] if transform is not None else rawpoint)[outputs]
        if F>1:
            for start in range(0,draws,chunk_size):
                multiplicity=multiplicities[start:start+chunk_size]
                numerator=np.einsum('nf,bfc->nbc',multiplicity,nums,optimize=False)
                denominator=np.einsum('nf,bfc->nbc',multiplicity,use,optimize=False)
                avg=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
                rawsample=avg.mean(1)
                transformed=operation(rawsample) if transform is not None else rawsample
                require(transformed.shape==(len(multiplicity),out_count),'transformed statistic output geometry differs')
                samples[start:start+len(multiplicity),outputs]=transformed[:,outputs]
    if transform is not None:valid=output_valid
    columns=output_columns;C=out_count
    den=valid.sum(1)
    supported=(den>0).all(0) if B else np.zeros(C,bool)
    # Undefined resamples remain part of the experiment. A nonlinear outcome
    # with any undefined draw has no finite interval; it cannot suppress an
    # independently supported linear outcome or produce conditional intervals.
    finite_draws=np.isfinite(samples).all(0)
    scale=np.full(C,np.nan)
    if F>1 and finite_draws.any():scale[finite_draws]=np.std(samples[:,finite_draws],axis=0,ddof=1)
    eligible = supported & (valid.any(0).sum(0)>1) & finite_draws & np.isfinite(point) & np.isfinite(scale) & (scale>1e-14)
    critical=None
    if eligible.any():
        z=np.abs((samples[:,eligible]-point[eligible])/scale[eligible])
        require(np.isfinite(z).all(),"undefined draws in declared simultaneous panel")
        critical=float(np.quantile(z.max(1),1-alpha))
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
            "dependencies":dict(zip(columns,dependencies)) if transform is not None else {c:[c] for c in columns},
            "coverage_scope":"declared panel conditional on frozen selections; not global across panels"}


def unknown_bounds(correct,finish_reason,answer_present):
    uncertain=correct is None or (correct is False and finish_reason=="stop" and answer_present)
    return float(correct) if correct is not None else 0.,1. if uncertain else float(correct)
