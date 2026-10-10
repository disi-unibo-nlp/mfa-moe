"""All-assignment family inference, shared resampling, and missingness bounds."""
from __future__ import annotations

from collections import defaultdict
from .contracts import require, seal
from .registry import MODEL_KEYS

CONTRASTS = (("target", "native"), ("target", "matched_random"))


def binary_bounds(values):
    require(values and all(v is None or type(v) is bool or v in (0,1) for v in values), "invalid binary population")
    n = len(values)
    return [sum(v is True or v == 1 for v in values if v is not None)/n,
            (sum(v is True or v == 1 for v in values if v is not None)+sum(v is None for v in values))/n]


def family_values(records, model, endpoint, left, right, families, enrolled=None):
    import numpy as np
    indexed = defaultdict(dict)
    for record in records:
        if record["model"] != model or record["experiment"] != ("conditional" if endpoint == "semantic" else "utility"):
            continue
        key = (record["family"], record["seed"], record["arm"])
        require(key not in indexed, "duplicate assigned outcome")
        indexed[key] = record
    result = np.full((len(families),2), np.nan)
    if endpoint == "semantic":
        if enrolled is None:
            require(not indexed, "conditional data lacks sealed enrollment")
            return result
        require(len(set(enrolled))==len(enrolled) and set(enrolled)<=set(families), "foreign/duplicate enrollment")
        require(all(key[0] in enrolled for key in indexed), "outcome from unenrolled conditional family")
    elif not indexed:
        return result
    for i,family in enumerate(families):
        if endpoint == "semantic" and family not in enrolled:
            continue
        bounds = {}
        absent = []
        for arm in (left,right):
            rows = [indexed.get((family,seed,arm)) for seed in (0,1)]
            absent.append(all(r is None for r in rows))
            if endpoint in ("semantic", "accuracy"):
                bounds[arm] = binary_bounds([r.get(endpoint) if r else None for r in rows])
            else:
                limit = 16384  # frozen cheap recognizer emits zero controller tokens
                require(all(r is None or r.get("controller_method") == "cheap" for r in rows),
                        "non-cheap policy needs its own certified finite token bound")
                values = [r.get("total_output_tokens") if r else None for r in rows]
                require(all(v is None or 0 <= v <= limit for v in values), "output cost exceeds contract")
                bounds[arm] = [sum(v or 0 for v in values)/2,
                               sum(v if v is not None else limit for v in values)/2]
        if len(bounds) == 2:
            result[i] = [bounds[left][0]-bounds[right][1], bounds[left][1]-bounds[right][0]]
    return result


def analyze(records, families, *, conditional_enrollment=None, draws=50000, seed=7307):
    import numpy as np
    require(len(families) == 96 and len(set(families)) == 96, "analysis requires sealed final families")
    matrices, names = [], []
    for endpoint in ("semantic", "accuracy", "tokens"):
        for model in MODEL_KEYS:
            for left,right in CONTRASTS:
                names.append((model,endpoint,left,right))
                matrices.append(family_values(records,model,endpoint,left,right,families,
                    (conditional_enrollment or {}).get(model)))
    values = np.stack(matrices, axis=1)  # family x42 x bound
    support = np.isfinite(values[:,:,0]).sum(0)
    means = np.divide(np.nansum(values,axis=0), support[:,None],
                      out=np.full((42,2),np.nan), where=support[:,None]>0)
    scales = np.zeros((42,2))
    for j in range(42):
        if support[j] > 1:
            scales[j] = np.nanstd(values[:,j,:],axis=0,ddof=1)/np.sqrt(support[j])
    usable = (support >= 10)[:,None] & (scales > 1e-12)
    rng = np.random.default_rng(seed)
    maxima = []
    # Resample the same 96 families for EVERY model and endpoint. Conditional
    # support differs across models; its enrolled family set stays fixed.
    for start in range(0, draws, 250):
        sample = values[rng.integers(0,len(families),size=(min(250,draws-start),len(families)))]
        counts = np.isfinite(sample[:,:,:,0]).sum(1)
        bootstrap_means = np.divide(np.nansum(sample,axis=1),counts[:,:,None],
            out=np.full((len(sample),42,2),np.nan),where=counts[:,:,None]>0)
        errors = np.divide(np.abs(bootstrap_means-means),scales,
                          out=np.zeros_like(bootstrap_means),where=usable)
        maxima.extend(np.nanmax(errors,axis=(1,2)).tolist())
    critical = float(np.quantile(maxima,.95)) if bool(usable.any()) else None
    output = []
    for j,(model,endpoint,left,right) in enumerate(names):
        available = support[j] > 0
        identified = means[j].tolist() if available else None
        # Degenerate bootstrap uncertainty is explicitly unavailable. Zero
        # variance is not evidence for a precise null or guaranteed success.
        interval = [float(means[j,0]-critical*scales[j,0]),float(means[j,1]+critical*scales[j,1])] if usable[j].all() else None
        if endpoint != "tokens" and interval:
            interval = [max(-1.,interval[0]),min(1.,interval[1])]
        output.append({"model":model,"endpoint":endpoint,"contrast":[left,right],
            "families":int(support[j]),"identified_mean_bounds":identified,"simultaneous_interval95":interval,
            "status":"ESTIMATED" if interval else "UNCERTAINTY_UNAVAILABLE" if available else "NOT_MEASURED",
            "missingness":"unknown outcomes retain worst-case bounds; all assignments preserved"})
    common=[]
    for endpoint in ("semantic","accuracy","tokens"):
        for contrast in CONTRASTS:
            cells=[r for r in output if r["endpoint"]==endpoint and tuple(r["contrast"])==contrast]
            if not all(r["identified_mean_bounds"] is not None for r in cells):
                common.append({"endpoint":endpoint,"contrast":list(contrast),"status":"ALL_SEVEN_REQUIRED_NOT_IDENTIFIED"})
                continue
            bounds={r["model"]:r["identified_mean_bounds"] for r in cells}
            def aggregate(weights):
                total=sum(weights.values())
                return [sum(weights[m]*bounds[m][i] for m in weights)/total for i in (0,1)]
            weights={m:1. for m in MODEL_KEYS}
            architecture={m:(.5 if m in ("qwen35","qwen36") else 1.) for m in MODEL_KEYS}
            common.append({"endpoint":endpoint,"contrast":list(contrast),"status":"DESCRIPTIVE_CHECKPOINT_MIXTURE",
                "equal_checkpoint_bounds":aggregate(weights),"qwen35_36_shared_architecture_bounds":aggregate(architecture),
                "above_zero_simultaneous":sum(r["simultaneous_interval95"] is not None and r["simultaneous_interval95"][0]>0 for r in cells),
                "below_zero_simultaneous":sum(r["simultaneous_interval95"] is not None and r["simultaneous_interval95"][1]<0 for r in cells),
                "not_a_new_confirmatory_contrast":True,"population":"declared checkpoint mixture, model-specific conditional eligibility where applicable"})
    return seal({"schema":"verification-analysis-v3","contrasts":output,"simultaneous_family_size":42,
        "common_direction_and_architecture_sensitivity":common,
        "bootstrap_draws":draws,"bootstrap_seed":seed,"critical_value":critical,
        "unit":"shared original-problem family; average both seeds within family",
        "conditional_population":"model-specific native-seed0 eligible prefixes",
        "utility_population":"all original-prompt families and assignments",
        "accuracy_retention_margin":None,"no_accuracy_retention_claim_from_nonsignificance":True,
        "unqualified_models":"missing causal cells, not zero effects",
        "cross_model_pooling":"per-checkpoint effects; seven checkpoints are not seven independent benchmark samples"})
