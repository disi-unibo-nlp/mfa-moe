"""Materialize primary executions once; alias actions, masks and native branches."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import json
import numpy as np

from .contracts import (CLASSES, CONFIGURATIONS, METHODS, REMOVALS, REPLICATE, SEED,
                        digest, execution_id, load, publish, require)
from .design import config_values, masks
from .runtime import remaining_allowance
from .inputs import read_source


@lru_cache(maxsize=4)
def _source(location):
    return read_source(location)


def row_for(campaign, model, source, *, mask=None, action=None, prefix_tokens=0, window=0, capture_layers=()):
    trace = _source(source["source_location"])
    replay = trace["metadata"]["token_replay"]
    contract = campaign["models"][model]
    prefix = replay["completion_token_ids"][:prefix_tokens]
    location={k:json.loads(source["source_location"])[k] for k in ("path","byte_offset","line_bytes","line_sha256")}
    value = {"model": model, "profile_sha256": contract["profile_sha256"],
        "source": {"location": location, "trace_sha256": source["trace_sha256"],
                   "attempt_id": source["attempt_id"], "prefix_tokens": prefix_tokens,
                   "prompt_sha256": digest(replay["prompt_token_ids"]), "prefix_sha256": digest(prefix)},
        "seed": int(trace["metadata"]["generation_config"]["seed"]),
        "replicate": REPLICATE, "mask": mask, "action": action,
        "cap": remaining_allowance(contract["card"], len(replay["prompt_token_ids"]), len(prefix)),
        "penalty_history_sha256": digest(prefix), "family": source["family"],
        "benchmark": source["benchmark"], "question": source["question"], "window": window,
        "prompt_tokens":len(replay["prompt_token_ids"]),
        "capture_layers": list(capture_layers)}
    value["uid"] = execution_id(value)
    from .masks import reference
    value["mask"] = reference(campaign,model,mask)
    return value


def freeze_masks(campaign, model, reap_scores):
    root = Path(campaign["root"])
    contract = campaign["models"][model]
    with np.load(root / model / "SCORES.npz", allow_pickle=False) as archive:
        scores = archive["scores"]
    require(reap_scores.shape == (5, len(contract["layers"]), contract["experts"]), "REAP fold score shape mismatch")
    records = []
    for fold in range(5):
        values = dict(zip(("state_informed", "all_reasoning", "matched_label", "random"), scores[fold]))
        values["reap"] = reap_scores[fold]
        for method in METHODS:
            for allocation in ("equal", "unequal"):
                for removal in REMOVALS:
                    if not np.isfinite(values[method]).all():
                        records.append({"fold": fold, "method": method, "allocation": allocation,
                                        "removal": removal, "status": "insufficient_support", "mask": None})
                        continue
                    mask, retained = masks(values[method], contract["expert_bytes"], removal,
                        contract["native_k"], unequal=allocation=="unequal", layers=contract["layers"])
                    records.append({"fold": fold, "method": method, "allocation": allocation, "removal": removal,
                        "mask": mask, "mask_sha256": digest(mask), "retained_serialized_bytes": retained,
                        "status": "frozen", "outcomes_used": False})
    return publish(root / model / "MASKS.json", {"records": records, "outcomes_used": False,
                   "unequal_rule": "unchanged method score per stored byte with native per-layer minimum"})


def task_table(campaign, model, configurations):
    root = Path(campaign["root"])
    source_rows = load(root / model / "SOURCES.json")["records"]
    sources = {s["attempt_id"]: s for s in source_rows}
    by_question = {s["question"]: s for s in source_rows}
    cohort = load(root / "COHORTS.json")
    masks_record = load(root / model / "MASKS.json")
    mask_lookup = {(r["fold"],r["method"],r["allocation"],r["removal"]):r for r in masks_record["records"]}
    executions, cells = {}, []
    for selected in cohort["pruning"]:
        source = by_question[selected["question"]]
        for method, allocation, removal in [("native", "shared",0.)] + [
            (method,allocation,removal) for method in METHODS for allocation in ("equal","unequal") for removal in REMOVALS]:
            value = mask_lookup.get((selected["fold"],method,allocation,removal))
            status = "pending" if method=="native" or value["status"]=="frozen" else value["status"]
            row = row_for(campaign, model, source, mask=value["mask"] if value else None) if status=="pending" else None
            if row:
                executions.setdefault(row["uid"], row)
            cells.append({"study":"memory", "model":model, **selected, "method":method,
                "allocation":allocation, "removal":removal, "uid":row["uid"] if row else None, "status":status})
    configs = {(r["fold"],r["class"],r["depth_third"],r["sign"],r["threshold_index"],r["arm"]):r
               for r in configurations}
    for selected in load(root / model / "PREFIXES.json")["records"]:
        if selected["prefix"] is None:
            cells.append({"study":"causal",**selected,"target":"native","arm":"native","threshold_index":0,"status":"upstream_unavailable","uid":None})
            for target in CLASSES:
                for depth in range(3):
                    for sign in (-1,1):
                        for threshold in selected["configuration_indices"]:
                            for arm in ("target","random"):
                                cells.append({"study":"causal",**selected,"target":target,"depth_third":depth,"sign":sign,
                                    "threshold_index":threshold,"arm":arm,"status":"upstream_unavailable","uid":None})
            continue
        source = sources[selected["prefix"]["attempt_id"]]
        start = selected["prefix"]["prefix_tokens"]
        possible = [r for r in configurations if r["fold"]==selected["fold"] and r["status"]=="frozen"]
        window = max((r["windows"][-1] for r in possible), default=0)
        sites = sorted({r["layer"] for r in possible})
        # One shared native continuation at the union of nominated sites, bounded
        # by the largest window; no all-depth continuation archive is requested.
        native = row_for(campaign,model,source,prefix_tokens=start,window=window,capture_layers=sites) if sites else None
        if native:
            executions.setdefault(native["uid"], native)
            cells.append({"study":"causal", **selected, "target":"native", "arm":"native",
                          "threshold_index":0, "windows":possible[0]["windows"],"uid":native["uid"], "status":"pending"})
        else:
            cells.append({"study":"causal",**selected,"target":"native","arm":"native","threshold_index":0,
                "uid":None,"status":"insufficient_support"})
        for target in CLASSES:
            for depth in range(3):
                for sign in (-1,1):
                    for threshold in selected["configuration_indices"]:
                        for arm in ("target","random"):
                            conf = configs[selected["fold"],target,depth,sign,threshold,arm]
                            row = None
                            if conf["status"]=="frozen":
                                action = {"layer":conf["layer"], "experts":conf["experts"],
                                    "bias":conf["bias"],"pulse_tokens":conf["pulse_tokens"],"start":start}
                                row = row_for(campaign,model,source,action=action,prefix_tokens=start,
                                              window=window,capture_layers=[conf["layer"]])
                                executions.setdefault(row["uid"],row)
                            cells.append({"study":"causal",**selected,"target":target,"depth_third":depth,
                                "sign":sign,"threshold_index":threshold,"arm":arm,
                                "windows":conf.get("windows"), "uid":row["uid"] if row else None,
                                "status":"pending" if row else conf["status"]})
    rows = sorted(executions.values(),key=lambda r:(r["mask"] is not None,digest(r["mask"]),r["source"]["prefix_tokens"],r["uid"]))
    for i,row in enumerate(rows):
        row["task_index"] = i
    return publish(root / model / "TASKS.json", {"records":rows,"cells":cells,"configuration_count":9,
        "execution_aliases":len([c for c in cells if c["uid"]])-len(rows), "replicate":REPLICATE,
        "array_index_mapping":"records[SLURM_ARRAY_TASK_ID]; immutable task_index",
        "analysis_windows_do_not_create_execution":True})
