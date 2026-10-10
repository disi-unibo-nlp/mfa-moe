"""Measured role prices and finite shards; no zero-cost placeholders."""
from __future__ import annotations

import math
from .contracts import require, seal, verify, save

ROLES = ("generator_load", "prefill", "generation", "controller", "capture", "reader_load",
         "eligibility_reader", "outcome_reader", "grading", "closeout", "recovery")


def price_sheet(model, samples, code_files):
    verify(model)
    indexed = {role: [] for role in ROLES}
    for sample in samples:
        require(sample["role"] in indexed and sample["model_sha256"] == model["sha256"] and
                sample["execution_sha256"] and sample["seconds"] >= 0 and
                sample["units"] > 0 and sample["receipt_path"], "unbound component price")
        require(sample.get("engineering_only") is not True or sample["role"] in
                ("generator_load", "prefill", "generation", "capture", "closeout", "recovery"),
                "engineering fixture cannot price semantic/scientific controller roles")
        indexed[sample["role"]].append(sample)
    components = {role: {"status": "MEASURED" if rows else "UNMEASURED",
        "seconds_per_unit_max": max((r["seconds"]/r["units"] for r in rows), default=None),
        "max_context_input_tokens":max((r.get("context_input_tokens",0) for r in rows),default=0),
        "max_total_context_tokens":max((r.get("total_context_tokens",0) for r in rows),default=0),
        "samples": rows} for role, rows in indexed.items()}
    return seal({"schema": "verification-prices-v3", "model_sha256": model["sha256"],
        "components": components, "code_files": code_files,
        "missing": [r for r in ROLES if not indexed[r]], "not_a_scientific_result": True})


def shard_budget(prices, work, *, hours=8, reserve_seconds=600):
    verify(prices)
    require(0 < hours <= 24 and work, "finite nonempty shard required")
    seconds = 0.
    for role, units in work.items():
        require(role in prices["components"] and units >= 0, "unknown resource role")
        if units:
            component = prices["components"][role]
            require(component["status"] == "MEASURED", "unmeasured required price: " + role)
            seconds += units * component["seconds_per_unit_max"]
    conservative = 1.25 * seconds + reserve_seconds
    require(conservative <= hours * 3600, "work does not fit finite allocation")
    return {"measured_seconds": seconds, "reservation_seconds": math.ceil(conservative),
        "walltime_seconds": hours * 3600, "gpu_count": 2,
        "billing_core_hours": 16 * hours, "price_sha256": prices["sha256"]}


def collect_prices(campaign_path,key,root,output):
    from .contracts import load,digest,check_files,require_generation_binding
    campaign=load(campaign_path);check_files(campaign["code_files"])
    model=load(campaign["models"][key]);base=root/key;samples=[]
    paths=list((base/"qualification/fixtures").glob("*.json"))
    for stage in ("discovery","validation","final_native","utility","conditional"):
        paths+=list((base/stage).glob("*.json"))
    def add(role,seconds,units,path,receipt,engineering=False):
        if seconds is not None and units>0:
            samples.append({"role":role,"seconds":seconds,"units":units,"model_sha256":model["sha256"],
                "execution_sha256":receipt.get("runtime_sha256",campaign["runtime_sha256"]),
                "receipt_path":str(path),"receipt_sha256":receipt["sha256"],"engineering_only":engineering,
                "context_input_tokens":receipt.get("prompt_input_tokens",0)+receipt.get("prefix_input_tokens",0),
                "total_context_tokens":receipt.get("prompt_input_tokens",0)+receipt.get("prefix_input_tokens",0)+receipt.get("generated_tokens",0)})
    for path in paths:
        receipt=load(path)
        if receipt.get("schema")!="verification-generation-receipt-v3":continue
        require_generation_binding(receipt,campaign,model)
        require(receipt["model_sha256"]==model["sha256"] and receipt["runtime_sha256"]==campaign["runtime_sha256"],"foreign component price")
        engineering=receipt["engineering_only"]
        add("generator_load",receipt["load_seconds"],1,path,receipt,engineering)
        add("prefill",receipt["prefill_seconds"],1,path,receipt,engineering)
        add("generation",receipt["decode_inclusive_seconds"],receipt["generated_tokens"],path,receipt,engineering)
        add("capture",receipt["capture_closeout_seconds"],1,path,receipt,engineering)
        add("closeout",receipt["capture"]["archive_seconds"],1,path,receipt,engineering)
        if receipt["controller_calls"] and not engineering:
            add("controller",receipt["controller_seconds"],receipt["controller_calls"],path,receipt)
        # A retry can consume one complete request plus another cold load.
        add("recovery",receipt["load_seconds"]+receipt["generation_seconds"]+receipt["capture"]["archive_seconds"],1,path,receipt,engineering)
    for directory,role in ((base/"eligibility-ratings","eligibility_reader"),(base/"outcome-ratings","outcome_reader")):
        runtime_path=directory/"RUNTIME.json"
        if runtime_path.exists():
            runtime=load(runtime_path);add("reader_load",runtime["load_seconds"],1,runtime_path,runtime)
        for path in directory.glob("*.reader-*.json"):
            receipt=load(path);add(role,receipt["seconds"],1,path,receipt)
    for stage in ("final_native","utility"):
        for path in (base/(stage+"-grades")).glob("*.json"):
            receipt=load(path)
            if receipt.get("schema")=="verification-grade-v3":add("grading",receipt["seconds"],1,path,receipt)
    return save(output,price_sheet(model,samples,campaign["code_files"]),replace=True)
