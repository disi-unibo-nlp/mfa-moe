"""Allocated stage implementations; committed assignments are never redrawn."""
from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import time
import numpy as np

from .artifacts import npz
from .contracts import (CLASSES, CONFIGURATIONS, Units, allocated, check_files, digest,
                        load, publish, require)
from .design import family_weights
from .inputs import read_source
from .runtime import Backend
from .tasks import freeze_masks, row_for, task_table


def measurement_rows(campaign, model):
    root = Path(campaign["root"])
    sources = load(root / model / "SOURCES.json")["records"]
    # Largest original shape is retained, not a synthetic repeat or new solution.
    ordered = sorted(sources, key=lambda s: (-s["completion_tokens"], s["attempt_id"]))
    records = [{"uid":digest(["measurement",model,s["attempt_id"],campaign["models"][model]["profile_sha256"]]),
                "source":s, "model":model, "task_index":i} for i,s in enumerate(ordered)]
    return publish(root / model / "MEASURE_TASKS.json", {"records":records,"pilot_index":0,
        "saved_data_only":True, "maximum_positions_per_attempt":64})


def reuse_native(campaign,model,deadline):
    root=Path(campaign["root"])
    sources=load(root/model/"SOURCES.json")["records"]
    lookup={s["question"]:s for s in sources}
    by_attempt={s["attempt_id"]:s for s in sources}
    rows={};cells=[]
    for selected in load(root/"COHORTS.json")["pruning"]:
        row=row_for(campaign,model,lookup[selected["question"]])
        rows[row["uid"]]=row
        cells.append({"study":"memory","model":model,**selected,"method":"native","allocation":"shared","removal":0.,"uid":row["uid"],"status":"pending"})
    nominations=load(root/model/"NOMINATIONS.json")["records"]
    for selected in load(root/model/"PREFIXES.json")["records"]:
        if selected["prefix"] is None:
            continue
        supported=[n for n in nominations if n["fold"]==selected["fold"] and n["status"]=="nominated"]
        if not supported:
            continue
        window=max(n["windows"][-1] for n in supported)
        layers=sorted({n["layer"] for n in supported})
        row=row_for(campaign,model,by_attempt[selected["prefix"]["attempt_id"]],
            prefix_tokens=selected["prefix"]["prefix_tokens"],window=window,capture_layers=layers)
        rows[row["uid"]]=row
        cells.append({"study":"causal",**selected,"target":"native","arm":"native","threshold_index":0,
            "windows":supported[0]["windows"],"uid":row["uid"],"status":"pending"})
    records=sorted(rows.values(),key=lambda r:r["uid"])
    for i,r in enumerate(records):r["task_index"]=i
    table=publish(root/model/"NATIVE_TASKS.json",{"records":records,"cells":cells,"no_new_generations":True})
    execute_rows(campaign,model,records,deadline)
    from .scoring import score
    score(campaign,model,records)
    from .annotation import prepare_contexts
    contexts=prepare_contexts(campaign,model,phase="native")
    return {"status":"complete","tasks_sha256":table["sha256"],"contexts_sha256":contexts["sha256"]}


def measure(campaign, model, shard, deadline):
    root = Path(campaign["root"])
    table = measurement_rows(campaign,model)
    units = Units(root / model / "measurements")
    if shard == 0:
        selected = table["records"][:1]
    else:
        slices = load(root / model / "MEASURE_SHARDS.json")["shards"]
        require(1 <= shard <= len(slices), "measurement shard outside immutable task table")
        selected = [table["records"][i] for i in slices[shard-1]]
    backend = None
    try:
        missing = [r for r in selected if units.state(r["uid"]) is None]
        if not missing:
            return {"status":"reused","units":len(selected)}
        backend = Backend(campaign,model,deadline=deadline)
        for row in missing:
            require(time.time() < deadline-60, "deadline before saved sequence")
            if not units.begin(row,os.environ["SLURM_JOB_ID"]):
                continue
            try:
                receipt = backend.forward(row["source"],read_source(row["source"]["source_location"]))
                layers = campaign["models"][model]["layers"]
                arrays = {}
                for l in layers:
                    cell = receipt["layers"][str(l)]
                    for field in ("scores","positions","native_ids","native_weights","weighted_norm_sum","selected_count"):
                        arrays[f"{l}_{field}"] = np.asarray(cell[field])
                artifact = npz(root / model / "banks" / (row["uid"]+".npz"), **arrays)
                units.commit(row,{"status":"complete","artifact":artifact,
                    "seconds":receipt["seconds"],"load_seconds":receipt["load_seconds"],
                    "saved_tokens":receipt["saved_tokens"],"natural_generations":0,
                    "last_source_token_excluded":receipt["last_source_token_excluded"],
                    "reasoning_endpoint_unmeasured":receipt["reasoning_endpoint_unmeasured"],
                    "native_dtypes":{str(l):{k:receipt["layers"][str(l)][k] for k in ("router_logits_dtype","hidden_states_dtype")} for l in layers},
                    "reap_supported":all(receipt["layers"][str(l)]["reap_supported"] for l in layers),
                    "expert_output_reconstruction_max_abs":max(c["reconstruction_max_abs"] for c in receipt["layers"].values())})
            except Exception as exc:
                units.commit(row,{"status":"execution_failed","error_type":type(exc).__name__,"error":str(exc)[:2000]})
                raise
        return {"status":"complete","units":len(selected)}
    finally:
        if backend:
            backend.close()


def freeze(campaign, model, deadline, shard=0):
    root = Path(campaign["root"])
    folds = load(root / "FOLDS.json")["folds"]
    measured = Units(root / model / "measurements")
    rows = measurement_rows(campaign,model)["records"]
    require(all((measured.state(r["uid"]) or {}).get("status")=="complete" for r in rows), "full saved native measurement incomplete")
    L, E = len(campaign["models"][model]["layers"]), campaign["models"][model]["experts"]
    source_rows, sums, supports = [], [], []
    banks = defaultdict(list)
    for row in rows:
        state = measured.state(row["uid"])
        check_files({state["artifact"]["path"]:state["artifact"]["sha256"]})
        source_rows.append(row["source"])
        with np.load(state["artifact"]["path"],allow_pickle=False) as archive:
            sums.append(np.stack([archive[f"{l}_weighted_norm_sum"] for l in campaign["models"][model]["layers"]]))
            supports.append(np.stack([archive[f"{l}_selected_count"] for l in campaign["models"][model]["layers"]]))
    sums, supports = np.stack(sums), np.stack(supports)
    reap = []
    for fold in range(5):
        indices = [i for i,s in enumerate(source_rows) if folds[s["family"]]!=fold]
        weights = family_weights([source_rows[i] for i in indices])
        means = np.divide(sums[indices],supports[indices],out=np.zeros_like(sums[indices]),where=supports[indices]>0)
        reap.append(np.einsum("n,nle->le",weights,means))
        if any(measured.state(rows[i]["uid"]).get("reasoning_endpoint_unmeasured") or not measured.state(rows[i]["uid"])["reap_supported"] for i in indices):
            reap[-1][:]=np.nan
    mask_record = freeze_masks(campaign,model,np.stack(reap))
    nominations = load(root / model / "NOMINATIONS.json")["records"]
    if shard == 0:
        configurations=[]
        for fold in range(5):
            configurations.extend(load(root/model/f"CONFIGURATIONS-fold-{fold}.json")["records"])
        configs=publish(root/model/"CONFIGURATIONS.json",{"records":configurations,
            "numerical_values_frozen_before_outcomes":True,"mask_sha256":mask_record["sha256"]})
        tasks=task_table(campaign,model,configurations)
        return {"status":"complete","tasks_sha256":tasks["sha256"],"configurations_sha256":configs["sha256"]}
    require(1<=shard<=5,"invalid calibration fold shard")
    nominations=[n for n in nominations if n["fold"]==shard-1]
    configurations = []
    backend = Backend(campaign,model,deadline=deadline)
    try:
        for nomination in nominations:
            bank_rows = []
            if nomination["status"]=="nominated":
                for row in rows:
                    if folds[row["source"]["family"]] == nomination["fold"]:
                        continue
                    state = measured.state(row["uid"])
                    l = nomination["layer"]
                    with np.load(state["artifact"]["path"],allow_pickle=False) as archive:
                        bank_rows.append({"scores":archive[f"{l}_scores"].tolist(),
                            "native_ids":archive[f"{l}_native_ids"].tolist(),
                            "native_weights":archive[f"{l}_native_weights"].tolist(),
                            "family":row["source"]["family"],"benchmark":row["source"]["benchmark"],
                            **state["native_dtypes"][str(l)]})
            for sign in (-1,1):
                calibration = []
                if nomination["status"]=="nominated":
                    for extent in range(3):
                        cache_id = digest([nomination,sign,extent])
                        path = root / model / "calibration" / (cache_id+".json")
                        if path.exists():
                            result = load(path)
                        else:
                            require(time.time()<deadline-120,"deadline before calibration unit")
                            reports = backend.llm.collective_rpc("calibrate",kwargs={"contract":backend.contract,
                                "banks":bank_rows,"target":{"layer":nomination["layer"],"experts":nomination["expert_sets"][extent]},
                                "control":nomination["control_sets"][extent],"fractions":(.25,.5,.75),"sign":sign})
                            require(digest(reports[0])==digest(reports[1]),"TP calibration disagreement")
                            result = publish(path,reports[0])
                        calibration.append(result)
                for index,name in enumerate(CONFIGURATIONS):
                    extent,strength,pulse = ["LMH".index(c) for c in name]
                    status = "frozen" if nomination["status"]=="nominated" and calibration[extent]["status"]=="complete" else "insufficient_support"
                    for arm in ("target","random"):
                        value = {"model":model,"fold":nomination["fold"],"class":nomination["class"],
                            "depth_third":nomination["depth_third"],"sign":sign,"threshold_index":index,
                            "configuration":name,"arm":arm,"layer":nomination["layer"],"status":status,
                            "windows":nomination["windows"],"outcomes_used":False}
                        if status=="frozen":
                            value.update(experts=nomination["expert_sets" if arm=="target" else "control_sets"][extent],
                                bias=calibration[extent]["target_doses" if arm=="target" else "control_doses"][strength],
                                pulse_tokens=nomination["pulse_tokens"][pulse],calibration_sha256=calibration[extent]["sha256"])
                        configurations.append(value)
    finally:
        backend.close()
    configs = publish(root / model / f"CONFIGURATIONS-fold-{shard-1}.json", {"records":configurations,
        "numerical_values_frozen_before_outcomes":True,"mask_sha256":mask_record["sha256"]})
    return {"status":"complete","configurations_sha256":configs["sha256"]}


def qualification_rows(campaign,model):
    root = Path(campaign["root"])
    masks = load(root / model / "MASKS.json")["records"]
    native = campaign["models"][model]
    # Different count vectors require distinct compact-loading qualification.
    unique = {}
    for r in masks:
        if r["status"]=="frozen":
            unique.setdefault(digest([len(r["mask"][str(l)]) for l in native["layers"]]),r["mask"])
    return publish(root/model/"QUALIFICATION_TASKS.json",{"records":[{"key":key,"mask":mask,"shard_index":i+1}
        for i,(key,mask) in enumerate([("native",None),*sorted(unique.items())])]})


def qualify(campaign, model, deadline, shard=0):
    root=Path(campaign["root"])
    native=campaign["models"][model]
    table=qualification_rows(campaign,model)["records"]
    if shard==0:
        cases=[]
        for r in table:
            p=root/model/"qualification"/(r["key"]+".json")
            cases.append(load(p) if p.exists() else {"key":r["key"],"status":"execution_failed","memory":[],"reason":"allocation ended without qualification commit"})
        cache={digest([s["cache_spec"] for s in c["memory"]]) for c in cases if c["status"]=="complete"}
        require(len(cache)<=1,"KV capacity differs between memory comparisons")
        native_pass=next(c for c in cases if c["key"]=="native")["status"]=="complete"
        return publish(root/model/"QUALIFICATION.json",{"status":"complete" if native_pass else "execution_failed",
            "cases":cases,"failed_shape_keys":[c["key"] for c in cases if c["status"]!="complete"],
            "independent_method_cells_continue":True,"sampled_replay_required":False})
    require(1<=shard<=len(table),"invalid qualification shard")
    cases,failures=[],[]
    for r in table[shard-1:shard]:
        key,mask=r["key"],r["mask"]
        path = root / model / "qualification" / (key+".json")
        if path.exists():
            old=load(path)
            cases.append(old)
            continue
        backend=None
        try:
            backend=Backend(campaign,model,mask=mask,deadline=deadline)
            status=backend.llm.collective_rpc("memory_status")
            require(all(s["memory"]["nvml_process_residence"] for s in status),"actual process residence unavailable")
            fidelity=backend.llm.collective_rpc("memory_qualify") if mask else None
            if fidelity:
                require(all(s["status"]=="PASS" for s in fidelity),"compact/full-resident masked equivalence failed")
            full_context=native["card"]["max_model_len"]
            token=backend.tokenizer.encode(" test",add_special_tokens=False)[0]
            fixture={"uid":digest([model,key,"full-context"]),"seed":20261008,"cap":2,"action":None,
                     "capture_layers":[],"window":0}
            output=backend.generate(fixture,[token]*(full_context-2),engineering_tokens=2)
            checks=[]
            if mask is None:
                for sign in (-1,1):
                    fixture["uid"]=digest([model,key,"signed",sign])
                    fixture["action"]={"layer":native["layers"][0],"experts":[0,1],"bias":float(sign),"start":0,"pulse_tokens":3}
                    fixture.update(capture_layers=[native["layers"][0]],window=8)
                    result=backend.generate(fixture,[token],engineering_tokens=8,audit=True)
                    capture=backend.last_capture
                    require(all(p["noop_equal"] for p in capture["parity"].values()),"within-execution native no-op fidelity failed")
                    active=next(iter(capture["capture"].values()))["active"]
                    require(active==[True]*3+[False]*5,"signed pulse boundaries differ")
                    checks.append({"sign":sign,"capture":capture})
            value=publish(path,{"status":"complete","key":key,"memory":status,"fidelity":fidelity,
                "full_context":full_context,"fixture":output,"signed_checks":checks})
            cases.append(value)
        except Exception as exc:
            failures.append({"key":key,"error_type":type(exc).__name__,"error":str(exc)[:2000]})
            publish(path,{"status":"execution_failed","key":key,"memory":[],"failures":failures})
        finally:
            if backend:
                backend.close()
    return {"status":"complete" if not failures else "execution_failed","shard":shard,"failures":failures}


def execute_rows(campaign, model, rows, deadline):
    root=Path(campaign["root"])
    units=Units(root/model/"executions")
    backend,current=None,None
    try:
        for row in rows:
            state=units.state(row["uid"])
            if state is not None:
                continue
            if row["mask"] is not None:
                shape=mask_shape(campaign,model,row["mask"])
                qualification=load(root/model/"QUALIFICATION.json")
                if shape in qualification["failed_shape_keys"]:
                    if units.begin(row,os.environ["SLURM_JOB_ID"]):
                        units.commit(row,{"status":"upstream_unavailable","qualification_shape":shape,
                            "reason":"frozen mask's physical count vector failed runtime qualification"})
                    continue
            if row["mask"] is None and row["action"] is None:
                if units.begin(row,os.environ["SLURM_JOB_ID"]):
                    trace=read_source(row["source"]["location"])
                    replay=trace["metadata"]["token_replay"]
                    contract=trace["metadata"]["generation_config"]
                    native=campaign["models"][model]
                    require(contract["revision"]==native["revision"] and
                        contract["sampler"]==native["card"]["sampler"] and
                        contract["max_model_len"]==native["card"]["max_model_len"] and
                        contract["max_tokens"]==native["card"]["max_tokens"],"native source profile incompatible")
                    start=row["source"]["prefix_tokens"]
                    tokens=replay["completion_token_ids"][start:]
                    offsets=replay["completion_offsets"]
                    char=offsets[start][0] if start<len(offsets) else len(trace["cot_text"])
                    capture={}
                    if row["capture_layers"]:
                        source=next(s for s in load(root/model/"SOURCES.json")["records"] if s["attempt_id"]==row["source"]["attempt_id"])
                        capture={"native_reference":{"tensor_path":source["tensor_path"],"packet_path":source["packet_path"],
                            "prefix_tokens":start,"window":row["window"],"layers":row["capture_layers"]},
                            "capture":{},"all_layer_capture_reused":True,"source_archive_copied":False}
                    units.commit(row,{"status":"complete","reused":True,"reuse_source":row["source"],
                        "completion_reference":{"source":row["source"],"prefix_tokens":start},"generated_tokens":len(tokens),
                        "finish_reason":trace["metadata"]["finish_reason"],"seconds":0.,"load_seconds":0.,
                        "memory_from_qualification":str(root/model/"QUALIFICATION.json"),
                        "native_suffix_is_original_within_execution":True,
                        "no_cross_run_sampled_replay_claim":True, "capture":capture})
                continue
            mask_key=digest(row["mask"])
            if backend is None or current!=mask_key:
                if backend:
                    backend.close()
                try:
                    backend=Backend(campaign,model,mask=row["mask"],deadline=deadline)
                except Exception as exc:
                    for unavailable in rows:
                        if digest(unavailable["mask"])==mask_key and units.state(unavailable["uid"]) is None and units.begin(unavailable,os.environ["SLURM_JOB_ID"]):
                            units.commit(unavailable,{"status":"upstream_unavailable","generation_started":False,
                                "reason":"qualified runtime could not initialize","error_type":type(exc).__name__,"error":str(exc)[:2000]})
                    raise
                current=mask_key
            if not units.begin(row,os.environ["SLURM_JOB_ID"]):
                continue
            try:
                trace=read_source(row["source"]["location"])
                replay=trace["metadata"]["token_replay"]
                prefix=replay["completion_token_ids"][:row["source"]["prefix_tokens"]]
                require(digest(prefix)==row["source"]["prefix_sha256"],"prefix source changed")
                result=backend.generate(row,replay["prompt_token_ids"],prefix)
                if row["capture_layers"]:
                    capture=backend.last_capture
                    result["capture"]=capture
                units.commit(row,result)
            except Exception as exc:
                units.commit(row,{**getattr(backend,"last_partial",{}),"capture":getattr(backend,"last_capture",None),
                    "status":"execution_failed","error_type":type(exc).__name__,"error":str(exc)[:2000]})
                # A CUDA/engine failure can poison later requests in this process.
                # Retain the failure; independent model jobs continue.
                raise
        return {"status":"complete","assignments":len(rows)}
    finally:
        if backend:
            backend.close()


def mask_shape(campaign,model,mask):
    return digest([len(mask[str(l)]) for l in campaign["models"][model]["layers"]])


def price_rows(campaign,model):
    root=Path(campaign["root"])
    table=load(root/model/"TASKS.json")
    rows=table["records"]
    grouped=defaultdict(list)
    for r in rows:
        if r["mask"] is None and r["action"] is None:
            continue
        shape=mask_shape(campaign,model,r["mask"]) if r["mask"] is not None else None
        if shape in load(root/model/"QUALIFICATION.json")["failed_shape_keys"]:
            continue
        role="memory:"+shape if shape else "causal" if r["source"]["prefix_tokens"] else "native"
        grouped[role].append(r)
    original=[]
    for role,choices in sorted(grouped.items()):
        original.append({"role":role,"row":max(choices,key=lambda r:(r.get("prompt_tokens",0)+r["source"]["prefix_tokens"],r["cap"],r["uid"])),"shard_index":len(original)+1})
    return publish(root/model/"PRICE_TASKS.json",{"records":original,"required_physical_shapes":True})


def price(campaign, model, deadline, shard=0):
    root=Path(campaign["root"])
    tasks=price_rows(campaign,model)["records"]
    if shard==0:
        records=[];failed=[]
        for t in tasks:
            p=root/model/"prices"/(t["role"]+".json")
            if p.exists():
                records.append(load(p))
            else:
                failed.append(t["role"])
        # A failed first original role has no replacement draw. Close only its
        # dependent cells and preserve independent priced methods/operators.
        units=Units(root/model/"executions")
        for row in load(root/model/"TASKS.json")["records"]:
            if row["mask"] is None and row["action"] is None:continue
            shape=mask_shape(campaign,model,row["mask"]) if row["mask"] is not None else None
            role="memory:"+shape if shape else "causal"
            if role in failed and units.state(row["uid"]) is None and units.begin(row,os.environ["SLURM_JOB_ID"]):
                units.commit(row,{"status":"upstream_unavailable","reason":"first original required role measurement failed",
                    "role":role,"generation_started":False,"redraw":False})
        return publish(root/model/"PRICES.json",{"status":"complete",
            "records":records,"missing_roles":failed,"original_assignments_retained":True})
    require(1<=shard<=len(tasks),"unknown retained role price shard")
    task=tasks[shard-1]
    original=[task["row"]]
    execute_rows(campaign,model,original,deadline)
    units=Units(root/model/"executions")
    measurements=[]
    for row in original:
        result=units.state(row["uid"])
        require(result["status"]=="complete","retained role measurement failed")
        measurements.append({"uid":row["uid"],"role":task["role"],
            "seconds":result["seconds"],"load_seconds":result["load_seconds"],"generated_tokens":result["generated_tokens"],
            "cap":row["cap"],"prefix_tokens":row["source"]["prefix_tokens"],"output_bytes":len(str(result).encode()),
            "runtime_shape":campaign["models"][model]["runtime_shape"],"retained_as_campaign_data":True})
    return publish(root/model/"prices"/(task["role"]+".json"),{"status":"complete",**measurements[0]})
