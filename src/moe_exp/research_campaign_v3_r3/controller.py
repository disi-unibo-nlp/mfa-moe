"""Bounded finite Slurm orchestration, scientific receipts and automatic admission."""
from __future__ import annotations
from collections import defaultdict
import os
from pathlib import Path
import time

from .contracts import MODELS, Units, check_files, digest, load, locked, publish, require
from .scheduler import Slurm, admission, array_card, card, never_started, protected_controller, submit_card, serial_capacity


def stage_key(stage,model="all",shard=0,threshold=0):
    return f"{stage}-{model}-{threshold:02d}-{shard:05d}"


def jobs(campaign,scheduler):
    root=Path(campaign["root"])
    records=[]
    for path in sorted((root/"submissions").glob("*/ACCEPTED.json")):
        accepted=load(path)
        value=load(root/"cards"/(accepted["card_sha256"]+".json"))
        assignments=[(str(accepted["job_id"])+"_"+str(i),sub) for i,sub in enumerate(value["array_cards"])] if value.get("array_cards") else [(accepted["job_id"],value)]
        for job_id,assignment in assignments:
            records.append({"card":assignment,"accepted":{**accepted,"job_id":job_id,
                "array_parent":accepted["job_id"] if value.get("array_cards") else None}})
    records=list({str(j["accepted"]["job_id"]):j for j in records}.values())
    for offset in range(0,len(records),200):
        batch=records[offset:offset+200]
        observations={}
        if hasattr(scheduler,"observe_many"):
            try:observations=scheduler.observe_many([j["accepted"]["job_id"] for j in batch])
            except Exception:pass
        for j in batch:
            job_id=j["accepted"]["job_id"]
            try:
                observation=observations.get(job_id) or scheduler.observe(job_id)
            except Exception as exc:
                observation={"job_id":job_id,"terminal":False,"state":"ACCOUNTING_PENDING",
                             "error":str(exc)[:500],"billing_hours":None}
            if observation.get('tres') and not observation.get('terminal'):
                from .scheduler import verified_resources
                verified_resources({'AllocTRES':observation['tres']},j['card'])
            j["accounting"]=observation
    return records


def status(campaign,scheduler=None,detailed=True):
    root=Path(campaign["root"])
    observed=jobs(campaign,scheduler or Slurm())
    inherited = _inherited_actual(campaign,scheduler or Slurm())
    results=[]
    for p in sorted((root/"stages").glob("*/RESULT.json")):
        results.append(load(p))
    terminal=load(root/"TERMINAL.json") if (root/"TERMINAL.json").exists() else None
    report=(root/"REPORT_COMPLETE.json").exists()
    science=load(root/"SCIENCE_COMPLETE.json")["complete"] if (root/"SCIENCE_COMPLETE.json").exists() else False
    if not detailed:
        from collections import Counter
        return {"campaign_id":campaign["campaign_id"],"root":str(root),"jobs":dict(Counter(j["accounting"]["state"] for j in observed)),
            "stages":dict(Counter(r["status"] for r in results)),"REPORT_COMPLETE":report,"SCIENCE_COMPLETE":science,
            "actual_local_hours":sum(j["accounting"].get("billing_hours") or 0 for j in observed)+inherited,
            "inherited_actual_local_hours":inherited,
            "receipt_locations":[str(root/n) for n in ("LAUNCH.json","OWNERSHIP_TRANSFER.json","handoffs","stages","output")],
            "terminal":terminal and {"status":terminal["status"]}}
    return {"campaign_id":campaign["campaign_id"],"root":str(root),"jobs":observed,"stages":results,
        "terminal":terminal,"REPORT_COMPLETE":report,"SCIENCE_COMPLETE":science,
        "actual_local_hours":sum(j["accounting"].get("billing_hours") or 0 for j in observed)+inherited,
        "inherited_actual_local_hours":inherited}


def _outstanding(snapshot,observed,scheduler):
    own={j["accepted"]["job_id"] for j in observed}
    reservation=sum(max(0,j["accepted"].get("readback",{}).get("resources",{}).get("reservation_hours",j["card"]["reservation_hours"])-(j["accounting"].get("billing_hours") or 0)) for j in observed if not j["accounting"]["terminal"])
    # Other outstanding reservations reduce the same finite spending authority.
    for job in snapshot["jobs"]:
        if job["job_id"] in own:
            continue
        import re
        text=scheduler.command(["scontrol","show","job","-o",job["job_id"]])
        fields=dict(re.findall(r"(\w+)=([^\s]+)",text))
        if fields.get("Account")!="iscrc_miosr":
            continue
        tres=fields.get("AllocTRES")
        if not tres or tres=="(null)":tres=fields.get("ReqTRES",fields.get("TRES",""))
        match=re.search(r"(?:^|,)billing=(\d+(?:\.\d+)?)",tres)
        require(match,"other outstanding job has no verifiable billing reservation")
        duration=fields["TimeLimit"]
        day,clock=duration.split("-",1) if "-" in duration else ("0",duration)
        hh,mm,ss=map(int,clock.split(":"))
        reservation+=float(match[1])*(int(day)*24+hh+mm/60+ss/3600)
    return reservation


def _check(campaign,scheduler,observed,*,hours,bytes_,prices=True,reserve=32):
    snapshot=scheduler.snapshot()
    hours+=_saved_cpu_residual(campaign,observed)
    return {"snapshot":snapshot,"saved_storage_scope":"bounded retained CPU pilots and measured remaining pair banks; production has a separate complete admission",**admission(snapshot,
        baseline_consumed=campaign["authorization_baseline_consumed"],
        campaign_actual=sum(j["accounting"].get("billing_hours") or 0 for j in observed)+_inherited_actual(campaign,scheduler),
        outstanding=_outstanding(snapshot,observed,scheduler),residual=hours,
        output_bytes=bytes_+_saved_output_residual(campaign),prices_complete=prices,reserve_hours=reserve)}


def _inherited_actual(campaign,scheduler):
    from .resume import ancestry
    seen={j['accepted']['job_id'] for j in jobs(campaign,scheduler)}
    total=0.
    for parent in ancestry(campaign):
        for j in jobs(parent,scheduler):
            jid=j['accepted']['job_id']
            if jid in seen:continue
            seen.add(jid);total+=j['accounting'].get('billing_hours') or 0.
    return total


def handoff_bound(campaign):
    root=Path(campaign['root']);shards=2
    # Each registered scientific shard can cause one handoff and at most one
    # proven pre-initialization recovery. Aggregate/snapshot/export are finite.
    for model in campaign.get('models',{}):
        shards+=12
        for name in ('COORDINATION_SHARDS','MEASURE_SHARDS','EXECUTION_SHARDS','SCORE_SHARDS','ANNOTATION_SHARDS'):
            path=root/model/(name+'.json')
            if path.exists():shards+=len(load(path)['shards'])
        for name in ('QUALIFICATION_TASKS','PRICE_TASKS'):
            path=root/model/(name+'.json')
            if path.exists():shards+=len(load(path).get('records',[]))
        shards+=len(list((root/model/'execution_fragments').glob('*.json')))
    return 2*shards+8


def _operational_residual(campaign,observed):
    # The registered bound already includes one infrastructure replacement
    # per scientific shard. Every primary consumes at most 2 CPU-hours.
    # A first guard has the same shape and occupies a remaining handoff;
    # the 12-hour final guard terminates rather than continuing a new chain.
    root=Path(campaign['root'])
    accepted=len(list((root/'handoffs').glob('*.json')))
    remaining=max(0,handoff_bound(campaign)-accepted)
    return remaining*2.+12.+12.  # terminal owner and snapshot/export reserve


def _saved_cpu_residual(campaign,observed):
    root=Path(campaign["root"])
    if campaign.get('compact_extraction'):
        from .observations import pair_tasks
        hours=0.
        for model in MODELS:
            if _failure(root,'derive',model):continue
            price=root/model/'ANALYSIS_PRICE.json'
            if not price.exists():hours+=12.;continue
            each=load(price)['pair_seconds']*4/3600
            hours+=sum(not (root/model/'coordination'/f"pair-{p['shard_index']:03d}"/'RESULT.json').exists()
                       for p in pair_tasks(campaign,model))*each
        return hours
    submitted={stage_key(j["card"]["stage"],j["card"]["model"],j["card"]["shard_index"],j["card"]["threshold_index"]) for j in observed}
    from .observations import pair_tasks
    hours=0.
    for model in MODELS:
        if _failure(root,"derive",model):continue
        if _stage_state(root,"derive",model) is None and stage_key("derive",model) not in submitted:hours+=12.
        if campaign.get("resume_from") and _stage_state(root,"mapping",model) is None and stage_key("mapping",model) not in submitted:hours+=12.
        if _stage_state(root,"supplementary",model) is None and stage_key("supplementary",model) not in submitted:hours+=12.
        pairs=pair_tasks(campaign,model)
        if not pairs:continue
        pilot=_stage_state(root,"coordination",model,pairs[0]["shard_index"])
        if pilot is None:
            if stage_key("coordination",model,pairs[0]["shard_index"]) not in submitted:hours+=12.
            continue  # First original price is a bounded measurement exception.
        if pilot["status"] not in ("complete","reused"):continue
        for pair in pairs:
            if _stage_state(root,"coordination",model,pair["shard_index"]) is None and stage_key("coordination",model,pair["shard_index"]) not in submitted:
                hours+=max(1,pilot["seconds"])*4/3600
    return hours


def _saved_output_residual(campaign):
    """Reserve uncompressed joint tables, frozen ledgers and final exports.

    Exact quota already counts committed output. Only absent artifacts enter
    this bound, so every subsequent wave protects the full saved-data scope.
    """
    root=Path(campaign["root"])
    total=0
    from .observations import pair_tasks
    for model in MODELS:
        E=campaign["models"][model]["experts"]
        unavailable = _failure(root, 'derive', model)
        if not campaign.get("compact_extraction") and campaign.get("resume_from") and not (root/model/"TOKEN_OWNERSHIP.json").exists() and _stage_state(root,"mapping",model) is None:
            source_path=root/model/"SOURCES.json"
            if not source_path.exists():source_path=Path(campaign["resume_from"])/model/"SOURCES.json"
            if source_path.exists():
                records=load(source_path)["records"]
                # Each token belongs to at most one class plus reasoning;
                # reserve both int64 arrays and per-attempt NPZ/manifest headers.
                total+=16*sum(r["completion_tokens"] for r in records)+8192*len(records)
            else:total+=4*1024**3
        pairs = pair_tasks(campaign, model)
        missing = []
        for pair in pairs:
            path=root/model/"coordination"/f"pair-{pair['shard_index']:03d}"/"RESULT.json"
            state=_stage_state(root,"coordination",model,pair["shard_index"])
            if not unavailable and not path.exists() and (campaign.get('compact_extraction') or state is None):
                missing.append(pair)
        if campaign.get('compact_extraction'):
            from .analysis import pair_storage_bound
            measured = pair_storage_bound(campaign, model)
            # Pricing is an expressly authorized bounded exception. Do not
            # charge an unmeasured full bank to that exception. Its bulk work
            # remains ineligible until the retained pilot prices it.
            total += len(missing)*measured if measured is not None else min(1,len(missing))*((14*8+5*6*14*4)*E*E+4*1024**2)
        else:
            total += len(missing)*((14*8+5*6*14*4)*E*E+4*1024**2)
        if not (root/model/"SUPPLEMENTARY.json").exists() and _stage_state(root,"supplementary",model) is None:
            total+=64*1024**2
        if not (root/model/"TASKS.json").exists():
            # Reserve finite execution/analysis ledgers and shared mask objects.
            total+=2*1024**3
    if campaign.get('compact_extraction'):
        peaks=[]
        from .resume import source_file
        for model in MODELS:
            if (root/model/'CACHE_RETIRED.json').exists():continue
            path=source_file(campaign,Path(model)/'TOKEN_OWNERSHIP.json')
            if path:
                manifest=load(path);n=sum(r['counts'][-1] for r in manifest['records'])
                contract=campaign['models'][model]
                peaks.append(n*(2*len(contract['layers'])*contract['native_k']+24)+
                    len(manifest['records'])*9*len(contract['layers'])*contract['experts']*4)
        total+=max(peaks,default=0)
    if not (root/"REPORT_COMPLETE.json").exists():total+=2*1024**3
    return total


def _stage_state(root,stage,model="all",shard=0,threshold=0):
    directory=root/"stages"/stage_key(stage,model,shard,threshold)
    return load(directory/"RESULT.json") if (directory/"RESULT.json").exists() else None


def _success(root,stage,model="all",shard=0):
    return (_stage_state(root,stage,model,shard) or {}).get("status") in ("complete","reused")


def _failure(root,stage,model="all",shard=0):
    state=_stage_state(root,stage,model,shard)
    return state is not None and state["status"] not in ("complete","reused")


def _mark(root,stage,model,status,reason,shard=0,scope="model"):
    return publish(root/"stages"/stage_key(stage,model,shard)/"RESULT.json",{
        "stage":stage,"model":model,"shard_index":shard,"threshold_index":0,
        "status":status,"reason":reason,"failure_scope":scope,"automatic_terminal_status":True})


def terminal(campaign,status_,reason):
    root=Path(campaign["root"])
    if not (root/"TERMINAL.json").exists():
        publish(root/"TERMINAL.json",{"status":status_,"reason":reason,"epoch":time.time(),
            "report_complete_is_not_science_complete":True})


def reconcile(campaign,scheduler,observed):
    root=Path(campaign["root"])
    for j in observed:
        c,a=j["card"],j["accounting"]
        if c["stage"]=="controller" or not a["terminal"]:
            continue
        directory=root/"stages"/stage_key(c["stage"],c["model"],c["shard_index"],c["threshold_index"])
        if (directory/"RESULT.json").exists():
            continue
        safe_cpu_recovery=(c['attempt']==0 and a['state'] in ('NODE_FAIL','PREEMPTED','BOOT_FAIL') and
            c['stage'] in ('derive','mapping','coordination','supplementary','analysis_finalize'))
        if never_started(a,(directory/"START.json").exists(),c["attempt"]) or safe_cpu_recovery:
            replacement=card(campaign,c["stage"],c["model"],c["shard_index"],c["threshold_index"],
                attempt=1,minutes=c["minutes"])
            if not (directory/"INFRASTRUCTURE_REPLACEMENT.json").exists():
                publish(directory/"INFRASTRUCTURE_REPLACEMENT.json",{"original":j,"replacement":replacement,
                    "proof":"restartable saved-data CPU shard after verified infrastructure failure" if safe_cpu_recovery else "zero-runtime accounting, unset Start, no Python or scientific-unit start receipt"})
            submit_card(campaign,replacement,scheduler)
        else:
            _mark(root,c["stage"],c["model"],"execution_failed",
                {"accounting":a,"scientific_redraw":False,"committed_units_retained":True},c["shard_index"])


def _measurement_forecast(campaign):
    root=Path(campaign["root"])
    hours=0.;bytes_=0;missing=[]
    for model in MODELS:
        if _failure(root,"derive",model) or _failure(root,"native",model) or _failure(root,"measure",model,0):
            continue
        tasks=load(root/model/"MEASURE_TASKS.json")["records"]
        units=Units(root/model/"measurements")
        pilot=units.state(tasks[0]["uid"])
        if (pilot or {}).get("status")!="complete":
            missing.append(model)
            continue
        # The largest original sequence prices prefill and the streamed operator;
        # charge its observed duration to every missing assignment conservatively.
        each=max(1.,pilot["seconds"])
        pending=[r["task_index"] for r in tasks if units.state(r["uid"]) is None]
        per_shard=max(1,int((22*3600-pilot["load_seconds"])/each))
        shards=[pending[i:i+per_shard] for i in range(0,len(pending),per_shard)]
        publish(root/model/"MEASURE_SHARDS.json",{"shards":shards,"price_uid":pilot["uid"],
            "seconds_per_assignment_upper":each,"load_seconds":pilot["load_seconds"],"all_original_sources":True})
        hours+=(len(pending)*each+len(shards)*pilot["load_seconds"])*16/3600
        bytes_+=len(pending)*pilot["artifact"]["bytes"]
    return hours,bytes_,missing


def _production_forecast(campaign,freeze_shards=True):
    root=Path(campaign["root"])
    residual=0.;storage=0;missing=[]
    for model in MODELS:
        if not _success(root,"price",model):
            upstream_failed=any(_failure(root,s,model) for s in ("derive","native","freeze","qualify","price")) or _failure(root,"measure",model,0)
            if not upstream_failed:
                missing.append(model)
            continue
        table=load(root/model/"TASKS.json")
        prices={p["role"]:p for p in load(root/model/"PRICES.json")["records"]}
        units=Units(root/model/"executions")
        scores=Units(root/model/"scores")
        # Each forecast reads execution receipts once and retains only terminal
        # statuses, rather than keeping token captures or re-reading them for
        # every role and annotator bound.
        execution_states={}
        for r in table["records"]:
            state=units.state(r["uid"])
            execution_states[r["uid"]]=state.get("status") if state else None
        failed_shapes=set(load(root/model/"QUALIFICATION.json").get("failed_shape_keys",[]))
        residual+=sum(scores.state(r["uid"]) is None for r in table["records"])*30*4/3600
        if not _stage_state(root,"contexts",model):residual+=12.
        grouped=defaultdict(list)
        for row in table["records"]:
            if execution_states[row["uid"]] is not None:
                continue
            from .workers import mask_shape
            shape=mask_shape(campaign,model,row["mask"]) if row["mask"] is not None else None
            if shape in failed_shapes:
                continue
            role="memory:"+shape if shape else "causal" if row["action"] is not None else "reused_native"
            grouped[role].append(row)
        shards=[]
        for role,rows in sorted(grouped.items()):
            if role=="reused_native":
                # Native branches share saved outcome and capture records.
                each,load_seconds,outbytes=2.,0.,200000
            elif role not in prices:
                missing.append(f"{model}:{role}");continue
            else:
                p=prices[role]
                rate=p["seconds"]/max(1,p["generated_tokens"])
                # Native cumulative allowances bound every remaining task. This
                # deliberately prices long continuations rather than assuming an
                # intervention cannot extend them.
                each=max(p["seconds"],max(r["cap"] for r in rows)*rate)
                load_seconds=p["load_seconds"]
                outbytes=p["output_bytes"]/max(1,p["generated_tokens"])*max(r["cap"] for r in rows)
            batches=defaultdict(list)
            for r in rows:
                batches[digest(r["mask"])].append(r["task_index"])
            size=max(1,int((22*3600-load_seconds)/max(1,each)))
            for batch in batches.values():
                shards.extend(batch[i:i+size] for i in range(0,len(batch),size))
            residual+=(len(rows)*each+sum((len(b)+size-1)//size for b in batches.values())*load_seconds)*16/3600
            storage+=len(rows)*outbytes
        if freeze_shards:
            publish(root/model/"EXECUTION_SHARDS.json",{"shards":shards,"prices_sha256":load(root/model/"PRICES.json")["sha256"],
                "complete_residual_forecast":True,"native_cumulative_allowances_used":True})
        # Bound all sentence contexts and the original annotator workload before
        # admitting production. Missing reader prices reject admission; no guessed
        # annotator seconds silently permit target generations.
        reader=root/"ANNOTATOR_PRICE.json"
        if not reader.exists() or load(reader).get("status")!="complete":
            missing.append("original-annotator")
        else:
            price=load(reader)
            maximum_sentences=sum(min(r["cap"],r["window"])+1 for r in table["records"] if r["source"]["prefix_tokens"] and execution_states[r["uid"]] is None)
            if not (root/model/"CONTEXTS.json").exists():
                maximum_sentences+=sum(min(r["cap"],r["window"])+1 for r in table["records"] if r["source"]["prefix_tokens"] and
                    execution_states[r["uid"]]=="complete")
            annotation_tasks=root/model/"ANNOTATION_TASKS.json"
            if annotation_tasks.exists():
                annotations=Units(root/model/"annotations")
                maximum_sentences+=sum(annotations.state(r["uid"]) is None for r in load(annotation_tasks)["records"])
            # One sentence can be one token; measured original maximum-input
            # reader batch bounds every fixed draw without shortening its cap.
            per_job=max(1,int((22*3600-price.get("load_seconds",0))/price["seconds_per_context_upper"]))
            jobs_needed=(maximum_sentences+per_job-1)//per_job
            residual+=(maximum_sentences*price["seconds_per_context_upper"]+jobs_needed*price.get("load_seconds",0))*16/3600
            storage+=maximum_sentences*price["bytes_per_context_upper"]
    return residual,int(storage),sorted(set(missing))


def ready(campaign,scheduler,observed):
    root=Path(campaign["root"])
    work=[]
    def want(stage,model="all",shard=0,minutes=None):
        key=stage_key(stage,model,shard)
        if _stage_state(root,stage,model,shard) is not None:
            return
        if any(stage_key(j["card"]["stage"],j["card"]["model"],j["card"]["shard_index"],j["card"]["threshold_index"])==key for j in observed):
            return
        work.append(card(campaign,stage,model,shard,minutes=minutes))
    def saved_work():
        from .observations import pair_tasks
        pilots=[];residual=0.
        for m in MODELS:
            if not _success(root,"derive",m):continue
            if campaign.get("resume_from"):
                want("mapping",m,minutes=180)
                if not _success(root,"mapping",m):continue
            want("supplementary",m,minutes=180)
            pairs=pair_tasks(campaign,m)
            if not pairs:continue
            pilot=_stage_state(root,"coordination",m,pairs[0]["shard_index"])
            pilots.append((m,pairs,pilot))
            if pilot is None:want("coordination",m,pairs[0]["shard_index"],minutes=180)
            elif pilot["status"] in ("complete","reused"):
                residual+=sum(_stage_state(root,"coordination",m,p["shard_index"]) is None for p in pairs)*max(1,pilot["seconds"])*4/3600
        if any(p is None for m,pairs,p in pilots):return
        if not (root/"SAVED_COORDINATION_ADMISSION.json").exists():
            check=_check(campaign,scheduler,observed,hours=0,bytes_=0)
            publish(root/"SAVED_COORDINATION_ADMISSION.json",{"retained_first_original_pair_per_model":True,**check})
        admitted=load(root/"SAVED_COORDINATION_ADMISSION.json")["admitted"]
        for m,pairs,pilot in pilots:
            supported=pilot["status"] in ("complete","reused")
            for pair in pairs:
                if _stage_state(root,"coordination",m,pair["shard_index"]) is not None:continue
                if admitted and supported:
                    import math
                    minutes=min(240,max(30,math.ceil(pilot["seconds"]*1.25/60)+5))
                    want("coordination",m,pair["shard_index"],minutes=minutes)
                else:
                    _mark(root,"coordination",m,"resource_limited" if not admitted else "upstream_unavailable",
                        "complete saved-data workload failed admission" if not admitted else "original pair operator price unavailable",pair["shard_index"])
    if (root/"REPORT_COMPLETE.json").exists():
        return []
    shared_failures=[load(p) for p in (root/'stages').glob('*/RESULT.json') if
        load(p).get('failure_scope')=='shared' and load(p).get('status') not in ('complete','reused')]
    if shared_failures and not (root/'SHARED_IMPLEMENTATION_FAILURE.json').exists():
        publish(root/'SHARED_IMPLEMENTATION_FAILURE.json',{'failures':shared_failures,'dependent_GPU_work_stops':True})
    if (root/"SHARED_IMPLEMENTATION_FAILURE.json").exists():
        terminal(campaign,"upstream_unavailable",{"shared_implementation_failure":str(root/"SHARED_IMPLEMENTATION_FAILURE.json")})
    if (root/"TERMINAL.json").exists():
        if _success(root,"resolve") and not campaign.get("compact_extraction") and not (root/"SHARED_IMPLEMENTATION_FAILURE.json").exists():
            saved_work()
        if work:
            return work
        if not any(not j["accounting"]["terminal"] and j["card"]["stage"] not in ("controller","snapshot") for j in observed):
            want("closeout",minutes=180)
        return work
    if not _success(root,"resolve"):
        if _failure(root,"resolve"):
            terminal(campaign,"upstream_unavailable","source resolution failed; no replacements scheduled")
            want("closeout",minutes=180)
        else:
            want("resolve",minutes=180)
        return work
    if campaign.get('compact_extraction'):
        from .analysis import ready as analysis_ready
        if not analysis_ready(campaign,scheduler,observed,want):return work
    else:
        for model in MODELS:
            want('derive',model,minutes=180)
            if campaign.get('resume_from'):want('mapping',model,minutes=180)
        if not all(_stage_state(root,'derive',m) for m in MODELS):return work
        saved_work()
    shared_failures=[_stage_state(root,'derive',m) for m in MODELS if _failure(root,'derive',m) and
                    (_stage_state(root,'derive',m) or {}).get('failure_scope','shared')=='shared']
    if shared_failures:
        terminal(campaign,'upstream_unavailable',{'shared_derivation_failures':shared_failures})
        return []
    for model in MODELS:
        if _success(root,"derive",model):
            want("native",model,minutes=180)
    if campaign.get("resume_from") and not _stage_state(root,"measure","qwen330b",0):
        if _failure(root,"derive","qwen330b") or _failure(root,"native","qwen330b"):
            publish(root/"stages"/stage_key("measure","qwen330b")/"RESULT.json",{
                "stage":"measure","model":"qwen330b","status":"upstream_unavailable",
                "failure_scope":"model","reason":"original Qwen3-30B inputs unavailable; independent models remain eligible"})
        else:
            if _success(root,"native","qwen330b"):want("measure","qwen330b",0,minutes=120)
            return work
    if campaign.get("resume_from") and _failure(root,"measure","qwen330b",0):
        state=_stage_state(root,"measure","qwen330b",0)
        if state.get("failure_scope","shared")!="model":
            terminal(campaign,"upstream_unavailable",{"shared_implementation_gate_failed":state})
            return work
    for model in MODELS:
        if _success(root,"native",model):
            want("measure",model,0,minutes=120)
    if not all(_failure(root,"derive",m) or _failure(root,"native",m) or _stage_state(root,"measure",m,0) for m in MODELS):
        return work
    if not (root/"PREPARATION_ADMISSION.json").exists():
        hours,bytes_,missing=_measurement_forecast(campaign)
        # Qualification and first role prices are bounded measurement exceptions,
        # not an unpriced final-production admission.
        check=_check(campaign,scheduler,observed,hours=hours,bytes_=bytes_,prices=not missing)
        publish(root/"PREPARATION_ADMISSION.json",{"scope":"all residual saved native measurements",**check})
        if not check["admitted"]:
            terminal(campaign,"resource_limited",check)
            return work
    for model in MODELS:
        if _failure(root,"derive",model) or _failure(root,"native",model) or _failure(root,"measure",model,0):
            continue
        shards=load(root/model/"MEASURE_SHARDS.json")["shards"]
        for i in range(1,len(shards)+1):
            want("measure",model,i,minutes=1440)
        if not all(_stage_state(root,"measure",model,i) for i in range(len(shards)+1)):
            continue
        if any(_failure(root,"measure",model,i) for i in range(len(shards)+1)):
            _mark(root,"freeze",model,"upstream_unavailable","missing native measurement remains terminal")
            continue
        for i in range(1,6):
            want("freeze",model,i,minutes=180)
        if not all(_stage_state(root,"freeze",model,i) for i in range(1,6)):
            continue
        if any(_failure(root,"freeze",model,i) for i in range(1,6)):
            _mark(root,"freeze",model,"upstream_unavailable","discovery calibration failed")
            continue
        want("freeze",model,0)
        if not _success(root,"freeze",model):
            continue
        from .workers import qualification_rows
        qualification=qualification_rows(campaign,model)["records"]
        if campaign.get("resume_from") and not (root/model/"QUALIFICATION_ADMISSION.json").exists():
            bound=sum(card(campaign,"qualify",model,r["shard_index"],minutes=120)["reservation_hours"]
                for r in qualification if _stage_state(root,"qualify",model,r["shard_index"]) is None)
            check=_check(campaign,scheduler,observed,hours=bound,bytes_=0)
            publish(root/model/"QUALIFICATION_ADMISSION.json",{"scope":"complete distinct native/compact shape qualification",**check})
            if not check["admitted"]:
                _mark(root,"qualify",model,"resource_limited",check)
                continue
        for r in qualification:
            want("qualify",model,r["shard_index"],minutes=120)
        if not all(_stage_state(root,"qualify",model,r["shard_index"]) for r in qualification):
            continue
        want("qualify",model,0)
        if _success(root,"qualify",model):
            from .workers import price_rows
            prices=price_rows(campaign,model)["records"]
            if campaign.get("resume_from") and not (root/model/"ROLE_MEASUREMENT_ADMISSION.json").exists():
                bound=sum(card(campaign,"price",model,r["shard_index"],minutes=1440)["reservation_hours"]
                    for r in prices if _stage_state(root,"price",model,r["shard_index"]) is None)
                check=_check(campaign,scheduler,observed,hours=bound,bytes_=0)
                publish(root/model/"ROLE_MEASUREMENT_ADMISSION.json",{"scope":"all required retained original target-role prices",**check})
                if not check["admitted"]:
                    _mark(root,"price",model,"resource_limited",check)
                    continue
            for r in prices:
                want("price",model,r["shard_index"],minutes=1440)
            if all(_stage_state(root,"price",model,r["shard_index"]) for r in prices):
                want("price",model,0)
    finished=lambda m: _failure(root,"derive",m) or _failure(root,"native",m) or _failure(root,"measure",m,0) or _failure(root,"freeze",m) or _failure(root,"qualify",m) or _stage_state(root,"price",m)
    if not all(finished(m) for m in MODELS):
        return work
    if campaign.get("resume_from") and any(not _stage_state(root,"mapping",m) for m in MODELS):
        return work
    if not any(_success(root,"price",m) for m in MODELS):
        terminal(campaign,"upstream_unavailable","no model supplied a qualified retained production role price")
        return work
    # Price the first genuinely missing original annotator context retained from
    # the original target-role measurement, then include it in residual tasks.
    if not (root/"ANNOTATOR_PRICE.json").exists():
        want("reader_price",minutes=1440)
        return work
    if not (root/"ADMISSION.json").exists():
        from .observations import pair_tasks
        if any(_success(root,"derive",m) and pair_tasks(campaign,m) and
            _stage_state(root,"coordination",m,1) is None for m in MODELS):return work
        hours,bytes_,missing=_production_forecast(campaign)
        # Bound the remaining registered all-ancestor owners through the finite handoff limit.
        latest=max((load(p)["generation"] for p in (root/"handoffs").glob("*.json")),default=0)
        hours+=_operational_residual(campaign,observed)
        check=_check(campaign,scheduler,observed,hours=hours,bytes_=bytes_,prices=not missing)
        publish(root/"ADMISSION.json",{"required_missing_prices":missing,**check})
        from .decisions import preproduction
        preproduction(campaign,load(root/'ADMISSION.json'),hours=hours,bytes_=bytes_,missing=missing)
        if not check["admitted"]:
            terminal(campaign,"resource_limited",check)
            return work
    for model in MODELS:
        if not _success(root,"price",model):
            continue
        shards=load(root/model/"EXECUTION_SHARDS.json")["shards"]
        for i in range(len(shards)):
            want("execute",model,i,minutes=1440)
        if not all(_stage_state(root,"execute",model,i) for i in range(len(shards))):
            continue
        fragments=sorted((root/model/"execution_fragments").glob("*.json"))
        for p in fragments:
            fragment=load(p);want("execute",model,fragment["shard_index"],minutes=1440)
        if not all(_stage_state(root,"execute",model,load(p)["shard_index"]) for p in fragments):continue
        table=load(root/model/"TASKS.json")["records"]
        units=Units(root/model/"executions")
        uncovered=[r for r in table if units.state(r["uid"]) is None]
        if uncovered:
            # Later fragments cover only untouched frozen assignments, never
            # started failures or an alternative source/dose. The finite graph
            # can expand by at most the number of original execution rows.
            grouped=defaultdict(list)
            for row in uncovered:grouped[digest(row["mask"])].append(row["task_index"])
            size=max((len(s) for s in shards),default=1)
            index=len(shards)+len(fragments)
            for group in grouped.values():
                for start in range(0,len(group),size):
                    publish(root/model/"execution_fragments"/f"{index:06d}.json",{
                        "shard_index":index,"indices":group[start:start+size],"only_proven_unstarted_frozen_units":True})
                    want("execute",model,index,minutes=1440);index+=1
            continue
        want("contexts",model,minutes=180)
        scoreplan=root/model/"SCORE_SHARDS.json"
        if not scoreplan.exists():
            scores=Units(root/model/"scores")
            table=load(root/model/"TASKS.json")["records"]
            pending=[r["task_index"] for r in table if scores.state(r["uid"]) is None]
            publish(scoreplan,{"shards":[pending[i:i+256] for i in range(0,len(pending),256)],
                "maximum_assignment_seconds":30,"original_strict_contract":True})
        scoring=load(scoreplan)["shards"]
        for i in range(len(scoring)):want("score",model,i+1,minutes=180)
        if all(_stage_state(root,"score",model,i+1) for i in range(len(scoring))):want("score",model,0,minutes=30)
        if _success(root,"contexts",model):
            table=load(root/model/"ANNOTATION_TASKS.json")["records"]
            count=max(1,int(22*3600/load(root/"ANNOTATOR_PRICE.json")["seconds_per_context_upper"]))
            slices=[list(range(i,min(i+count,len(table)))) for i in range(0,len(table),count)]
            publish(root/model/"ANNOTATION_SHARDS.json",{"shards":slices})
            for i in range(len(slices)):
                want("annotate",model,i,minutes=1440)
    all_finished=True
    for model in MODELS:
        if not _success(root,"price",model):
            continue
        if not _stage_state(root,"score",model) or not _stage_state(root,"contexts",model):
            all_finished=False;continue
        if _success(root,"contexts",model):
            count=len(load(root/model/"ANNOTATION_SHARDS.json")["shards"])
            all_finished &= all(_stage_state(root,"annotate",model,i) for i in range(count))
    if all_finished:
        terminal(campaign,"production_terminal","all admitted independent assignments have terminal evidence")
    return work


def retire_guards(campaign,scheduler,*,before_generation=None,all_=False):
    root=Path(campaign["root"])
    for p in sorted((root/"handoffs").glob("*.json")):
        handoff=load(p)
        if not all_ and handoff["generation"]>=before_generation:
            continue
        for guard in handoff["guards"]:
            proofpath=root/"guard_retirements"/(guard["job_id"]+".json")
            if proofpath.exists():
                continue
            observation=scheduler.observe(guard["job_id"])
            if observation["terminal"] or observation["start"] not in ("None","Unknown","N/A","",None):
                continue
            require(observation["elapsed_seconds"]==0,"running failure guard cannot be superseded")
            scheduler.command(["scancel",guard["job_id"]])
            after=scheduler.observe(guard["job_id"])
            require(after["state"]=="CANCELLED" and after["elapsed_seconds"]==0,"guard retirement not verified")
            publish(proofpath,{"before":observation,"after":after,"protected_successor_or_terminal_report":True})


def launch(campaign,scheduler=None):
    scheduler=scheduler or Slurm()
    check_files(campaign["code_files"])
    root=Path(campaign["root"])
    with locked(root,"controller"):
        if (root/"LAUNCH.json").exists():
            return load(root/"LAUNCH.json")
        snapshot=scheduler.snapshot()
        require(len([j for j in snapshot["jobs"] if j["partition"]=="lrd_all_serial"])<=5,"insufficient serial submission capacity for protected launch")
        check=_check(campaign,scheduler,[],hours=7*12.+5*32.+24.,bytes_=512*1024**2)
        require(check["admitted"],"bounded source resolution and retained measurements cannot be admitted")
        if campaign.get("resume_from"):
            require((root/"READINESS.json").exists(),"repair CPU smoke gates are not sealed")
            check_files(load(root/"READINESS.json")["files"])
            old=load(Path(campaign["resume_from"])/"CAMPAIGN.json")
            inherited=jobs(old,scheduler)
            running_owners=[j for j in inherited if not j["accounting"]["terminal"] and
                j["card"]["stage"] in ("controller","snapshot") and j["accounting"]["state"]!="PENDING"]
            require(not running_owners,"running old dispatcher must finish before ownership transfer")
            wait=[j["accepted"]["job_id"] for j in inherited if not j["accounting"]["terminal"] and
                j["card"]["stage"] not in ("controller","snapshot")]
            initial=card(campaign,"resolve",minutes=180,hold=True,
                dependency="afterany:"+":".join(wait) if wait else None)
            accepted=submit_card(campaign,initial,scheduler)
            milestone_card=card(campaign,"snapshot",minutes=180,hold=True,begin="2026-10-15T21:59:59")
            milestone=submit_card(campaign,milestone_card,scheduler)
            handoff=protected_controller(campaign,scheduler,generation=0,wait_ids=[accepted["job_id"]],release=False)
            from .resume import transfer
            transferred=transfer(campaign,scheduler,handoff)
            protected_controller(campaign,scheduler,generation=0,release=True)
            for value,receipt in ((initial,accepted),(milestone_card,milestone)):
                ledger=root/"submissions"/value["sha256"]/"RELEASED.json"
                if not ledger.exists():
                    scheduler.release(receipt["job_id"])
                    proof=scheduler.verify(receipt["job_id"],value)
                    require(proof["fields"].get("Reason")!="JobHeldUser","new owner remains held")
                    publish(ledger,{"job_id":receipt["job_id"],"proof":proof})
        else:
            initial=card(campaign,"resolve",minutes=180)
            accepted=submit_card(campaign,initial,scheduler)
            milestone=submit_card(campaign,card(campaign,"snapshot",minutes=180,begin="2026-10-15T21:59:59"),scheduler)
            handoff=protected_controller(campaign,scheduler,generation=0,wait_ids=[accepted["job_id"]])
        return publish(root/"LAUNCH.json",{"campaign_id":campaign["campaign_id"],"initial_jobs":[accepted,milestone,*[handoff["primary"],*handoff["guards"]]],
            "output_location":campaign["root"],"snapshot":snapshot,"bounded_initial_admission":check,
            "complete_finite_graph_registered":True})


def capacity_events(snapshot, observed, work, active, accepted, current):
    """Wait on other jobs only when their partition actually defers dispatch."""
    if active or accepted or not work:
        return []
    partitions = {c["partition"] for c in work}
    operational = {j["accepted"]["job_id"] for j in observed
                   if j["card"]["stage"] in ("controller", "snapshot")}
    return sorted({j["job_id"] for j in snapshot["jobs"]
                   if j["job_id"] != current and j["job_id"] not in operational
                   and j["partition"] in partitions})


def control(campaign,value,scheduler=None):
    scheduler=scheduler or Slurm()
    root=Path(campaign["root"])
    check_files(campaign["code_files"])
    with locked(root,"controller"):
        observed=jobs(campaign,scheduler)
        reconcile(campaign,scheduler,observed)
        if (root/"REPORT_COMPLETE.json").exists():
            refreshed=jobs(campaign,scheduler)
            milestone=[j["accepted"]["job_id"] for j in refreshed if j["card"]["stage"]=="snapshot" and not j["accounting"]["terminal"]]
            if milestone:
                current=os.environ.get("SLURM_JOB_ID")
                generation=value["generation"]+1
                handoff=protected_controller(campaign,scheduler,generation=generation,wait_ids=milestone+([current] if current else []))
                retire_guards(campaign,scheduler,before_generation=generation)
                return {"status":"complete","REPORT_COMPLETE":True,"milestone_owner":handoff}
            retire_guards(campaign,scheduler,all_=True)
            if not (root/"FINAL_ACCOUNTING.json").exists():
                publish(root/"FINAL_ACCOUNTING.json",{"observed":status(campaign,scheduler),
                    "scientific_and_milestone_jobs_terminal":True,"current_final_controller_charge_pending":True})
            return {"status":"complete","REPORT_COMPLETE":True}
        work=ready(campaign,scheduler,observed)
        if (root/"ADMISSION.json").exists() and load(root/"ADMISSION.json")["admitted"] and not (root/"TERMINAL.json").exists():
            hours,bytes_,missing=_production_forecast(campaign,freeze_shards=False)
            recheck=_check(campaign,scheduler,observed,hours=hours,bytes_=bytes_,prices=not missing)
            if not recheck["admitted"]:
                terminal(campaign,"resource_limited",recheck)
                work=ready(campaign,scheduler,observed)
        priorities={"mapping":10,"derive":0,"native":1,"measure":2,"freeze":3,"qualify":4,"price":5,
            "reader_price":6,"execute":7,"contexts":8,"score":8,"annotate":9,"supplementary":10,"coordination":11}
        work.sort(key=lambda c:(priorities.get(c["stage"],0),c["model"],c["shard_index"]))
        snapshot=scheduler.snapshot()
        serial=len([j for j in snapshot["jobs"] if j["partition"]=="lrd_all_serial"])
        gpu=len([j for j in snapshot["jobs"] if j["partition"]=="boost_usr_prod"])
        accepted=[];wave_cards=[]
        for c in work:
            if c["partition"]=="lrd_all_serial":
                if not all(serial_capacity(snapshot,[*wave_cards,c]).values()):
                    continue
                serial+=1
            elif gpu>=24:
                continue
            else:
                gpu+=1
            # Live accounting and exact quota are rechecked before each wave.
            wave=_check(campaign,scheduler,observed,hours=c["reservation_hours"]+sum(x["reservation_hours"] for x in wave_cards),bytes_=0)
            if not wave["admitted"] and c["stage"]!="closeout":
                if wave_cards:
                    break  # A bounded smaller wave owns capacity deferral.
                terminal(campaign,"resource_limited",wave)
                if c["stage"] in ("coordination","supplementary"):
                    for unavailable in work:
                        if unavailable["stage"] in ("coordination","supplementary"):
                            _mark(root,unavailable["stage"],unavailable["model"],"resource_limited",wave,unavailable["shard_index"])
                break
            wave_cards.append(c)
        grouped=defaultdict(list)
        for c in wave_cards:
            grouped[c["stage"],c["model"],c["minutes"]].append(c)
        for group in grouped.values():
            parent=array_card(group) if len(group)>1 else group[0]
            accepted.append(submit_card(campaign,parent,scheduler))
        refreshed=jobs(campaign,scheduler)
        active=[j["accepted"]["job_id"] for j in refreshed if not j["accounting"]["terminal"] and j["card"]["stage"] not in ("controller","snapshot")]
        # An afterany successor owns completion; bounded begin deferral owns
        # scheduler saturation without a login daemon or repeated watchdog.
        current=os.environ.get("SLURM_JOB_ID")
        wait_ids=active+([current] if current else [])
        events=capacity_events(snapshot,refreshed,work,active,accepted,current)
        wait_ids=sorted(set(wait_ids+events))
        begin=None
        if not active and not accepted and not events and work and value['shard_index']!=2:
            terminal(campaign,'resource_limited','capacity deferral has no owned scheduler event')
        if not active and not accepted and not events and not work and not (root/'TERMINAL.json').exists():
            terminal(campaign,'upstream_unavailable','finite graph has no eligible work or owned completion event')
        generation=value["generation"]+1
        if generation>handoff_bound(campaign):
            terminal(campaign,"resource_limited","finite controller handoff capacity exhausted")
            from .export import closeout
            closeout(campaign)
            return {"status":"complete","REPORT_COMPLETE":True}
        handoff=protected_controller(campaign,scheduler,generation=generation,wait_ids=wait_ids,begin=begin)
        retire_guards(campaign,scheduler,before_generation=generation)
        return {"status":"complete","submitted":accepted,"successor":handoff,
            "scheduler_capacity_deferred":len(work)-len(accepted)}
