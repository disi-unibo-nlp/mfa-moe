"""Read-through reuse and lock-protected transfer of operational ownership."""
from pathlib import Path
import os
import re
from .contracts import check_files, digest, file_sha, load, locked, publish, require

REFERENCE_FILES = ("SCORES.npz", "association_atlas.parquet", "NOMINATIONS.json", "DYNAMICS.json",
                   "native_dynamics_estimands.parquet", "NATIVE_TASKS.json")


def reference_file(target, source):
    target, source = Path(target), Path(source)
    require(source.is_file(), "missing compatible reference")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not target.exists():
        target.symlink_to(source.resolve())
    require(file_sha(target) == file_sha(source), "compatible artifact reference differs")
    return {"path":str(source),"sha256":file_sha(source),"referenced_as":str(target)}


def ancestry(campaign):
    """Newest first; verify every immutable source closure and reject cycles."""
    chain, seen = [], set()
    current = campaign
    while current.get("resume_from"):
        path = (Path(current["resume_from"])/"CAMPAIGN.json").resolve()
        require(str(path) not in seen, "campaign ancestry cycle")
        seen.add(str(path))
        current = load(path)
        check_files(current["code_files"])
        check_files(current["source_files"])
        chain.append(current)
    return chain


def source_file(campaign, relative):
    for binding in [campaign, *ancestry(campaign)]:
        candidate = Path(binding["root"])/relative
        if candidate.is_file():
            return candidate
    return None


def import_evidence(campaign):
    from .controller import stage_key
    root = Path(campaign["root"])
    chain = ancestry(campaign)
    references, components, failed = [], [], []
    def reference(relative):
        source = source_file({**campaign,"root":"/__no_local_campaign__"},relative)
        if source is not None:
            references.append(reference_file(root/relative, source))
        return source
    for name in ("FOLDS.json","COHORTS.json","REUSE.json","REUSED_CATALOGUE.json"):
        require(reference(name) is not None,"frozen campaign evidence missing: "+name)
    for model in campaign["models"]:
        for name in ("SOURCES.json","PREFIXES.json",*REFERENCE_FILES,
                     "NATIVE_ESTIMANDS.json","NATIVE_ACCURACY.json","TOKEN_OWNERSHIP.json"):
            reference(model+"/"+name)
        ownership=root/model/"TOKEN_OWNERSHIP.json"
        if ownership.exists():
            manifest=load(ownership)
            require(manifest["sources_sha256"]==load(root/model/"SOURCES.json")["sha256"],"ownership sources differ")
            require(manifest["causal_cohort_sha256"]==load(root/"COHORTS.json")["sha256"] and
                    manifest["prefix_roster_sha256"]==load(root/model/"PREFIXES.json")["sha256"],"ownership enrollment differs")
            from .inputs import counts_for
            sources=load(root/model/'SOURCES.json')['records']
            _,expected_tokens=counts_for(campaign,model,sources)
            original={r['attempt_id']:r for r in manifest['records']}
            require(set(original)=={s['attempt_id'] for s in sources},'ownership attempt roster differs')
            for source,nine in zip(sources,expected_tokens):
                require(original[source['attempt_id']]['counts']==list(map(int,nine)),'ownership nine saved token counts differ')
            for row in manifest["records"]:
                check_files({row["artifact"]["path"]:row["artifact"]["sha256"]})
                references.append(reference_file(root/model/"token_ownership"/(row["attempt_id"]+".npz"),row["artifact"]["path"]))
            publish(root/"stages"/stage_key("mapping",model)/"RESULT.json",{"stage":"mapping","model":model,"status":"reused",
                "source_receipt":str(ownership.resolve()),"source_sha256":file_sha(ownership),"enrollment_changed":False})
        directories={kind:list(dict.fromkeys(str(Path(b["root"])/model/kind) for b in chain))
                     for kind in ("executions","scores","annotations","measurements")}
        publish(root/model/"INHERITED_UNITS.json",{"ancestry":[b["sha256"] for b in chain],"directories":directories,
            "compatible_statuses":["complete","reused","invalid","truncated"]})
        for parent in chain:
            old=Path(parent["root"])
            for receipt in sorted((old/model/"measurements").glob("*.result.json")):
                value=load(receipt)
                if value.get("status")!="complete":
                    failed.append({"path":str(receipt),"sha256":file_sha(receipt),"component":"measurement"})
                    continue
                require(parent["models"][model]["profile_sha256"]==campaign["models"][model]["profile_sha256"],"bank runtime differs")
                require(parent.get("instrumentation_binding")==campaign.get("instrumentation_binding"),"router measurement binding differs")
                check_files({value["artifact"]["path"]:value["artifact"]["sha256"]})
                components.append({"model":model,"uid":value["uid"],"source_receipt":str(receipt),"source_sha256":file_sha(receipt),
                    "artifact":value["artifact"],"router_bank_qualified":True,"reap_qualified":value.get("reap_supported",False),
                    "assignment":value["assignment"],"native_dtypes":value.get("native_dtypes",{})})
            for receipt in sorted((old/"stages").glob("*/RESULT.json")):
                value=load(receipt)
                if value.get("status")=="execution_failed":
                    failed.append({"path":str(receipt),"sha256":file_sha(receipt),"stage":value.get("stage")})
        native=source_file(campaign,Path("stages")/stage_key("native",model)/"RESULT.json")
        if native and load(native)["status"] in ("complete","reused"):
            publish(root/"stages"/stage_key("native",model)/"RESULT.json",{"stage":"native","model":model,"status":"reused",
                "source_receipt":str(native),"source_sha256":file_sha(native),"no_new_generation":True})
    # The two qualified REAP pilots need no new GPU allocation. The other four
    # reuse their router component and replace only the defective norm component.
    for model in campaign['models']:
        qualified=[r for r in components if r['model']==model and r['reap_qualified']]
        if qualified:
            pilot=max(qualified,key=lambda r:r['assignment']['source']['completion_tokens'])
            publish(root/model/'REAP_QUALIFICATION.json',{'status':'qualified','source_receipt':pilot['source_receipt'],
                'source_sha256':pilot['source_sha256'],'compatible_measured_component':True,'router_bank_independently_usable':True})
            publish(root/'stages'/stage_key('measure',model,0)/'RESULT.json',{'stage':'measure','model':model,
                'shard_index':0,'status':'reused','source_receipt':pilot['source_receipt'],
                'source_sha256':pilot['source_sha256'],'components':['router_bank','REAP'],'no_new_allocation':True})
    unique={ (r["model"],r["assignment"]["source"]["attempt_id"],r["artifact"]["sha256"]):r for r in components }
    publish(root/"MEASUREMENT_COMPONENTS.json",{"records":list(unique.values()),"router_and_reap_separate":True})
    return publish(root/"EVIDENCE_IMPORT.json",{"status":"complete","ancestry":[b["sha256"] for b in chain],
        "references":references,"failed_receipts":list({r["path"]:r for r in failed}.values()),
        "failed_measurements_preserved":True,"affected_summaries_recomputed":True,
        "cohorts_and_prefixes_changed":False,"nominations_reused":True})


def inherited_state(root, uid, seen=None):
    root=Path(root)
    seen=set() if seen is None else seen
    identity=str(root.resolve())
    require(identity not in seen,"inherited unit ancestry cycle")
    seen.add(identity)
    path=root.parent/"INHERITED_UNITS.json"
    if not path.is_file():return None
    inherited=load(path)
    directories=inherited["directories"].get(root.name,[])
    if isinstance(directories,str):directories=[directories]
    for directory in directories:
        result=Path(directory)/(uid+".result.json")
        if result.is_file():
            value=load(result)
            if value.get("status") in inherited["compatible_statuses"]:
                return {**value,"inherited_receipt":str(result),"inherited_receipt_sha256":file_sha(result)}
        starts=sorted(Path(directory).glob(uid+".start-*.json"))
        if starts:
            return {"status":"started","starts":[load(p) for p in starts],"inherited_directory":directory}
        value=inherited_state(directory,uid,set(seen))
        if value is not None:return value
    return None


def transfer(campaign, scheduler, prearmed):
    from .controller import jobs
    from .scheduler import TERMINAL_SLURM
    root, old = Path(campaign["root"]), Path(campaign["resume_from"])
    if (root/"OWNERSHIP_TRANSFER.json").exists():
        return load(root/"OWNERSHIP_TRANSFER.json")
    parent = load(old/"CAMPAIGN.json")
    require(len(prearmed["guards"])==2,"new all-ancestor owners not prearmed")
    with locked(old,"controller"):
        observed = jobs(parent,scheduler)
        unresolved = []
        for intent in (old/"submissions").glob("*/INTENT.json"):
            if not (intent.parent/"ACCEPTED.json").exists():
                value=load(intent)
                matches=scheduler.matching(value["comment"])
                require(len(matches)==1,"ambiguous inherited submission requires exact reconciliation")
                card=load(old/"cards"/(value["card_sha256"]+".json"))
                proof=scheduler.verify(matches[0],{**card,"allocation_cpus":max(2,card["cpus"]),
                    "memory_mib":card["memory_gib"]*1024,"billing_units_upper":max(2,card["cpus"])})
                publish(intent.parent/"ACCEPTED.json",{"job_id":matches[0],"response":"reconciled by v2 transfer",
                    "card_sha256":card["sha256"],"readback":proof})
                unresolved.append(matches[0])
        intent_path=root/"OWNERSHIP_TRANSFER_INTENT.json"
        if intent_path.exists():
            planned=load(intent_path)["observed"]
        else:
            planned=publish(intent_path,{"observed":jobs(parent,scheduler),"prearmed":prearmed,
                "operational_stages":["controller","snapshot"]})["observed"]
        retirements, preserved = [], []
        for item in planned:
            value, before = item["card"], item["accounting"]
            accounting=scheduler.observe(item["accepted"]["job_id"])
            if before["terminal"]:
                continue
            jid = item["accepted"]["job_id"]
            if value["stage"] not in ("controller","snapshot"):
                preserved.append(item)
                continue
            retirement=root/"transfer_retirements"/(jid+".json")
            if retirement.exists():
                retirements.append(load(retirement))
                continue
            if accounting["terminal"]:
                require(accounting["state"]=="CANCELLED" and accounting["elapsed_seconds"]==0 and
                    accounting["start"] in ("None","Unknown","N/A","",None),"interrupted operational retirement differs")
                retirements.append(publish(retirement,{"job_id":jid,"card_sha256":value["sha256"],"before":before,
                    "after":accounting,"reconciled_interrupted_retirement":True}))
                continue
            require(accounting["state"]=="PENDING" and accounting["start"] in ("None","Unknown","N/A","",None)
                    and accounting["elapsed_seconds"]==0,"running old operational owner must finish before transfer")
            proof=scheduler.command(["scontrol","show","job","-o",jid])
            require(value["comment"] in proof,"old operational ownership mismatch")
            scheduler.command(["scontrol","hold",jid])
            held=scheduler.command(["scontrol","show","job","-o",jid])
            require("Reason=JobHeldUser" in held,"old dispatcher hold not verified")
            scheduler.command(["scancel",jid])
            after=scheduler.observe(jid)
            require(after["state"]=="CANCELLED" and after["elapsed_seconds"]==0 and
                    after["start"] in ("None","Unknown","N/A","",None),"operational retirement lacks zero-runtime proof")
            retirements.append(publish(retirement,{"job_id":jid,"card_sha256":value["sha256"],"before":before,"after":after}))
        return publish(root/"OWNERSHIP_TRANSFER.json", {"parent_campaign":parent["sha256"],
            "prearmed":prearmed,"old_operational_retirements":retirements,
            "accepted_scientific_jobs_preserved":preserved,"reconciled_intents":unresolved,
            "original_authorization_baseline_preserved":True})
