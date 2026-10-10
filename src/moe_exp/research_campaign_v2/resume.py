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


def import_evidence(campaign):
    from .controller import jobs, stage_key
    from .scheduler import Slurm
    root, old = Path(campaign["root"]), Path(campaign["resume_from"])
    parent = load(old/"CAMPAIGN.json")
    check_files(parent["code_files"])
    check_files(parent["source_files"])
    references = []
    for name in ("FOLDS.json","COHORTS.json","REUSE.json","REUSED_CATALOGUE.json"):
        references.append(reference_file(root/name, old/name))
    for model in campaign["models"]:
        for name in ("SOURCES.json","PREFIXES.json",*REFERENCE_FILES):
            if (old/model/name).is_file():
                references.append(reference_file(root/model/name,old/model/name))
        # Only matching original scientific identities read through. Never write
        # to an inherited directory, or import failed instrumentation units.
        publish(root/model/"INHERITED_UNITS.json", {"parent_campaign":parent["sha256"],
            "directories":{kind:str(old/model/kind) for kind in ("executions","scores","annotations")},
            "compatible_statuses":["complete","reused","invalid","truncated"]})
        native = old/"stages"/stage_key("native",model)/"RESULT.json"
        if native.exists() and load(native)["status"] in ("complete","reused"):
            publish(root/"stages"/stage_key("native",model)/"RESULT.json", {
                "stage":"native","model":model,"status":"reused","source_receipt":str(native),
                "source_sha256":file_sha(native),"no_new_generation":True})
    return publish(root/"EVIDENCE_IMPORT.json", {"status":"complete","parent_campaign":parent["sha256"],
        "references":references,"failed_measurements_preserved":True,"affected_summaries_recomputed":True,
        "cohorts_and_prefixes_changed":False,"nominations_reused":True})


def inherited_state(root, uid):
    path = Path(root).parent/"INHERITED_UNITS.json"
    if not path.is_file():
        return None
    inherited = load(path)
    directory = inherited["directories"].get(Path(root).name)
    if directory is None:
        return None
    result = Path(directory)/(uid+".result.json")
    if result.is_file():
        value = load(result)
        if value.get("status") in inherited["compatible_statuses"]:
            return {**value,"inherited_receipt":str(result),"inherited_receipt_sha256":file_sha(result)}
    # A started inherited original unit must never be redrawn after a lost reply.
    starts = sorted(Path(directory).glob(uid+".start-*.json"))
    if starts:
        return {"status":"started","starts":[load(p) for p in starts],"inherited_directory":directory}
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
