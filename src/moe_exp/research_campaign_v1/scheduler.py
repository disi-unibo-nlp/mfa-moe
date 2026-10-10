"""Finite submission journals, exact readback, admission and protected handoffs."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from .contracts import (CEILING, MARGIN, REPO, STORAGE_RESERVE, check_files, digest,
                        load, locked, publish, require, seal)

TERMINAL_SLURM = frozenset(("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"))


def clean_environment():
    return {**{k:v for k,v in os.environ.items() if not k.startswith(("SLURM_","SRUN_","SBATCH_"))},"TZ":"UTC"}


class Slurm:
    def command(self, args):
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=45,
                              env=clean_environment() if args[0] == "sbatch" else None).stdout.strip()

    def snapshot(self):
        require(getpass.getuser() == "lmolfett" and "leonardo" in socket.getfqdn(), "wrong cluster identity")
        balance = self.command(["saldo", "-b"])
        line = next((s for s in balance.splitlines() if s.strip().startswith("IscrC_MIOSR ")), None)
        require(line is not None, "project balance unavailable")
        fields = line.split()
        total, consumed = float(fields[3]), float(fields[4])
        require(total >= consumed and len(fields[2]) == 8, "invalid balance report")
        assoc = self.command(["sacctmgr", "-nP", "show", "assoc", "where", "user=lmolfett", "format=Account,Partition,QOS,DefaultQOS"])
        require(any(s.startswith("iscrc_miosr|") and "normal" in s.split("|")[2].split(",") for s in assoc.splitlines()), "normal QoS entitlement absent")
        gpu = self.command(["scontrol", "show", "partition", "boost_usr_prod"])
        serial = self.command(["scontrol", "show", "partition", "lrd_all_serial"])
        require("State=UP" in gpu and "State=UP" in serial, "required partition down")
        qos = self.command(["sacctmgr", "-nP", "show", "qos", "where", "name=lrd_all_serial,boost_usr_prod,normal", "format=Name,MaxJobsPU,MaxSubmitPU,MaxWall"])
        serial_limits=next((s.split("|") for s in qos.splitlines() if s.startswith("lrd_all_serial|")),None)
        require(serial_limits is not None and int(serial_limits[1])>=2 and int(serial_limits[2])>=10,
            "observed serial limits no longer support the sealed protected handoff shape")
        quota = self.command(["lfs", "quota", "-p", "20149688", "/leonardo_work"])
        qline = next((s for s in quota.splitlines() if s.strip().startswith("/leonardo_work ")), None)
        require(qline is not None, "exact project quota unavailable")
        q = qline.split()
        used, quota_kib = int(q[1].rstrip("*")), int(q[2])
        require(quota_kib > 0, "finite WORK project quota unavailable")
        jobs = self.queue()
        return {"epoch": time.time(), "balance_text": balance, "total_local_hours": total,
                "consumed_local_hours": consumed, "remaining_local_hours": total-consumed,
                "quota_text": quota, "storage_headroom_bytes": (quota_kib-used)*1024,
                "entitlement_text": assoc, "gpu_partition": gpu, "serial_partition": serial,
                "qos_text": qos, "jobs": jobs}

    def queue(self):
        raw = self.command(["squeue", "--array", "-u", "lmolfett", "-h", "-o", "%i|%P|%T|%k"])
        return [dict(zip(("job_id", "partition", "state", "comment"), s.split("|", 3))) for s in raw.splitlines() if s]

    def observe(self, job_id):
        require(re.fullmatch(r"\d+(?:_\d+)?", str(job_id)), "invalid Slurm job ID")
        return self.observe_many([str(job_id)])[str(job_id)]

    def observe_many(self,job_ids):
        require(job_ids and all(re.fullmatch(r"\d+(?:_\d+)?",str(j)) for j in job_ids),"invalid accounting batch")
        raw = self.command(["sacct", "-nP", "-X", "-j", ",".join(map(str,job_ids)),
                            "--format=JobID%64,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,Comment%200"])
        rows = [s.split("|") for s in raw.splitlines() if s]
        return {row[0]:self._accounting(row) for row in rows if row[0] in set(map(str,job_ids))}

    @staticmethod
    def _accounting(row):
        values = dict(zip(("job_id", "state", "exit_code", "elapsed_seconds", "cpus", "tres", "start", "comment"), row))
        values["state"] = values["state"].split()[0].rstrip("+")
        values["terminal"] = values["state"] in TERMINAL_SLURM
        values["elapsed_seconds"] = int(values["elapsed_seconds"] or 0)
        billing = re.search(r"(?:^|,)billing=(\d+(?:\.\d+)?)", values["tres"])
        values["billing_hours"] = float(billing[1]) * values["elapsed_seconds"] / 3600 if billing else None
        return values

    def matching(self, comment):
        ids = [r["job_id"] for r in self.queue() if r["comment"] == comment]
        raw = self.command(["sacct", "-nP", "-X", "-u", "lmolfett", "-S", "2026-10-08",
                            "--format=JobID%64,Comment%200"])
        ids += [s.split("|")[0] for s in raw.splitlines() if len(s.split("|")) >= 2 and s.split("|")[1] == comment]
        return sorted({jid.split("_",1)[0] for jid in ids})

    def verify(self, job_id, card):
        raw = self.command(["scontrol", "show", "job", "-o", str(job_id)])
        fields = dict(re.findall(r"(\w+)=([^\s]+)", raw))
        require(fields.get("UserId", "").startswith("lmolfett(") and fields.get("Account") == "iscrc_miosr" and
                fields.get("Partition") == card["partition"] and fields.get("Comment") == card["comment"], "accepted job readback mismatch")
        return {"raw": raw, "fields": fields}

    def submit(self, card, *, test_only=False):
        args = ["sbatch", "--test-only" if test_only else "--parsable", "--account=iscrc_miosr",
                "--qos=normal", "--partition=" + card["partition"], "--nodes=1", "--ntasks=1",
                "--cpus-per-task=" + str(card["cpus"]), "--mem=" + str(card["memory_gib"]) + "G",
                "--time=" + str(card["minutes"]), "--no-requeue", "--job-name=rc1-" + card["stage"],
                "--comment=" + card["comment"], "--chdir=" + str(REPO),
                "--output=" + card["root"] + "/logs/%x-%j.out", "--error=" + card["root"] + "/logs/%x-%j.err"]
        if card.get("gpus"):
            args += ["--gres=gpu:" + str(card["gpus"])]
        if card.get("hold"):
            args += ["--hold"]
        if card.get("dependency"):
            args += ["--dependency=" + card["dependency"]]
        if card.get("begin"):
            args += ["--begin=" + card["begin"]]
        if card.get("array_cards"):
            args += [f"--array=0-{len(card['array_cards'])-1}%{card['array_concurrency']}"]
        args += [str(REPO / "scripts/research_campaign_v1/job.sbatch"),
                 str(Path(card["root"]) / "cards" / (card["sha256"] + ".json"))]
        raw = self.command(args)
        if test_only:
            return {"test_only": True, "response": raw}
        job_id = raw.split(";", 1)[0]
        require(job_id.isdigit(), "ambiguous submission response")
        return {"job_id": job_id, "response": raw}

    def release(self, job_id):
        self.command(["scontrol", "release", str(job_id)])


def card(campaign, stage, model="all", shard=0, threshold=0, *, dependency=None, hold=False, attempt=0, minutes=None, ancestors=(), generation=0, begin=None):
    gpu = stage in ("measure", "execute", "annotate","reader_price") or (stage in ("qualify","freeze","price") and shard>0)
    emergency=stage=="controller" and shard==2
    value = {"root": campaign["root"], "campaign_sha256": campaign["sha256"], "stage": stage,
             "model": model, "shard_index": shard, "threshold_index": threshold,
             "partition": "boost_usr_prod" if gpu else "lrd_all_serial",
             "gpus": 2 if gpu else 0, "cpus": 16 if gpu else (1 if stage == "controller" and not emergency else 4),
             "memory_gib": 120 if gpu else 24,
             "minutes": minutes or (120 if gpu else 180 if stage in ("derive", "aggregate","closeout","supplementary","coordination","contexts","score") else 30),
             "dependency": dependency, "hold": hold, "attempt": attempt,
             "ancestors": list(ancestors), "generation": generation, "begin": begin}
    value["reservation_hours"] = value["minutes"] / 60 * value["cpus"]
    value["comment"] = "rc1:" + digest(value)
    return seal(value)


def never_started(accounting, worker_start, attempt):
    return (attempt==0 and not worker_start and accounting["terminal"] and
        accounting["state"] in ("NODE_FAIL","BOOT_FAIL","CANCELLED","FAILED") and
        accounting["elapsed_seconds"]==0 and accounting["start"] in ("None","Unknown","N/A","",None))


def array_card(cards):
    require(cards and all((c["stage"],c["model"],c["partition"],c["minutes"],c["gpus"],c["cpus"],c["memory_gib"])
        ==(cards[0]["stage"],cards[0]["model"],cards[0]["partition"],cards[0]["minutes"],cards[0]["gpus"],cards[0]["cpus"],cards[0]["memory_gib"]) for c in cards),
        "array groups require identical qualified resources")
    value={k:v for k,v in cards[0].items() if k not in ("sha256","comment")}
    value.update(array_cards=cards,array_concurrency=min(8,len(cards)) if cards[0]["gpus"] else 1,
        reservation_hours=sum(c["reservation_hours"] for c in cards),
        array_index_mapping="array_cards[SLURM_ARRAY_TASK_ID] -> immutable stage/threshold/shard")
    value["comment"]="rc1:"+digest(value)
    return seal(value)


def submit_card(campaign, value, scheduler):
    root = Path(campaign["root"])
    (root / "logs").mkdir(parents=True, exist_ok=True, mode=0o700)
    publish(root / "cards" / (value["sha256"] + ".json"), value)
    directory = root / "submissions" / value["sha256"]
    with locked(root / "locks", "submissions"):
        if (directory / "ACCEPTED.json").exists():
            return load(directory / "ACCEPTED.json")
        if (directory / "INTENT.json").exists():
            matches = scheduler.matching(value["comment"])
            require(len(matches) == 1, "unresolved submission; no blind resubmission")
            response = {"job_id": matches[0], "response": "reconciled by exact unique job comment"}
        else:
            publish(directory / "INTENT.json", {"card_sha256": value["sha256"], "comment": value["comment"], "epoch": time.time()})
            response = scheduler.submit(value)
        proof = scheduler.verify(response["job_id"], value)
        return publish(directory / "ACCEPTED.json", {**response, "card_sha256": value["sha256"],
                                                     "readback": proof})


def protected_controller(campaign, scheduler, *, generation, wait_ids=(), begin=None):
    """Accept the primary and all-ancestor failure guards before any release."""
    require(generation <= 1024, "finite controller handoff limit reached")
    intent_path=Path(campaign["root"])/"handoff_intents"/f"{generation:04d}.json"
    if intent_path.exists():
        intent=load(intent_path)
        wait_ids,begin=intent["wait_ids"],intent["begin"]
    else:
        intent=publish(intent_path,{"generation":generation,"wait_ids":sorted(set(map(str,wait_ids))),"begin":begin})
        wait_ids=intent["wait_ids"]
    dependency = "afterany:" + ":".join(sorted(set(map(str, wait_ids)))) if wait_ids else None
    primary = card(campaign, "controller", hold=True, minutes=60, dependency=dependency,
                   generation=generation, begin=begin)
    accepted = [submit_card(campaign, primary, scheduler)]
    ancestors = [accepted[0]["job_id"]]
    guards = []
    for index in (1, 2):
        guard = card(campaign, "controller", shard=index, hold=True, minutes=180 if index==2 else 60,
                     generation=generation, ancestors=ancestors,
                     dependency="afternotok:" + ":".join(reversed(ancestors)))
        receipt = submit_card(campaign, guard, scheduler)
        guards.append(guard)
        accepted.append(receipt)
        ancestors.append(receipt["job_id"])
    handoff = publish(Path(campaign["root"]) / "handoffs" / f"{generation:04d}.json", {
        "primary": accepted[0], "guards": accepted[1:], "wait_ids": list(wait_ids),
        "generation": generation, "all_ancestor_protection": True})
    for value, receipt in reversed(list(zip([primary, *guards], accepted))):
        if not (Path(campaign["root"]) / "submissions" / value["sha256"] / "RELEASED.json").exists():
            scheduler.release(receipt["job_id"])
            proof = scheduler.verify(receipt["job_id"], value)
            require(proof["fields"].get("Reason") != "JobHeldUser", "controller remains held")
            publish(Path(campaign["root"]) / "submissions" / value["sha256"] / "RELEASED.json", {"job_id": receipt["job_id"], "proof": proof})
    return handoff


def admission(snapshot, *, baseline_consumed, campaign_actual, outstanding, residual, output_bytes, prices_complete=True, reserve_hours=32.):
    require(all(v >= 0 for v in (campaign_actual, outstanding, residual, output_bytes)), "negative resource forecast")
    # The larger of observed campaign consumption and current balance delta also
    # captures charges from independent project activity since authorization.
    expended = max(campaign_actual, snapshot["consumed_local_hours"] - baseline_consumed)
    available = min(CEILING - expended, snapshot["remaining_local_hours"]) - outstanding - reserve_hours
    forecast = MARGIN * residual
    storage = snapshot["storage_headroom_bytes"] - STORAGE_RESERVE
    reasons = []
    if not prices_complete:
        reasons.append("required runtime/shape/operator price is unavailable")
    if forecast > available:
        reasons.append("complete residual workload exceeds remaining authorized local hours")
    if output_bytes > storage:
        reasons.append("complete residual output exceeds exact project storage headroom after reserve")
    return {"admitted": not reasons, "reasons": reasons, "forecast_local_hours": forecast,
            "available_local_hours": available, "campaign_actual_local_hours": campaign_actual,
            "outstanding_reservations_local_hours": outstanding, "residual_output_bytes": output_bytes,
            "available_output_bytes": storage, "reserve_local_hours": reserve_hours,
            "scope": "complete residual campaign", "reservation_margin": MARGIN}
