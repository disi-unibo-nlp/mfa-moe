"""Campaign v3 stage gates. Each gate runs as a short Slurm CPU job: it validates its stage,
resubmits drained work (bounded rounds) or, on acceptance, writes a receipt and submits the
next stage. There is no long-running orchestrator: every transition is a finite Slurm job.

  stage.py gen-gate  --model M [--round R]
  stage.py cont-gate --model M --phase pilot|main [--round R]
  stage.py submit-gen --model M          (after an accepted smoke)
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

# CODE is the tree this script runs from (the frozen T1 snapshot in production); outputs stay
# under the live checkout REPO/results. Every job submitted here gets CAMPAIGN_CODE=CODE.
CODE = Path(__file__).resolve().parents[2]
REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
sys.path.insert(0, str(CODE / "src"))
from moe_exp.correlation_pipeline.model_profiles import CARD_PROFILES  # noqa: E402

D = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3")
CARD_JOB = CODE / "sbatch/card_job.sbatch"
CPU_JOB = D / "sbatch/cpu_boost.sbatch"
GEN_TABLE = ["aime24:0:2", "aime24:1:2", "aime25:0:2", "aime25:1:2", "amc23:0:2", "amc23:1:2",
             "olympiad:0:2", "olympiad:1:2", "math500:0:1", "minerva:0:1"]
EXPECTED = {"math500": 500, "aime24": 960, "aime25": 960, "olympiad": 675, "amc23": 1280, "minerva": 272}
MAX_ROUNDS = 6
CLIENT_PY = "/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python"


def reviewed() -> bool:
    """Full runs start only after the T1 code is signed, and only from the signed snapshot.

    A gate still running the live checkout therefore holds (writes its waiting marker) and
    `release` has to be run from the snapshot.
    """
    path = D / "review" / "T1.json"
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "approved" or Path(receipt.get("code_snapshot", "")).resolve() != CODE:
        return False
    manifest = json.loads((CODE / "MANIFEST.json").read_text())
    return manifest["tree_sha256"] == receipt.get("tree_sha256")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sbatch(args: list[str], stage: str) -> int:
    out = subprocess.run(["sbatch", "--parsable", f"--export=ALL,CAMPAIGN_CODE={CODE}", *args], capture_output=True, text=True, check=True)
    job = int(out.stdout.strip().split(";")[0])
    shown = subprocess.run(["scontrol", "show", "job", str(job)], capture_output=True, text=True).stdout
    with (D / "ledger.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(stage=stage, job_id=job, submitted_utc=now(),
                                     verified="JobId=%d" % job in shown, args=args)) + "\n")
    print(f"submitted {stage} {job}", flush=True)
    return job


def gpu_slot() -> list[str]:
    """Job-name/dependency args that put a GPU job in the shared campaign slot pool.

    Label parts and capture shards (the capture driver uses the same file) take names
    v3-gpu-<k> with --dependency=singleton, so at most `slots` of them run at once whatever
    the model; the pool size is read from gpu_slots.json at submit time and can be widened
    as generation drains without refreezing code.
    """
    config = D / "gpu_slots.json"
    slots = int(json.loads(config.read_text())["slots"]) if config.exists() else 6
    with (D / "gpu_slots.counter").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        n = int(handle.read().strip() or 0)
        handle.seek(0)
        handle.truncate()
        handle.write(str(n + 1))
    return [f"--job-name=v3-gpu-{n % slots}", "--dependency=singleton"]


def gate_job(command: str, name: str, dependency: str | None) -> int:
    dep = [f"--dependency={dependency}", "--kill-on-invalid-dep=yes"] if dependency else []
    return sbatch([f"--job-name={name}", "--time=02:00:00", *dep, str(CPU_JOB), command], name)


def slug(model: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "--", CARD_PROFILES[model]["model"]).strip("-")


def gen_dir(model: str) -> Path:
    return REPO / "results/correlation_pipeline/card-v3" / model / "generation" / slug(model)


def task_done(model: str, task: int) -> bool:
    dataset, shard, shards = GEN_TABLE[task].split(":")
    base = gen_dir(model) / dataset
    if shards == "1":
        return (base / "manifest.json").exists() and json.loads((base / "manifest.json").read_text()).get("status") == "complete"
    marker = base / f"shard_{int(shard):02d}_of_{int(shards):02d}.json"
    return marker.exists() and json.loads(marker.read_text()).get("status") == "shard_complete"


def submit_gen(args) -> None:
    receipt = D / "smoke" / args.model / "smoke_receipt.json"
    if json.loads(receipt.read_text()).get("status") != "complete":
        raise SystemExit(f"smoke not accepted for {args.model}")
    if not reviewed():
        print(f"{args.model}: smoke accepted; waiting for T1 review before full generation")
        return
    todo = [t for t in range(len(GEN_TABLE)) if not task_done(args.model, t)]
    array = sbatch([f"--job-name=v3-gen-{args.model}", f"--array={','.join(map(str, todo))}%2",
                    str(CARD_JOB), args.model, "gen"], f"gen-{args.model}")
    gate_job(f"$CLIENT_PY {CODE}/scripts/campaign_v3/stage.py gen-gate --model {args.model} --round 1",
             f"v3-gengate-{args.model}", f"afterany:{array}")


def contract_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in config.items() if k != "seed"},
                                     sort_keys=True).encode()).hexdigest()


def validate_corpus(model: str) -> dict:
    problems, stats = [], {}
    contracts = Counter()
    for dataset, expected in EXPECTED.items():
        path = gen_dir(model) / dataset / "traces.jsonl"
        rows = [json.loads(line) for line in path.open()] if path.exists() else []
        if len(rows) != expected:
            problems.append(f"{dataset}: {len(rows)} traces, expected {expected}")
        replay = sum(bool(r["metadata"].get("token_replay", {}).get("completion_token_ids")) for r in rows)
        if replay != len(rows):
            problems.append(f"{dataset}: token replay on {replay}/{len(rows)}")
        bad_cap = fallback = capped = 0
        for r in rows:
            meta = r["metadata"]
            contracts[contract_hash(meta["generation_config"])] += 1
            n = len(meta.get("token_replay", {}).get("completion_token_ids") or [])
            if meta.get("termination") == "length":
                capped += 1
                if meta.get("effective_max_tokens") and n != meta["effective_max_tokens"]:
                    bad_cap += 1
            if r.get("scoring_method") == "normalized_exact_numeric_fallback" and \
                    meta.get("scoring_status") not in ("prediction_parse_failed",):
                fallback += 1
        if bad_cap:
            problems.append(f"{dataset}: {bad_cap} length stops not at the effective cap")
        if rows and fallback / len(rows) > 0.005:
            problems.append(f"{dataset}: math_verify fallback {fallback}/{len(rows)}")
        stats[dataset] = dict(traces=len(rows), capped=capped, fallback=fallback,
                              correct=sum(r.get("is_correct") is True for r in rows))
    if len(contracts) != 1:
        problems.append(f"{len(contracts)} distinct generation contracts")
    return dict(problems=problems, stats=stats, contracts=dict(contracts))


def gen_gate(args) -> None:
    missing = [t for t in range(len(GEN_TABLE)) if not task_done(args.model, t)]
    if missing:
        if args.round >= MAX_ROUNDS:
            raise SystemExit(f"{args.model}: tasks {missing} incomplete after {args.round} rounds")
        array = sbatch([f"--job-name=v3-gen-{args.model}", f"--array={','.join(map(str, missing))}%2",
                        str(CARD_JOB), args.model, "gen"], f"gen-{args.model}-r{args.round + 1}")
        gate_job(f"$CLIENT_PY {CODE}/scripts/campaign_v3/stage.py gen-gate --model {args.model} "
                 f"--round {args.round + 1}", f"v3-gengate-{args.model}", f"afterany:{array}")
        return
    for dataset in ("aime24", "aime25", "amc23", "olympiad"):
        subprocess.run([CLIENT_PY, "-m", "moe_exp.correlation_pipeline.generate", "--profile", args.model,
                        "--save-token-ids", "--datasets", dataset, "--assemble-only", "--seed", "20260925",
                        "--output-dir", str(gen_dir(args.model).parent)], check=True)
    result = validate_corpus(args.model)
    status = "complete" if not result["problems"] else "failed"
    receipt = dict(status=status, model=args.model, checked_utc=now(), **result)
    out = D / "acceptance" / f"gen-{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt)[:2000])
    if status != "complete":
        raise SystemExit(3)
    after_gen(args.model)


def after_gen(model: str) -> None:
    """Accepted corpus -> cohorts -> 4 label parts -> label gate; J1 on the residual set."""
    gen = gen_dir(model)
    cohorts = D / "cohorts" / model
    c = sbatch(["--job-name=v3-cohorts-" + model, "--time=04:00:00", "--cpus-per-task=4", str(CPU_JOB),
                f"$CLIENT_PY -m moe_exp.correlation_pipeline.sample_cohorts --model {model} "
                f"--gen-root {gen} --out {cohorts}"], f"cohorts-{model}")
    gate_job(f"$CLIENT_PY {CODE}/scripts/campaign_v3/stage.py submit-labels --model {model}",
             f"v3-sublabels-{model}", f"afterok:{c}")
    gate_job(f"$CLIENT_PY {CODE}/scripts/campaign_v3/stage.py capture-b --model {model}",
             f"v3-capb-{model}", f"afterok:{c}")
    j1 = D / "j1" / f"new_{model}_items.jsonl"
    b = sbatch(["--job-name=v3-j1build-" + model, "--time=02:00:00", str(CPU_JOB),
                f"$CLIENT_PY -m moe_exp.correlation_pipeline.answer_equivalence build-corpus "
                f"--model {model} --gen-root {gen} --out {j1}"], f"j1-build-new-{model}")
    sbatch([f"--job-name=v3-j1-new-{model}", f"--dependency=afterok:{b}", "--kill-on-invalid-dep=yes",
            str(CODE / "sbatch/j1_judge.sbatch"), str(j1), str(D / "j1" / f"new_{model}_verdicts.jsonl")],
           f"j1-new-{model}")


def submit_labels(args) -> None:
    parts = json.loads((D / "cohorts" / args.model / "parts.json").read_text())["parts"]
    jobs = []
    for part in parts:
        out = D / "labels" / args.model / f"part-{part['part']:02d}"
        if (out / "summary.json").exists() and json.loads((out / "summary.json").read_text()).get("status") == "complete":
            continue
        jobs.append(sbatch([*gpu_slot(), str(CODE / "sbatch/label_job.sbatch"),
                            args.model, str(part["part"]), part["trace_root"], str(out), str(part["total"])],
                           f"label-{args.model}-{part['part']}-r{args.round}"))
    dependency = "afterany:" + ":".join(map(str, jobs)) if jobs else None
    gate_job(f"$CLIENT_PY {CODE}/scripts/campaign_v3/stage.py label-gate --model {args.model} --round {args.round}",
             f"v3-labelgate-{args.model}", dependency)


def label_gate(args) -> None:
    parts = json.loads((D / "cohorts" / args.model / "parts.json").read_text())["parts"]
    cohorts = json.loads((D / "cohorts" / args.model / "cohorts.json").read_text())
    status, problems, labeled = {}, [], set()
    classes = {"Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor"}
    for part in parts:
        base = D / "labels" / args.model / f"part-{part['part']:02d}"
        summary = base / "summary.json"
        info = json.loads(summary.read_text()) if summary.exists() else {}
        ok = info.get("status") == "complete"
        status[part["part"]] = ok
        if ok:
            if not info.get("completed") == info.get("expected") == part["total"]:
                problems.append(f"part {part['part']}: completed/expected {info.get('completed')}/"
                                f"{info.get('expected')} vs planned {part['total']}")
            records = json.loads((base / "annotations.json").read_text())
            unknown = sum(r.get("label") not in classes for r in records)
            if records and unknown / len(records) > 0.02:
                problems.append(f"part {part['part']}: unknown labels {unknown}/{len(records)} > 2%")
            labeled.update((r["identity"]["dataset"], r["identity"]["problem_id"],
                            r["identity"]["sentence_index"]) for r in records)
    if not all(status.values()):
        if args.round >= MAX_ROUNDS:
            raise SystemExit(f"{args.model}: label parts {status} incomplete after {args.round} rounds")
        args.round += 1
        submit_labels(args)
        return
    # coverage: labeled (trace, sentence) pairs are exactly the cohort selection, and every
    # cohort attempt keeps an unfolded outcome state (capped is its own stratum)
    wanted = {(r["dataset"], r["problem_id"], i) for r in cohorts["rows"]
              for i in r["sentence_selection"]["indices"]}
    if labeled != wanted:
        problems.append(f"selection coverage: {len(wanted - labeled)} selected sentences unlabeled, "
                        f"{len(labeled - wanted)} labeled outside the selection")
    unselected = [(r["dataset"], r["problem_id"]) for r in cohorts["rows"]
                  if not r["sentence_selection"]["indices"]]
    if unselected:
        problems.append(f"{len(unselected)} cohort attempts without selected sentences: {unselected[:5]}")
    states = Counter(r["stratum"] for r in cohorts["rows"])
    if set(states) - {"correct", "finished_wrong", "capped", "unscored"}:
        problems.append(f"unexpected cohort strata {dict(states)}")
    (D / "acceptance" / f"labels-{args.model}.json").write_text(json.dumps(dict(
        status="complete" if not problems else "failed", model=args.model, parts=status,
        strata=dict(states), labeled_sentences=len(labeled), problems=problems,
        checked_utc=now()), indent=1))
    if problems:
        raise SystemExit(f"{args.model}: label gate failed: {problems}")
    capture_handoff(args.model, "A")


def capture_handoff(model: str, cohort: str) -> None:
    """Captures are the routing-study stage: start them once its T2 review is signed.

    review/T2.json names the frozen capture bundle and the driver that plans, verifies and
    submits the capture jobs; until it is approved a waiting marker records the pending
    cohort, and `release-capture` (run after T2) picks it up.
    """
    t2 = D / "review" / "T2.json"
    receipt = json.loads(t2.read_text()) if t2.exists() else {}
    marker = D / "acceptance" / f"capture-waiting-{model}-{cohort}.json"
    if receipt.get("status") == "approved" and Path(receipt.get("driver", "")).is_file():
        subprocess.run([CLIENT_PY, receipt["driver"], "start", "--model", model, "--cohort", cohort,
                        "--bundle", receipt["bundle"]], check=True)
        marker.unlink(missing_ok=True)
    else:
        marker.write_text(json.dumps(dict(status="waiting_for_T2", model=model, cohort=cohort,
                                          checked_utc=now())))


def capture_gate_b(args) -> None:
    """Cohorts file accepted: cohort B needs no labels, so its capture can start now."""
    capture_handoff(args.model, "B")


def release_capture(args) -> None:
    """After T2: start every capture cohort whose handoff found T2 unsigned."""
    for cohort in ("B", "A"):
        if (D / "acceptance" / f"capture-waiting-{args.model}-{cohort}.json").exists():
            capture_handoff(args.model, cohort)


def cont_gate(args) -> None:
    out = D / "continuation" / args.model
    plan = json.loads((out / "plan.json").read_text())
    rows = [json.loads(line) for line in (out / "continuations.jsonl").open()] \
        if (out / "continuations.jsonl").exists() else []
    done = {r["key"] for r in rows if r.get("status") == "complete"}
    eligible = [p for p in plan["parents"] if p["exclusion"] is None]
    if args.phase == "pilot":
        problems = []
        pilot = [r for r in rows if r.get("status") == "complete"]
        if len(pilot) < min(8, len(eligible)):
            problems.append(f"pilot produced {len(pilot)} branches")
        if not all(r.get("prompt_echo_verified") for r in pilot):
            problems.append("prompt echo not verified on every branch")
        receipt = dict(status="complete" if not problems else "failed", model=args.model,
                       branches=len(pilot), suffix_lens=[r["suffix_len"] for r in pilot],
                       finish=[r["finish_reason"] for r in pilot],
                       seconds=[r["seconds"] for r in pilot], problems=problems, checked_utc=now())
        (D / "acceptance").mkdir(exist_ok=True)
        (D / "acceptance" / f"cont-pilot-{args.model}.json").write_text(json.dumps(receipt, indent=1))
        print(json.dumps(receipt))
        if problems:
            raise SystemExit(3)
        if not reviewed():
            print(f"{args.model}: pilot accepted; waiting for T1 review before the main continuation")
            return
        args.round = 0
    remaining = [p for p in eligible if f"{p['dataset']}|{p['problem_id']}|{p['sample_id']}" not in done]
    if remaining:
        if args.round >= MAX_ROUNDS:
            raise SystemExit(f"{args.model}: {len(remaining)} continuations left after {args.round} rounds")
        job = sbatch([f"--job-name=v3-cont-{args.model}", str(CARD_JOB), args.model, "cont"],
                     f"cont-{args.model}-r{args.round + 1}")
        gate_job(f"$CLIENT_PY {CODE}/scripts/campaign_v3/stage.py cont-gate --model {args.model} "
                 f"--phase main --round {args.round + 1}", f"v3-contgate-{args.model}", f"afterany:{job}")
        return
    subprocess.run([sys.executable, "-m", "moe_exp.correlation_pipeline.continuation", "score",
                    "--model", args.model, "--out", str(out)], check=True)
    (D / "acceptance" / f"cont-{args.model}.json").write_text(json.dumps(dict(
        status="complete", model=args.model, branches=len(done), eligible=len(eligible),
        checked_utc=now()), indent=1))
    items = D / "j1" / f"cont_{args.model}_items.jsonl"
    subprocess.run([sys.executable, "-m", "moe_exp.correlation_pipeline.answer_equivalence", "build-cont",
                    "--model", args.model, "--cont-dir", str(out), "--out", str(items)], check=True)
    if items.stat().st_size:
        sbatch([f"--job-name=v3-j1-cont-{args.model}", "--time=04:00:00", str(CODE / "sbatch/j1_judge.sbatch"),
                str(items), str(D / "j1" / f"cont_{args.model}_verdicts.jsonl")], f"j1-cont-{args.model}")


def queued(name: str) -> bool:
    out = subprocess.run(["squeue", "-u", "lmolfett", "-h", "-o", "%j"], capture_output=True, text=True)
    return name in out.stdout.split()


def release(args) -> None:
    """After review/T1.json is approved: start stages whose gates exited while on hold."""
    if not reviewed():
        raise SystemExit("T1 review not approved")
    smoke = D / "smoke" / args.model / "smoke_receipt.json"
    if smoke.exists() and json.loads(smoke.read_text()).get("status") == "complete" \
            and not queued(f"v3-gen-{args.model}") and not queued(f"v3-gengate-{args.model}") \
            and not (D / "acceptance" / f"gen-{args.model}.json").exists():
        submit_gen(args)
    pilot = D / "acceptance" / f"cont-pilot-{args.model}.json"
    if pilot.exists() and json.loads(pilot.read_text()).get("status") == "complete" \
            and not queued(f"v3-cont-{args.model}") and not queued(f"v3-contgate-{args.model}") \
            and not (D / "acceptance" / f"cont-{args.model}.json").exists():
        args.phase, args.round = "main", 0
        cont_gate(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["gen-gate", "cont-gate", "submit-gen", "submit-labels",
                                         "label-gate", "after-gen", "release", "capture-b",
                                         "release-capture"])
    ap.add_argument("--model", required=True, choices=sorted(CARD_PROFILES))
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--phase", choices=["pilot", "main"], default="main")
    args = ap.parse_args()
    {"gen-gate": gen_gate, "cont-gate": cont_gate, "submit-gen": submit_gen,
     "submit-labels": submit_labels, "label-gate": label_gate,
     "after-gen": lambda a: after_gen(a.model), "release": release, "capture-b": capture_gate_b,
     "release-capture": release_capture}[args.command](args)


if __name__ == "__main__":
    main()
