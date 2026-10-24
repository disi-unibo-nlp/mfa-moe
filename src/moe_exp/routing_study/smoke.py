"""One explicitly authorized debug job; synthetic expert targets are engineering checks."""
from __future__ import annotations
import os
import time
from pathlib import Path
from .common import (CHECKPOINT, SAMPLING, digest, seal, verify, rows, read_json, write_json,
                     require_compute, code_hashes, private_path, file_hash, problem_key)
from .policy import freeze
from .execution import make_engine, run_wave
from .vllm_backend import backend_fingerprint


def run(fixtures, cpu_report, output):
    require_compute()
    cpu = verify(read_json(cpu_report))
    if cpu["status"] != "passed" or cpu["code"] != code_hashes() or cpu["skipped"] or not cpu.get("tensor_tests"):
        raise ValueError("A passing, current, no-skip CPU verification report is required")
    output = private_path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Smoke output must be new")
    output.mkdir(parents=True, exist_ok=True)
    sources = list(rows(fixtures))
    if len(sources) != 4:
        raise ValueError("Smoke uses exactly four existing engineering fixtures")
    specs = []
    for i in range(8):
        source = sources[i % 4]
        replay = source["metadata"]["token_replay"]
        ids = replay["prompt_token_ids"]
        if len(ids)>2048:
            raise ValueError("Engineering prompt exceeds the capped smoke envelope")
        specs.append(dict(id=digest(["smoke",i]), problem_key=problem_key(source), prompt_token_ids=ids,
            replay_token_ids=ids+replay["completion_token_ids"][:512], is_correct=bool(source["is_correct"]),
            gold_answer=source["gold_answer"], answer_type="math",
            sampling={**SAMPLING,"max_tokens":128,"seed":42000+i}))
    code = code_hashes()
    binding = {"engineering_only": True}
    scores = [1.] + [0.] * 255
    native, zero = freeze("native"), freeze("identity",0)
    checks = [freeze("identity",a,parameters={"layers":{"0":{"scores":scores}}},calibration=binding)
              for a in (-1,-.5,.5,1)]
    checks += [freeze(a,1,targets=[(0,0),(39,255)],parameters={"epsilon":.01},calibration=binding)
               for a in ("steer_promote","steer_suppress")]
    checks += [freeze("rice",b,targets=[(0,0),(39,255)],parameters={"renormalize":norm},calibration=binding)
               for b,norm in ((.5,True),(2,True),(4,True),(2,False))]
    checks += [freeze("temperature",t) for t in (.5,2)]
    checks += [freeze("margin",s,parameters={"layers":{"0":{"lower":0.,"upper":0.}}},
                      calibration=binding) for s in (.5,1)]
    from .vllm_backend import checkpoint_fingerprint
    print("[routing-study smoke] validating checkpoint contents", flush=True)
    checksum_began = time.perf_counter()
    checkpoint_fingerprint(verify_weights=True)
    backend = backend_fingerprint()
    write_json(output/"checkpoint-validation.json", seal(dict(
        checkpoint=CHECKPOINT, backend=backend, code=code, job_id=os.environ["SLURM_JOB_ID"],
        checkpoint_content_verified=True, seconds=time.perf_counter()-checksum_began)), immutable=True)
    print("[routing-study smoke] checkpoint validated; starting engine", flush=True)
    began = time.perf_counter()
    llm = make_engine(backend)
    startup = time.perf_counter()-began
    print(f"[routing-study smoke] engine started in {startup:.1f}s; configuring dispatch hooks", flush=True)
    llm.collective_rpc("study_configure", kwargs={"policy":native,"backend":backend,
        "checkpoint":CHECKPOINT,"validation":True})
    saved = []
    for index, policy in enumerate((native, zero, native, *checks)):
        print(f"[routing-study smoke] condition {index}: {policy['action']} {policy['strength']}", flush=True)
        if index:
            llm.collective_rpc("study_set_validation_policy", kwargs={"policy":policy})
        directory = output / (policy["sha256"] + ("-replay" if index==2 else ""))
        directory.mkdir()
        replay_mode = index >= 2
        requests = [{**r,"sampling":{"max_tokens":1,"temperature":0.,"seed":42}} if replay_mode else
                    {k:v for k,v in r.items() if k != "replay_token_ids"} for r in specs]
        started = time.perf_counter()
        records = list(run_wave(llm, list(enumerate(requests)),policy,directory,replay=replay_mode,
                                study_sha256="engineering_smoke"))
        saved.append(dict(policy=policy,phase="replay" if replay_mode else "generation",input_tokens=sum(r["observed_input_tokens"] for r in records),seconds=time.perf_counter()-started,
                          generated_tokens=sum(r["generated_token_count"] for r in records),
                          hook_seconds=max(r["hook_seconds_cumulative"] for r in records)))
        if index == 0:
            baseline = {r["id"]: r for r in records}
        elif index == 1:
            for r in records:
                b = baseline[r["id"]]
                if r["generated_token_ids"] != b["generated_token_ids"] or r["dispatch"] != b["dispatch"]:
                    raise RuntimeError("Actual zero-strength identity differs from native")
        else:
            if index>2 and not any(d["diagnostics"]["changed_weights"] for r in records for d in r["dispatch"].values()):
                raise RuntimeError("Smoke action did not change any executed reasoning weights")
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file())>2*1024**3:
            raise RuntimeError("Smoke artifacts exceed the 2 GiB cap")
    storage = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    result = seal(dict(schema=1,status="passed",checkpoint=CHECKPOINT,backend=backend,code=code,
        job_id=os.environ["SLURM_JOB_ID"],cpu_report_sha256=cpu["sha256"],fixtures_sha256=file_hash(Path(fixtures)),
        dispatch_verified=True,zero_matches_native=True,weight_ids_preserved=True,checkpoint_content_verified=True,
        startup_seconds=startup,elapsed_seconds=time.perf_counter()-began,
        native_generation_tokens_per_second=saved[0]["generated_tokens"]/saved[0]["seconds"],
        native_replay_input_tokens_per_second=saved[2]["input_tokens"]/saved[2]["seconds"],
        conservative_replay_input_tokens_per_second=min(s["input_tokens"]/s["seconds"] for s in saved[2:]),
        observed_storage_bytes=storage,conditions=saved,
        limitation="Engineering targets; short traces. Long-run throughput/storage must retain a safety margin. No study accuracy finding."))
    write_json(output/"acceptance.json",result,immutable=True)
    return result


def resource_proposal(acceptance, study, *, conditions, safety_factor=2.):
    verify(acceptance)
    verify(study)
    if acceptance["status"] != "passed" or safety_factor < 1:
        raise ValueError("Measured smoke required")
    rate=acceptance["native_generation_tokens_per_second"]
    if rate <= 0:
        raise ValueError("No measured throughput")
    if not 1<=conditions<=16:
        raise ValueError("Static grid contains at most 16 conditions")
    tuning=len(study["attempts"]["tuning"])
    test=len(study["attempts"]["test"])
    replay_tokens=sum(a["replay_tokens"] for a in study["calibration_sources"].values())
    prefill_rate=acceptance["conservative_replay_input_tokens_per_second"]
    generation_rate=min(s["generated_tokens"]/s["seconds"] for s in acceptance["conditions"] if s["phase"]=="generation")
    phases={"calibration":dict(input_tokens=replay_tokens,loads=1,seconds=replay_tokens/prefill_rate),
            "static_tuning":dict(generated_tokens=tuning*32768*conditions,loads=conditions,seconds=tuning*32768*conditions/generation_rate),
            "adaptive_tuning":dict(generated_tokens=tuning*32768*18,loads=18,seconds=tuning*32768*18/generation_rate),
            "final":dict(generated_tokens=test*32768*14,loads=14,seconds=test*32768*14/generation_rate)}
    for phase in phases.values():
        phase["proposed_gpu_hours"]=2*safety_factor*(phase["seconds"]+phase["loads"]*acceptance["startup_seconds"])/3600
    measured_tokens=sum(s["input_tokens"]+s["generated_tokens"] for s in acceptance["conditions"])
    bytes_per_token=acceptance["observed_storage_bytes"]/measured_tokens
    total_tokens=replay_tokens+sum(p.get("generated_tokens",0) for p in phases.values())
    return seal(dict(schema=1,status="proposal_not_authorized",
        smoke_sha256=acceptance["sha256"],study_sha256=study["sha256"],phases=phases,
        safety_factor=safety_factor,proposed_total_gpu_hours=sum(p["proposed_gpu_hours"] for p in phases.values()),
        proposed_storage_bytes=int(safety_factor*bytes_per_token*total_tokens),
        storage_method="conservative short-smoke artifact bytes per observed token; caches separate",
        budget_method="hard 32768-token caps; includes startup per condition; checkpointed 24h-or-shorter normal-QoS jobs",
        limitations="Decode action overhead beyond native/zero must be measured before accepting the full budget; reserve additional startup for each walltime chunk. Short smokes do not guarantee long-run rates.",
        pilot_budget_reserved_gpu_hours=32,approved_study_gpu_hours=0))
