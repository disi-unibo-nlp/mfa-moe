"""Size verified long captures and produce a capped full-collection resource draft."""
import hashlib
import json
import math
import os
import time
from pathlib import Path
from measurement_study.artifact_store import pack
ROOT=Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study")
def main():
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("Slurm required")
    dest=ROOT/"results/measurement-study/long-storage-sizing"
    dest.mkdir(parents=True,exist_ok=True)
    selected={"gpt":"smoke/gpt-58487262","glm":"smoke/glm-58487265","qwen330b":"smoke/qwen330b-58487288","qwen36":"native-replay/qwen36-58472802","nemotron":"native-replay/nemotron-58474338"}
    campaign=json.loads((ROOT/"report/measurement-study/campaign.json").read_text())
    collection_path=ROOT/"results/measurement-study/source-join-v2/collection.json"
    collection=json.loads(collection_path.read_text());groups={g["model"]:g for g in collection["model_groups"]}
    results=[]
    for source,rel in selected.items():
        start=time.monotonic();path=ROOT/"results/measurement-study"/rel
        result=json.loads((path/"result.json").read_text());assert result["status"]=="native_capture_and_long_trace_passed"
        model=campaign["sources"][source]["generation_model"]
        corpus=groups.get(model)
        max_context=0
        if corpus is None:
            spec=campaign["sources"][source];root=Path(spec["sampling_root"]);manifest=json.loads((root/"sampling_manifest.json").read_text())
            wanted={(x["dataset"],x["problem_id"],x["sample_id"]) for x in manifest["completions"] if x["selected_sentences"]>0}
            found=set();total=0
            for p in sorted((root/"generation"/model.replace("/","--")).glob("*/traces.jsonl")):
                for line in p.open():
                    row=json.loads(line);key=(row["dataset"],row["problem_id"],row["sample_id"])
                    if key not in wanted:continue
                    assert key not in found;found.add(key)
                    replay=row["metadata"]["token_replay"];n=len(replay["completion_token_ids"]);assert n>0
                    total+=n;max_context=max(max_context,n+len(replay["prompt_token_ids"]))
            assert found==wanted
            corpus=dict(unique_content_attempts=len(found),completion_tokens=total)
        files=[]
        for file in sorted((path/"forward").rglob("*")):
            if file.is_file():files.append(pack(file,dest/source/(str(file.relative_to(path/"forward"))+".gz")))
        measured_tokens=result.get("completion_tokens") or sum(x["tokens"] for x in result["attempts"])
        # vLLM forward_seconds excludes reduction: use observed complete post-load
        # phases from timestamped logs (136 s Qwen3.6, 104 s Nemotron).
        phase_seconds=result.get("forward_seconds",{"qwen36":136,"nemotron":104}.get(source))
        shards=math.ceil(corpus["completion_tokens"]/200000)
        predicted_seconds=corpus["completion_tokens"]*phase_seconds/measured_tokens+shards*result["load_seconds"]
        wall_seconds=math.ceil(predicted_seconds*1.5/1800)*1800
        packed=sum(x["packed_bytes"] for x in files)
        row=dict(source=source,model=model,corpus=corpus,measured_tokens=measured_tokens,measured_phase_seconds=phase_seconds,measured_load_seconds=result["load_seconds"],planned_shards=shards,shard_target_tokens=200000,estimated_total_wall_hours=predicted_seconds/3600,proposed_gpu_hours_cap=2*wall_seconds/3600,estimated_packed_bytes=math.ceil(corpus["completion_tokens"]*packed/measured_tokens),packed_fixture_bytes=packed,source_fixture_bytes=sum(x["source_bytes"] for x in files),sizing_seconds=time.monotonic()-start,files=files,max_context_pending_sources=max_context or None)
        results.append(row);(dest/"progress.json").write_text(json.dumps(results,indent=2)+"\n")
    proposal=dict(status="measured_resource_draft_not_submitted",source_join_sha256=hashlib.sha256(collection_path.read_bytes()).hexdigest(),gpus_per_job=2,qos="normal",concurrent_jobs_cap=4,models=results,total_proposed_gpu_hours_cap=sum(r["proposed_gpu_hours_cap"] for r in results),estimated_packed_bytes=sum(r["estimated_packed_bytes"] for r in results),assumptions=["1.5x measured phase plus repeated startup; rounded up to 30 minutes per model", "200000 completion tokens per provisional shard", "compression estimated from three real traces per model", "memory and timing must be checked for out-of-range longest traces"],outstanding=["production runner and immutable per-shard input manifests", "storage headroom against live WORK quota", "all class-cohort views and calibration-only boundary thresholds", "Gemma checkpoint transfer and native GPU acceptance", "older GPTQ/GGUF checkpoint and token recovery", "full production GPU resource authorization"])
    (dest/"result.json").write_text(json.dumps(proposal,indent=2)+"\n")
    print(json.dumps({k:v for k,v in proposal.items() if k!="models"}),flush=True)
if __name__=="__main__":main()
