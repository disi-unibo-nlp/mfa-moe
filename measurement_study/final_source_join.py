"""Extend the verified saved-label join with final GLM/Qwen3 labels."""
import hashlib
import json
import os
from pathlib import Path
from measurement_study.join_collection import join_cohort, atomic
ROOT=Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study")
def main():
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("Slurm required")
    prior=ROOT/"results/measurement-study/source-join-v2/collection.json"
    sha=hashlib.sha256(prior.read_bytes()).hexdigest()
    if sha!="eb07af23d96dfdec344f9e0ecb25910b0034ef3421d9e81497874ab992b20fa2":raise ValueError("Previous collection changed")
    result=json.loads(prior.read_text());out=ROOT/"results/measurement-study/source-join-final"
    bindings={"previous_collection_sha256":sha,"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"new_inputs":{}}
    ids={row["id"] for row in result["attempts"]}
    for source in ("glm","qwen330b"):
        root=ROOT/"results/measurement-study/completed-labels"/source
        status=json.loads((root/"status.json").read_text())
        assert status["status"]=="complete" and status["unmatched"]==status["rejected_candidates"]==status["conflicts"]==0
        path=root/"input-manifest.json";spec=json.loads(path.read_text());bindings["new_inputs"][source]=hashlib.sha256(path.read_bytes()).hexdigest()
        report=join_cohort(spec,out)
        assert report["unmatched"]==report["rejected"]==report["label_conflicts"]==0
        result["cohorts"].append(report)
        for row in json.loads((out/spec["id"]/"attempts.json").read_text()):
            assert row["id"] not in ids,"Unexpected overlap with previously absent model"
            ids.add(row["id"])
            target={k:v for k,v in row.items() if k not in ("annotation","locations","metadata_variants")}
            target["cohorts"]=[dict(id=spec["id"],annotation=row["annotation"],locations=row["locations"],identity=row["identity"],metadata_variants=row["metadata_variants"])]
            result["attempts"].append(target)
    result["model_groups"]=[]
    for model in sorted({r["model"] for r in result["attempts"]}):
        rows=[r for r in result["attempts"] if r["model"]==model]
        result["model_groups"].append(dict(model=model,unique_content_attempts=len(rows),completion_tokens=sum(r["completion_tokens"] or 0 for r in rows),exact_replay_attempts=sum(r["exact_token_replay"] for r in rows),benchmarks=sorted({r["dataset"] for r in rows})))
    result["pending_primary_labels"]=[];result["binding"]=bindings
    atomic(out/"collection.json",result)
    print(json.dumps(dict(status=result["status"],model_groups=result["model_groups"],cohorts=len(result["cohorts"]),attempts=len(result["attempts"]))),flush=True)
if __name__=="__main__":main()
