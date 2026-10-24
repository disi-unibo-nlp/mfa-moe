"""Validate completed 100k-label corpus, export annotations and run class analyses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from moe_exp.correlation_pipeline.forward_adapter import adapt_source
from moe_exp.correlation_pipeline.labels_export import SOURCES
import analyze_available_metrics as metrics
P=Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe")
ROOT=P/"routing-study"
OLD=Path("/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-stratified")
def main():
    parser=argparse.ArgumentParser();parser.add_argument("source",choices=["glm","qwen330b"]);a=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("Slurm allocation required")
    code=Path(__file__).parent
    hashes=json.loads((code/"SHA256.json").read_text())
    for rel,sha in hashes.items():
        if hashlib.sha256((code/rel).read_bytes()).hexdigest()!=sha:raise ValueError("Frozen code changed: "+rel)
    out=ROOT/"results/measurement-study/completed-labels"/a.source
    out.mkdir(parents=True,exist_ok=True)
    labelroot=out/"parts";labelroot.mkdir(exist_ok=True)
    for part in range(4):
        name=f"part-{part:02d}"
        src=(ROOT/"results/measurement-study/label-resume-v1/qwen330b"/name if a.source=="qwen330b" and part<2 else OLD/a.source/name)
        dst=labelroot/name
        if dst.is_symlink():assert dst.resolve()==src.resolve()
        elif dst.exists():raise ValueError("Unexpected existing part output")
        else:dst.symlink_to(src,target_is_directory=True)
    sampling=Path(SOURCES[a.source]["sampling_root"])
    # Frozen code is relocated: reconstruct the original source path explicitly.
    sub={"glm":"glm-4.7-flash","qwen330b":"qwen3-30b-a3b"}[a.source]
    sampling=P/"repo/results/correlation_pipeline"/sub/"reasoning-vllm-v1/sampling-100k-stratified"
    report=adapt_source(a.source,labelroot,sampling,out/"annotations")
    slug=SOURCES[a.source]["generation_model"].replace("/","--")
    spec=dict(id="qwen38-mtp3-stratified-"+a.source,kind="labels",format="batch",annotations=[metrics.file_info(p) for p in sorted(labelroot.glob("part-*/checkpoints/batch-*.json"))],traces=[metrics.file_info(p) for p in sorted((sampling/"generation"/slug).glob("*/traces.jsonl"))])
    (out/"input-manifest.json").write_text(json.dumps(spec,indent=2)+"\n")
    result=metrics.labels_job(spec,out)
    if result["unmatched"] or result["rejected_candidates"] or result["conflicts"]:raise ValueError("Incomplete label analysis: "+str(result))
    (out/"status.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
