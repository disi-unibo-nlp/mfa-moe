"""Validate both failed label shards and size context without changing prompts."""
import hashlib
import json
import math
import shutil
from pathlib import Path
from moe_exp.correlation_pipeline.annotation_partition import enumerate_items, DATASETS
from moe_exp.correlation_pipeline.annotation_batch import run_part, digest
from moe_exp.correlation_pipeline.batch_predictor import load_program_and_adapter, render_conversation
from transformers import AutoTokenizer

P = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe")
OUT = P / "routing-study/results/measurement-study/label-resume-v1"
OLD = Path("/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-stratified/qwen330b")
MODEL = Path("/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
PROGRAM = P / "repo/results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json"


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    _, predict, adapter = load_program_and_adapter(str(PROGRAM))
    for part in (0, 1):
        name = f"part-{part:02d}"
        source = P / f"repo/results/correlation_pipeline/qwen3-30b-a3b/reasoning-vllm-v1/sampling-100k-stratified/parts/{name}/generation/Qwen--Qwen3-30B-A3B"
        old = OLD / name
        target = OUT / "qwen330b" / name
        items = enumerate_items(source, DATASETS, limit=25000)
        first = json.loads((old / "checkpoints/batch-000000.json").read_text())
        config = first["binding"]["config"]
        assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == config["program_sha256"]
        plan = dict(schema_version=1, trace_root=str(source), datasets=list(DATASETS), include_traces_sha256=None, source_identities=len(items), selected_identities=len(items), part_size=25000, parts=1, part_counts=[len(items)], selected_sha256=hashlib.sha256("\n".join(digest(x[0]) for x in items).encode()).hexdigest(), first_identity=items[0][0], last_identity=items[-1][0])
        assert digest(plan) == config["source_plan_sha256"], "Source plan changed"
        check = run_part(items, config, output_dir=old, classify=None, expected_count=25000, dry_run=True)
        completed = check["completed"]
        max_bound = 0
        max_exact = 0
        max_identity = None
        overflowing = []
        # UTF-8 bytes bound byte-BPE token count. Exact tokenize only long rendered
        # prompts; normal short prompts cannot overflow the existing context.
        for i in range(completed, len(items)):
            identity, _, _, inputs = items[i]
            conv = render_conversation(adapter, predict, inputs)
            rendered = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort="low", preserve_thinking=False)
            bound = len(rendered.encode("utf-8"))
            if bound + config["max_tokens"] <= 49152:
                max_bound = max(max_bound, bound)
            else:
                count = len(tokenizer.encode(rendered, add_special_tokens=False))
                if count > max_exact:
                    max_exact, max_identity = count, identity
                if count + config["max_tokens"] > 49152:
                    overflowing.append(dict(offset=i, prompt_tokens=count, identity=identity))
            if (i + 1) % 5000 == 0:
                print(json.dumps(dict(part=part, checked=i+1, overflows=len(overflowing))), flush=True)
        capacity = max(49152, math.ceil((max(max_bound, max_exact) + config["max_tokens"]) / 8192) * 8192)
        assert capacity <= 262144, f"Prompt exceeds native context: {capacity}"
        target.mkdir(parents=True, exist_ok=True)
        checkpoints = target / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        copied = []
        for src in sorted((old / "checkpoints").glob("*.json")):
            dst = checkpoints / src.name
            sha = hashlib.sha256(src.read_bytes()).hexdigest()
            if dst.exists():
                assert hashlib.sha256(dst.read_bytes()).hexdigest() == sha
            else:
                shutil.copy2(src, dst)
            assert hashlib.sha256(dst.read_bytes()).hexdigest() == sha
            copied.append(dict(name=src.name, sha256=sha))
        assert run_part(items, config, output_dir=target, classify=None, expected_count=25000, dry_run=True) == check
        result = dict(status="ready", source=str(source), original_output=str(old), output=str(target), resumed_records=completed, expected_records=25000, remaining_records=25000-completed, binding_sha256=check["binding_sha256"], max_model_len=capacity, previous_max_model_len=49152, max_exact_long_prompt_tokens=max_exact, max_short_prompt_byte_bound=max_bound, max_prompt_identity=max_identity, overflowing_prompts=overflowing, checkpoint_hashes=copied, judge_prompt_and_sampling_unchanged=True)
        (target / "preflight.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k not in ("overflowing_prompts", "checkpoint_hashes")}), flush=True)

if __name__ == "__main__":
    main()
