from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

from moe_exp.jsonl import iter_jsonl
from moe_exp.moe_guiding.run import load_prompts
from .calibration import fit, problem_key, validate_policy

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_reports(reports, policy, condition):
    if not reports:
        raise RuntimeError("No worker diagnostics")
    for report in reports:
        if report["condition"] != condition:
            raise RuntimeError("Worker condition mismatch")
        expected = set(policy["layers"]) if condition == "guided" else set()
        if set(report["layers"]) != expected:
            raise RuntimeError("Missing routing hooks")
        if any(s["token_evaluations"] == 0 for s in report["layers"].values()):
            raise RuntimeError("Native backend bypassed the identity gate hook")


def generate(args):
    policy = json.loads(args.policy.read_text())
    validate_policy(policy, args.model)
    rows = load_prompts(args.prompts)
    if not math.isfinite(args.strength):
        raise ValueError("strength must be finite")
    if any(value < 1 for value in (args.max_tokens, args.max_model_len,
                                   args.max_num_seqs, args.tensor_parallel_size)):
        raise ValueError("Token limits, concurrency and tensor parallel size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("GPU memory utilization must be in (0,1]")
    overlaps = set(policy["calibration_problems"]) & {problem_key(row) for row in rows}
    if overlaps and not args.allow_calibration_overlap:
        raise ValueError("Evaluation overlaps calibration problems; split by problem or explicitly "
                         "use --allow-calibration-overlap for an exploratory in-sample run")
    from moe_exp.correlation_pipeline.model_profiles import model_profile

    engine = dict(model=args.model, revision=args.revision, tokenizer_revision=args.revision,
                  dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
                  max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  seed=args.seed, enforce_eager=True, enable_prefix_caching=False,
                  generation_config="vllm",
                  language_model_only=model_profile(args.model).language_model_only,
                  worker_extension_cls="moe_exp.moe_identity_guiding.routing.IdentityWorkerExtension")
    sampling = dict(max_tokens=args.max_tokens, temperature=args.temperature,
                    top_p=args.top_p, top_k=(-1 if getattr(args, "top_k", 0) == 0 else args.top_k),
                    seed=args.seed)
    from .sampling import request_sampling
    per_request = request_sampling(rows, sampling,
                                   require_original=getattr(args, "require_original_sampling", False),
                                   model=args.model)
    manifest = dict(experiment="moe_identity_guiding", status="running", condition=args.condition,
                    strength=args.strength, policy=policy, policy_sha256=digest(args.policy),
                    prompts_sha256=digest(args.prompts), engine_args=engine, sampling_args=sampling,
                    calibration_overlap=sorted(overlaps), token_scope="prefill_and_decode",
                    scoring_contract="correlation_pipeline.score_completion",
                    seed_strategy="original_seed_plus_offset_or_id_hash_v1",
                    versions={name: version(name) for name in ("torch", "vllm", "transformers")})
    from moe_exp.moe_identity_guiding.execution import execute
    execute(args, manifest, rows, per_request, "identity", check_reports)


def compare(baseline: Path, guided: Path):
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (baseline, guided)]
    for manifest, condition in zip(manifests, ("baseline", "guided")):
        if manifest["status"] != "complete" or manifest["condition"] != condition:
            raise ValueError("Expected complete baseline and guided runs")
    for key in ("prompts_sha256", "policy_sha256", "engine_args", "sampling_args", "versions", "scoring_contract"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Runs differ in {key}; use matched conditions")
    if "reused_baseline" in manifests[0]:
        from .execution import matching_baseline, file_hash
        provenance = manifests[0]["reused_baseline"]
        if (not matching_baseline(provenance["manifest"], manifests[0])
                or file_hash(baseline / "generations.jsonl") != provenance["generations_sha256"]):
            raise ValueError("Reused baseline provenance or completion hash differs")
    groups = []
    for path in (baseline, guided):
        rows = list(iter_jsonl(path / "generations.jsonl"))
        group = {row["id"]: row for row in rows}
        if len(group) != len(rows) or not rows:
            raise ValueError("Empty or duplicate comparison IDs")
        groups.append(group)
    if groups[0].keys() != groups[1].keys():
        raise ValueError("Baseline and guided runs must contain exactly the same IDs")
    for key in groups[0]:
        a, b = groups[0][key], groups[1][key]
        if a.get("sampling_args") != b.get("sampling_args"):
            raise ValueError("Paired per-attempt sampling settings differ")
        if a["input"] != b["input"] or a["prompt_token_ids"] != b["prompt_token_ids"]:
            raise ValueError("Paired prompts differ")
        if any(type(row["is_correct"]) is not bool for row in (a, b)):
            raise ValueError("All comparison rows need scored gold answers")
    result = {"num_examples": len(groups[0]),
              "calibration_overlap": manifests[0]["calibration_overlap"]}
    for name, group in zip(("baseline", "guided"), groups):
        result[name] = {
            "accuracy": sum(r["is_correct"] for r in group.values()) / len(group),
            "mean_generated_tokens": sum(r["generated_token_count"] for r in group.values()) / len(group),
            "truncated": sum(r["finish_reason"] == "length" for r in group.values())}
    result["accuracy_delta"] = result["guided"]["accuracy"] - result["baseline"]["accuracy"]
    result["mean_generated_tokens_delta"] = (
        result["guided"]["mean_generated_tokens"] - result["baseline"]["mean_generated_tokens"])
    result["wrong_to_right"] = sum(not groups[0][k]["is_correct"] and groups[1][k]["is_correct"]
                                   for k in groups[0])
    result["right_to_wrong"] = sum(groups[0][k]["is_correct"] and not groups[1][k]["is_correct"]
                                   for k in groups[0])
    return result


def main():
    parser = argparse.ArgumentParser(description="MoE expert-identity accuracy guiding")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Split scored correlation traces by problem")
    prep.add_argument("--traces", type=Path, nargs="+", required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--calibration-fraction", type=float, default=.7)
    prep.add_argument("--seed", type=int, default=42)
    global_prep = sub.add_parser("prepare-global", help="Prepare a global policy and full-corpus holdout")
    global_prep.add_argument("--generations", type=Path, nargs="+", required=True)
    global_prep.add_argument("--traces", type=Path, nargs="+", required=True)
    global_prep.add_argument("--output-dir", type=Path, required=True)
    global_prep.add_argument("--calibration-fraction", type=float, default=.7)
    global_prep.add_argument("--seed", type=int, default=42)
    sampling_prep = sub.add_parser("prepare-sampling", help="Attach original per-attempt sampling metadata")
    sampling_prep.add_argument("--prompts", type=Path, required=True)
    sampling_prep.add_argument("--generations", type=Path, nargs="+", required=True)
    sampling_prep.add_argument("--output", type=Path, required=True)
    fit_parser = sub.add_parser("fit", help="Learn a frozen policy from scored routing traces")
    fit_parser.add_argument("--traces", type=Path, nargs="+", required=True)
    fit_parser.add_argument("--tensor-base-dir", type=Path, default=Path("."))
    fit_parser.add_argument("--model", default=DEFAULT_MODEL,
                            help="Target checkpoint whose expert IDs the tensors represent")
    fit_parser.add_argument("--num-experts", type=int, default=None)
    fit_parser.add_argument("--top-k", type=int, default=None)
    fit_parser.add_argument("--min-support", type=int, default=4)
    fit_parser.add_argument("--max-experts", type=int, default=8)
    fit_parser.add_argument("--output", type=Path, required=True)
    gen = sub.add_parser("generate")
    from moe_exp.moe_identity_guiding.execution import add_execution_args
    add_execution_args(gen)
    gen.add_argument("--model", default=DEFAULT_MODEL)
    gen.add_argument("--revision")
    gen.add_argument("--policy", type=Path, required=True)
    gen.add_argument("--prompts", type=Path, required=True)
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--condition", choices=("baseline", "guided"), required=True)
    gen.add_argument("--strength", type=float, default=1.0,
                     help="Signed router bias multiplier; negative values penalize favored experts")
    gen.add_argument("--allow-calibration-overlap", action="store_true")
    gen.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    # Match correlation_pipeline generation: completion budget plus prompt headroom.
    gen.add_argument("--max-tokens", type=int, default=32768)
    gen.add_argument("--max-model-len", type=int, default=49152)
    gen.add_argument("--max-num-seqs", type=int, default=1)
    gen.add_argument("--tensor-parallel-size", type=int, default=1)
    gen.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    gen.add_argument("--temperature", type=float, default=0.6)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--require-original-sampling", action="store_true")
    gen.add_argument("--seed", type=int, default=0, help="Offset added to original per-attempt seeds")
    comp = sub.add_parser("compare")
    comp.add_argument("--baseline", type=Path, required=True)
    comp.add_argument("--guided", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            from .prepare import prepare
            print(json.dumps(prepare(args.traces, args.output_dir,
                                     args.calibration_fraction, args.seed), indent=2))
        elif args.command == "prepare-global":
            from .prepare import prepare_global
            print(json.dumps(prepare_global(args.generations, args.traces, args.output_dir,
                                            args.calibration_fraction, args.seed), indent=2))
        elif args.command == "prepare-sampling":
            from .sampling import prepare_sampling
            print(json.dumps(prepare_sampling(args.prompts, args.generations, args.output), indent=2))
        elif args.command == "fit":
            from .profiles import expert_defaults
            if args.num_experts is None or args.top_k is None:
                experts, top_k = expert_defaults(args.model)
                args.num_experts = experts if args.num_experts is None else args.num_experts
                args.top_k = top_k if args.top_k is None else args.top_k
            rows = [row for path in args.traces for row in iter_jsonl(path)]
            policy = fit(rows, model=args.model, num_experts=args.num_experts, top_k=args.top_k,
                         tensor_base_dir=args.tensor_base_dir, min_support=args.min_support,
                         max_experts=args.max_experts)
            policy["sources"] = [{"path": str(p), "sha256": digest(p)} for p in args.traces]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as handle:
                json.dump(policy, handle, indent=2)
            print(f"Saved policy to {args.output}")
        elif args.command == "generate":
            generate(args)
            print(f"Saved run to {args.output_dir}")
        else:
            print(json.dumps(compare(args.baseline, args.guided), indent=2))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"moe_identity_guiding: {error}\n")


if __name__ == "__main__":
    main()
