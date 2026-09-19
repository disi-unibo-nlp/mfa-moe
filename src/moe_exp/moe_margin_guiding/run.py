from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

from moe_exp.jsonl import iter_jsonl
from moe_exp.moe_guiding.run import _write_json, load_prompts
from .calibration import fit, problem_key, validate_policy
from moe_exp.moe_identity_guiding.run import compare

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
            raise RuntimeError("Native backend bypassed the margin gate hook")


def generate(args):
    policy = json.loads(args.policy.read_text())
    validate_policy(policy, args.model)
    rows = load_prompts(args.prompts)
    if not math.isfinite(args.strength) or not 0 <= args.strength <= 1:
        raise ValueError("strength must be finite and in [0, 1]")
    if any(value < 1 for value in (args.max_tokens, args.max_model_len,
                                   args.max_num_seqs, args.tensor_parallel_size)):
        raise ValueError("Token limits, concurrency and tensor parallel size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("GPU memory utilization must be in (0,1]")
    overlaps = set(policy["calibration_problems"]) & {problem_key(row) for row in rows}
    if overlaps and not args.allow_calibration_overlap:
        raise ValueError("Evaluation overlaps calibration problems; split by problem or explicitly "
                         "use --allow-calibration-overlap for an exploratory in-sample run")
    from vllm import LLM, SamplingParams
    from moe_exp.correlation_pipeline.model_profiles import model_profile
    from moe_exp.correlation_pipeline.scoring import score_completion

    engine = dict(model=args.model, revision=args.revision, tokenizer_revision=args.revision,
                  dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
                  max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  seed=args.seed, enforce_eager=True, enable_prefix_caching=False,
                  generation_config="vllm",
                  language_model_only=model_profile(args.model).language_model_only,
                  worker_extension_cls="moe_exp.moe_margin_guiding.routing.MarginWorkerExtension")
    sampling = dict(max_tokens=args.max_tokens, temperature=args.temperature,
                    top_p=args.top_p, top_k=(-1 if getattr(args, "top_k", 0) == 0 else args.top_k),
                    seed=args.seed)
    from moe_exp.moe_identity_guiding.sampling import request_sampling
    per_request = request_sampling(rows, sampling,
                                   require_original=getattr(args, "require_original_sampling", False),
                                   model=args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = dict(experiment="moe_margin_guiding", status="running", condition=args.condition,
                    strength=args.strength, policy=policy, policy_sha256=digest(args.policy),
                    prompts_sha256=digest(args.prompts), engine_args=engine, sampling_args=sampling,
                    calibration_overlap=sorted(overlaps), token_scope="prefill_and_decode",
                    scoring_contract="correlation_pipeline.score_completion",
                    seed_strategy="original_seed_plus_offset_or_id_hash_v1",
                    versions={name: version(name) for name in ("torch", "vllm", "transformers")})
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
    try:
        llm = LLM(**engine)
        tokenizer = llm.get_tokenizer()
        rendered = []
        for row in rows:
            messages = row.get("generation_messages") or row.get("messages")
            if messages is None:
                messages = []
                if row.get("system_prompt"):
                    messages.append({"role": "system", "content": row["system_prompt"]})
                messages.append({"role": "user", "content": row["prompt"]})
            rendered.append(tokenizer.apply_chat_template(messages, tokenize=False,
                                                          add_generation_prompt=True,
                                                          **(row.get("original_generation_config", {}).get(
                                                              "chat_template_kwargs") or {})))
        inputs = [{"prompt_token_ids": tokenizer.encode(p, add_special_tokens=False)}
                  for p in rendered]
        llm.collective_rpc("margin_configure", kwargs=dict(
            policy=policy, strength=args.strength, condition=args.condition))
        outputs = llm.generate(inputs, [SamplingParams(**p) for p in per_request])
        reports = llm.collective_rpc("margin_diagnostics")
        manifest["routing_diagnostics"] = reports
        check_reports(reports, policy, args.condition)
        if len(outputs) != len(rows):
            raise RuntimeError("Generation count does not match prompts")
        with (args.output_dir / "generations.jsonl").open("x") as handle:
            for row, prompt, output, request in zip(rows, rendered, outputs, per_request, strict=True):
                completion = output.outputs[0]
                answer, correct, method = score_completion(
                    row, answer_type=row.get("answer_type", "math"), model_text=completion.text)
                record = dict(id=row["id"], input=row, rendered_prompt=prompt,
                              condition=args.condition, text=completion.text, sampling_args=request,
                              model_answer=answer, is_correct=correct, scoring_method=method,
                              prompt_token_ids=output.prompt_token_ids,
                              generated_token_ids=list(completion.token_ids),
                              generated_token_count=len(completion.token_ids),
                              finish_reason=completion.finish_reason)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description="MoE routing-margin accuracy guiding")
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
    sampling_prep = sub.add_parser("prepare-sampling")
    sampling_prep.add_argument("--prompts", type=Path, required=True)
    sampling_prep.add_argument("--generations", type=Path, nargs="+", required=True)
    sampling_prep.add_argument("--output", type=Path, required=True)
    fit_parser = sub.add_parser("fit", help="Learn a frozen policy from scored routing traces")
    fit_parser.add_argument("--traces", type=Path, nargs="+", required=True)
    fit_parser.add_argument("--tensor-base-dir", type=Path, default=Path("."))
    fit_parser.add_argument("--model", default=DEFAULT_MODEL,
                            help="Target generation checkpoint")
    fit_parser.add_argument("--num-experts", type=int, default=None)
    fit_parser.add_argument("--top-k", type=int, default=None)
    fit_parser.add_argument("--min-support", type=int, default=4)
    fit_parser.add_argument("--bins", type=int, default=5)
    fit_parser.add_argument("--metric", choices=("router_boundary_margin", "router_margin"),
                            default="router_boundary_margin")
    fit_parser.add_argument("--output", type=Path, required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--model", default=DEFAULT_MODEL)
    gen.add_argument("--revision")
    gen.add_argument("--policy", type=Path, required=True)
    gen.add_argument("--prompts", type=Path, required=True)
    gen.add_argument("--output-dir", type=Path, required=True)
    gen.add_argument("--condition", choices=("baseline", "guided"), required=True)
    gen.add_argument("--strength", type=float, default=1.0)
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
            from moe_exp.moe_identity_guiding.sampling import prepare_sampling
            print(json.dumps(prepare_sampling(args.prompts, args.generations, args.output), indent=2))
        elif args.command == "fit":
            from moe_exp.moe_identity_guiding.profiles import expert_defaults
            if args.num_experts is None or args.top_k is None:
                experts, top_k = expert_defaults(args.model)
                args.num_experts = experts if args.num_experts is None else args.num_experts
                args.top_k = top_k if args.top_k is None else args.top_k
            rows = [row for path in args.traces for row in iter_jsonl(path)]
            policy = fit(rows, model=args.model, num_experts=args.num_experts, top_k=args.top_k,
                         tensor_base_dir=args.tensor_base_dir, min_support=args.min_support,
                         bins=args.bins, metric=args.metric)
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
        parser.exit(1, f"moe_margin_guiding: {error}\n")


if __name__ == "__main__":
    main()
