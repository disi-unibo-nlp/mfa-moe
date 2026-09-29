from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from tqdm import tqdm

from moe_exp.analysis.classifier import classify_trace
from moe_exp.analysis.step_splitter import split_steps
from moe_exp.correlation_pipeline.benchmarks import (
    BENCHMARKS,
    DEFAULT_BENCHMARKS,
    generation_messages,
    sample_variant,
)
from moe_exp.correlation_pipeline.client import generate_completion
from moe_exp.correlation_pipeline.defaults import (
    DEFAULT_FORWARD_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MTP_MODEL,
)
from moe_exp.correlation_pipeline.scoring import score_completion_detailed
from moe_exp.correlation_pipeline.model_profiles import CARD_PROFILES, model_profile
from moe_exp.schemas import TraceRecord
from moe_exp.models.token_replay import TOKEN_REPLAY_VERSION, make_token_replay
from moe_exp.utils import write_jsonl

logger = logging.getLogger(__name__)
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")
_SCORING_CONTRACT_VERSION = 3
_GENERATION_CONTRACT_VERSION = 3


@lru_cache(maxsize=8)
def _replay_tokenizer(model: str, revision: str | None = None):
    from transformers import AutoTokenizer
    # a pinned revision resolves offline even when the hub cache has no refs/ entry
    return AutoTokenizer.from_pretrained(model, revision=revision) if revision else \
        AutoTokenizer.from_pretrained(model)


def _tokenizer_ref(args: argparse.Namespace) -> tuple[str, str | None]:
    """(name or local path, pinned revision or None) for the replay tokenizer."""
    if getattr(args, "tokenizer", None):
        return args.tokenizer, None
    card = CARD_PROFILES.get(getattr(args, "profile", None) or "")
    if card is not None:
        if card["weights"]:
            return card["weights"], None
        return card["tokenizer"] or card["model"], card["revision"]
    return args.model, None


def _model_slug(model: str) -> str:
    return _SAFE_FILENAME.sub("--", model).strip("-") or "local-llamacpp"


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _scoring_text(completion: Any) -> tuple[str, str]:
    """Return the model's submitted answer rather than private reasoning.

    Reasoning-capable llama.cpp models expose private reasoning and the final
    answer in separate fields.  Scoring their concatenation can turn an
    intermediate answer in a length-truncated reasoning trace into a nominally
    correct final response.  Models without a separate reasoning field retain
    the legacy behavior because their content is the complete response.
    """
    if completion.reasoning_content:
        return completion.content, "assistant_content"
    return completion.text, "combined_assistant_text"


def _load_reusable_shard(
    path: Path,
    *,
    example_sha256: str,
    generation_sha256: str,
) -> TraceRecord | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        trace = TraceRecord(**payload)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if (
        trace.metadata.get("example_sha256") != example_sha256
        or trace.metadata.get("generation_sha256") != generation_sha256
    ):
        return None
    return trace


def _load_assembled(job: dict[str, Any]) -> TraceRecord | None:
    """Load a finished shard file for assembly, re-verifying its example and contract hashes."""
    contract = _job_contract(job["dataset"], job["example"], job["example_index"],
                             job["sample_id"], job["args"])
    return _load_reusable_shard(job["shard_path"], example_sha256=contract["example_sha256"],
                                generation_sha256=contract["generation_sha256"])


def _job_contract(
    dataset: str, example: dict[str, Any], example_index: int, sample_id: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Example and generation contract of one attempt (shared by generation and assembly)."""
    example = sample_variant(dataset, example, sample_id)
    messages = generation_messages(example)
    example_sha256 = _digest(
        {
            "dataset": dataset,
            "problem_id": example["problem_id"],
            "prompt": example["prompt"],
            "gold_answer": example.get("gold_answer"),
            "messages": messages,
        }
    )
    sample_seed = args.seed + example_index * 1000 + sample_id
    card = CARD_PROFILES[args.profile] if getattr(args, "profile", None) else None
    extra_sampling: dict[str, Any] = {}
    if card is None:
        generation_config = {
            "base_url": args.base_url,
            "served_model": args.model,
            "target_model_id": args.target_model_id,
            "draft_model_id": args.draft_model_id,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": sample_seed,
            "scoring_contract_version": _SCORING_CONTRACT_VERSION,
        }
        template_kwargs = (
            {"enable_thinking": True} if model_profile(args.model).family == "gemma4" else {}
        )
        if template_kwargs:
            generation_config["chat_template_kwargs"] = template_kwargs
    else:
        # v3 card contract: every sampler/context/runtime field that can change the
        # output is hashed; the server address is not (resume must survive new ports).
        sampler = card["sampler"]
        extra_sampling = {key: sampler.get(key) for key in
                          ("min_p", "presence_penalty", "repetition_penalty")}
        extra_sampling["reasoning_effort"] = card["reasoning_effort"]
        template_kwargs = dict(card["template_kwargs"])
        generation_config = {
            "generation_contract_version": _GENERATION_CONTRACT_VERSION,
            "profile": args.profile,
            "served_model": card["model"],
            "revision": card["revision"],
            "max_tokens": card["max_tokens"],
            "max_model_len": card["max_model_len"],
            "sampler": dict(sampler),
            "reasoning_effort": card["reasoning_effort"],
            "chat_template_kwargs": template_kwargs,
            "speculation": card["speculation"],
            "weights": card["weights"],
            "attention": card["attention"],
            "language_model_only": card["language_model_only"],
            "server_extra_args": list(card["extra_args"]),
            "kv_cache_dtype": "bfloat16",
            "vllm_version": args.vllm_version,
            "card": card["card"],
            "seed": sample_seed,
            "scoring_contract_version": _SCORING_CONTRACT_VERSION,
        }
    save_token_ids = getattr(args, "save_token_ids", False)
    if save_token_ids:
        generation_config["token_replay_version"] = TOKEN_REPLAY_VERSION
    generation_sha256 = _digest(generation_config)
    return dict(example=example, messages=messages, example_sha256=example_sha256,
                sample_seed=sample_seed, card=card, extra_sampling=extra_sampling,
                template_kwargs=template_kwargs, generation_config=generation_config,
                generation_sha256=generation_sha256, save_token_ids=save_token_ids)


def _generate_trace(
    *,
    dataset: str,
    example: dict[str, Any],
    example_index: int,
    sample_id: int,
    samples_per_problem: int,
    spec: Any,
    args: argparse.Namespace,
    shard_path: Path,
) -> TraceRecord:
    contract = _job_contract(dataset, example, example_index, sample_id, args)
    example, messages = contract["example"], contract["messages"]
    example_sha256, sample_seed = contract["example_sha256"], contract["sample_seed"]
    card, extra_sampling = contract["card"], contract["extra_sampling"]
    template_kwargs, generation_config = contract["template_kwargs"], contract["generation_config"]
    generation_sha256, save_token_ids = contract["generation_sha256"], contract["save_token_ids"]
    reusable = _load_reusable_shard(
        shard_path,
        example_sha256=example_sha256,
        generation_sha256=generation_sha256,
    )
    if reusable is not None:
        return reusable
    deadline = getattr(args, "deadline_epoch", None)
    if deadline and time.time() > deadline:
        return None  # drain: leave for the resumed array task

    completion = generate_completion(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        messages=messages,
        max_tokens=card["max_tokens"] if card else args.max_tokens,
        temperature=card["sampler"]["temperature"] if card else args.temperature,
        top_p=card["sampler"]["top_p"] if card else args.top_p,
        top_k=card["sampler"]["top_k"] if card else args.top_k,
        seed=sample_seed,
        timeout=args.timeout,
        max_retries=args.max_retries,
        **({"return_token_ids": True} if save_token_ids else {}),
        **({"chat_template_kwargs": template_kwargs} if template_kwargs else {}),
        **({"extra_sampling": extra_sampling} if extra_sampling else {}),
    )
    cot_text = completion.text
    replay = None
    if save_token_ids:
        cot_text, replay = make_token_replay(
            _replay_tokenizer(*_tokenizer_ref(args)), completion.prompt_token_ids,
            completion.token_ids,
        )
    scoring_text, scoring_input = _scoring_text(completion)
    scored = score_completion_detailed(
        example,
        answer_type=spec.answer_type,
        model_text=scoring_text,
    )
    model_answer, is_correct, scoring_method = (
        scored["model_answer"], scored["is_correct"], scored["method"])
    prompt_tokens = (completion.usage or {}).get("prompt_tokens")
    if card is not None:
        room = card["max_model_len"] - prompt_tokens if prompt_tokens else None
        requested = card["max_tokens"]
        effective_cap = min(requested, room) if (requested and room) else (requested or room)
    else:
        effective_cap = args.max_tokens
    steps = split_steps(cot_text)
    source_problem_id = str(example["problem_id"])
    problem_id = f"{source_problem_id}__sample_{sample_id:02d}"
    source_metadata = dict(example.get("metadata") or {})
    source_metadata.update(
        {
            "benchmark_source": spec.source,
            "benchmark_split": spec.split,
            "evaluation_metric": (
                f"avg@{samples_per_problem}" if samples_per_problem > 1 else "pass@1"
            ),
            "answer_type": spec.answer_type,
            "example_sha256": example_sha256,
            "generation_sha256": generation_sha256,
            "generation_config": generation_config,
            "finish_reason": completion.finish_reason,
            "usage": completion.usage,
            "assistant_content": completion.content,
            "reasoning_content": completion.reasoning_content,
            "scoring_contract_version": _SCORING_CONTRACT_VERSION,
            "scoring_input": scoring_input,
            "scoring_status": scored["math_verify_status"],
            "termination": completion.finish_reason,
            "effective_max_tokens": effective_cap,
        }
    )
    if replay is not None:
        source_metadata["token_replay"] = replay
    if example.get("first_error_step") is not None:
        source_metadata["reference_first_error_step"] = example["first_error_step"]
    if example.get("solution_is_correct") is not None:
        source_metadata["reference_solution_is_correct"] = example["solution_is_correct"]

    trace = TraceRecord(
        dataset=dataset,
        problem_id=problem_id,
        source_problem_id=source_problem_id,
        sample_id=sample_id,
        prompt=str(example["prompt"]),
        generation_messages=messages,
        gold_answer=str(example.get("gold_answer") or ""),
        model_id=args.model,
        model_answer=model_answer,
        is_correct=is_correct,
        cot_text=cot_text,
        steps=steps,
        step_labels=classify_trace(steps, cot_text),
        scoring_method=scoring_method,
        metadata=source_metadata,
        task_type="reasoning",
    )
    _write_json_atomic(trace.model_dump(mode="json"), shard_path)
    return trace


def _dataset_summary(
    dataset: str,
    traces: list[TraceRecord],
    *,
    samples_per_problem: int,
) -> dict[str, Any]:
    scored = [trace for trace in traces if trace.is_correct is not None]
    correct = sum(trace.is_correct is True for trace in scored)
    problem_ids = {trace.source_problem_id or trace.problem_id for trace in traces}
    return {
        "dataset": dataset,
        "status": "complete",
        "problems": len(problem_ids),
        "samples_per_problem": samples_per_problem,
        "traces": len(traces),
        "scored_traces": len(scored),
        "correct_traces": correct,
        "accuracy": (correct / len(scored)) if scored else None,
    }


def generate_dataset(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    spec = BENCHMARKS[dataset]
    examples = [example for example in spec.loader(args.max_items) if example.get("prompt")]
    if not examples:
        raise RuntimeError(f"Benchmark {dataset!r} produced zero usable examples")
    if getattr(args, "save_token_ids", False) and not getattr(args, "assemble_only", False):
        _replay_tokenizer(*_tokenizer_ref(args))  # Load once before concurrent requests.
    samples_per_problem = args.samples_per_problem or spec.default_samples
    dataset_dir = args.output_dir / _model_slug(args.model) / dataset
    shard_dir = dataset_dir / "generation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        safe_id = _SAFE_FILENAME.sub("_", str(example["problem_id"]))
        for sample_id in range(samples_per_problem):
            jobs.append(
                {
                    "dataset": dataset,
                    "example": example,
                    "example_index": example_index,
                    "sample_id": sample_id,
                    "samples_per_problem": samples_per_problem,
                    "spec": spec,
                    "args": args,
                    "shard_path": shard_dir / f"{safe_id}__sample_{sample_id:02d}.json",
                }
            )

    num_shards = getattr(args, "num_shards", 1) or 1
    if getattr(args, "assemble_only", False):
        traces = [_load_assembled(job) for job in jobs]
        missing = [job["shard_path"].name for job, trace in zip(jobs, traces) if trace is None]
        if missing:
            raise RuntimeError(f"{dataset}: {len(missing)} of {len(jobs)} traces missing or stale, "
                               f"e.g. {missing[:3]}")
    else:
        if num_shards > 1:
            jobs = [job for index, job in enumerate(jobs) if index % num_shards == args.shard_index]
        if args.workers == 1:
            traces = [
                _generate_trace(**job)
                for job in tqdm(jobs, desc=f"Generating {dataset}")
            ]
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                traces = list(
                    tqdm(
                        executor.map(lambda job: _generate_trace(**job), jobs),
                        total=len(jobs),
                        desc=f"Generating {dataset}",
                    )
                )
        pending = sum(trace is None for trace in traces)
        if pending or num_shards > 1:
            status = "drained" if pending else "shard_complete"
            summary = {"dataset": dataset, "status": status, "shard_index": args.shard_index,
                       "num_shards": num_shards, "jobs": len(jobs), "pending": pending}
            _write_json_atomic(summary, dataset_dir / f"shard_{args.shard_index:02d}_of_{num_shards:02d}.json")
            if pending:
                raise SystemExit(f"{dataset}: {pending} traces left for the resumed task (deadline)")
            return summary

    traces.sort(key=lambda trace: (trace.source_problem_id or trace.problem_id, trace.sample_id))
    trace_path = dataset_dir / "traces.jsonl"
    write_jsonl(traces, trace_path)
    summary = _dataset_summary(
        dataset,
        traces,
        samples_per_problem=samples_per_problem,
    )
    summary.update(
        {
            "model": args.model,
            "target_model_id": args.target_model_id,
            "draft_model_id": args.draft_model_id,
            "source": spec.source,
            "split": spec.split,
            "trace_path": trace_path.as_posix(),
        }
    )
    _write_json_atomic(summary, dataset_dir / "manifest.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate correlation-pipeline benchmark traces with an OpenAI-compatible server"
    )
    parser.add_argument("--datasets", nargs="+", choices=tuple(BENCHMARKS), default=DEFAULT_BENCHMARKS)
    parser.add_argument("--output-dir", type=Path, default=Path("results/correlation_pipeline/generation"))
    parser.add_argument("--base-url", default="http://127.0.0.1:41800/v1")
    parser.add_argument("--api-key", default="local-vllm-key")
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--target-model-id", default=DEFAULT_FORWARD_MODEL)
    parser.add_argument("--draft-model-id", default=DEFAULT_MTP_MODEL)
    parser.add_argument("--save-token-ids", action="store_true",
                        help="Require vLLM token IDs for exact model-native forward replay")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--samples-per-problem",
        type=int,
        default=None,
        help="Override defaults (AIME24/25 and AMC23 use 32; GPQA-D uses 10; others use 1).",
    )
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base sampling seed (SPIRAL uses 0).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--profile", choices=sorted(CARD_PROFILES), default=None,
                        help="Card-sampler v3 contract (overrides model/sampler/max-tokens)")
    parser.add_argument("--tokenizer", default=None, help="Replay tokenizer path/id override")
    parser.add_argument("--vllm-version", default=os.environ.get("VLLM_VERSION_RECORD", "unknown"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--deadline-epoch", type=float, default=None,
                        help="Do not start new requests after this UNIX time (drain mode)")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max-retries", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.samples_per_problem is not None and args.samples_per_problem < 1:
        raise ValueError("--samples-per-problem must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not 0 <= args.shard_index < max(1, args.num_shards):
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.profile:
        args.model = CARD_PROFILES[args.profile]["model"]
    summaries = [generate_dataset(dataset, args) for dataset in args.datasets]
    if args.num_shards > 1 and not args.assemble_only:
        return
    summary_path = args.output_dir / _model_slug(args.model) / "summary.json"
    _write_json_atomic({"status": "complete", "datasets": summaries}, summary_path)
    logger.info("Generation complete: %s", summary_path)


if __name__ == "__main__":
    main()
