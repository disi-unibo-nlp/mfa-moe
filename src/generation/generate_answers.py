"""Generate benchmark traces of one model with its model-card sampler, through a running vLLM server.

Inputs: --model (the Hugging Face id of the model, e.g. openai/gpt-oss-20b), --output-dir, --base-url of the
server, --datasets (default: the six math benchmarks) and, to split the work over jobs, --num-shards / --shard-index.
Command: sbatch scripts/generation/generate_answers.sbatch HF_MODEL DATASET SHARD NSHARDS OUT_DIR
Outputs, in --output-dir/<served model>/<dataset>/: generation_shards/ (one JSON per attempt; a finished attempt
is reused when its contract hash still matches, so an interrupted job resumes) and, once the dataset is
complete, traces.jsonl and manifest.json. A sharded run writes only its shards; run once more with
--assemble-only (no server needed) to collect them into traces.jsonl. Each run also records its datasets in
--output-dir/<served model>/summary.json, next to the datasets that earlier runs recorded there.
Every trace keeps the exact prompt and completion token IDs (metadata.token_replay) for the forward replay.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
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

from generation.benchmarks import BENCHMARKS, DEFAULT_BENCHMARKS, generation_messages, sample_variant
from generation.client import generate_completion, slurm_deadline
from generation.model_profiles import CARD_PROFILES, load_tokenizer
from generation.scoring import score_completion_detailed
from models.roster import resolve
from models.token_replay import TOKEN_REPLAY_VERSION, make_token_replay
from schemas import TraceRecord
from store import write_jsonl

logger = logging.getLogger(__name__)
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")
_SCORING_CONTRACT_VERSION = 3
_GENERATION_CONTRACT_VERSION = 3
SEED = 20260925  # an attempt uses SEED + 1000 * problem index + sample index
TIMEOUT_S = 21600  # wait for one reply


@lru_cache(maxsize=None)
def vllm_version() -> str:
    """Installed vLLM release without its build tag (0.29.0+cu129 -> 0.29.0); recorded in the generation contract."""
    return importlib.metadata.version("vllm").split("+")[0]


def _model_slug(model: str) -> str:
    return _SAFE_FILENAME.sub("--", model).strip("-") or "model"


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


def _load_reusable_shard(path: Path, contract: dict[str, Any]) -> TraceRecord | None:
    """The finished attempt saved at `path`, if it was made for this very example and generation contract."""
    try:
        trace = TraceRecord(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):  # no file, bad JSON, or a record that does not validate
        return None
    same = all(trace.metadata.get(key) == contract[key] for key in ("example_sha256", "generation_sha256"))
    return trace if same else None


def _job_contract(
    dataset: str, example: dict[str, Any], example_index: int, sample_id: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Example and generation contract of one attempt (shared by generation and assembly).

    Everything that can change the output is hashed (sampler, context, server settings, vLLM release, seed);
    the server address is not, so a resumed job may run on a new port.
    """
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
    card = CARD_PROFILES[args.profile]
    generation_config = copy.deepcopy({
        "generation_contract_version": _GENERATION_CONTRACT_VERSION,
        "profile": args.profile,
        "served_model": card["model"],
        "revision": card["revision"],
        "max_tokens": card["max_tokens"],
        "max_model_len": card["max_model_len"],
        "sampler": card["sampler"],
        "reasoning_effort": card["reasoning_effort"],
        "chat_template_kwargs": card["template_kwargs"],
        "speculation": card["speculation"],
        "weights": card["weights"],
        "attention": card["attention"],
        "language_model_only": card["language_model_only"],
        "server_extra_args": card["extra_args"],
        "kv_cache_dtype": "bfloat16",
        "vllm_version": vllm_version(),
        "card": card["card"],
        "seed": SEED + example_index * 1000 + sample_id,
        "scoring_contract_version": _SCORING_CONTRACT_VERSION,
        "token_replay_version": TOKEN_REPLAY_VERSION,
    })  # a copy: the profile's own dicts are never shared with a trace
    return dict(example=example, messages=messages, example_sha256=example_sha256,
                generation_config=generation_config, generation_sha256=_digest(generation_config))


def _generate_trace(
    dataset: str,
    args: argparse.Namespace,
    deadline: float | None,
    example_index: int,
    example: dict[str, Any],
    sample_id: int,
    shard_path: Path,
) -> TraceRecord | None:
    spec, card = BENCHMARKS[dataset], CARD_PROFILES[args.profile]
    contract = _job_contract(dataset, example, example_index, sample_id, args)
    example, messages = contract["example"], contract["messages"]
    reusable = _load_reusable_shard(shard_path, contract)
    if reusable is not None:
        return reusable
    if deadline and time.time() > deadline:
        return None  # drain: leave for the resumed array task

    completion = generate_completion(
        base_url=args.base_url,
        model=card["model"],
        messages=messages,
        max_tokens=card["max_tokens"],
        sampler=card["sampler"],
        reasoning_effort=card["reasoning_effort"],
        seed=contract["generation_config"]["seed"],
        timeout=TIMEOUT_S,
        chat_template_kwargs=card["template_kwargs"],
    )
    cot_text, replay = make_token_replay(
        load_tokenizer(args.profile), completion.prompt_token_ids, completion.token_ids,
    )
    # A reasoning model returns its reasoning and its final answer in separate fields. Scoring their
    # concatenation could turn an intermediate answer in a length-truncated reasoning trace into a nominally
    # correct final response, so the answer alone is scored.
    scored = score_completion_detailed(example, answer_type=spec.answer_type, model_text=completion.content)
    prompt_tokens = completion.usage.get("prompt_tokens")
    room = card["max_model_len"] - prompt_tokens if prompt_tokens else None
    requested = card["max_tokens"]
    effective_cap = min(requested, room) if (requested and room) else (requested or room)
    source_problem_id = str(example["problem_id"])
    trace = TraceRecord(
        dataset=dataset,
        problem_id=f"{source_problem_id}__sample_{sample_id:02d}",
        source_problem_id=source_problem_id,
        sample_id=sample_id,
        prompt=str(example["prompt"]),
        generation_messages=messages,
        gold_answer=str(example.get("gold_answer") or ""),
        model_id=card["model"],
        model_answer=scored["model_answer"],
        is_correct=scored["is_correct"],
        cot_text=cot_text,
        scoring_method=scored["method"],
        metadata={
            **example["metadata"],
            "benchmark_source": spec.source,
            "benchmark_split": spec.split,
            "evaluation_metric": f"avg@{spec.default_samples}" if spec.default_samples > 1 else "pass@1",
            "answer_type": spec.answer_type,
            "example_sha256": contract["example_sha256"],
            "generation_sha256": contract["generation_sha256"],
            "generation_config": contract["generation_config"],
            "finish_reason": completion.finish_reason,
            "usage": completion.usage,
            "assistant_content": completion.content,
            "reasoning_content": completion.reasoning_content,
            "scoring_contract_version": _SCORING_CONTRACT_VERSION,
            "scoring_input": "assistant_content" if completion.reasoning_content else "combined_assistant_text",
            "scoring_status": scored["math_verify_status"],
            "termination": completion.finish_reason,
            "effective_max_tokens": effective_cap,
            "token_replay": replay,
        },
    )
    _write_json_atomic(trace.model_dump(mode="json"), shard_path)
    return trace


def generate_dataset(dataset: str, args: argparse.Namespace, deadline: float | None = None) -> dict[str, Any]:
    spec, card = BENCHMARKS[dataset], CARD_PROFILES[args.profile]
    examples = [example for example in spec.loader(args.max_items) if example.get("prompt")]
    if not examples:
        raise RuntimeError(f"Benchmark {dataset!r} produced zero usable examples")
    dataset_dir = args.output_dir / _model_slug(card["model"]) / dataset
    shard_dir = dataset_dir / "generation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # one job per attempt: (problem index, problem, sample index, file of the finished attempt)
    jobs = [
        (example_index, example, sample_id,
         shard_dir / f"{_SAFE_FILENAME.sub('_', str(example['problem_id']))}__sample_{sample_id:02d}.json")
        for example_index, example in enumerate(examples)
        for sample_id in range(spec.default_samples)
    ]
    if args.assemble_only:
        traces = [_load_reusable_shard(path, _job_contract(dataset, example, index, sample_id, args))
                  for index, example, sample_id, path in jobs]
        missing = [job[3].name for job, trace in zip(jobs, traces) if trace is None]
        if missing:
            raise RuntimeError(f"{dataset}: {len(missing)} of {len(jobs)} traces missing or stale, "
                               f"e.g. {missing[:3]}")
    else:
        load_tokenizer(args.profile)  # once, before the concurrent requests
        if args.num_shards > 1:
            jobs = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_index]
        with ThreadPoolExecutor(max_workers=card["max_num_seqs"]) as executor:
            traces = list(
                tqdm(
                    executor.map(lambda job: _generate_trace(dataset, args, deadline, *job), jobs),
                    total=len(jobs),
                    desc=f"Generating {dataset}",
                )
            )
        pending = sum(trace is None for trace in traces)
        if pending or args.num_shards > 1:
            summary = {"dataset": dataset, "status": "drained" if pending else "shard_complete",
                       "shard_index": args.shard_index, "num_shards": args.num_shards,
                       "jobs": len(jobs), "pending": pending}
            _write_json_atomic(
                summary, dataset_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json")
            if pending:
                raise SystemExit(f"{dataset}: {pending} traces left for the resumed task (deadline)")
            return summary

    traces.sort(key=lambda trace: (trace.source_problem_id or trace.problem_id, trace.sample_id))
    trace_path = dataset_dir / "traces.jsonl"
    write_jsonl(traces, trace_path)
    scored = [trace.is_correct for trace in traces if trace.is_correct is not None]
    summary = {
        "dataset": dataset,
        "status": "complete",
        "problems": len(examples),
        "samples_per_problem": spec.default_samples,
        "traces": len(traces),
        "scored_traces": len(scored),
        "correct_traces": sum(scored),
        "accuracy": sum(scored) / len(scored) if scored else None,
        "model": card["model"],
        "source": spec.source,
        "split": spec.split,
        "trace_path": trace_path.as_posix(),
    }
    _write_json_atomic(summary, dataset_dir / "manifest.json")
    return summary


def write_summary(model_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    """Record `summaries` in model_dir/summary.json. The datasets that earlier runs recorded stay there, and a
    dataset that is run again replaces its own entry, so assembling one dataset at a time keeps all of them."""
    path = model_dir / "summary.json"
    recorded = json.loads(path.read_text(encoding="utf-8"))["datasets"] if path.exists() else []
    by_dataset = {summary["dataset"]: summary for summary in recorded}
    by_dataset.update((summary["dataset"], summary) for summary in summaries)
    _write_json_atomic({"status": "complete", "datasets": list(by_dataset.values())}, path)
    return path


def _model_key(name: str) -> str:
    """The short key of a model given as its Hugging Face id (or its short key)."""
    try:
        return resolve(name)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", dest="profile", required=True, type=_model_key, metavar="HF_MODEL",
                        help="Hugging Face id of the model (e.g. openai/gpt-oss-20b); its card in model_profiles.py "
                             "sets the sampler, token cap and recorded server settings")
    parser.add_argument("--datasets", nargs="+", choices=tuple(BENCHMARKS), default=DEFAULT_BENCHMARKS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", help="OpenAI-compatible URL of the running server (not needed to assemble)")
    parser.add_argument("--max-items", type=int, default=None, help="first N problems only (a quick check)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--assemble-only", action="store_true",
                        help="collect finished shards into traces.jsonl without generating")
    parser.add_argument("--deadline-margin-hours", type=float, default=3,
                        help="inside a Slurm job, start no request in the last N hours so running ones can finish")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if not args.assemble_only and not args.base_url:
        parser.error("--base-url is required unless --assemble-only")
    deadline = slurm_deadline(args.deadline_margin_hours)
    summaries = [generate_dataset(dataset, args, deadline) for dataset in args.datasets]
    if args.num_shards > 1 and not args.assemble_only:
        return
    summary_path = write_summary(args.output_dir / _model_slug(CARD_PROFILES[args.profile]["model"]), summaries)
    logger.info("Generation complete: %s", summary_path)


if __name__ == "__main__":
    main()
