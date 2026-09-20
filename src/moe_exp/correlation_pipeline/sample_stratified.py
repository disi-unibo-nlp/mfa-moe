"""Select judge-labelling sentences stratified by correctness and reasoning length.

The sampler reads completed benchmark generations from
``<generation-dir>/<model-slug>/<dataset>/traces.jsonl`` and writes exactly
``--total`` sentence identities (default 100,000) in a layout that the existing
production labelling harness (``annotation_batch.py``) already understands.

Selection policy
----------------

* Only scored traces (``is_correct is not None``) with a
  ``metadata.usage.completion_tokens_details.reasoning_tokens`` value are
  eligible. Truncated/unscored traces are dropped and counted in the manifest.
* ``reasoning_tokens`` is split into four equal-count quartiles *per dataset*
  (ties broken by the stable trace digest), producing the strata
  ``dataset x correct|incorrect x Q1..Q4``.
* Exactly one completion is chosen per source problem (``source_problem_id``,
  falling back to ``problem_id``). Problems with a single eligible attempt are
  fixed; problems with several attempts are assigned by a deterministic greedy
  plus local-swap pass that minimises the maximum deviation between the chosen
  universe's within-dataset cell sentence shares and the full eligible corpus
  shares.
* Cell quotas are proportional to the chosen universe's sentence supply
  (``quota = total * cell_sentences / chosen_sentences``, largest-remainder
  rounding). Inside a cell the quota is spread across problems with an integer
  max-min fair water-filling allocation, so every problem receives the same
  share until its completion's sentences run out. Sentence indices inside a
  completion are drawn uniformly without replacement with a per-stratum seeded
  RNG; because the quota never exceeds the chosen universe's supply, no problem
  needs a second completion.
* Each cell quota is split evenly across ``--parts`` parts (default four) and a
  balancing pass guarantees every part holds exactly ``--part-size`` identities
  (default 25,000). Boundary traces are duplicated across adjacent part roots
  with their part-specific indices.

Outputs
-------

* ``<output-dir>/generation/<model-slug>/<dataset>/traces.jsonl``: combined
  100k selection, one row per chosen completion with
  ``metadata.sentence_selection = {schema_version, manifest_sha256, stratum,
  indices}`` and a per-dataset ``manifest.json``.
* ``<output-dir>/parts/part-XX/generation/<model-slug>/<dataset>/traces.jsonl``:
  self-contained, internally stratified 25k part roots. Point the labelling
  harness at one root per part, e.g. ``annotation_batch.py --trace-root
  <output-dir>/parts/part-00/generation/<model-slug> --part 0 --parts 1
  --part-size 25000 --total 25000 ... --plan-only``.
* ``<output-dir>/sampling_manifest.json``: full audit record (config, seed,
  source manifest hashes, per-cell supplies/quotas, per-problem chosen attempt,
  deviations, part counts, plan SHA-256).

The module is CPU-only and needs no GPU::

    PYTHONPATH=src python -m moe_exp.correlation_pipeline.sample_stratified \
        --generation-dir results/correlation_pipeline/nemotron-nvfp4-dspark/generation \
        --generation-model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
        --output-dir results/correlation_pipeline/nemotron-nvfp4-dspark/reasoning-vllm-v1/sampling-100k-stratified \
        --dry-run

Remove ``--dry-run`` to write the sampled roots. Re-running with the same
options and seed is idempotent; re-running with a different plan into the same
output directory is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from moe_exp.correlation_pipeline.annotation_partition import capped_quotas
from moe_exp.correlation_pipeline.generate import _model_slug, _write_json_atomic
from moe_exp.correlation_pipeline.spans import digest, sentence_spans, trace_digest
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord

DEFAULT_DATASETS = ["math500", "aime24", "aime25", "olympiad", "amc23", "minerva"]
DEFAULT_GENERATION_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
DEFAULT_TOTAL = 100_000
DEFAULT_PARTS = 4
DEFAULT_PART_SIZE = 25_000
DEFAULT_SEED = 42
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TraceInfo:
    """Small pass-one record: no generation text is retained in memory."""

    dataset: str
    problem_id: str
    source_problem_id: str | None
    sample_id: int
    is_correct: bool
    reasoning_tokens: int
    sentences: int
    trace_sha256: str

    @property
    def problem(self) -> str:
        return self.source_problem_id or self.problem_id

    @property
    def identity(self) -> tuple[str, int]:
        return self.problem_id, self.sample_id


@dataclass(frozen=True)
class SelectedTrace:
    """One chosen completion plus its part assignment."""

    info: TraceInfo
    cell: str
    quartile: int
    combined: tuple[int, ...]
    parts: tuple[tuple[int, ...], ...]


def _priority(seed: int, *parts: Any) -> str:
    """Deterministic ordering key that does not depend on dict iteration order."""
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _trace_key(dataset: str, problem_id: str, sample_id: int) -> str:
    return f"{dataset}|{problem_id}|{sample_id}"


def _cell_id(dataset: str, correct: bool, quartile: int) -> str:
    return f"{dataset}|{'correct' if correct else 'incorrect'}|Q{quartile + 1}"


def server_reasoning_tokens(trace: TraceRecord) -> int | None:
    """Return the server-reported reasoning-token count, or None when absent."""
    usage = (trace.metadata or {}).get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    value = details.get("reasoning_tokens")
    if type(value) is not int or value < 0:
        return None
    return value


def replay_reasoning_tokens(trace: TraceRecord) -> int | None:
    """Count reasoning tokens from the saved token replay, or None when impossible.

    ``completion_offsets`` are character spans into ``cot_text``, so locating the
    parsed ``reasoning_content`` inside it identifies exactly which completion
    tokens belong to the reasoning segment. This is model-agnostic: it needs no
    tokenizer and no chat-format knowledge.
    """
    metadata = trace.metadata or {}
    replay = metadata.get("token_replay") or {}
    offsets = replay.get("completion_offsets")
    reasoning = metadata.get("reasoning_content")
    text = trace.cot_text
    if not isinstance(offsets, list) or not isinstance(reasoning, str) or not reasoning:
        return None
    if not isinstance(text, str) or not text:
        return None
    start = text.find(reasoning)
    if start < 0:
        return None
    end = start + len(reasoning)
    count = 0
    for span in offsets:
        if not isinstance(span, list) or len(span) != 2:
            return None
        if span[0] < end and span[1] > start:
            count += 1
    return count


def reasoning_tokens(trace: TraceRecord) -> tuple[int, str] | None:
    """Resolve the reasoning-length metric and say where it came from.

    vLLM does not populate ``usage.completion_tokens_details.reasoning_tokens``
    for every model family (the gpt-oss harmony path reports a constant 0 while
    still parsing ``reasoning_content`` correctly), so a zero server value falls
    back to the token replay rather than collapsing the whole corpus into one
    length stratum.
    """
    server = server_reasoning_tokens(trace)
    if server is not None and server > 0:
        return server, "server"
    derived = replay_reasoning_tokens(trace)
    if derived is not None and derived > 0:
        return derived, "token_replay"
    if server is not None:
        return server, "server"
    return None


def quartile_groups(ordered_keys: list[str]) -> dict[str, int]:
    """Assign equal-count quartiles (0..3) to keys already in length order."""
    if not ordered_keys:
        raise ValueError("Cannot build quartiles from an empty trace set")
    total = len(ordered_keys)
    return {
        key: min(3, (4 * rank) // total)
        for rank, key in enumerate(ordered_keys)
    }


def waterfill(caps: dict[str, int], quota: int, order: list[str]) -> dict[str, int]:
    """Integer max-min fair allocation of ``quota`` under per-problem caps."""
    if quota < 0 or set(caps) != set(order):
        raise ValueError("waterfill requires a non-negative quota and matching caps/order")
    if any(type(cap) is not int or cap < 0 for cap in caps.values()):
        raise ValueError("waterfill caps must be non-negative integers")
    if quota > sum(caps.values()):
        raise ValueError(f"waterfill quota {quota} exceeds supply {sum(caps.values())}")
    if not caps:
        return {}
    low, high = 0, max(caps.values())
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(cap, middle) for cap in caps.values()) <= quota:
            low = middle
        else:
            high = middle - 1
    allocation = {key: min(caps[key], low) for key in order}
    remaining = quota - sum(allocation.values())
    for key in order:
        if remaining == 0:
            break
        if allocation[key] < caps[key]:
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("waterfill failed to allocate the remaining quota")
    return allocation


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid dataset manifest: {path}")
    if manifest.get("status") != "complete":
        raise ValueError(
            f"Dataset manifest {path} has status {manifest.get('status')!r}; "
            "sampling requires completed generations"
        )
    return manifest


def _load_infos(path: Path, dataset: str) -> tuple[list[TraceInfo], dict[str, int]]:
    """First streaming pass: validation, eligibility and sentence counts."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing traces file: {path}")
    infos: list[TraceInfo] = []
    dropped_unscored = 0
    dropped_tokens = 0
    metric_sources: Counter[str] = Counter()
    seen: set[tuple[str, int]] = set()
    for row in iter_jsonl(path):
        trace = TraceRecord(**row)
        if trace.dataset != dataset:
            raise ValueError(f"Trace dataset {trace.dataset!r} does not match directory {dataset!r}")
        key = (trace.problem_id, trace.sample_id)
        if key in seen:
            raise ValueError(f"Duplicate trace identity {key} in {path}")
        seen.add(key)
        if trace.is_correct is None:
            dropped_unscored += 1
            continue
        resolved = reasoning_tokens(trace)
        if resolved is None:
            dropped_tokens += 1
            continue
        tokens, metric_source = resolved
        metric_sources[metric_source] += 1
        units = sentence_spans(trace)
        infos.append(
            TraceInfo(
                dataset=dataset,
                problem_id=trace.problem_id,
                source_problem_id=trace.source_problem_id,
                sample_id=trace.sample_id,
                is_correct=bool(trace.is_correct),
                reasoning_tokens=tokens,
                sentences=len(units),
                trace_sha256=trace_digest(trace),
            )
        )
    if not infos:
        raise ValueError(f"No eligible scored traces with reasoning_tokens in {path}")
    lengths = {info.reasoning_tokens for info in infos}
    if len(lengths) == 1:
        raise ValueError(
            f"Every eligible trace in {path} reports the same reasoning length "
            f"({lengths.pop()}); the length quartiles would be meaningless. The "
            "server reported no usable reasoning_tokens and the token replay could "
            "not supply one."
        )
    return infos, {
        "dropped_unscored": dropped_unscored,
        "dropped_missing_reasoning_tokens": dropped_tokens,
        "length_metric_source": dict(sorted(metric_sources.items())),
    }


def _quartile_assignment(infos: list[TraceInfo]) -> dict[tuple[str, int], int]:
    ordered = sorted(infos, key=lambda info: (info.reasoning_tokens, info.trace_sha256))
    keys = [_trace_key(info.dataset, info.problem_id, info.sample_id) for info in ordered]
    groups = quartile_groups(keys)
    return {
        (info.problem_id, info.sample_id): groups[_trace_key(info.dataset, info.problem_id, info.sample_id)]
        for info in infos
    }


def _deviation(
    masses: dict[str, int],
    total: int,
    targets: dict[str, Fraction],
    cells: list[str],
) -> tuple[Fraction, Fraction]:
    if total <= 0:
        return Fraction(1), Fraction(1)
    deviations = [
        abs(Fraction(masses.get(cell, 0), total) - targets[cell])
        for cell in cells
    ]
    return max(deviations), sum(deviations)


def _choose_completions(
    infos: list[TraceInfo],
    quartiles: dict[tuple[str, int], int],
    *,
    dataset: str,
    seed: int,
) -> tuple[dict[str, TraceInfo], dict[str, int], dict[str, Any]]:
    """Pick exactly one eligible attempt per source problem.

    The objective is the maximum absolute deviation between the chosen
    universe's cell sentence shares and the full eligible corpus shares within
    the dataset. A greedy first pass is followed by deterministic local swaps.
    """
    problems: dict[str, list[tuple[TraceInfo, str]]] = defaultdict(list)
    supplies: Counter[str] = Counter()
    for info in infos:
        cell = _cell_id(dataset, info.is_correct, quartiles[info.identity])
        problems[info.problem].append((info, cell))
        supplies[cell] += info.sentences
    corpus_total = sum(supplies.values())
    cells = sorted(supplies)
    targets = {cell: Fraction(supplies[cell], corpus_total) for cell in cells}

    chosen: dict[str, TraceInfo] = {}
    masses: Counter[str] = Counter()
    total = 0
    flexible: list[str] = []
    for problem in sorted(problems):
        options = problems[problem]
        if len(options) == 1:
            info, cell = options[0]
            chosen[problem] = info
            masses[cell] += info.sentences
            total += info.sentences
        else:
            flexible.append(problem)
    flexible.sort(key=lambda problem: _priority(seed, "flexible", problem))

    for problem in flexible:
        best_score: tuple[Fraction, Fraction, str] | None = None
        best_info: TraceInfo | None = None
        for info, cell in problems[problem]:
            trial = Counter(masses)
            trial[cell] += info.sentences
            score = _deviation(trial, total + info.sentences, targets, cells)
            tie = _priority(seed, "tie", problem, info.sample_id)
            candidate = (score[0], score[1], tie)
            if best_score is None or candidate < best_score:
                best_score, best_info = candidate, info
        if best_info is None:
            raise RuntimeError(f"No candidate attempt for problem {problem}")
        cell = _cell_id(dataset, best_info.is_correct, quartiles[best_info.identity])
        chosen[problem] = best_info
        masses[cell] += best_info.sentences
        total += best_info.sentences

    for _ in range(3):
        changed = False
        for problem in flexible:
            current = chosen[problem]
            current_cell = _cell_id(dataset, current.is_correct, quartiles[current.identity])
            current_score = _deviation(masses, total, targets, cells)
            best_score = (
                current_score[0],
                current_score[1],
                _priority(seed, "tie", problem, current.sample_id),
            )
            best_info: TraceInfo | None = None
            best_cell = current_cell
            for info, cell in problems[problem]:
                if info is current:
                    continue
                candidate_masses = Counter(masses)
                candidate_masses[current_cell] -= current.sentences
                candidate_masses[cell] += info.sentences
                score = _deviation(
                    candidate_masses,
                    total - current.sentences + info.sentences,
                    targets,
                    cells,
                )
                tie = _priority(seed, "tie", problem, info.sample_id)
                candidate = (score[0], score[1], tie)
                if candidate < best_score:
                    best_score, best_info, best_cell = candidate, info, cell
            if best_info is not None:
                masses[current_cell] -= current.sentences
                masses[best_cell] += best_info.sentences
                total += best_info.sentences - current.sentences
                chosen[problem] = best_info
                changed = True
        if not changed:
            break

    deviations = {
        cell: float(abs(Fraction(masses.get(cell, 0), total) - targets[cell]))
        for cell in cells
    }
    details = {
        "cells": cells,
        "supplies": dict(supplies),
        "targets": {cell: float(targets[cell]) for cell in cells},
        "chosen_sentences": dict(masses),
        "chosen_total": total,
        "max_share_deviation": max(deviations.values()),
        "share_deviations": deviations,
        "forced_problems": sum(len(problems[problem]) == 1 for problem in problems),
        "flexible_problems": len(flexible),
    }
    return chosen, dict(masses), details


def _split_part_quotas(
    quotas: dict[str, int],
    cells: list[str],
    *,
    parts: int,
    part_size: int,
    seed: int,
) -> dict[str, list[int]]:
    """Split every cell quota across parts, then balance exact part totals."""
    split: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        base, remainder = divmod(quotas[cell], parts)
        split[cell] = [
            base + (1 if (part - index) % parts < remainder else 0)
            for part in range(parts)
        ]
    totals = [sum(split[cell][part] for cell in cells) for part in range(parts)]
    moves = 0
    limit = 4 * part_size * parts
    while max(totals) > part_size:
        over = max(range(parts), key=lambda part: (totals[part], -part))
        under = min(range(parts), key=lambda part: (totals[part], part))
        candidates = [cell for cell in cells if split[cell][over] > 0]
        if not candidates:
            raise RuntimeError("Unable to balance part quotas")
        cell = max(
            candidates,
            key=lambda name: (
                split[name][over] - split[name][under],
                _priority(seed, "balance", name),
            ),
        )
        split[cell][over] -= 1
        split[cell][under] += 1
        totals[over] -= 1
        totals[under] += 1
        moves += 1
        if moves > limit:
            raise RuntimeError("Part quota balancing did not converge")
    if any(total != part_size for total in totals):
        raise RuntimeError("Part quotas do not sum to the requested part size")
    return split


def _sample_indices(problem: str, sentences: int, count: int, seed: int, salt: str) -> list[int]:
    if count < 0 or count > sentences:
        raise ValueError("Sentence selection count exceeds supply")
    if count == sentences:
        return list(range(sentences))
    rng = random.Random(f"{seed}:{salt}:{problem}")
    return rng.sample(range(sentences), count)


def _dataset_manifest(
    source: dict[str, Any],
    *,
    trace_path: Path,
    source_manifest: Path,
    plan_hash: str,
    traces: int,
    scored: int,
    correct: int,
    selected_sentences: int,
) -> dict[str, Any]:
    return {
        **source,
        "status": "complete",
        "problems": traces,
        "traces": traces,
        "scored_traces": scored,
        "correct_traces": correct,
        "accuracy": correct / scored if scored else None,
        "samples_per_problem": 1,
        "source_samples_per_problem": source.get("samples_per_problem"),
        "trace_path": str(trace_path),
        "sampling_manifest_sha256": plan_hash,
        "source_manifest": str(source_manifest),
        "sampling": "stratified-accuracy-reasoning-length",
        "selected_sentences": selected_sentences,
    }


def _select(args: argparse.Namespace) -> tuple[dict[str, Any], dict[tuple[str, str], SelectedTrace], dict[str, Any]]:
    """Build the plan, its selected traces and auxiliary per-dataset details."""
    if args.total != args.parts * args.part_size:
        raise ValueError("--total must equal --parts * --part-size")
    if args.total < 1 or args.parts < 1 or args.part_size < 1:
        raise ValueError("--total, --parts and --part-size must be positive")
    slug = _model_slug(args.generation_model)
    generation_root = args.generation_dir / slug
    datasets = list(args.datasets)
    if len(set(datasets)) != len(datasets):
        raise ValueError("--datasets contains duplicates")

    infos_by_dataset: dict[str, list[TraceInfo]] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    eligible_report: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        directory = generation_root / dataset
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file() and (directory / "generation_shards").is_dir():
            raise ValueError(
                f"Dataset {dataset} is still generating: {manifest_path} does not exist yet"
            )
        manifest = _read_manifest(manifest_path)
        infos, dropped = _load_infos(directory / "traces.jsonl", dataset)
        infos_by_dataset[dataset] = infos
        source_manifests[dataset] = {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "status": manifest.get("status"),
        }
        eligible_report[dataset] = {
            "traces": len(infos),
            "problems": len({info.problem for info in infos}),
            "sentences": sum(info.sentences for info in infos),
            **dropped,
        }

    quartiles = {dataset: _quartile_assignment(infos) for dataset, infos in infos_by_dataset.items()}
    chosen_by_dataset: dict[str, dict[str, TraceInfo]] = {}
    details_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        chosen, _masses, details = _choose_completions(
            infos_by_dataset[dataset], quartiles[dataset], dataset=dataset, seed=args.seed
        )
        chosen_by_dataset[dataset] = chosen
        details_by_dataset[dataset] = details

    cell_supply: dict[str, int] = {}
    cell_of: dict[tuple[str, str], str] = {}
    for dataset in datasets:
        for problem, info in chosen_by_dataset[dataset].items():
            cell = _cell_id(dataset, info.is_correct, quartiles[dataset][info.identity])
            cell_of[(dataset, problem)] = cell
            cell_supply[cell] = cell_supply.get(cell, 0) + info.sentences
    chosen_total = sum(cell_supply.values())
    if chosen_total < args.total:
        raise ValueError(
            f"Chosen one-completion universe has {chosen_total} sentences, fewer than --total {args.total}"
        )
    quotas, _rounds = capped_quotas(
        dict(cell_supply), {cell: cell_supply[cell] for cell in cell_supply}, args.total
    )
    cells = sorted(cell_supply)
    part_quotas = _split_part_quotas(
        quotas, cells, parts=args.parts, part_size=args.part_size, seed=args.seed
    )

    selected: dict[tuple[str, str], SelectedTrace] = {}
    completions: list[dict[str, Any]] = []
    cell_problem_count: Counter[str] = Counter()
    for dataset in datasets:
        quartile = quartiles[dataset]
        chosen = chosen_by_dataset[dataset]
        by_cell: dict[str, list[str]] = defaultdict(list)
        for problem in chosen:
            by_cell[cell_of[(dataset, problem)]].append(problem)
        for cell in sorted(by_cell):
            order = sorted(by_cell[cell], key=lambda problem: _priority(args.seed, "cell", cell, problem))
            caps = {problem: chosen[problem].sentences for problem in order}
            allocation = waterfill(caps, quotas[cell], order)
            if sum(allocation.values()) != quotas[cell]:
                raise RuntimeError(f"Cell {cell} allocation does not match its quota")
            part_remaining = list(part_quotas[cell])
            part_index = 0
            for problem in order:
                info = chosen[problem]
                count = allocation[problem]
                cell_problem_count[cell] += 1
                completions.append(
                    {
                        "dataset": dataset,
                        "source_problem_id": info.source_problem_id,
                        "problem_id": info.problem_id,
                        "sample_id": info.sample_id,
                        "cell": cell,
                        "quartile": quartile[info.identity] + 1,
                        "is_correct": info.is_correct,
                        "reasoning_tokens": info.reasoning_tokens,
                        "sentences": info.sentences,
                        "selected_sentences": count,
                        "trace_sha256": info.trace_sha256,
                    }
                )
                if count <= 0:
                    continue
                picked = _sample_indices(problem, info.sentences, count, args.seed, f"sentences:{cell}")
                part_lists: list[list[int]] = [[] for _ in range(args.parts)]
                for index in picked:
                    while part_index < args.parts and part_remaining[part_index] == 0:
                        part_index += 1
                    if part_index >= args.parts:
                        raise RuntimeError("Sentence assignment ran past the last part")
                    part_lists[part_index].append(index)
                    part_remaining[part_index] -= 1
                selected[(dataset, problem)] = SelectedTrace(
                    info=info,
                    cell=cell,
                    quartile=quartile[info.identity],
                    combined=tuple(sorted(picked)),
                    parts=tuple(tuple(sorted(indices)) for indices in part_lists),
                )
            if any(part_remaining):
                raise RuntimeError(f"Cell {cell} did not fill its part quotas")

    part_dataset: list[Counter[str]] = [Counter() for _ in range(args.parts)]
    part_cells: list[Counter[str]] = [Counter() for _ in range(args.parts)]
    part_totals: list[int] = [0] * args.parts
    for (dataset, _problem), item in selected.items():
        for part, indices in enumerate(item.parts):
            if indices:
                part_dataset[part][dataset] += 1
                part_cells[part][item.cell] += len(indices)
                part_totals[part] += len(indices)
    if part_totals != [args.part_size] * args.parts:
        raise RuntimeError(f"Part totals {part_totals} do not match {args.part_size}")

    cell_report = {}
    for cell in cells:
        dataset, correctness, quartile = cell.split("|")
        cell_report[cell] = {
            "dataset": dataset,
            "correct": correctness == "correct",
            "quartile": int(quartile[1:]),
            "supply_sentences": details_by_dataset[dataset]["supplies"].get(cell, 0),
            "chosen_sentences": cell_supply[cell],
            "chosen_problems": cell_problem_count[cell],
            "quota": quotas[cell],
            "selected": quotas[cell],
        }

    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selection": "stratified-accuracy-reasoning-length",
        "generation_dir": str(args.generation_dir),
        "generation_model": args.generation_model,
        "generation_slug": slug,
        "datasets": datasets,
        "seed": args.seed,
        "total_sentences": args.total,
        "parts": args.parts,
        "part_size": args.part_size,
        "eligibility": {
            "correctness": "traces with is_correct is not None only",
            "length_metric": "metadata.usage.completion_tokens_details.reasoning_tokens",
            "length_bins": "per-dataset equal-count quartiles over eligible traces",
            "diversity": "one completion per source problem; extra completions only if a cell supply requires it",
        },
        "source_manifests": source_manifests,
        "eligible": eligible_report,
        "quartiles": {
            dataset: {
                "counts": {
                    f"Q{quartile + 1}": sum(
                        1 for info in infos_by_dataset[dataset]
                        if quartiles[dataset][info.identity] == quartile
                    )
                    for quartile in range(4)
                },
                "reasoning_tokens_min": min(info.reasoning_tokens for info in infos_by_dataset[dataset]),
                "reasoning_tokens_max": max(info.reasoning_tokens for info in infos_by_dataset[dataset]),
            }
            for dataset in datasets
        },
        "cells": cell_report,
        "deviation": {
            dataset: {
                "max_share_deviation": details_by_dataset[dataset]["max_share_deviation"],
                "share_deviations": details_by_dataset[dataset]["share_deviations"],
                "forced_problems": details_by_dataset[dataset]["forced_problems"],
                "flexible_problems": details_by_dataset[dataset]["flexible_problems"],
            }
            for dataset in datasets
        },
        "completions": sorted(
            completions,
            key=lambda row: (row["dataset"], row["source_problem_id"] or "", row["sample_id"]),
        ),
        "part_breakdown": [
            {
                "part": part,
                "identities": part_totals[part],
                "datasets": dict(sorted(part_dataset[part].items())),
                "cells": dict(sorted(part_cells[part].items())),
            }
            for part in range(args.parts)
        ],
        "selected_sentences": args.total,
        "chosen_problems": sum(len(chosen_by_dataset[dataset]) for dataset in datasets),
        "selected_traces": len(selected),
        "completions_per_problem": {
            "0": sum(len(chosen_by_dataset[dataset]) for dataset in datasets) - len(selected),
            "1": len(selected),
        },
    }
    plan_hash = digest(plan)
    manifest = {**plan, "manifest_sha256": plan_hash}
    aux = {
        "quartiles": quartiles,
        "source_manifests": source_manifests,
    }
    return manifest, selected, aux


def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "selection": manifest["selection"],
        "generation_model": manifest["generation_model"],
        "datasets": manifest["datasets"],
        "seed": manifest["seed"],
        "total_sentences": manifest["total_sentences"],
        "selected_sentences": manifest["selected_sentences"],
        "chosen_problems": manifest["chosen_problems"],
        "selected_traces": manifest["selected_traces"],
        "parts": [
            {"part": row["part"], "identities": row["identities"]} for row in manifest["part_breakdown"]
        ],
        "eligible": manifest["eligible"],
        "max_share_deviation": {
            dataset: manifest["deviation"][dataset]["max_share_deviation"]
            for dataset in manifest["datasets"]
        },
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _write_outputs(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    selected: dict[tuple[str, str], SelectedTrace],
    aux: dict[str, Any],
) -> None:
    """Second streaming pass: write combined and per-part sampled trace roots."""
    slug = manifest["generation_slug"]
    plan_hash = manifest["manifest_sha256"]
    generation_root = args.generation_dir / slug
    roots: list[tuple[str, Path]] = [("combined", args.output_dir / "generation")]
    roots.extend(
        (f"part-{part:02d}", args.output_dir / "parts" / f"part-{part:02d}" / "generation")
        for part in range(manifest["parts"])
    )
    source_manifests = aux["source_manifests"]
    for dataset in manifest["datasets"]:
        dataset_selected = {
            (item.info.problem_id, item.info.sample_id): item
            for (name, _problem), item in selected.items()
            if name == dataset
        }
        writers: dict[str, Any] = {}
        temporaries: dict[str, tuple[Path, Path]] = {}
        for name, root in roots:
            path = root / slug / dataset / "traces.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            writers[name] = temporary.open("w", encoding="utf-8")
            temporaries[name] = (temporary, path)
        try:
            for row in iter_jsonl(generation_root / dataset / "traces.jsonl"):
                key = (row.get("problem_id"), row.get("sample_id", 0))
                item = dataset_selected.get(key)
                if item is None:
                    continue
                base = {
                    "schema_version": SCHEMA_VERSION,
                    "manifest_sha256": plan_hash,
                    "selection": manifest["selection"],
                    "stratum": {
                        "dataset": item.info.dataset,
                        "correct": item.info.is_correct,
                        "quartile": item.quartile + 1,
                        "reasoning_tokens": item.info.reasoning_tokens,
                    },
                }
                combined_row = {**row, "metadata": {**(row.get("metadata") or {})}}
                combined_row["metadata"]["sentence_selection"] = {
                    **base,
                    "indices": list(item.combined),
                }
                writers["combined"].write(json.dumps(combined_row, ensure_ascii=False) + "\n")
                for part, indices in enumerate(item.parts):
                    if not indices:
                        continue
                    part_row = {**row, "metadata": {**(row.get("metadata") or {})}}
                    part_row["metadata"]["sentence_selection"] = {**base, "indices": list(indices)}
                    writers[f"part-{part:02d}"].write(
                        json.dumps(part_row, ensure_ascii=False) + "\n"
                    )
        finally:
            for handle in writers.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        for temporary, path in temporaries.values():
            os.replace(temporary, path)

        source_manifest_path = Path(source_manifests[dataset]["path"])
        source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        for name, root in roots:
            part = None if name == "combined" else int(name.split("-")[1])
            items = [
                item for item in dataset_selected.values()
                if part is None or item.parts[part]
            ]
            selected_sentences = sum(
                len(item.combined) if part is None else len(item.parts[part])
                for item in items
            )
            _write_json_atomic(
                _dataset_manifest(
                    source,
                    trace_path=root / slug / dataset / "traces.jsonl",
                    source_manifest=source_manifest_path,
                    plan_hash=plan_hash,
                    traces=len(items),
                    scored=len(items),
                    correct=sum(item.info.is_correct for item in items),
                    selected_sentences=selected_sentences,
                ),
                root / slug / dataset / "manifest.json",
            )
    for row in manifest["part_breakdown"]:
        part = row["part"]
        _write_json_atomic(
            {
                "schema_version": SCHEMA_VERSION,
                "part": part,
                "identities": row["identities"],
                "datasets": row["datasets"],
                "cells": row["cells"],
                "sampling_manifest_sha256": plan_hash,
                "trace_root": str(args.output_dir / "parts" / f"part-{part:02d}" / "generation" / slug),
            },
            args.output_dir / "parts" / f"part-{part:02d}" / "part_manifest.json",
        )
    _write_json_atomic(manifest, args.output_dir / "sampling_manifest.json")


def sample(args: argparse.Namespace) -> dict[str, Any]:
    """Build (and unless ``--dry-run`` is set, write) one stratified plan."""
    manifest, selected, aux = _select(args)
    if args.dry_run:
        return _summary(manifest)
    manifest_path = args.output_dir / "sampling_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Output contains a different sampling plan; choose a new output directory")
    _write_outputs(args, manifest, selected, aux)
    return _summary(manifest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=Path("results/correlation_pipeline/nemotron-nvfp4-dspark/generation"),
        help="Root containing <model-slug>/<dataset>/traces.jsonl",
    )
    parser.add_argument(
        "--generation-model",
        default=DEFAULT_GENERATION_MODEL,
        help="Model identifier whose slug names the generation subtree",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Benchmarks to sample (default: the six SPIRAL math sets)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--parts", type=int, default=DEFAULT_PARTS)
    parser.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the plan without writing any output",
    )
    print(json.dumps(sample(parser.parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
