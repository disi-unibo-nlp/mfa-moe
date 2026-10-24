"""Pick the 100,000 sentences the judge labels for one source model, stratified by correctness and
reasoning length.

Input: the finished benchmark generations `GENERATION_DIR/<model-slug>/<dataset>/traces.jsonl` (+
`manifest.json`). Output in OUT: `sampling_manifest.json` (the plan, with its SHA-256),
`generation/<slug>/<dataset>/traces.jsonl` (the 100k selection: one row per chosen completion, with
`metadata.sentence_selection`) and four self-contained part roots
`parts/part-0X/generation/<slug>/...` of exactly 25,000 sentences each, which
`labeling.annotation_batch` labels.

    python -m labeling.sample_stratified --output-dir OUT --generation-model openai/gpt-oss-20b
        --generation-dir results/correlation_pipeline/gpt-oss-20b/generation [--dry-run]

How the sentences are chosen (seed 42, CPU only):
* Only scored traces with a reasoning length are eligible. The length is the server's
  `reasoning_tokens`, or, where the server reports 0 (gpt-oss), the count derived from the saved
  token replay. A dataset whose traces all have the same length is refused, since its quartiles
  would mean nothing.
* Each dataset's traces are split into four equal-count length quartiles, giving the cells
  dataset x correct|incorrect x Q1..Q4.
* One completion is chosen per source problem, by a deterministic greedy pass plus local swaps that
  keeps each dataset's cell shares of sentences as close as possible to the full corpus.
* A cell's quota is proportional to the chosen completions' sentences in it. Inside a cell the quota
  is spread over problems as evenly as their sentence counts allow, and the sentences are drawn
  uniformly without replacement.
* Each cell's quota is split evenly over the four parts, and a balancing pass makes every part
  exactly 25,000.

The same plan into the same OUT is idempotent; a different plan into an existing OUT is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from generation.generate import _model_slug, _write_json_atomic
from labeling.spans import digest, sentence_spans, trace_digest
from schemas import TraceRecord
from store import DATASETS, file_sha, iter_jsonl

PARTS = 4
PART_SIZE = 25_000
SEED = 42
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

    def outputs(self) -> dict[int | None, tuple[int, ...]]:
        """The sentence indices of each output file: None is the combined one, then every part that
        has any."""
        parts = {part: indices for part, indices in enumerate(self.parts) if indices}
        return {None: self.combined, **parts}


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


def waterfill(caps: dict[str, int], quota: int) -> dict[str, int]:
    """Integer max-min fair allocation of ``quota`` under per-problem caps (spare units go to the
    first problems)."""
    if quota < 0 or quota > sum(caps.values()):
        raise ValueError(
            f"waterfill quota {quota} must be between 0 and the supply {sum(caps.values())}")
    if not caps:
        return {}
    low, high = 0, max(caps.values())
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(cap, middle) for cap in caps.values()) <= quota:
            low = middle
        else:
            high = middle - 1
    allocation = {key: min(cap, low) for key, cap in caps.items()}
    remaining = quota - sum(allocation.values())
    for key, cap in caps.items():
        if remaining == 0:
            break
        if allocation[key] < cap:
            allocation[key] += 1
            remaining -= 1
    return allocation


def proportional_quotas(supply: dict[str, int], total: int) -> dict[str, int]:
    """Split `total` over the cells in proportion to their sentence supply (largest remainders
    first, ties by name).

    No cell gets more than its own supply: a share is at most the supply, and a rounded-up share
    stays within it.
    """
    all_supply = sum(supply.values())
    if total <= 0 or total > all_supply:
        raise ValueError(f"cannot select {total} sentences from a supply of {all_supply}")
    shares = {cell: Fraction(total * count, all_supply) for cell, count in supply.items()}
    quotas = {cell: int(share) for cell, share in shares.items()}
    left = total - sum(quotas.values())
    for cell in sorted(shares, key=lambda cell: (-(shares[cell] - quotas[cell]), cell))[:left]:
        quotas[cell] += 1
    return quotas


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


def _deviation(masses: dict[str, int], targets: dict[str, Fraction]) -> tuple[Fraction, Fraction]:
    """Largest and total absolute gap between the chosen cell shares and the corpus shares."""
    total = sum(masses.values())
    if total <= 0:
        return Fraction(1), Fraction(1)
    deviations = [abs(Fraction(masses.get(cell, 0), total) - targets[cell]) for cell in targets]
    return max(deviations), sum(deviations)


def _choose_completions(
    infos: list[TraceInfo],
    quartiles: dict[tuple[str, int], int],
    *,
    dataset: str,
    seed: int,
) -> tuple[dict[str, TraceInfo], dict[str, Any]]:
    """Pick exactly one eligible attempt per source problem.

    The objective is the maximum absolute deviation between the chosen
    universe's cell sentence shares and the full eligible corpus shares within
    the dataset. A greedy first pass is followed by deterministic local swaps.
    """
    problems: dict[str, list[TraceInfo]] = defaultdict(list)
    cell_of = {info.identity: _cell_id(dataset, info.is_correct, quartiles[info.identity])
               for info in infos}
    supplies: Counter[str] = Counter()
    for info in infos:
        problems[info.problem].append(info)
        supplies[cell_of[info.identity]] += info.sentences
    corpus_total = sum(supplies.values())
    targets = {cell: Fraction(supplies[cell], corpus_total) for cell in sorted(supplies)}

    chosen: dict[str, TraceInfo] = {}
    masses: Counter[str] = Counter()
    flexible: list[str] = []
    for problem in sorted(problems):
        if len(problems[problem]) == 1:
            chosen[problem] = problems[problem][0]
            masses[cell_of[chosen[problem].identity]] += chosen[problem].sentences
        else:
            flexible.append(problem)
    flexible.sort(key=lambda problem: _priority(seed, "flexible", problem))

    def score(problem: str, info: TraceInfo, replacing: TraceInfo | None = None) -> tuple:
        """The objective if `info` were chosen for `problem` (in place of `replacing`); ties go to
        the tie hash."""
        trial = Counter(masses)
        if replacing is not None:
            trial[cell_of[replacing.identity]] -= replacing.sentences
        trial[cell_of[info.identity]] += info.sentences
        return (*_deviation(trial, targets), _priority(seed, "tie", problem, info.sample_id))

    for problem in flexible:
        info = min(problems[problem], key=lambda option: score(problem, option))
        chosen[problem] = info
        masses[cell_of[info.identity]] += info.sentences

    for _ in range(3):
        changed = False
        for problem in flexible:
            current = chosen[problem]
            best, best_score = current, score(problem, current, current)
            for info in problems[problem]:
                if info is current:
                    continue
                candidate = score(problem, info, current)
                if candidate < best_score:
                    best, best_score = info, candidate
            if best is not current:
                masses[cell_of[current.identity]] -= current.sentences
                masses[cell_of[best.identity]] += best.sentences
                chosen[problem] = best
                changed = True
        if not changed:
            break

    total = sum(masses.values())
    deviations = {
        cell: float(abs(Fraction(masses.get(cell, 0), total) - targets[cell]))
        for cell in targets
    }
    details = {
        "supplies": dict(supplies),
        "max_share_deviation": max(deviations.values()),
        "share_deviations": deviations,
        "forced_problems": sum(len(options) == 1 for options in problems.values()),
        "flexible_problems": len(flexible),
    }
    return chosen, details


def _split_part_quotas(
    quotas: dict[str, int], *, parts: int, part_size: int, seed: int
) -> dict[str, list[int]]:
    """Split every cell quota across parts, then move units between parts until each holds exactly
    `part_size`."""
    cells = sorted(quotas)
    split: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        base, remainder = divmod(quotas[cell], parts)
        split[cell] = [
            base + (1 if (part - index) % parts < remainder else 0)
            for part in range(parts)
        ]
    totals = [sum(split[cell][part] for cell in cells) for part in range(parts)]
    while max(totals) > part_size:
        over = max(range(parts), key=lambda part: (totals[part], -part))
        under = min(range(parts), key=lambda part: (totals[part], part))
        cell = max(
            (cell for cell in cells if split[cell][over] > 0),
            key=lambda name: (
                split[name][over] - split[name][under],
                _priority(seed, "balance", name),
            ),
        )
        split[cell][over] -= 1
        split[cell][under] += 1
        totals[over] -= 1
        totals[under] += 1
    return split


def _sample_indices(problem: str, sentences: int, count: int, seed: int, salt: str) -> list[int]:
    if count < 0 or count > sentences:
        raise ValueError("Sentence selection count exceeds supply")
    if count == sentences:
        return list(range(sentences))
    rng = random.Random(f"{seed}:{salt}:{problem}")
    return rng.sample(range(sentences), count)


def build_plan(
    generation_dir: Path,
    generation_model: str,
    datasets: Sequence[str] = DATASETS,
    *,
    parts: int = PARTS,
    part_size: int = PART_SIZE,
    seed: int = SEED,
) -> tuple[dict[str, Any], dict[tuple[str, str], SelectedTrace]]:
    """Build the plan and the selected traces: `parts` parts of `part_size` sentences each."""
    datasets = list(datasets)  # a list: the plan must equal itself after a round trip through JSON
    total = parts * part_size
    slug = _model_slug(generation_model)
    generation_root = generation_dir / slug

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
            "sha256": file_sha(manifest_path),
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
        chosen, details = _choose_completions(
            infos_by_dataset[dataset], quartiles[dataset], dataset=dataset, seed=seed
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
    if chosen_total < total:
        raise ValueError(
            f"Chosen one-completion universe has {chosen_total} sentences, "
            f"fewer than the {total} to select"
        )
    quotas = proportional_quotas(cell_supply, total)
    cells = sorted(cell_supply)
    part_quotas = _split_part_quotas(quotas, parts=parts, part_size=part_size, seed=seed)

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
            order = sorted(by_cell[cell],
                           key=lambda problem: _priority(seed, "cell", cell, problem))
            caps = {problem: chosen[problem].sentences for problem in order}
            allocation = waterfill(caps, quotas[cell])
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
                picked = _sample_indices(problem, info.sentences, count, seed, f"sentences:{cell}")
                part_lists: list[list[int]] = [[] for _ in range(parts)]
                for index in picked:
                    while part_index < parts and part_remaining[part_index] == 0:
                        part_index += 1
                    if part_index >= parts:
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

    part_dataset: list[Counter[str]] = [Counter() for _ in range(parts)]
    part_cells: list[Counter[str]] = [Counter() for _ in range(parts)]
    part_totals: list[int] = [0] * parts
    for (dataset, _problem), item in selected.items():
        for part, indices in enumerate(item.parts):
            if indices:
                part_dataset[part][dataset] += 1
                part_cells[part][item.cell] += len(indices)
                part_totals[part] += len(indices)
    if part_totals != [part_size] * parts:
        raise RuntimeError(f"Part totals {part_totals} do not match {part_size}")

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
        "generation_dir": str(generation_dir),
        "generation_model": generation_model,
        "generation_slug": slug,
        "datasets": datasets,
        "seed": seed,
        "total_sentences": total,
        "parts": parts,
        "part_size": part_size,
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
            for part in range(parts)
        ],
        "selected_sentences": total,
        "chosen_problems": sum(len(chosen_by_dataset[dataset]) for dataset in datasets),
        "selected_traces": len(selected),
        "completions_per_problem": {
            "0": sum(len(chosen_by_dataset[dataset]) for dataset in datasets) - len(selected),
            "1": len(selected),
        },
    }
    return {**plan, "manifest_sha256": digest(plan)}, selected


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
    generation_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    selected: dict[tuple[str, str], SelectedTrace],
) -> None:
    """Second streaming pass: write the combined and the per-part sampled trace roots."""
    slug = manifest["generation_slug"]
    plan_hash = manifest["manifest_sha256"]
    generation_root = generation_dir / slug
    # None is the combined 100k root, 0.. are the part roots
    roots: dict[int | None, Path] = {None: output_dir / "generation"}
    for part in range(manifest["parts"]):
        roots[part] = output_dir / "parts" / f"part-{part:02d}" / "generation"
    source_manifests = manifest["source_manifests"]
    for dataset in manifest["datasets"]:
        dataset_selected = {
            (item.info.problem_id, item.info.sample_id): item
            for (name, _problem), item in selected.items()
            if name == dataset
        }
        writers: dict[int | None, Any] = {}
        temporaries: dict[int | None, tuple[Path, Path]] = {}
        for out, root in roots.items():
            path = root / slug / dataset / "traces.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            writers[out] = temporary.open("w", encoding="utf-8")
            temporaries[out] = (temporary, path)
        try:
            for row in iter_jsonl(generation_root / dataset / "traces.jsonl"):
                item = dataset_selected.get((row.get("problem_id"), row.get("sample_id", 0)))
                if item is None:
                    continue
                selection = {
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
                for out, indices in item.outputs().items():
                    metadata = {**(row.get("metadata") or {}),
                                "sentence_selection": {**selection, "indices": list(indices)}}
                    line = json.dumps({**row, "metadata": metadata}, ensure_ascii=False)
                    writers[out].write(line + "\n")
        finally:
            for handle in writers.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        for temporary, path in temporaries.values():
            os.replace(temporary, path)

        source_manifest_path = Path(source_manifests[dataset]["path"])
        source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        for out, root in roots.items():
            items = [item for item in dataset_selected.values() if out in item.outputs()]
            selected_sentences = sum(len(item.outputs()[out]) for item in items)
            correct = sum(item.info.is_correct for item in items)
            _write_json_atomic(
                {
                    **source,
                    "status": "complete",
                    "problems": len(items),
                    "traces": len(items),
                    "scored_traces": len(items),
                    "correct_traces": correct,
                    "accuracy": correct / len(items) if items else None,
                    "samples_per_problem": 1,
                    "source_samples_per_problem": source.get("samples_per_problem"),
                    "trace_path": str(root / slug / dataset / "traces.jsonl"),
                    "sampling_manifest_sha256": plan_hash,
                    "source_manifest": str(source_manifest_path),
                    "sampling": "stratified-accuracy-reasoning-length",
                    "selected_sentences": selected_sentences,
                },
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
                "trace_root": str(output_dir / "parts" / f"part-{part:02d}" / "generation" / slug),
            },
            output_dir / "parts" / f"part-{part:02d}" / "part_manifest.json",
        )
    _write_json_atomic(manifest, output_dir / "sampling_manifest.json")


def sample(
    generation_dir: Path,
    generation_model: str,
    output_dir: Path,
    *,
    dry_run: bool = False,
    **plan_options: Any,
) -> dict[str, Any]:
    """Build (and unless `dry_run`, write) one stratified plan. `plan_options` are those of
    `build_plan`."""
    manifest, selected = build_plan(generation_dir, generation_model, **plan_options)
    if dry_run:
        return _summary(manifest)
    manifest_path = output_dir / "sampling_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Output contains a different sampling plan; choose a new output directory")
    _write_outputs(generation_dir, output_dir, manifest, selected)
    return _summary(manifest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation-dir", type=Path, required=True,
                        help="Root containing <model-slug>/<dataset>/traces.jsonl")
    parser.add_argument("--generation-model", required=True,
                        help="Model identifier whose slug names the generation subtree")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print the plan, write nothing")
    args = parser.parse_args(argv)
    summary = sample(args.generation_dir, args.generation_model, args.output_dir,
                     dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
