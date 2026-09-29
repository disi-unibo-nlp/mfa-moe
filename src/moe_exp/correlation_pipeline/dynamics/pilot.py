"""Freeze the Qwen3.6 fully labeled pilot without launching a workload."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import random
from .common import DEFAULT_CONFIG, digest, identity, termination, write_json
from moe_exp.correlation_pipeline.spans import sentence_spans, trace_digest, reasoning_ranges
from moe_exp.correlation_pipeline.provenance import code_provenance, file_sha256
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord

SOURCE = "Qwen/Qwen3.6-35B-A3B-FP8"
TARGET = "unsloth/Qwen3.6-35B-A3B"
REVISION = "2ab40a9acc6d567889ca4d4e59feb2da56121454"
JUDGE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SLUG = "Qwen--Qwen3.6-35B-A3B-FP8"
LAUNCHERS = ("scripts/run_routing_dynamics.py", "scripts/routing_dynamics_env.sh",
             "sbatch/routing_dynamics_label.sbatch", "sbatch/routing_dynamics_replay.sbatch")


def launcher_hashes():
    repo = Path(__file__).resolve().parents[4]
    return {name: file_sha256(repo / name) for name in LAUNCHERS}


JUDGE = dict(judge_model="Qwen/Qwen3.8-27B", batch_size=64, enable_thinking=True,
             max_tokens=16384, temperature=1., top_p=.95, top_k=20, min_p=0.,
             presence_penalty=0., repetition_penalty=1., reasoning_effort="low",
             preserve_thinking=False)


def select(traces, excluded=(), *, seed=42, per_quartile=16):
    excluded = set(excluded)
    candidates, seen = [], set()
    for trace in traces:
        question = (trace.dataset, trace.source_problem_id or trace.problem_id)
        if question in seen:
            raise ValueError("Pilot source must have one existing attempt per question")
        seen.add(question)
        if digest(identity(trace)) in excluded:
            continue
        replay = trace.metadata.get("token_replay", {})
        offsets = replay.get("completion_offsets")
        if offsets is None or len(offsets) != len(replay.get("completion_token_ids", [])):
            raise ValueError("Pilot source lacks exact reasoning token alignment")
        ranges = reasoning_ranges(trace)
        length = sum(right > left and any(right > a and left < b for a, b in ranges)
                     for left, right in offsets)
        if length <= 0:
            raise ValueError("Pilot source has no reasoning tokens")
        candidates.append((length, digest(identity(trace)), trace))
    candidates.sort(key=lambda x: (x[0], x[1]))
    rng, selected, bins = random.Random(seed), [], []
    for q in range(4):
        bucket = candidates[len(candidates) * q // 4:len(candidates) * (q + 1) // 4]
        if len(bucket) < per_quartile:
            raise ValueError("Insufficient traces for each quartile")
        chosen = rng.sample(bucket, per_quartile)
        selected.extend((q + 1, length, trace) for length, _, trace in chosen)
        bins.append(dict(quartile=q + 1, supply=len(bucket), minimum=bucket[0][0],
                         maximum=bucket[-1][0], selected=len(chosen)))
    return sorted(selected, key=lambda item: (item[0], digest(identity(item[2])))), bins


def prepare(traces_path, fixtures_path, program, output, *, seed=42):
    traces = [TraceRecord(**r) for r in iter_jsonl(traces_path)]
    if any(t.model_id != SOURCE or t.dataset != "math500" for t in traces):
        raise ValueError("Pilot requires the existing Qwen3.6 MATH-500 source")
    fixtures = [digest(identity(TraceRecord(**r))) for r in iter_jsonl(fixtures_path)]
    selected, quartiles = select(traces, fixtures, seed=seed)
    program = Path(program)
    output = Path(output)
    source_manifest = Path(traces_path).with_name("manifest.json")
    original_manifest = json.loads(source_manifest.read_text())
    if original_manifest.get("target_model_id") != TARGET:
        raise ValueError("Unexpected forward checkpoint in generation manifest")
    rows, membership = [], []
    for quartile, length, original in selected:
        trace = original.model_copy(deep=True)
        # The pilot requests every sentence, independent of old sampled annotations.
        for name in ("sentence_selection", "reasoning_annotation"):
            trace.metadata.pop(name, None)
        rows.append(trace.model_dump())
        membership.append(dict(identity=identity(trace), trace_sha256=trace_digest(trace),
            source_record_sha256=digest(original.model_dump()), quartile=quartile,
            reasoning_tokens=length, completion_tokens=len(trace.metadata["token_replay"]["completion_token_ids"]),
            sentences=len(sentence_spans(trace)), termination=termination(trace), is_correct=trace.is_correct))
    token_count = sum(r["completion_tokens"] for r in membership)
    sentence_count = sum(r["sentences"] for r in membership)
    window_count = sum(sum(max(0, (r["reasoning_tokens"] - w) // 64 + 1) for w in (64, 256))
                       for r in membership) * 40
    manifest = dict(schema_version=1, status="prepared_not_submitted", seed=seed,
        selection="equal-count exact reasoning-token quartiles, 16 per quartile, excluding engineering fixtures; correctness unused",
        source_sha256=file_sha256(traces_path), source_manifest_sha256=file_sha256(source_manifest),
        fixture_sha256=file_sha256(fixtures_path), excluded_fixtures=fixtures,
        source_model=SOURCE, target_model=TARGET, forward_revision=REVISION, quantization="bnb-4bit",
        judge={**JUDGE, "program_sha256": file_sha256(program), "revision": JUDGE_REVISION},
        code=code_provenance(), launchers=launcher_hashes(), metrics=DEFAULT_CONFIG, quartiles=quartiles, traces=membership,
        records_sha256=digest(rows), program_file="selected_program.json",
        resources=dict(label=dict(hours=12, gpus=2, cpus=16, memory_gb=120),
                       replay=dict(hours=4, gpus=2, cpus=16, memory_gb=120), maximum_gpu_hours=32),
        storage_estimate=dict(completion_tokens=token_count, reasoning_sentences=sentence_count,
            selected_expert_payload_bytes=token_count * 40 * 8 * 2,
            raw_router_bytes_avoided=token_count * 40 * 256 * 4,
            window_rows=window_count,
            compact_windows_upper_bytes=window_count * 20000,
            sparse_sentence_upper_bytes=sentence_count * 40 * 5000,
            judge_text_upper_bytes=sentence_count * 16384 * 6,
            conservative_total_bytes=(window_count * 20000 * 3 + sentence_count * 40 * 5000
                                      + sentence_count * 16384 * 6 * 3 + token_count * 40 * 8 * 2
                                      + 5_000_000_000),
            note="Conservative uncompressed JSON bounds; shared cached weights reused, no downloads."))
    manifest["manifest_sha256"] = digest(manifest)
    if (output / "pilot.json").exists():
        if json.loads((output / "pilot.json").read_text()) != manifest:
            raise ValueError("Pilot source/code/config changed; choose a new directory")
        verify(output)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError("Pilot output must be empty")
    directory = output / "generation" / SLUG / "math500"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "traces.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_json(directory / "manifest.json", {**original_manifest, "pilot_manifest_sha256": manifest["manifest_sha256"],
                                           "pilot_traces": len(rows)})
    (output / "selected_program.json").write_bytes(program.read_bytes())
    write_json(output / "pilot.json", manifest)
    verify(output)
    return manifest


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "pilot.json").read_text())
    contract = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if digest(contract) != manifest["manifest_sha256"]:
        raise ValueError("Pilot manifest integrity failure")
    rows = list(iter_jsonl(root / "generation" / SLUG / "math500" / "traces.jsonl"))
    if digest(rows) != manifest["records_sha256"] or len(rows) != 64:
        raise ValueError("Pilot record integrity failure")
    if file_sha256(root / "selected_program.json") != manifest["judge"]["program_sha256"]:
        raise ValueError("Frozen classifier integrity failure")
    if Counter(r["quartile"] for r in manifest["traces"]) != {1: 16, 2: 16, 3: 16, 4: 16}:
        raise ValueError("Pilot quartile mismatch")
    if set(digest(r["identity"]) for r in manifest["traces"]) & set(manifest["excluded_fixtures"]):
        raise ValueError("Engineering fixture entered pilot")
    return manifest


def rebind_code(root):
    """Refresh only code/launcher binding after tests; never alter selected records/settings."""
    root = Path(root)
    manifest = verify(root)
    if any((root / p).exists() for p in ("label-checkpoints", "forward", "label-acceptance.json")):
        raise ValueError("Cannot rebind an executed pilot")
    manifest["code"] = code_provenance()
    manifest["launchers"] = launcher_hashes()
    size = manifest["storage_estimate"]
    size["conservative_total_bytes"] = (size["compact_windows_upper_bytes"] * 3
        + size["sparse_sentence_upper_bytes"] + size["judge_text_upper_bytes"] * 3
        + size["selected_expert_payload_bytes"] + 5_000_000_000)
    manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    generation = root / "generation" / SLUG / "math500/manifest.json"
    payload = json.loads(generation.read_text())
    payload["pilot_manifest_sha256"] = manifest["manifest_sha256"]
    write_json(generation, payload)
    write_json(root / "pilot.json", manifest)
    verify(root)
    return manifest
