"""Problem-grouped seed-42 split; outcomes never influence membership."""
from __future__ import annotations
from collections import defaultdict
import math
import random
import re
import unicodedata
from .common import (BENCHMARKS, CHECKPOINT, ENGINE, SAMPLING, REPETITIONS, MODEL,
                     digest, seal, verify, rows, problem_key, attempt_key, write_json,
                     code_hashes, file_hash, bounded_input)


def normalized_question(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def split_problems(problems, *, pilot=(), fixtures=(), duplicate_pairs=(), seed=42):
    """Allocate floors of group counts; the rounding remainder belongs to test.

    Exact normalized questions are grouped across benchmarks, as are user-supplied
    duplicate pairs. Cross-benchmark duplicate groups are conservatively assigned
    calibration before stratified allocation. Actual counts/deviations are reported.
    Engineering fixtures count in calibration and can never enter either holdout.
    """
    by_key = {problem_key(r): r for r in problems}
    if len(by_key) != len(problems) or {r["dataset"] for r in problems} != set(BENCHMARKS):
        raise ValueError("Need unique problems from all six benchmarks")
    parent = {key: key for key in by_key}
    def find(key):
        if key not in parent:
            raise ValueError("Unknown pilot, fixture, or duplicate problem")
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    texts = {}
    for key, row in by_key.items():
        text = normalized_question(row["prompt"])
        if not text:
            raise ValueError("Empty question")
        if text in texts:
            union(key, texts[text])
        texts[text] = key
    for a, b in duplicate_pairs:
        union(a, b)
    groups = defaultdict(list)
    for key in sorted(by_key):
        groups[find(key)].append(key)
    forced = {find(k) for k in (*pilot, *fixtures)}
    forced |= {g for g, members in groups.items()
               if len({by_key[k]["dataset"] for k in members}) > 1}
    membership, counts = {}, {}
    for benchmark in BENCHMARKS:
        candidates = sorted(g for g, members in groups.items()
                            if any(by_key[k]["dataset"] == benchmark for k in members))
        quotas = {"calibration": math.floor(.6 * len(candidates)),
                  "tuning": math.floor(.2 * len(candidates))}
        quotas["test"] = len(candidates) - sum(quotas.values())
        fixed = [g for g in candidates if g in forced]
        if len(fixed) > quotas["calibration"]:
            raise ValueError("Forced calibration groups exceed 60% quota")
        rest = [g for g in candidates if g not in forced]
        random.Random(f"{seed}:{benchmark}").shuffle(rest)
        cut = quotas["calibration"] - len(fixed)
        selected = {"calibration": fixed + rest[:cut],
                    "tuning": rest[cut:cut + quotas["tuning"]],
                    "test": rest[cut + quotas["tuning"]:]}
        for split, gs in selected.items():
            for g in gs:
                for key in groups[g]:
                    if key in membership and membership[key] != split:
                        raise ValueError("Cross-benchmark duplicate split conflict")
                    membership[key] = split
        counts[benchmark] = {"groups": len(candidates), "group_counts": quotas,
            "problem_counts": {s: sum(by_key[k]["dataset"] == benchmark and membership.get(k) == s
                                    for k in by_key) for s in selected}}
    return seal(dict(schema=1, seed=seed, membership=membership, groups=dict(groups),
                     counts=counts, pilot=sorted(pilot), fixtures=sorted(fixtures),
                     duplicate_policy="NFKC_whitespace_exact_plus_explicit_pairs; cross_benchmark_to_calibration"))


def validate_split(split):
    verify(split)
    seen = set()
    for members in split["groups"].values():
        if seen.intersection(members) or len({split["membership"][k] for k in members}) != 1:
            raise ValueError("Duplicate group leakage")
        seen.update(members)
    if seen != set(split["membership"]):
        raise ValueError("Split group coverage mismatch")
    if any(split["membership"][k] != "calibration" for k in split["pilot"] + split["fixtures"]):
        raise ValueError("Pilot/fixture leakage")


def collect(paths):
    """Streaming source adapter. No correctness/text is returned for holdout attempts."""
    problems, attempts = {}, {}
    for path in paths:
        for row in rows(path):
            if row["model_id"] != MODEL:
                raise ValueError("Source generation checkpoint differs")
            key, aid = problem_key(row), attempt_key(row)
            if aid in attempts:
                raise ValueError("Duplicate source attempt")
            replay = row["metadata"].get("token_replay", {})
            ids = replay.get("prompt_token_ids")
            if not ids or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError("Exact rendered prompt IDs required")
            config = row["metadata"]["generation_config"]
            sampling = {k: config.get(k) for k in SAMPLING}
            sampling["top_k"] = -1 if sampling["top_k"] == 0 else sampling["top_k"]
            if sampling != SAMPLING or type(config.get("seed")) is not int:
                raise ValueError("Source sampling differs from prespecified protocol")
            problem = {k: row.get(k) for k in ("dataset", "problem_id", "source_problem_id",
                "prompt", "gold_answer", "system_prompt", "generation_messages")}
            # Source problem_id includes the attempt suffix; group metadata must not.
            problem["problem_id"] = row.get("source_problem_id") or row["problem_id"]
            problem.update(prompt_token_ids=ids, answer_type=row["metadata"].get("answer_type", "math"),
                           tokenizer_sha256=replay.get("tokenizer_sha256"))
            if problem["gold_answer"] is None or not str(problem["gold_answer"]).strip():
                raise ValueError("Known gold required")
            if key in problems and problems[key] != problem:
                differing = sorted(k for k in problem if problems[key].get(k) != problem[k])
                raise ValueError(f"Attempts disagree on question/gold/rendered tokens: {key}; fields={differing}")
            problems[key] = problem
            attempts[aid] = dict(problem_key=key, sample_id=row.get("sample_id", 0), seed=config["seed"],
                                 source=str(path), original_problem_id=row["problem_id"],
                                 replay_tokens=len(ids)+len(replay.get("completion_token_ids", [])))
    return problems, attempts


def resolve_pilot(problems, attempts, pilot):
    """Resolve legacy attempt identities using original IDs, then group problems."""
    import hashlib, json
    original = defaultdict(set)
    fixture_keys = set()
    for attempt in attempts.values():
        key = attempt["problem_key"]
        dataset = problems[key]["dataset"]
        original[(dataset, attempt["original_problem_id"])].add(key)
        legacy = hashlib.sha256(json.dumps([MODEL, dataset,
            attempt["original_problem_id"], attempt["sample_id"]],
            sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if legacy in pilot["excluded_fixtures"]:
            fixture_keys.add(key)
    pilot_keys = set()
    for item in pilot["traces"]:
        identity = item["identity"]
        matches = original.get((identity[1], identity[2]), set())
        if identity[0] != MODEL or len(matches) != 1:
            raise ValueError("Pilot identity missing/ambiguous in source population")
        pilot_keys.update(matches)
    return pilot_keys, fixture_keys


def prepare(paths, pilot_path, output, *, duplicate_pairs=()):
    bounded_input(paths)
    pilot = verify_legacy_pilot(pilot_path)
    problems, attempts = collect(paths)
    pilot_keys, fixture_keys = resolve_pilot(problems, attempts, pilot)
    if len(pilot_keys) != 64 or len(fixture_keys) != 4:
        raise ValueError("Must resolve exactly 64 pilot problems and four engineering fixtures")
    split = split_problems(list(problems.values()), pilot=pilot_keys, fixtures=fixture_keys,
                           duplicate_pairs=duplicate_pairs)
    validate_split(split)
    # Freeze source seeds when present; extra tuning attempts get stable new seeds.
    source_seeds = {(a["problem_key"], a["sample_id"]): a["seed"] for a in attempts.values()}
    planned = {}
    for phase in ("tuning", "test"):
        planned[phase] = []
        for key in sorted(problems):
            if split["membership"][key] != phase:
                continue
            p = problems[key]
            for sample in range(4 if phase == "tuning" else REPETITIONS[p["dataset"]]):
                seed = source_seeds.get((key, sample), int(digest([42, key, sample])[:8], 16))
                planned[phase].append({**p, "problem_key": key, "sample_id": sample,
                    "id": digest([key, sample]), "sampling": {**SAMPLING, "seed": seed}})
    sources = [{"path": str(p), "sha256": file_hash(p)} for p in paths]
    from .vllm_backend import backend_fingerprint
    manifest = seal(dict(schema=1, checkpoint=CHECKPOINT, engine=ENGINE, sampling=SAMPLING,
        backend=backend_fingerprint(),
        split=split, sources=sources, pilot_sha256=file_hash(pilot_path),
        calibration_sources={aid: a for aid, a in attempts.items()
                             if split["membership"][a["problem_key"]] == "calibration"},
        problems=problems, attempts=planned, code=code_hashes(),
        token_scope="generated_reasoning", seed_strategy="saved_seed_else_sha256(42,problem,sample)",
        final_test_state="sealed_until_frontier_freeze"))
    write_json(output, manifest, immutable=True)
    return manifest


def verify_legacy_pilot(path):
    import hashlib, json
    from .common import read_json
    value = read_json(path)
    unsigned = {k: v for k, v in value.items() if k != "manifest_sha256"}
    actual = hashlib.sha256(json.dumps(unsigned, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if actual != value["manifest_sha256"]:
        raise ValueError("Pilot integrity failure")
    return value
