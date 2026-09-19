"""Convert correlation traces to a reproducible problem-disjoint experiment."""
import hashlib
import json
from pathlib import Path

from moe_exp.jsonl import iter_jsonl
from .calibration import problem_key


def has_margin_data(row):
    features = (row.get("metadata") or {}).get("correlation_features")
    return bool(features or (row.get("model_logs") or {}).get("router_logits"))


def prepare(paths, output_dir: Path, calibration_fraction=.7, seed=42):
    if not 0 < calibration_fraction < 1:
        raise ValueError("Calibration fraction must be between zero and one")
    traces, seen = [], set()
    excluded = {"unscored": 0, "missing_routing": 0}
    for path in paths:
        for row in iter_jsonl(path):
            if type(row.get("is_correct")) is not bool:
                excluded["unscored"] += 1
                continue
            if not has_margin_data(row):
                excluded["missing_routing"] += 1
                continue
            identity = (row["dataset"], row["problem_id"], row.get("sample_id", 0))
            if identity in seen:
                raise ValueError(f"Duplicate trace: {identity}")
            seen.add(identity)
            traces.append(row)
    keys = sorted({problem_key(r) for r in traces}, key=lambda k:
                  hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
    if len(keys) < 2:
        raise ValueError("Need at least two scored problems with routing tensors")
    cutoff = max(1, min(len(keys)-1, int(len(keys) * calibration_fraction)))
    calibration_keys = set(keys[:cutoff])
    calibration, prompts = [], []
    for row in traces:
        if problem_key(row) in calibration_keys:
            calibration.append(row)
        else:
            prompts.append({
                "id": json.dumps([row["dataset"], row["problem_id"], row.get("sample_id", 0)]),
                **{key: row.get(key) for key in ("dataset", "problem_id", "source_problem_id",
                                                 "prompt", "gold_answer", "system_prompt",
                                                 "generation_messages")},
                "answer_type": row.get("metadata", {}).get("answer_type", "math"),
            })
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, records in (("calibration.jsonl", calibration), ("prompts.jsonl", prompts)):
        with (output_dir / name).open("x") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = dict(seed=seed, calibration_fraction=calibration_fraction, excluded=excluded,
                    calibration_problems=len(calibration_keys), evaluation_problems=len(keys)-cutoff,
                    calibration_traces=len(calibration), evaluation_prompts=len(prompts),
                    sources=[{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                             for p in paths])
    (output_dir / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_global(generation_paths, routing_paths, output_dir: Path,
                   calibration_fraction=.7, seed=42):
    """Split the full generation population, retaining all calibration attempts.

    Evaluation eligibility never depends on saved correctness or routing coverage.
    Write both one-per-problem and all-attempt prompt files for the same holdout.
    """
    from collections import Counter, defaultdict

    if not 0 < calibration_fraction < 1:
        raise ValueError("Calibration fraction must be between zero and one")
    generation_paths, routing_paths = list(generation_paths), list(routing_paths)
    if not generation_paths or not routing_paths:
        raise ValueError("Generation and routing files are both required")
    problems, models, seen = defaultdict(list), set(), set()
    for path in generation_paths:
        for row in iter_jsonl(path):
            identity = (row["dataset"], row["problem_id"], row.get("sample_id", 0))
            if identity in seen:
                raise ValueError(f"Duplicate generation trace: {identity}")
            seen.add(identity)
            models.add(row["model_id"])
            key = problem_key(row)
            prompt = {
                "id": json.dumps(identity),
                **{name: row.get(name) for name in ("dataset", "problem_id", "source_problem_id",
                   "prompt", "gold_answer", "system_prompt", "generation_messages")},
                "sample_id": row.get("sample_id", 0),
                "answer_type": row.get("metadata", {}).get("answer_type", "math"),
            }
            if not prompt["prompt"] or not str(prompt["gold_answer"] or "").strip():
                raise ValueError(f"Missing prompt/gold answer: {identity}")
            if problems[key]:
                previous = problems[key][0]
                for field in ("prompt", "gold_answer", "system_prompt", "generation_messages",
                              "answer_type"):
                    if previous[field] != prompt[field]:
                        raise ValueError(f"Attempts disagree on {field}: {key}")
            problems[key].append(prompt)
    if len(models) != 1:
        raise ValueError("Use generations from exactly one checkpoint")
    datasets = defaultdict(list)
    for key, attempts in problems.items():
        datasets[attempts[0]["dataset"]].append(key)
    calibration_keys = set()
    counts = {}
    for dataset, keys in sorted(datasets.items()):
        if len(keys) < 2:
            raise ValueError(f"Need at least two problems for dataset {dataset}")
        keys.sort(key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
        cutoff = max(1, min(len(keys)-1, int(len(keys)*calibration_fraction)))
        calibration_keys.update(keys[:cutoff])
        counts[dataset] = dict(total_problems=len(keys), calibration_problems=cutoff,
                               evaluation_problems=len(keys)-cutoff)
    calibration, routing_seen = [], set()
    exclusions = Counter()
    for path in routing_paths:
        for row in iter_jsonl(path):
            key = problem_key(row)
            if key not in calibration_keys:
                continue
            identity = (row["dataset"], row["problem_id"], row.get("sample_id", 0))
            if identity not in seen or row.get("model_id") not in models:
                raise ValueError(f"Routing trace does not match generation population: {identity}")
            if identity in routing_seen:
                raise ValueError(f"Duplicate routing trace: {identity}")
            routing_seen.add(identity)
            if type(row.get("is_correct")) is not bool:
                exclusions["unscored_calibration"] += 1
                continue
            if not has_margin_data(row):
                exclusions["missing_calibration_margin_data"] += 1
                continue
            calibration.append({name: row.get(name) for name in (
                "dataset", "problem_id", "source_problem_id", "sample_id", "model_id",
                "is_correct", "model_logs", "metadata")})
    supported = {problem_key(r) for r in calibration}
    if {r["is_correct"] for r in calibration} != {False, True}:
        raise ValueError("Global calibration needs correct and incorrect scored routing traces")
    for dataset in counts:
        group = [r for r in calibration if r["dataset"] == dataset]
        if not group:
            raise ValueError(f"No usable calibration routing for {dataset}")
        counts[dataset]["usable_calibration_problems"] = len({problem_key(r) for r in group})
        counts[dataset]["calibration_attempts"] = len(group)
    single, full = [], []
    for key in sorted(problems):
        if key in calibration_keys:
            continue
        attempts = sorted(problems[key], key=lambda r: (r["sample_id"], r["id"]))
        single.append(attempts[0])
        full.extend(attempts)
    manifest = dict(seed=seed, calibration_fraction=calibration_fraction,
                    policy_scope="global_problem_balanced_margin", generation_models=sorted(models),
                    datasets=counts, excluded=dict(exclusions),
                    calibration_problems=len(calibration_keys),
                    usable_calibration_problems=len(supported), calibration_traces=len(calibration),
                    evaluation_problems=len(single), evaluation_attempts=len(full))
    sources = []
    for kind, paths in (("generation", generation_paths), ("routing", routing_paths)):
        for path in paths:
            with path.open("rb") as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                sha = digest.hexdigest()
            sources.append(dict(kind=kind, path=str(path), sha256=sha))
    manifest["sources"] = sources
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, records in (("calibration.jsonl", calibration), ("prompts.single.jsonl", single),
                          ("prompts.full.jsonl", full)):
        with (output_dir/name).open("x") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False)+"\n")
    (output_dir/"split.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest
