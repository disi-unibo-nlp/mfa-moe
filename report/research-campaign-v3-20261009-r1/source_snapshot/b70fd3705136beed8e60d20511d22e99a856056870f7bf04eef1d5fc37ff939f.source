"""Campaign-v3 adapter. No imports from mutable staging analysis packages.

The metadata inventory is cheap. Packet verification and feature extraction are
separate compute stages; missing checks are never reported as successful checks.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

from .common import digest, write_json
from .contracts import deduplicate, file_binding, load, sampling_weight, save, seal
from .contracts import question_id
from ..provenance import code_provenance, file_sha256


def read_location(location):
    with Path(location["path"]).open("rb") as handle:
        handle.seek(location["byte_offset"])
        raw = handle.read(location["line_bytes"])
    if hashlib.sha256(raw).hexdigest() != location["line_sha256"]:
        raise ValueError("Source trace line changed")
    return json.loads(raw)


def _capture_artifacts(marker):
    files = {}
    for record in marker["files"]:
        path = record.get("path") or record.get("file")
        if not path:
            raise ValueError("Capture artifact has no path")
        kind = "tensor" if path.endswith(".pt.gz") else "packet" if path.endswith(".json.gz") else None
        if kind is None or kind in files:
            raise ValueError("Ambiguous capture artifacts")
        files[kind] = dict(path=path, bytes=record["bytes"], sha256=record["sha256"])
    if set(files) != {"tensor", "packet"}:
        raise ValueError("Incomplete capture artifacts")
    return files


def freeze_inventory(manifests, acceptance, outcomes_path, *, corpus="card-v3"):
    """Freeze accepted manifests/receipts and outcome joins, without loading packets."""
    import pandas as pd
    outcomes_binding = file_binding(outcomes_path)
    outcomes = pd.read_parquet(outcomes_path)
    outcomes = outcomes[outcomes.corpus == corpus]
    outcome_index = {}
    for value in outcomes.to_dict("records"):
        key = (value["model"], value["dataset"], value["problem_id"], int(value["sample_id"]))
        if key in outcome_index:
            raise ValueError("Duplicate outcome identity within budget population")
        outcome_index[key] = value
    inputs = {"outcomes": outcomes_binding}
    rows, cohorts, missing = [], [], []
    for ready_path in sorted(Path(manifests).glob("*/[AB]/ready.json")):
        model, cohort = ready_path.parent.parent.name, ready_path.parent.name
        receipt_path = Path(acceptance) / f"capture-{model}-{cohort}.json"
        if not receipt_path.exists():
            missing.append(dict(model=model, cohort=cohort, reason="no_acceptance_receipt"))
            continue
        ready, receipt = json.loads(ready_path.read_text()), json.loads(receipt_path.read_text())
        if receipt.get("status") != "complete":
            missing.append(dict(model=model, cohort=cohort, reason="capture_not_accepted"))
            continue
        ready_binding = file_binding(ready_path)
        if ready_binding["sha256"] != receipt["ready_sha256"]:
            raise ValueError("Ready manifest differs from accepted capture")
        inputs[str(ready_path)] = ready_binding
        inputs[str(receipt_path)] = file_binding(receipt_path)
        cohort_binding = ready["cohort_binding"]
        if file_sha256(cohort_binding["path"]) != cohort_binding["file_sha256"]:
            raise ValueError("Cohort probabilities changed after capture planning")
        inputs[cohort_binding["path"]] = file_binding(cohort_binding["path"])
        cohort_source = json.loads(Path(cohort_binding["path"]).read_text())
        sampling = {(r["cohort"], r["dataset"], r["problem_id"], r["sample_id"]): r
                    for r in cohort_source["rows"]}
        if len(sampling) != len(cohort_source["rows"]):
            raise ValueError("Duplicate membership in sampling manifest")
        spec, count, tokens = ready["spec"], 0, 0
        for shard in ready["shards"]:
            path = ready_path.parent / f"{shard['index']}.json"
            manifest = json.loads(path.read_text())
            if manifest["spec"] != spec:
                raise ValueError("Shard native model specification differs from ready manifest")
            if manifest["binding"] != shard["binding"] or digest(
                    {k: v for k, v in manifest.items() if k != "binding"}) != shard["binding"]:
                raise ValueError("Shard content/binding mismatch")
            inputs[str(path)] = file_binding(path)
            result_path = Path(manifest["capture_dir"]) / "result.json"
            if receipt["results"].get(str(result_path)) != file_sha256(result_path):
                raise ValueError("Shard result changed after acceptance")
            inputs[str(result_path)] = file_binding(result_path)
            for attempt in manifest["attempts"]:
                ident = attempt["identity"]
                out = outcome_index[(model, ident["dataset"], ident["problem_id"], int(ident["sample_id"]))]
                if out["trace_sha256"] != attempt["trace_sha256"] or int(out["completion_tokens"]) != attempt["completion_tokens"]:
                    raise ValueError("Outcome is bound to a different trace")
                if out["source_problem_id"] != ident["source_problem_id"] or out["termination"] != ident["termination"]:
                    raise ValueError("Outcome question/termination differs from capture")
                marker_path = Path(manifest["capture_dir"]) / (attempt["id"] + ".complete.json")
                marker = json.loads(marker_path.read_text())
                if (marker["id"] != attempt["id"] or marker["manifest_binding"] != shard["binding"]
                        or marker["trace_sha256"] != attempt["trace_sha256"]
                        or marker["tokens"] != attempt["completion_tokens"]):
                    raise ValueError("Capture receipt identity/binding mismatch")
                member = next(c for c in attempt["cohorts"] if c["id"] == cohort)
                sampled = sampling[(cohort, ident["dataset"], ident["problem_id"], ident["sample_id"])]
                if any(sampled.get(k) != member.get(k) for k in
                       ("inclusion_probability", "arm_inclusion_probability", "stratum")):
                    raise ValueError("Capture membership differs from original sampling probabilities")
                correct = out["adjudicated_correct"]
                row = dict(attempt_id=attempt["id"], model=model, model_id=attempt["model"],
                    dataset=ident["dataset"], problem_id=ident["problem_id"],
                    source_problem_id=ident["source_problem_id"], sample_id=int(ident["sample_id"]),
                    question_id=question_id(ident["dataset"], ident["source_problem_id"]),
                    trace_sha256=attempt["trace_sha256"], token_sha256=None,
                    completion_tokens=attempt["completion_tokens"], reasoning_tokens=attempt.get("reasoning_tokens"),
                    is_correct=None if pd.isna(correct) else bool(correct),
                    termination=ident["termination"], capped=bool(ident["capped"]),
                    outcome_version=outcomes_binding["sha256"], budget_population="original",
                    model_revision=spec["revision"], precision=spec["precision"], backend=spec["backend"],
                    layers=spec["layers"], top_k=spec["top_k"], num_experts=spec.get("num_experts", spec.get("experts")),
                    memberships=[dict(cohort=cohort, stratum=member.get("stratum"),
                        inclusion_probability=member.get("inclusion_probability"),
                        arm_inclusion_probability=member.get("arm_inclusion_probability"))],
                    captures=[dict(receipt=file_binding(marker_path), manifest_binding=shard["binding"],
                        artifacts=_capture_artifacts(marker), cohort=cohort,
                        source_location=attempt.get("source_location") or member["locations"][0],
                        annotation_binding=ready.get("annotation_binding"),
                        annotation=member.get("annotation"), seconds=marker.get("seconds"))],
                    integrity="receipts_verified_packet_and_tokens_pending")
                sampling_weight(row, cohort)
                rows.append(row)
                count += 1
                tokens += row["completion_tokens"]
        if count != ready["attempts"] or count != receipt["attempts"] or tokens != receipt["tokens"]:
            raise ValueError("Accepted capture coverage does not match manifests")
        cohorts.append(dict(model=model, cohort=cohort, attempts=count, tokens=tokens,
                            sizing=ready.get("sizing")))
    unique = deduplicate(rows)
    if not unique:
        raise ValueError("No accepted cohorts")
    return seal("inventory", dict(attempts=unique, cohorts=cohorts, unaccepted=missing,
        coverage=coverage(unique), integrity="metadata_only"), inputs=inputs,
        config=dict(corpus=corpus, packet_verification=False), population="original_budget_A_and_B_union")


def coverage(rows):
    result = []
    for model, cohort in sorted({(r["model"], m["cohort"]) for r in rows for m in r["memberships"]}):
        group = [r for r in rows if r["model"] == model and any(m["cohort"] == cohort for m in r["memberships"])]
        result.append(dict(model=model, cohort=cohort, attempts=len(group),
            questions=len({r["question_id"] for r in group}), scored=sum(r["is_correct"] is not None for r in group),
            capped=sum(r["capped"] for r in group), capped_correct=sum(r["capped"] and r["is_correct"] is True for r in group),
            windows=sum(bool(r.get("availability", {}).get("windows")) for r in group),
            native_ids=sum(bool(r.get("availability", {}).get("native_ids")) for r in group),
            executed_occupancy=sum(bool(r.get("availability", {}).get("executed_occupancy")) for r in group),
            gaps=sum(bool(r.get("availability", {}).get("gaps")) for r in group),
            labels=sum(bool(r.get("availability", {}).get("labels")) for r in group),
            verified=sum(r.get("integrity") == "verified" for r in group)))
    return result


def read_packet(capture, *, max_uncompressed_bytes=2 << 30):
    """Verify compressed bytes before JSON decoding; never unpickle to read windows."""
    source = capture["artifacts"]["packet"]
    if Path(source["path"]).stat().st_size != source["bytes"] or file_sha256(source["path"]) != source["sha256"]:
        raise ValueError("Packet hash/size mismatch")
    with gzip.open(source["path"], "rb") as handle:
        raw = handle.read(max_uncompressed_bytes + 1)
    if len(raw) > max_uncompressed_bytes:
        raise ValueError("Packet exceeds configured memory bound")
    packet = json.loads(raw)
    if packet["manifest_binding"] != capture["manifest_binding"]:
        raise ValueError("Packet manifest binding mismatch")
    return packet


def extract_attempt(row, *, landmarks=(1024, 2048, 4096), include_weights=False):
    """Read retained windows first; tensor access is opt-in for missing occupancies."""
    from .evaluation import latest_windows, text_features
    from ..spans import trace_digest
    from ..continuation import original_generation_config
    from moe_exp.schemas import TraceRecord
    # Choose a capture deterministically; duplicated captures retain separate receipts.
    capture = sorted(row["captures"], key=lambda c: (c["cohort"], c["receipt"]["path"]))[0]
    if file_sha256(capture["receipt"]["path"]) != capture["receipt"]["sha256"]:
        raise ValueError("Receipt changed since inventory freeze")
    source = read_location(capture["source_location"])
    packet = read_packet(capture)
    trace = packet["trace"]
    if packet["id"] != row["attempt_id"] or trace_digest(TraceRecord(**trace)) != row["trace_sha256"]:
        raise ValueError("Packet trace mismatch")
    replay = trace["metadata"]["token_replay"]
    source_replay = source["metadata"]["token_replay"]
    token_key = lambda r: [r["prompt_token_ids"], r["completion_token_ids"]]
    if token_key(replay) != token_key(source_replay):
        raise ValueError("Capture differs from exact source tokens")
    tokens = replay["completion_token_ids"]
    if len(tokens) != row["completion_tokens"]:
        raise ValueError("Completion token count mismatch")
    dynamics = trace["metadata"].get("routing_dynamics") or {}
    if dynamics.get("trace_sha256") != row["trace_sha256"]:
        raise ValueError("Stored routing windows are not bound to this trace")
    windows = dynamics["windows"]
    for w in windows:
        if w["layer"] not in row["layers"] or not 0 <= w["start_token"] < w["end_token"] <= len(tokens):
            raise ValueError("Window has invalid native layer or token bounds")
        if w["window"] != w["end_token"] - w["start_token"]:
            raise ValueError("Window width mismatch")
    endpoints = sorted(set(landmarks))
    wanted = {(w["layer"], w["window"], w["end_token"]): dict(w)
              for endpoint in endpoints for w in latest_windows(windows, endpoint)}
    scopes = None
    if include_weights:
        scopes = _add_weight_occupancy(row, capture, wanted.values())
    verified = {**row, "token_sha256": digest(token_key(replay)), "integrity": "verified",
        "generation_config": original_generation_config(source),
        "original_generation_config": source["metadata"]["generation_config"],
        "scope_summaries": scopes,
        "forward_provenance": trace["metadata"].get("forward_provenance"),
        "availability": dict(windows=bool(windows), native_ids=bool(dynamics.get("semantics", {}).get("native_selection")),
            executed_occupancy=bool(wanted) and all("expert_weight_occupancy" in w for w in wanted.values()),
            gaps=bool(windows) and all(w.get("boundary_gaps") is not None for w in windows),
            labels=any(v.get("annotation") for v in packet.get("cohorts", {}).values()))}
    budget = verified["generation_config"]["max_tokens"]
    if len(tokens) > budget:
        raise ValueError("Extended continuation in original-budget inventory")
    offsets = replay["completion_offsets"]
    if len(offsets) != len(tokens):
        raise ValueError("Token offset count mismatch")
    features = []
    for endpoint in endpoints:
        if endpoint >= len(tokens):
            continue
        end_char = max(b for _, b in offsets[:endpoint])
        available = latest_windows(list(wanted.values()), endpoint)
        # A window must actually reach the landmark. A ended reasoning window
        # cannot silently become a current reasoning-phase observation.
        available = [w for w in available if w["end_token"] == endpoint]
        if not available:
            continue
        features.append(dict(attempt_id=row["attempt_id"], trace_id=row["attempt_id"],
            model=row["model"], dataset=row["dataset"], question_id=row["question_id"],
            budget_population="original", decision_token=endpoint, feature_end_token=endpoint,
            base={"token_position": endpoint, "dataset_" + row["dataset"]: 1.},
            text=text_features(trace["cot_text"][:end_char]), state={},
            routing_windows=available, is_correct=row["is_correct"],
            remaining_tokens=len(tokens) - endpoint, capped=row["capped"],
            completion_tokens=len(tokens), memberships=row["memberships"],
            token_sha256=verified["token_sha256"], input_receipt=capture["receipt"]["sha256"]))
    return verified, features


def _add_weight_occupancy(row, capture, windows):
    """Load one accepted selected-ID/weight tensor bundle, reduce bounded slices.

    There is no replay and no reconstruction of unselected probabilities. Loading
    uses weights_only=True and only after verification of the frozen hash.
    """
    import torch
    source = capture["artifacts"]["tensor"]
    if file_sha256(source["path"]) != source["sha256"]:
        raise ValueError("Tensor hash mismatch")
    with gzip.open(source["path"], "rb") as handle:
        tensors = torch.load(handle, map_location="cpu", weights_only=True)
    expected = (len(row["layers"]), row["completion_tokens"], row["top_k"])
    if tuple(tensors["ids"].shape) != expected or tuple(tensors["weights"].shape) != expected:
        raise ValueError("Native tensor shape mismatch")
    for w in windows:
        index = row["layers"].index(w["layer"])
        ids = tensors["ids"][index, w["start_token"]:w["end_token"]].long()
        weights = tensors["weights"][index, w["start_token"]:w["end_token"]].double()
        if (ids < 0).any() or (ids >= row["num_experts"]).any() or (ids.sort(1).values.diff(dim=1) == 0).any():
            raise ValueError("Invalid native expert IDs")
        if not torch.isfinite(weights).all() or (weights < 0).any() or (weights.sum(1) <= 0).any():
            raise ValueError("Invalid dispatched weights")
        counts = torch.bincount(ids.flatten(), minlength=row["num_experts"]).double() / len(ids)
        stored = torch.tensor([w["expert_rates"].get(str(i), 0.) for i in range(row["num_experts"])])
        if not torch.allclose(counts, stored.double(), atol=1e-6):
            raise ValueError("Stored window IDs disagree with tensor IDs")
        weights /= weights.sum(1, keepdim=True)
        occupancy = torch.zeros(row["num_experts"], dtype=torch.float64)
        occupancy.scatter_add_(0, ids.flatten(), weights.flatten())
        w["expert_weight_occupancy"] = {str(i): float(v / len(ids)) for i, v in enumerate(occupancy) if v}
    return _scope_summaries(tensors, row)


def _scope_summaries(tensors, row):
    """Identical native layers/metrics for whole completion and reasoning only."""
    import torch
    reasoning = tensors["reasoning_tokens"].long()
    if (reasoning < 0).any() or (reasoning >= row["completion_tokens"]).any() or (
            reasoning.numel() > 1 and (reasoning.diff() <= 0).any()):
        raise ValueError("Invalid retained reasoning-token indices")
    result = []
    for name, indices in (("whole_completion", torch.arange(row["completion_tokens"])),
                          ("reasoning_only", reasoning)):
        if not len(indices):
            continue
        for position, layer in enumerate(row["layers"]):
            counts = torch.zeros(row["num_experts"], dtype=torch.float64)
            occupancy = torch.zeros_like(counts)
            local = 0.
            for start in range(0, len(indices), 2048):
                selected = indices[start:start + 2048]
                ids = tensors["ids"][position, selected].long()
                w = tensors["weights"][position, selected].double()
                if (w < 0).any() or not torch.isfinite(w).all() or (w.sum(-1) <= 0).any():
                    raise ValueError("Invalid full-scope dispatched weights")
                w /= w.sum(-1, keepdim=True)
                counts += torch.bincount(ids.flatten(), minlength=row["num_experts"])
                occupancy.scatter_add_(0, ids.flatten(), w.flatten())
                local += float(-(w * w.clamp_min(1e-300).log()).sum())
            occupancy /= len(indices)
            marginal = float(-(occupancy * occupancy.clamp_min(1e-300).log()).sum())
            result.append(dict(scope=name, layer=layer, tokens=len(indices),
                expert_rates=(counts / len(indices)).tolist(),
                expert_weight_occupancy=occupancy.tolist(), mixture_local_entropy=local / len(indices),
                mixture_marginal_entropy=marginal, mixture_entropy_difference=marginal - local / len(indices)))
    return result


def extract(inventory_path, output, *, limit_per_cohort=None, include_weights=False,
            landmarks=(1024, 2048, 4096), shard_index=0, shard_count=1, max_output_bytes=None):
    inventory = load(inventory_path, "inventory")
    rows = inventory["payload"]["attempts"]
    if type(shard_count) is not int or type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid extraction shard")
    if max_output_bytes is not None and (type(max_output_bytes) is not int or max_output_bytes <= 0):
        raise ValueError("Invalid extraction storage limit")
    if limit_per_cohort is not None and shard_count != 1:
        raise ValueError("Sizing must cover all cohorts in one measured job")
    if limit_per_cohort is not None:
        if limit_per_cohort < 1:
            raise ValueError("Sizing sample must be positive")
        picked = set()
        for key in sorted({(r["model"], m["cohort"]) for r in rows for m in r["memberships"]}):
            eligible = [r for r in rows if r["model"] == key[0] and any(m["cohort"] == key[1] for m in r["memberships"])]
            # Short and longest inputs bound sizing, not an efficacy sample.
            eligible.sort(key=lambda r: (r["completion_tokens"], r["attempt_id"]))
            order = [eligible[-1], *eligible[:-1]]
            picked.update((r["model"], r["attempt_id"]) for r in order[:limit_per_cohort])
        rows = [r for r in rows if (r["model"], r["attempt_id"]) in picked]
    elif shard_count > 1:
        rows = [r for i, r in enumerate(sorted(rows, key=lambda r: (r["model"], r["attempt_id"])))
                if i % shard_count == shard_index]
    if not rows:
        raise ValueError("Empty extraction shard")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(limit_per_cohort=limit_per_cohort, include_weights=include_weights, landmarks=list(landmarks))
    if shard_count > 1:
        config.update(shard_index=shard_index, shard_count=shard_count)
    if max_output_bytes is not None:
        config["max_output_bytes"] = max_output_bytes
    code = code_provenance()
    population = "sizing_only" if limit_per_cohort else "original_budget_A_and_B_union"
    run_binding = seal("analysis", {"stage": "extraction"},
        inputs={"inventory": inventory["binding"]}, config=config, population=population, code=code)
    save(output / "run.json", run_binding)
    # Each attempt is an independent immutable artifact; resume cannot append duplicates.
    metrics, paths, verified_rows, output_bytes = [], [], [], 0
    for row in rows:
        path = output / row["model"] / (row["attempt_id"] + ".json")
        if path.exists():
            artifact = load(path, "features")
            if (artifact["inputs"] != {"inventory": inventory["binding"]}
                    or artifact["config"] != config or artifact["code"] != code):
                raise ValueError("Incompatible resumed extraction")
        else:
            start = time.monotonic()
            verified, features = extract_attempt(row, landmarks=landmarks, include_weights=include_weights)
            artifact = seal("features", dict(attempt=verified, rows=features,
                measurement=dict(seconds=time.monotonic() - start,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
                inputs={"inventory": inventory["binding"]}, config=config,
                population=population, code=code)
            serialized_bytes = len(json.dumps(artifact, indent=2, allow_nan=False).encode()) + 1
            if max_output_bytes is not None and output_bytes + serialized_bytes > max_output_bytes:
                raise RuntimeError("Predefined extraction storage limit reached; completed artifacts remain resumable")
            save(path, artifact)
        output_bytes += path.stat().st_size
        if max_output_bytes is not None and output_bytes > max_output_bytes:
            raise RuntimeError("Existing artifacts exceed the frozen extraction storage limit")
        paths.append(file_binding(path))
        verified_rows.append(artifact["payload"]["attempt"])
        metrics.append(dict(model=row["model"], attempt_id=row["attempt_id"],
            completion_tokens=row["completion_tokens"], output_bytes=path.stat().st_size,
            **artifact["payload"]["measurement"]))
    result = seal("analysis", dict(features=paths, coverage=coverage(verified_rows), measurements=metrics),
        inputs={"inventory": inventory["binding"]}, config=config,
        population=population, code=code)
    save(output / "manifest.json", result)
    return result


def merge_extractions(inventory_path, manifests):
    """Verify every immutable shard and require exactly the full input union."""
    inventory = load(inventory_path, "inventory")
    expected = {(r["model"], r["attempt_id"]) for r in inventory["payload"]["attempts"]}
    seen, shards, paths, measurements, attempts = set(), set(), [], [], []
    config, code = None, None
    for path in manifests:
        manifest = load(path, "analysis")
        if manifest["population"] != "original_budget_A_and_B_union" or manifest["inputs"] != {"inventory": inventory["binding"]}:
            raise ValueError("Incompatible extraction population or inventory")
        current = dict(manifest["config"])
        shard = current.pop("shard_index", 0)
        count = current.pop("shard_count", 1)
        if (count, shard) in shards or not 0 <= shard < count:
            raise ValueError("Duplicate or invalid extraction shard")
        shards.add((count, shard))
        if config is None:
            config, code = current, manifest["code"]
        if config != current or code != manifest["code"]:
            raise ValueError("Extraction shards differ in configuration or code")
        for ref in manifest["payload"]["features"]:
            if file_binding(ref["path"]) != ref:
                raise ValueError("Feature file changed after extraction")
            artifact = load(ref["path"], "features")
            if (artifact["inputs"] != manifest["inputs"] or artifact["config"] != manifest["config"]
                    or artifact["code"] != code or artifact["population"] != manifest["population"]):
                raise ValueError("Feature is not part of this frozen shard")
            attempt = artifact["payload"]["attempt"]
            key = (attempt["model"], attempt["attempt_id"])
            if key in seen or key not in expected:
                raise ValueError("Duplicate or unexpected extracted attempt")
            seen.add(key)
            attempts.append(attempt)
            paths.append(ref)
        measurements.extend(manifest["payload"]["measurements"])
    if not shards or len({n for n, _ in shards}) != 1 or len(shards) != next(iter(shards))[0] or seen != expected:
        raise ValueError("Incomplete extraction shard union")
    metric_keys = [(r["model"], r["attempt_id"]) for r in measurements]
    if len(metric_keys) != len(seen) or set(metric_keys) != seen:
        raise ValueError("Measurements do not cover the extracted attempts exactly once")
    return seal("analysis", dict(features=paths, coverage=coverage(attempts), measurements=measurements,
        shards=[file_binding(p) for p in manifests]), inputs={"inventory": inventory["binding"]},
        config=config, code=code, population="original_budget_A_and_B_union")


def measured_resource_estimate(inventory, extraction, *, startup_attempt_id=None):
    """Conservative per-model CPU estimates from the observed short/long sizing batch."""
    if extraction["inputs"]["inventory"] != inventory["binding"]:
        raise ValueError("Sizing and inventory inputs disagree")
    measured = extraction["payload"]["measurements"]
    startup = None
    if startup_attempt_id is not None:
        # Historical sizing timed lazy imports inside the first attempt. Its
        # entire observed duration is a conservative once-per-worker allowance;
        # the initialization fraction itself was not separately measured.
        if not measured or measured[0]["attempt_id"] != startup_attempt_id:
            raise ValueError("Only the first timed attempt can include process initialization")
        startup = measured[0]
    measurements = defaultdict(list)
    for row in measured:
        if startup is row:
            continue
        measurements[row["model"]].append(row)
    estimates = []
    for model, points in sorted(measurements.items()):
        if len({p["completion_tokens"] for p in points}) < 2:
            estimates.append(dict(model=model, status="insufficient_sizing_range"))
            continue
        shortest = min(points, key=lambda p: p["completion_tokens"])
        longest = max(points, key=lambda p: p["completion_tokens"])
        slope = max(0., (longest["seconds"] - shortest["seconds"]) /
                    (longest["completion_tokens"] - shortest["completion_tokens"]))
        overhead = max(0., max(p["seconds"] - slope * p["completion_tokens"] for p in points))
        population = [r for r in inventory["payload"]["attempts"] if r["model"] == model]
        seconds = sum(overhead + slope * r["completion_tokens"] for r in population)
        all_points = [p for p in measured if p["model"] == model]
        output_bytes = max(p["output_bytes"] for p in all_points) * len(population)
        estimates.append(dict(model=model, status="measured_projection", attempts=len(population),
            observed_points=points, seconds_per_token=slope, per_attempt_seconds=overhead,
            projected_cpu_seconds=seconds, proposed_wall_seconds_with_25pct_margin=1.25 * seconds,
            conservative_output_bytes=output_bytes,
            observed_job_peak_rss_kib=max(p["peak_rss_kib"] for p in points),
            gpu_hours=0, new_labels=0, tensor_replay=False))
    return seal("sizing", dict(models=estimates, authorization="not_granted_by_this_artifact",
        startup_allowance_seconds_per_worker=startup["seconds"] if startup else 0.,
        startup_observation=startup,
        startup_caveat="First duration includes lazy imports; initialization fraction not separately measured" if startup else None),
        inputs={"inventory": inventory["binding"], "sizing": extraction["binding"]},
        config=dict(runtime_margin=1.25, estimator="nonnegative_short_long_envelope",
                    startup_attempt_id=startup_attempt_id),
        population=inventory["population"])


def inventory_report(inventory, path):
    rows = inventory["payload"]["attempts"]
    b = [r for r in rows if any(m["cohort"] == "B" for m in r["memberships"])]
    lines = ["# Frozen routing inventory", "", f"Binding: `{inventory['binding']}`", "",
        "Receipt and outcome bindings were checked. Packet/tensor bytes, windows and exact token hashes still require extraction.", "",
        "| Model | Cohort | Attempts | Questions | Scored | Capped | Capped correct |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in inventory["payload"]["coverage"]:
        lines.append("| " + " | ".join(str(r[k]) for k in
            ("model", "cohort", "attempts", "questions", "scored", "capped", "capped_correct")) + " |")
    lines += ["", f"B contains {len(b)} unique model-attempts and {len({r['question_id'] for r in b})} distinct questions across models.",
        "", "B arm weights are conditional on historical eligibility. Capped-correct attempts retain their adjudicated correctness and their historical sampling stratum.",
        "", "A/B duplicate captures remain separately auditable; each model-attempt enters analysis once. Extended continuations are excluded.", ""]
    Path(path).write_text("\n".join(lines))
