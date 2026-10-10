"""Seal missing-label inventory and execute multiple original shards per load.

The historical frame, exact prompts, sampler, batches and assignment bindings
stay fixed. Recovery writes a new overlay; it never edits the old label tree.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import time

import generated_dense_fresh_metadata_recovery_v1 as adapter
import generated_dense_pipeline_v1 as native
from moe_exp.routing_control.study_v1 import digest, file_sha, require, save, sealed

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "report/routing-study-v1"
SOURCE = native.ROOT / "runs/routing-control-v1/generated-dense-fresh-metadata-recovery-v1/dense-generated-v1-6b5aa286e46c-a77287237f3a"


def compute(gpu=False):
    native.require_step(gpu)
    require(os.environ.get("TMPDIR", "").startswith("/leonardo_work/IscrC_MIOSR/lmolfett/"),
            "Slurm TMPDIR must be owned WORK staging")


def historical_inputs(source):
    # Validation remains the exact frozen historical adapter in a fresh process.
    adapter.install_adapter()
    return adapter.inputs(source)


def codes():
    return {str(p): file_sha(p) for p in (Path(__file__).resolve(),
        REPO / "src/moe_exp/routing_control/study_v1.py",
        Path(__file__).with_name("routing_study_v1.sbatch"))}


def inventory(source):
    frame, price = historical_inputs(source)
    require(frame["sentences"] == 187524, "fresh dense assignment count changed")
    rows, shards, reused, ambiguous = [], [], [], []
    for index, shard in enumerate(price["shards"]):
        out = source / f"labels-shard-{index:03d}"
        body = native.label_binding(frame, price, index)
        binding = {**body, "sha256": digest(body)}
        if (out / "BINDING.json").exists():
            require(sealed(out / "BINDING.json") == binding, "historical shard rebound")
        missing = []
        for start in range(shard["start"], shard["stop"], native.BATCH):
            stop = min(start + native.BATCH, shard["stop"])
            batch = native.committed_labels(out, start, stop, binding, price["prompt_records"])
            if batch is not None:
                rows.extend(batch["records"])
                reused.append({"shard": index, "start": start, "stop": stop,
                    "batch_sha256": batch["sha256"],
                    "batch_file_sha256": file_sha(out / "batches" / f"{start:07d}.json"),
                    "attempt_file_sha256": file_sha(out / "attempts" / f"{start:07d}.json")})
            else:
                item = {"shard": index, "start": start, "stop": stop,
                        "blind_ids": [r["blind_id"] for r in frame["records"][start:stop]]}
                attempt_path = out / "attempts" / f"{start:07d}.json"
                if attempt_path.exists():
                    attempt = sealed(attempt_path)
                    require(attempt["binding_sha256"] == binding["sha256"] and
                            attempt["start"] == start and attempt["stop"] == stop and
                            attempt["blind_ids"] == item["blind_ids"], "ambiguous attempt identity changed")
                    item["preserved_attempt"] = {"path": str(attempt_path), "sha256": attempt["sha256"],
                                                 "job_id": attempt["job_id"]}
                    ambiguous.append(item["preserved_attempt"])
                missing.append(item)
        shards.append({"shard": index, "missing_batches": missing})
    require(len({r["blind_id"] for r in rows}) == len(rows), "duplicate committed sentence labels")
    missing_count = sum(b["stop"] - b["start"] for s in shards for b in s["missing_batches"])
    require(len(rows) + missing_count == frame["sentences"], "inventory misses assigned labels")
    return {"schema": "routing-study-dense-recovery-inventory-v2", "source": str(source),
        "frame_sha256": frame["sha256"], "price_sha256": price["sha256"],
        "assigned": frame["sentences"], "committed": len(rows), "missing": missing_count,
        "valid_stopped": sum(r["finish_reason"] == "stop" and r["label"] in native.qualified.CLASSES for r in rows),
        "reused_batches": reused, "shards": shards, "ambiguous_calls": ambiguous,
        "code_files": codes(), "scope": "Every original sentence assigned exactly once; invalid labels retain gaps."}


def pack(inventory_value, price, max_wall=86400):
    """Price all missing exact batches, with one recovery load per packed job."""
    load = price["cold_load_seconds"]
    shutdown = price["shutdown_seconds"]
    overhead = price["per_shard_overhead_seconds"]
    fixed = 2 * (load + shutdown) + overhead
    require(max_wall > fixed, "job wall cannot contain load and recovery reserve")
    groups, current, seconds = [], [], 0.
    for shard in inventory_value["shards"]:
        batches = shard["missing_batches"]
        if not batches:
            continue
        prompts = sum(sum(p["prompt_tokens"] for p in price["prompt_records"][b["start"]:b["stop"]]) for b in batches)
        count = sum(b["stop"] - b["start"] for b in batches)
        work = price["repeat_factor"] * (prompts / price["bounded_prefill_tokens_per_second"] +
                                        count * price["max_tokens"] / price["bounded_decode_tokens_per_second"])
        require(work + fixed <= max_wall, "one original shard exceeds packed wall")
        if current and seconds + work + fixed > max_wall:
            groups.append({"shards": current, "work_seconds": seconds,
                           "wall_seconds": math.ceil(seconds + fixed)})
            current, seconds = [], 0.
        current.append(shard["shard"])
        seconds += work
    if current:
        groups.append({"shards": current, "work_seconds": seconds,
                       "wall_seconds": math.ceil(seconds + fixed)})
    return {"schema": "routing-study-dense-recovery-price-v2", "inventory_sha256": inventory_value["sha256"],
        "historical_price_sha256": price["sha256"], "groups": groups, "gpus": 2,
        "max_wall_seconds": max_wall, "remaining_labels": inventory_value["missing"],
        "loads_including_bounded_recovery": 2 * len(groups),
        "maximum_GPU_hours": sum(2 * g["wall_seconds"] / 3600 for g in groups),
        "reserved_GPU_hours": len(groups) * 2 * max_wall / 3600,
        "qualification_GPU_hours": 1., "sampler": {"temperature": 0., "max_tokens": native.MAX_TOKENS},
        "status": "PRICED_PENDING_PACKAGING_QUALIFICATION"}


def validate(directory):
    manifest = sealed(directory / "MANIFEST.json")
    require(manifest["code_files"] == codes(), "recovery sources changed")
    inv = sealed(directory / "INVENTORY.json")
    price = sealed(directory / "PRICE.json")
    require(manifest["inventory_sha256"] == inv["sha256"] and price["inventory_sha256"] == inv["sha256"],
            "recovery manifest inventory changed")
    source = Path(inv["source"])
    frame, old_price = historical_inputs(source)
    require(frame["sha256"] == inv["frame_sha256"] and old_price["sha256"] == inv["price_sha256"],
            "historical frame or price changed")
    return manifest, inv, price, frame, old_price


def prepare(directory, source=SOURCE):
    compute()
    require(not directory.exists() or not (directory / "MANIFEST.json").exists(), "already prepared recovery")
    inv = save(directory / "INVENTORY.json", inventory(source))
    frame, old_price = historical_inputs(source)
    price = save(directory / "PRICE.json", pack(inv, old_price))
    # Use read-only references to immutable source metadata for native analysis.
    for name in ("BLIND_FRAME.json", "LABEL_PRICE.json", "ARM_MAP.json", "NORMALIZED_SOURCE_CONTEXT.json"):
        (directory / name).symlink_to(source / name)
    for shard in inv["shards"]:
        index = shard["shard"]
        out = directory / f"labels-shard-{index:03d}"
        for name in ("batches", "attempts", "loads"):
            (out / name).mkdir(parents=True, exist_ok=True)
        save(out / "BINDING.json", native.label_binding(frame, old_price, index))
    for row in inv["reused_batches"]:
        old = source / f'labels-shard-{row["shard"]:03d}'
        new = directory / old.name
        for folder, expected in (("batches", row["batch_file_sha256"]), ("attempts", row["attempt_file_sha256"])):
            name = f'{row["start"]:07d}.json'
            require(file_sha(old / folder / name) == expected, "reused batch changed since inventory")
            (new / folder / name).symlink_to(old / folder / name)
    samples, seen = [], set()
    for batch in inv["reused_batches"]:
        if batch["shard"] not in seen:
            samples.append(batch)
            seen.add(batch["shard"])
        if len(samples) == 2:
            break
    manifest = save(directory / "MANIFEST.json", {"schema": "routing-study-dense-recovery-manifest-v2",
        "inventory_sha256": inv["sha256"], "price_sha256": price["sha256"], "code_files": codes(),
        "measurement_changed": False, "max_labels": 187524, "qualification_batches": samples,
        "historical_writes": False, "execution": "Original batches and shard identities; shared model across original shards."})
    print(json.dumps({"status": "PREPARED", "manifest_sha256": manifest["sha256"],
        "committed": inv["committed"], "missing": inv["missing"], "ambiguous_calls": len(inv["ambiguous_calls"]),
        "packed_jobs": len(price["groups"]), "maximum_GPU_hours": price["maximum_GPU_hours"]}), flush=True)


def model_and_params(directory):
    from vllm import LLM, SamplingParams
    began = time.monotonic()
    model = LLM(model=str(native.qualified.MODEL), tokenizer=str(native.qualified.MODEL), **native.PROFILE)
    save(directory / ("LOAD-" + os.environ["SLURM_JOB_ID"] + ".json"), {
        "schema": "routing-study-recovery-load-v2", "job_id": os.environ["SLURM_JOB_ID"],
        "load_seconds": time.monotonic() - began, "model_profile": native.PROFILE})
    return model, SamplingParams(temperature=0., max_tokens=native.MAX_TOKENS)


def call(model, params, block, prompts):
    outputs = model.chat([native.qualified.messages(r, native.instruction()) for r in block],
                         sampling_params=params,
                         chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"}, use_tqdm=False)
    require(len(outputs) == len(block), "dense label output count differs")
    records = []
    for row, output, prompt in zip(block, outputs, prompts, strict=True):
        require(len(output.outputs) == 1 and digest(list(output.prompt_token_ids)) == prompt["prompt_ids_sha256"],
                "packed live prompt differs from historical exact prompt")
        one = output.outputs[0]
        require(one.finish_reason in ("stop", "length") and len(one.token_ids) <= native.MAX_TOKENS,
                "packed finish reason or cap changed")
        records.append({"blind_id": row["blind_id"], "label": native.qualified.parse_label(one.text),
            "finish_reason": one.finish_reason, "generated_tokens": len(one.token_ids), "raw_completion": one.text,
            "prompt_tokens": len(output.prompt_token_ids), "prompt_ids_sha256": digest(list(output.prompt_token_ids))})
    return records


def qualify(directory):
    compute(True)
    manifest, inv, _, frame, price = validate(directory)
    require(len(manifest["qualification_batches"]) == 2 and
            len({b["shard"] for b in manifest["qualification_batches"]}) == 2,
            "packaging qualification requires two original shards")
    path = directory / "PACKAGING_QUALIFICATION.json"
    require(not path.exists() and not (directory / "QUALIFICATION_ATTEMPT.json").exists(),
            "qualification already attempted; reconcile before another load")
    save(directory / "QUALIFICATION_ATTEMPT.json", {"manifest_sha256": manifest["sha256"],
        "job_id": os.environ["SLURM_JOB_ID"], "batches": manifest["qualification_batches"]})
    model, params = model_and_params(directory)
    results = []
    for batch in manifest["qualification_batches"]:
        start, stop = batch["start"], batch["stop"]
        rows = call(model, params, frame["records"][start:stop], price["prompt_records"][start:stop])
        old = sealed(Path(inv["source"]) / f'labels-shard-{batch["shard"]:03d}/batches/{start:07d}.json')
        require(old["sha256"] == batch["batch_sha256"], "qualification reference changed")
        results.extend({"blind_id": r["blind_id"], "prompt_ids_sha256": r["prompt_ids_sha256"],
            "label": r["label"], "finish_reason": r["finish_reason"], "reference_label": o["label"],
            "reference_finish_reason": o["finish_reason"], "agree": r["label"] == o["label"]}
            for r, o in zip(rows, old["records"], strict=True))
    valid = sum(r["finish_reason"] == "stop" and r["label"] in native.qualified.CLASSES for r in results)
    agree = sum(r["agree"] for r in results)
    passed = valid >= math.ceil(.95 * len(results)) and agree >= math.ceil(.95 * len(results))
    result = save(path, {"schema": "routing-study-dense-packaging-qualification-v2",
        "manifest_sha256": manifest["sha256"], "job_id": os.environ["SLURM_JOB_ID"],
        "records": results, "status": "PASS" if passed else "FAILED",
        "model_loads": 1, "original_shards": 2, "prompt_contract_exact": True,
        "valid_stopped": valid, "label_agreement": agree, "threshold_fraction": .95})
    require(passed, "packed execution qualification failed; full recovery held")
    print(json.dumps({"status": result["status"], "valid": valid, "agreed": agree}), flush=True)


def run(directory, group_index):
    compute(True)
    manifest, inv, price, frame, historical_price = validate(directory)
    qualification = sealed(directory / "PACKAGING_QUALIFICATION.json")
    require(qualification["status"] == "PASS" and qualification["manifest_sha256"] == manifest["sha256"],
            "packaging is not qualified")
    require(type(group_index) is int and 0 <= group_index < len(price["groups"]), "unknown packed group")
    group = price["groups"][group_index]
    with ExitStack() as stack:
        pending = []
        for index in group["shards"]:
            out = directory / f"labels-shard-{index:03d}"
            lock = stack.enter_context((out / "WRITER.lock").open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            binding = sealed(out / "BINDING.json")
            for batch in inv["shards"][index]["missing_batches"]:
                start, stop = batch["start"], batch["stop"]
                if native.committed_labels(out, start, stop, binding, historical_price["prompt_records"]) is not None:
                    continue
                require(not (out / "attempts" / f"{start:07d}.json").exists(),
                        "uncommitted recovery call requires new bounded recovery inventory")
                pending.append((out, binding, batch))
        model, params = model_and_params(directory) if pending else (None, None)
        for out, binding, batch in pending:
            start, stop = batch["start"], batch["stop"]
            attempt = save(out / "attempts" / f"{start:07d}.json", {
                "schema": "generated-dense-label-attempt-v1", "binding_sha256": binding["sha256"],
                "job_id": os.environ["SLURM_JOB_ID"], "start": start, "stop": stop,
                "blind_ids": batch["blind_ids"], "state": "before_model_chat",
                "preserved_historical_attempt": batch.get("preserved_attempt"),
                "recovery_manifest_sha256": manifest["sha256"]})
            began = time.monotonic()
            rows = call(model, params, frame["records"][start:stop], historical_price["prompt_records"][start:stop])
            save(out / "batches" / f"{start:07d}.json", {"schema": "generated-dense-label-batch-v1",
                "binding_sha256": binding["sha256"], "attempt_sha256": attempt["sha256"],
                "job_id": os.environ["SLURM_JOB_ID"], "start": start, "stop": stop,
                "wall_seconds": time.monotonic() - began, "records": rows})
            native.committed_labels(out, start, stop, binding, historical_price["prompt_records"])
            print(json.dumps({"group": group_index, "shard": batch["shard"], "complete_through": stop}), flush=True)
        for index in group["shards"]:
            native.seal_label_shard(directory, frame, historical_price, index)


def analyze(directory):
    compute()
    _, inv, _, frame, price = validate(directory)
    # Complete previously finished historical shards in the new overlay only.
    for index in range(len(price["shards"])):
        native.seal_label_shard(directory, frame, price, index)
    native.inputs = lambda requested: (frame, price) if Path(requested).resolve() == directory.resolve() else historical_inputs(requested)
    native.save = save
    result = native.analyze(directory)
    save(directory / "RECOVERY_COMPLETION.json", {"schema": "routing-study-dense-recovery-completion-v2",
        "inventory_sha256": inv["sha256"], "assigned": inv["assigned"], "reused": inv["committed"],
        "recovered": inv["missing"], "unresolved_execution": 0, "analysis_sha256": result["sha256"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "qualify", "run", "analyze"))
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--group-index", type=int)
    args = p.parse_args()
    require(args.directory.resolve().is_relative_to(REPO / "report/routing-study-v1"), "unowned recovery output")
    if args.stage == "prepare": prepare(args.directory)
    elif args.stage == "qualify": qualify(args.directory)
    elif args.stage == "run": run(args.directory, args.group_index)
    else: analyze(args.directory)


if __name__ == "__main__":
    main()
