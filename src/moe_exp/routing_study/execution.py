"""Durable study generation/replay using the colleague's locking/atomic infrastructure."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from .common import (CHECKPOINT, ENGINE, MODEL, REVISION, digest, verify, read_json, write_json,
    rows, file_hash, run_lock, require_compute, private_path, code_hashes, attempt_key)
from .policy import validate
from .prepare import validate_split
from .vllm_backend import backend_fingerprint, verify_reports

MATCH = ("study_sha256", "checkpoint", "engine", "phase", "planned_sha256", "backend",
         "code", "token_scope", "scoring_contract")
SESSION_ID = digest([os.getpid(), time.time_ns(), os.environ.get("SLURM_JOB_ID")])
SCORING = "correlation_pipeline.score_completion; known_gold_missing_answer_incorrect; caps_retained_v1"


def check_study(study):
    verify(study)
    validate_split(study["split"])
    if study["checkpoint"] != CHECKPOINT or study["engine"] != ENGINE:
        raise ValueError("Study execution/checkpoint changed")
    if study["code"] != code_hashes():
        raise ValueError("Study code changed; prepare a new frozen manifest")


def validate_acceptance(acceptance, study, backend):
    verify(acceptance)
    if (acceptance["status"] != "passed" or acceptance["code"] != code_hashes()
            or acceptance["checkpoint"] != CHECKPOINT or acceptance["backend"] != backend
            or not acceptance.get("job_id") or not acceptance.get("dispatch_verified")
            or not acceptance.get("checkpoint_content_verified")
            or not acceptance.get("zero_matches_native") or not acceptance.get("weight_ids_preserved")):
        raise ValueError("Authorized smoke acceptance is absent/stale/incomplete")


def planned_rows(study, phase, selection=None, policy=None):
    if phase not in ("tuning", "test"):
        raise ValueError("Unsupported generation phase")
    if phase == "test":
        if selection is None:
            raise ValueError("Final-test outcomes remain sealed until a tuning frontier is frozen")
        verify(selection)
        if (selection["study_sha256"] != study["sha256"]
                or selection["phase"] != "tuning" or selection["stage"] != "final"
                or policy["sha256"] not in selection["allowed_policy_hashes"]):
            raise ValueError("Policy is not in the frozen final shortlist/controls")
    return study["attempts"][phase]


def replay_rows(study):
    wanted = study["calibration_sources"]
    seen = set()
    for source in study["sources"]:
        if file_hash(Path(source["path"])) != source["sha256"]:
            raise ValueError("Calibration source file changed")
        for row in rows(source["path"]):
            aid = attempt_key(row)
            if aid not in wanted:
                continue
            replay = row["metadata"]["token_replay"]
            ids, completion = replay["prompt_token_ids"], replay["completion_token_ids"]
            offsets = replay["completion_offsets"]
            if len(offsets) != len(completion) or not completion:
                raise ValueError("Calibration requires exact saved token alignment")
            correct = row["is_correct"]
            if type(correct) is not bool:
                # Known gold but absent/unparseable source answers are incorrect.
                correct = False
            seen.add(aid)
            yield dict(id=aid, problem_key=wanted[aid]["problem_key"], prompt_token_ids=ids,
                replay_token_ids=ids + completion, is_correct=correct,
                source_record_sha256=digest(row), sampling=dict(max_tokens=1, temperature=0., seed=42))
    if seen != set(wanted):
        raise ValueError("Missing planned calibration attempts")


def check_record(record, expected, policy_sha256):
    if (record["id"] != expected["id"] or record["input_sha256"] != digest(expected)
            or record["policy_sha256"] != policy_sha256
            or record["prompt_token_ids"] != expected["prompt_token_ids"]
            or record["sampling_args"] != expected["sampling"]
            or record["generated_token_count"] > expected["sampling"]["max_tokens"]
            or any(type(t) is not int or t < 0 for t in record["generated_token_ids"])
            or record["generated_token_count"] != len(record["generated_token_ids"])
            or type(record["is_correct"]) is not bool or record["finish_reason"] not in ("stop", "length")
            or ("replay_token_ids" not in expected and record.get("model_answer") in (None, "") and record["is_correct"])):
        raise ValueError("Saved attempt inputs/coverage/scoring changed")


def saved_records(path, expected, policy_sha256):
    saved = {}
    if path.exists():
        for record in rows(path):
            key = record["id"]
            if key in saved or key not in expected:
                raise ValueError("Duplicate/unplanned saved attempt")
            check_record(record, expected[key], policy_sha256)
            saved[key] = record
    return saved


def append_record(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def repair_interrupted_tail(path):
    """Only discard a non-newline-terminated final record after acquiring the run lock."""
    if not path.exists():
        return
    with path.open("r+b") as handle:
        offset = 0
        for line in handle:
            if not line.endswith(b"\n"):
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
                return
            json.loads(line)  # An invalid complete line is corruption, never repaired.
            offset += len(line)


def make_engine(backend):
    # Check runtime placement and runner selection before expensive vLLM imports.
    import tempfile
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise RuntimeError("Set VLLM_USE_V2_MODEL_RUNNER=0 for the verified V1 hooks")
    if private_path(tempfile.gettempdir()) != private_path(os.environ["TMPDIR"]):
        raise RuntimeError("Python temporary directory differs from private Slurm staging")
    rpc = private_path(os.environ["VLLM_RPC_BASE_PATH"])
    if len(str(rpc).encode()) + 37 >= 108:
        raise RuntimeError("Private IPC path exceeds the Unix socket name limit")
    if os.environ.get("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION") != "1":
        raise RuntimeError("Set VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 for registered study request IDs")
    from vllm import LLM
    return LLM(model=MODEL, revision=REVISION, tokenizer_revision=REVISION,
        **ENGINE, enable_chunked_prefill=True,
        worker_extension_cls="moe_exp.routing_study.vllm_backend.StudyWorkerExtension")


def score_attempt(row, text, scorer=None):
    if row.get("gold_answer") is None or not str(row["gold_answer"]).strip():
        raise ValueError("A known gold answer is required")
    if scorer is None:
        from moe_exp.correlation_pipeline.scoring import score_completion
        scorer = score_completion
    answer, correct, method = scorer(row, answer_type=row.get("answer_type", "math"), model_text=text)
    return answer, bool(correct) if answer not in (None, "") else False, method


def wave_specs(indexed):
    """Globally unique external IDs, registered before asynchronous scheduling."""
    from uuid import uuid4
    wave = uuid4().hex
    specs = {}
    for index, row in indexed:
        key = f"study-{wave}-{index}"
        if key in specs:
            raise ValueError("Duplicate request index within a wave")
        specs[key] = row
    return specs


def run_wave(llm, indexed, policy, output, *, replay=False, study_sha256=None):
    """Keep at most eight live requests and checkpoint every finished attempt.

    Streams cumulative IDs so infrastructure failures retain observed token costs.
    Infrastructure failures may be retried; completed, capped or unparseable attempts
    are never selectively resampled.
    """
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    engine = llm.llm_engine
    specs = wave_specs(indexed)
    llm.collective_rpc("study_register", kwargs={"requests": specs})
    partial, waiting, started, finished = {}, set(specs), {}, set()
    try:
        for key, row in specs.items():
            ids = row["replay_token_ids"] if replay else row["prompt_token_ids"]
            internal_id = engine.add_request(key, {"prompt_token_ids": ids},
                               SamplingParams(**row["sampling"], output_kind=RequestOutputKind.CUMULATIVE))
            if internal_id != key:
                engine.abort_request([internal_id], internal=True)
                raise RuntimeError("Engine rewrote the registered study request ID")
            started[key] = time.perf_counter()
        while engine.has_unfinished_requests():
            for result in engine.step():
                key = result.request_id
                if key not in waiting or len(result.outputs) != 1:
                    raise RuntimeError("Unexpected generation output")
                completion = result.outputs[0]
                partial[key] = list(completion.token_ids)
                if not result.finished:
                    continue
                finished.add(key)
                row = specs[key]
                reports = llm.collective_rpc("study_snapshot", kwargs={"key": key, "full": replay,
                                                                      "study_sha256": study_sha256})
                native = verify_reports(reports)
                expected_inputs = len(row["replay_token_ids"]) if replay else (
                    len(row["prompt_token_ids"]) + max(0, len(completion.token_ids)-1))
                if native["observed_input_tokens"] != expected_inputs:
                    raise RuntimeError("Incomplete native input-token dispatch coverage")
                if replay:
                    sample = next(r["sample"] for r in reports if r["rank"] == 0)
                    write_json(output / "calibration" / (row["id"] + ".json"), sample, immutable=True)
                    correct, answer, method = row["is_correct"], None, "saved_calibration_outcome"
                else:
                    answer, correct, method = score_attempt(row, completion.text)
                record = dict(id=row["id"], execution_session=SESSION_ID, input_sha256=digest(row), policy_sha256=policy["sha256"],
                    problem_key=row["problem_key"], dataset=json.loads(row["problem_key"])[0],
                    prompt_token_ids=row["prompt_token_ids"], sampling_args=row["sampling"],
                    generated_token_ids=list(completion.token_ids), generated_token_count=len(completion.token_ids),
                    text=completion.text, model_answer=answer, is_correct=correct, scoring_method=method,
                    finish_reason=completion.finish_reason, latency_seconds=time.perf_counter()-started[key],
                    dispatch=native["layers"], infrastructure_failure=False,
                    # This is CPU-observed synchronization+instrumentation time; both replicas counted once.
                    hook_seconds_cumulative=max(r["hook_seconds"] for r in reports),
                    observed_input_tokens=native["observed_input_tokens"],
                    reasoning_tokens=native["reasoning_tokens"])
                append_record(output / "attempts.jsonl", record)
                waiting.remove(key)
                yield record
        if waiting:
            raise RuntimeError("Engine lost planned requests")
    except BaseException as error:
        for key in sorted(waiting):
            append_record(output / "failures.jsonl", dict(id=specs[key]["id"],
                input_sha256=digest(specs[key]), generated_token_ids=partial.get(key, []),
                generated_token_count=len(partial.get(key, [])), error=type(error).__name__ + ": " + str(error),
                token_count_status="exact" if key in finished else "observed_lower_bound_if_worker_lost",
                infrastructure_failure=True, retryable=True))
        raise
    finally:
        # Worker failure may make the RPC unavailable; preserve the original exception.
        try:
            llm.collective_rpc("study_release", kwargs={"keys": list(specs)})
        except Exception:
            pass


def matching_baseline(source, target):
    return (source.get("status") == "complete" and source["policy"]["action"] == "native"
            and target["policy"]["action"] == "native"
            and all(source.get(k) == target.get(k) for k in MATCH)
            and source.get("attempts_sha256") and source.get("dispatch_verified"))


def execute(study, policy, phase, output, acceptance, *, selection=None, registration=None, resume=False, baseline_from=None):
    require_compute()
    check_study(study)
    validate(policy, current_code=True)
    backend = backend_fingerprint()
    if study.get("backend") != backend:
        raise ValueError("Prepared backend fingerprint changed")
    validate_acceptance(acceptance, study, backend)
    replay = phase == "calibration"
    if not replay:
        if registration is None:
            raise ValueError("Register paired comparisons before policy generation")
        verify(registration)
        if (registration["study_sha256"] != study["sha256"] or registration["phase"] != phase
                or policy["sha256"] not in registration["policy_hashes"]):
            raise ValueError("Policy is not in the prespecified comparison registration")
    if replay and baseline_from is not None:
        raise ValueError("Calibration replay needs its own verified capture bundle")
    if replay and policy["action"] != "native":
        raise ValueError("Calibration replay must use native FP8 routing")
    planned = list(replay_rows(study)) if replay else planned_rows(study, phase, selection, policy)
    if not planned:
        raise ValueError("Empty planned workload")
    binding = policy.get("calibration")
    if binding and binding.get("split_sha256") != study["split"]["sha256"]:
        raise ValueError("Policy calibration split differs")
    manifest = dict(schema=1, study_sha256=study["sha256"], checkpoint=CHECKPOINT, engine=ENGINE,
        phase=phase, planned_sha256=digest(planned), policy=policy, backend=backend,
        final_selection=selection if phase=="test" else None,
        code=code_hashes(), token_scope=policy["token_scope"], scoring_contract=SCORING,
        registration_sha256=registration["sha256"] if registration else None,
        status="prepared", expected_count=len(planned), completed_count=0,
        batching="ordered_waves_of_at_most_eight; resume_pending_only",
        job_id=os.environ["SLURM_JOB_ID"])
    output = private_path(output)
    with run_lock(output):
        manifest_path, attempts_path = output / "manifest.json", output / "attempts.jsonl"
        if manifest_path.exists():
            old = read_json(manifest_path)
            if not resume or any(old.get(k) != manifest.get(k) for k in (*MATCH, "policy", "registration_sha256")):
                raise ValueError("Cannot resume changed policy/study/backend/inputs")
            if old["status"] != "complete":
                repair_interrupted_tail(attempts_path)
                repair_interrupted_tail(output / "failures.jsonl")
            else:
                if file_hash(attempts_path) != old["attempts_sha256"]:
                    raise ValueError("Complete output hash changed")
        elif attempts_path.exists():
            raise ValueError("Attempts without provenance")
        if baseline_from is not None:
            import shutil
            source = read_json(Path(baseline_from) / "manifest.json")
            if not matching_baseline(source, manifest):
                raise ValueError("Incompatible baseline reuse")
            if file_hash(Path(baseline_from) / "attempts.jsonl") != source["attempts_sha256"]:
                raise ValueError("Baseline output changed")
            if attempts_path.exists():
                raise ValueError("Baseline target already contains attempts")
            shutil.copyfile(Path(baseline_from) / "attempts.jsonl", attempts_path)
            if file_hash(attempts_path) != source["attempts_sha256"]:
                raise ValueError("Baseline changed while being copied")
            if source.get("failures_sha256"):
                source_failures = Path(baseline_from) / "failures.jsonl"
                if file_hash(source_failures) != source["failures_sha256"] or (output/"failures.jsonl").exists():
                    raise ValueError("Baseline failure ledger changed or target already exists")
                shutil.copyfile(source_failures, output/"failures.jsonl")
            manifest["reused_from"] = {"path": str(baseline_from), "manifest": source}
        expected = {r["id"]: r for r in planned}
        saved = saved_records(attempts_path, expected, policy["sha256"])
        if replay:
            for key in saved:
                sample = verify(read_json(output / "calibration" / (key + ".json")))
                if sample["id"] != key or sample["study_sha256"] != study["sha256"]:
                    raise ValueError("Resumed calibration capture differs")
        if manifest_path.exists() and old["status"] == "complete":
            if len(saved) != len(planned):
                raise ValueError("Complete run has missing attempts")
            if old.get("failures_sha256") and file_hash(output/"failures.jsonl") != old["failures_sha256"]:
                raise ValueError("Completed failure ledger changed")
            return old
        if manifest_path.exists():
            manifest["previous_sessions"] = old.get("previous_sessions", []) + [
                {k:old.get(k) for k in ("job_id", "status", "session_seconds", "startup_seconds", "completed_count")}]
        missing = [(i, row) for i, row in enumerate(planned) if row["id"] not in saved]
        began = time.perf_counter()
        startup = 0.
        try:
            if missing:
                write_json(manifest_path, {**manifest, "status": "running", "completed_count": len(saved)})
                llm = make_engine(backend)
                startup = time.perf_counter() - began
                llm.collective_rpc("study_configure", kwargs={"policy": policy, "backend": backend, "checkpoint": CHECKPOINT})
                for offset in range(0, len(missing), 8):
                    for record in run_wave(llm, missing[offset:offset+8], policy, output,
                                           replay=replay, study_sha256=study["sha256"]):
                        saved[record["id"]] = record
                        write_json(manifest_path, {**manifest, "status": "running", "completed_count": len(saved)})
            manifest.update(status="complete", completed_count=len(saved), dispatch_verified=True,
                attempts_sha256=file_hash(attempts_path), startup_seconds=startup,
                session_seconds=time.perf_counter()-began,
                failures_sha256=file_hash(output/"failures.jsonl") if (output/"failures.jsonl").exists() else None)
            if len(saved) != len(planned):
                raise ValueError("Incomplete planned coverage")
            write_json(manifest_path, manifest)
            return manifest
        except BaseException as error:
            manifest.update(status="failed", completed_count=len(saved), error=repr(error), startup_seconds=startup, session_seconds=time.perf_counter()-began)
            write_json(manifest_path, manifest)
            raise
