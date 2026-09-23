"""Shared durable generation and policy-independent baseline reuse."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shutil


WORKERS = {
    "moe_exp.moe_identity_guiding.routing.IdentityWorkerExtension",
    "moe_exp.moe_margin_guiding.routing.MarginWorkerExtension",
}
MATCH_FIELDS = ("prompts_sha256", "sampling_args", "versions", "scoring_contract",
                "seed_strategy", "token_scope")


def add_execution_args(parser):
    parser.add_argument("--template-date", type=date.fromisoformat, default=None,
                        help="Frozen date; inherit saved paired date or default to 2026-09-22")
    parser.add_argument("--diagnostics", choices=("minimal", "full"), default="minimal",
                        help="Minimal keeps hook activity checks; full adds router statistics")
    parser.add_argument("--resume", action="store_true",
                        help="Resume matching incremental runs; skip validated complete runs")
    parser.add_argument("--baseline-search-root", type=Path, action="append",
                        help="Search completed guiding baselines here (repeatable)")
    parser.add_argument("--no-reuse-baseline", action="store_true")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_engine(engine):
    """Only these known extensions install no hooks for baseline conditions."""
    result = dict(engine)
    worker = result.pop("worker_extension_cls", None)
    if worker is not None and worker not in WORKERS:
        raise ValueError("Unknown baseline worker extension")
    return result


def matching_baseline(source, target):
    return (source.get("status") == "complete"
            and source.get("condition") == "baseline"
            and all(source.get(k) == target.get(k) for k in MATCH_FIELDS)
            and native_engine(source["engine_args"]) == native_engine(target["engine_args"])
            and bool(source.get("routing_diagnostics"))
            and all(r.get("condition") == "baseline" and r.get("layers") == {}
                    for r in source["routing_diagnostics"]))


def read_records(path, rows, requests, condition, *, repair_tail=False):
    """Validate saved attempts; only an interrupted final line may be discarded."""
    expected = {row["id"]: (row, request) for row, request in zip(rows, requests, strict=True)}
    saved = set()
    if not path.exists():
        return saved
    with path.open("r+b" if repair_tail else "rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if repair_tail:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                    break
                raise ValueError("Incomplete generations JSONL tail")
            record = json.loads(line)
            key = record["id"]
            if key in saved or key not in expected:
                raise ValueError("Duplicate or unexpected saved attempt ID")
            row, request = expected[key]
            if (record.get("input") != row or record.get("sampling_args") != request
                    or record.get("condition") != condition):
                raise ValueError("Saved attempt inputs or sampling differ")
            if (not isinstance(record.get("text"), str)
                    or not isinstance(record.get("prompt_token_ids"), list)
                    or not isinstance(record.get("generated_token_ids"), list)
                    or record.get("generated_token_count") != len(record["generated_token_ids"])
                    or type(record.get("is_correct")) is not bool
                    or record.get("finish_reason") not in ("stop", "length")):
                raise ValueError("Incomplete or unscored saved attempt")
            saved.add(key)
    return saved


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".generation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is writing this run") from error
        yield


def reuse_baseline(args, manifest, rows, requests, rendered, inputs):
    if args.condition != "baseline" or getattr(args, "no_reuse_baseline", False):
        return False
    roots = getattr(args, "baseline_search_root", None) or [
        Path("results/moe_identity_guiding"), Path("results/moe_margin_guiding")]
    for path in sorted({p for root in roots for p in root.rglob("manifest.json")}):
        if path.parent.resolve() == args.output_dir.resolve():
            continue
        source_file = path.parent / "generations.jsonl"
        try:
            source = json.loads(path.read_text())
            if not matching_baseline(source, manifest):
                continue
            with run_lock(path.parent):
                # Recheck under lock, and preserve an auditable snapshot of the source.
                source = json.loads(path.read_text())
                if not matching_baseline(source, manifest):
                    continue
                saved = read_records(source_file, rows, requests, "baseline")
                if len(saved) != len(rows):
                    continue
                validate_rendered(source_file, rows, rendered, inputs)
                provenance = dict(path=str(path.parent.resolve()), manifest=source,
                                  manifest_sha256=file_hash(path),
                                  generations_sha256=file_hash(source_file))
                temp = args.output_dir / "generations.jsonl.tmp"
                shutil.copyfile(source_file, temp)
                with temp.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temp, args.output_dir / "generations.jsonl")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            continue
        manifest.update(status="complete", completed_count=len(saved), expected_count=len(rows),
                        reused_baseline=provenance,
                        routing_diagnostics=source["routing_diagnostics"])
        atomic_json(args.output_dir / "manifest.json", manifest)
        print(f"Reused compatible baseline: {path.parent}", flush=True)
        return True
    return False


def finished_outputs(llm, inputs, requests, pending, sampling_class):
    """Keep continuous batching while yielding each completed request immediately."""
    from vllm.sampling_params import RequestOutputKind

    engine = llm.llm_engine
    waiting = set(pending)
    for index in pending:
        engine.add_request(str(index), inputs[index], sampling_class(**requests[index], output_kind=RequestOutputKind.FINAL_ONLY))
    while engine.has_unfinished_requests():
        for output in engine.step():
            if not output.finished:
                continue
            index = int(output.request_id)
            if index not in waiting:
                raise RuntimeError("Unexpected or duplicate engine completion")
            waiting.remove(index)
            yield index, output
    if waiting:
        raise RuntimeError("Generation count does not match pending prompts")


def load_tokenizer(engine):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(engine["model"], revision=engine.get("tokenizer_revision"))


def render_inputs(tokenizer, rows, template_date):
    template = getattr(tokenizer, "chat_template", None)
    template_args = {}
    if isinstance(template, str) and "strftime_now" in template:
        template = template.replace('strftime_now("%Y-%m-%d")', 'guiding_template_date')
        template = template.replace("strftime_now('%Y-%m-%d')", 'guiding_template_date')
        if "strftime_now" in template:
            raise ValueError("Unsupported dynamic chat-template clock; freeze it explicitly")
        template_args["chat_template"] = template
    rendered = []
    for row in rows:
        messages = row.get("generation_messages") or row.get("messages")
        if messages is None:
            messages = []
            if row.get("system_prompt"):
                messages.append({"role": "system", "content": row["system_prompt"]})
            messages.append({"role": "user", "content": row["prompt"]})
        options = dict((row.get("original_generation_config") or {}).get("chat_template_kwargs") or {})
        options.update(template_args)
        options["guiding_template_date"] = str(template_date)
        rendered.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **options))
    inputs = [{"prompt_token_ids": tokenizer.encode(p, add_special_tokens=False)} for p in rendered]
    return rendered, inputs


def validate_rendered(path, rows, rendered, inputs):
    expected = {row["id"]: (prompt, tokens["prompt_token_ids"])
                for row, prompt, tokens in zip(rows, rendered, inputs, strict=True)}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if expected.get(record["id"]) != (record.get("rendered_prompt"), record.get("prompt_token_ids")):
                raise ValueError("Rendered prompts differ; use a matched template date/tokenizer")


def execute(args, manifest, rows, requests, kind, check_reports):
    with run_lock(args.output_dir):
        _execute_locked(args, manifest, rows, requests, kind, check_reports)


def _execute_locked(args, manifest, rows, requests, kind, check_reports):
    path = args.output_dir / "manifest.json"
    generations = args.output_dir / "generations.jsonl"
    manifest.update(execution_schema=1, diagnostics=getattr(args, "diagnostics", "minimal"))
    requested_date = getattr(args, "template_date", None)
    partner = args.output_dir.parent / ("guided" if args.condition == "baseline" else "baseline")
    template_date = str(requested_date or "2026-09-22")
    if requested_date is None:
        for existing in (generations, partner / "generations.jsonl"):
            if existing.exists():
                with existing.open() as handle:
                    line = handle.readline()
                if line:
                    record = json.loads(line)
                    found = re.search(r"Current date: (\d{4}-\d{2}-\d{2})", record.get("rendered_prompt", ""))
                    if found:
                        template_date = found[1]
                        break
        if path.exists():
            template_date = json.loads(path.read_text()).get("template_date", template_date)
    manifest["template_date"] = template_date
    rendered, inputs = render_inputs(load_tokenizer(manifest["engine_args"]), rows, template_date)
    # Compare against the other condition before loading the GPU model or reusing data.
    partner = args.output_dir.parent / ("guided" if args.condition == "baseline" else "baseline")
    if (partner / "generations.jsonl").exists():
        validate_rendered(partner / "generations.jsonl", rows, rendered, inputs)
    saved = set()
    if path.exists():
        old = json.loads(path.read_text())
        if not getattr(args, "resume", False):
            raise FileExistsError(f"Run exists: {path}; use --resume")
        for key in (*MATCH_FIELDS, "engine_args", "condition", "policy_sha256", "strength"):
            if old.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume: {key} differs")
        if old.get("status") != "complete" and old.get("execution_schema") != 1:
            raise ValueError("Cannot resume a legacy incomplete run; it may still be active. Use a fresh output directory")
        if old.get("status") != "complete" and old.get("diagnostics") != manifest["diagnostics"]:
            raise ValueError("Cannot change diagnostics while resuming")
        saved = read_records(generations, rows, requests, args.condition,
                             repair_tail=old.get("status") != "complete")
        if generations.exists():
            validate_rendered(generations, rows, rendered, inputs)
        if old.get("status") == "complete":
            if len(saved) != len(rows):
                raise ValueError("Complete manifest has missing attempts")
            check_reports(old.get("routing_diagnostics"), manifest["policy"], args.condition)
            print(f"Run already complete: {args.output_dir}", flush=True)
            return
        # Reports from earlier sessions retain their meaning; counters are not summed.
        manifest["previous_sessions"] = old.get("previous_sessions", []) + [
            {k: old[k] for k in ("completed_count", "routing_diagnostics", "error") if k in old}]
    elif generations.exists():
        raise ValueError("Generations exist without a manifest; refusing to overwrite")
    elif reuse_baseline(args, manifest, rows, requests, rendered, inputs):
        return
    manifest.update(status="running", completed_count=len(saved), expected_count=len(rows))
    atomic_json(path, manifest)
    if len(saved) == len(rows):
        sessions = manifest.get("previous_sessions", [])
        reports = next((s["routing_diagnostics"] for s in reversed(sessions)
                        if s.get("routing_diagnostics")), None)
        check_reports(reports, manifest["policy"], args.condition)
        manifest.update(status="complete", routing_diagnostics=reports)
        atomic_json(path, manifest)
        return
    try:
        from vllm import LLM, SamplingParams
        from tqdm import tqdm
        from moe_exp.correlation_pipeline.scoring import score_completion
        llm = LLM(**manifest["engine_args"])
        tokenizer = llm.get_tokenizer()
        engine_rendered, engine_inputs = render_inputs(tokenizer, rows, template_date)
        if engine_rendered != rendered or engine_inputs != inputs:
            raise ValueError("CPU and engine tokenizer outputs differ")
        llm.collective_rpc(f"{kind}_configure", kwargs=dict(
            policy=manifest["policy"], strength=args.strength, condition=args.condition,
            diagnostics=manifest["diagnostics"]))
        pending = [i for i, row in enumerate(rows) if row["id"] not in saved]
        with generations.open("a") as handle, tqdm(
            total=len(rows), initial=len(saved),
            desc=f"{kind.capitalize()} {args.condition}", unit="attempt",
            dynamic_ncols=True, mininterval=1.0,
        ) as progress:
            for i, output in finished_outputs(llm, inputs, requests, pending, SamplingParams):
                if len(output.outputs) != 1:
                    raise RuntimeError("Expected one completion per attempt")
                row = rows[i]
                completion = output.outputs[0]
                answer, correct, method = score_completion(
                    row, answer_type=row.get("answer_type", "math"), model_text=completion.text)
                reports = llm.collective_rpc(f"{kind}_diagnostics")
                check_reports(reports, manifest["policy"], args.condition)
                record = dict(id=row["id"], input=row, rendered_prompt=rendered[i],
                              condition=args.condition, text=completion.text, sampling_args=requests[i],
                              model_answer=answer, is_correct=correct, scoring_method=method,
                              prompt_token_ids=output.prompt_token_ids,
                              generated_token_ids=list(completion.token_ids),
                              generated_token_count=len(completion.token_ids),
                              finish_reason=completion.finish_reason)
                # Save the validated hook report before committing the completion.
                manifest["routing_diagnostics"] = reports
                atomic_json(path, manifest)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                saved.add(row["id"])
                manifest["completed_count"] = len(saved)
                atomic_json(path, manifest)
                progress.update(1)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(path, manifest)
