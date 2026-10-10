"""Answer and termination metadata belong to each exact original draw."""
import hashlib
import json
from pathlib import Path
from .contracts import digest, load, publish, require
from .statistics import bootstrap_panel


def attempt_key(row):
    return (row["dataset"], row["problem_id"], int(row["sample_id"]), row["trace_sha256"])


def joined_accuracy(outcomes, metadata, qf):
    from .scoring import unknown_bounds
    rows = []
    require(len(metadata) == len(outcomes), "exact native metadata population differs")
    for outcome in outcomes:
        key = attempt_key(outcome)
        require(key in metadata, "exact original attempt metadata missing")
        source = metadata[key]
        require(source["completion_tokens"] == int(outcome["completion_tokens"]) and
                source["finish_reason"] == outcome["termination"], "exact native attempt metadata mismatch")
        question = outcome["dataset"] + "|" + outcome["source_problem_id"]
        require(question in qf, "native outcome absent from family graph")
        strict = outcome["strict_correct"]
        strict = None if strict is None or (isinstance(strict, float) and strict != strict) else bool(strict)
        lower, upper = unknown_bounds(strict, source["finish_reason"], source["answer_present"])
        adjudicated = outcome["adjudicated_correct"]
        adjudicated = None if adjudicated is None or (isinstance(adjudicated, float) and adjudicated != adjudicated) else float(adjudicated)
        rows.append({"benchmark": outcome["dataset"], "family": qf[question],
                     "attempt_identity": list(key), "attempt_id": digest(key),
                     "strict": None if strict is None else float(strict),
                     "historical_adjudicated": adjudicated,
                     "unknown_lower": lower, "unknown_upper": upper,
                     "tokens": int(outcome["completion_tokens"]),
                     "termination": source["finish_reason"], "answer_present": source["answer_present"],
                     "original_source": source["location"]})
    return rows


def native_accuracy(campaign, model):
    import pandas as pd
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest
    from moe_exp.correlation_pipeline.scoring import extract_answer
    root = Path(campaign["root"])
    outcomes = pd.read_parquet(campaign["outcomes"])
    outcomes = outcomes[(outcomes.corpus == "card-v3") & (outcomes.model == model)].to_dict("records")
    require(len({attempt_key(r) for r in outcomes}) == len(outcomes), "duplicate native outcome identity")
    expected = {attempt_key(r) for r in outcomes}
    sources = load(root/model/"SOURCES.json")["records"]
    paths = sorted({Path(json.loads(r["source_location"])["path"]) for r in sources})
    metadata = {}
    for path in paths:
        with path.open("rb") as stream:
            offset = 0
            for raw in stream:
                trace = json.loads(raw)
                key = (trace["dataset"], trace["problem_id"], int(trace["sample_id"]), trace_digest(TraceRecord(**trace)))
                require(key in expected and key not in metadata, "native source contains an unexpected or duplicate attempt")
                replay = trace["metadata"]["token_replay"]
                final = trace["metadata"].get("assistant_content") or trace["metadata"].get("reasoning_content") or ""
                metadata[key] = {"completion_tokens": len(replay["completion_token_ids"]),
                                 "finish_reason": trace["metadata"]["finish_reason"],
                                 "answer_present": bool(extract_answer(final, "math")),
                                 "location": {"path": str(path), "byte_offset": offset, "line_bytes": len(raw),
                                              "line_sha256": hashlib.sha256(raw).hexdigest()}}
                offset += len(raw)
    rows = joined_accuracy(outcomes, metadata, load(root/"FOLDS.json")["question_families"])
    estimands = publish(root/model/"NATIVE_ESTIMANDS.json", {"records": rows,
        "historical_adjudicated_not_used": True, "exact_attempt_metadata": True})
    summary = publish(root/model/"NATIVE_ACCURACY.json", bootstrap_panel(rows,
        ("strict", "unknown_lower", "unknown_upper", "tokens"), panel="native:"+model))
    return estimands, summary
