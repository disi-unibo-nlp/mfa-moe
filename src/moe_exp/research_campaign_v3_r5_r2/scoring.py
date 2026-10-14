"""Strict original outcome contract and unknown bounds, with no J1 allocations."""
from pathlib import Path
import pandas as pd
from .contracts import Units, load, publish
from .inputs import read_source
from .recovery import DeterministicUnits


def unknown_bounds(correct,finish_reason,answer_present):
    uncertain=correct is None or (correct is False and finish_reason=="stop" and answer_present)
    return float(correct) if correct is not None else 0.,1. if uncertain else float(correct)


def score(campaign,model,rows):
    from transformers import AutoTokenizer
    from .annotation import rebuilt
    from moe_exp.correlation_pipeline.scoring import score_completion_detailed
    root=Path(campaign["root"])
    units=Units(root/model/"executions")
    scores=DeterministicUnits(root/model/"scores")
    native=pd.read_parquet(campaign["outcomes"])
    native=native[(native.corpus=="card-v3") & (native.model==model)].set_index("trace_sha256")
    tokenizer=None
    for row in rows:
        if (scores.state(row["uid"]) or {}).get("status") not in (None,"started"):
            continue
        result=units.state(row["uid"])
        if not scores.begin(row,"strict-scoring"):
            continue
        if (result or {}).get("status")!="complete":
            scores.commit(row,{"status":"upstream_unavailable","strict_correct":None,"unknown_lower":0.,"unknown_upper":1.})
            continue
        source=read_source(row["source"]["location"])
        if result.get("reused"):
            value=native.loc[row["source"]["trace_sha256"],"strict_correct"]
            correct=None if pd.isna(value) else bool(value)
            from moe_exp.correlation_pipeline.scoring import extract_answer
            text=source["metadata"].get("assistant_content") or source["metadata"].get("reasoning_content") or ""
            detail={"method":"reused_original_strict_outcome","model_answer":extract_answer(text,"math")}
        else:
            if tokenizer is None:
                tokenizer=AutoTokenizer.from_pretrained(campaign["models"][model]["local_path"],local_files_only=True)
            if result["token_ids"]:
                trace,replay=rebuilt(campaign,model,row,result,tokenizer)
                text=trace.metadata.get("assistant_content") or trace.metadata.get("reasoning_content") or ""
            else:
                text=""
            detail=score_completion_detailed({"gold_answer":source.get("gold_answer")},
                answer_type="math",model_text=text)
            # Original strict accepts remain accepted. Normally stopped rejects
            # without a compatible successful adjudicator remain unknown.
            raw=detail["is_correct"]
            correct=bool(raw) if raw is not None else None
        lower,upper=unknown_bounds(correct,result["finish_reason"],bool(detail.get("model_answer")))
        scores.commit(row,{"status":"complete","strict_correct":correct,
            "unknown_lower":lower,"unknown_upper":upper,
            "strict_detail":detail,"scorer_contract_version":3,
            "cumulative_tokens":row["source"]["prefix_tokens"]+result["generated_tokens"],
            "new_tokens":result["generated_tokens"],"termination":result["finish_reason"],"j1_used":False})
    return {"status":"complete","assignments":len(rows)}
