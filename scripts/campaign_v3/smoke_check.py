"""Acceptance for a card-profile smoke job (reads env SERVER_LOG, SOURCE, OUT, JOB)."""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from moe_exp.correlation_pipeline.model_profiles import CARD_PROFILES  # noqa: E402

source, out, log = os.environ["SOURCE"], Path(os.environ["OUT"]), Path(os.environ["SERVER_LOG"])
card = CARD_PROFILES[source]
traces = [json.loads(line) for f in sorted((out / "generation").glob("*/*/traces.jsonl")) for line in f.open()]
text = log.read_text(errors="replace")
kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
conc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text)
accept = re.findall(r"Mean acceptance length: ([\d.]+)", text)
problems = []
if len(traces) != 8:
    problems.append(f"expected 8 smoke traces, found {len(traces)}")
required = ("generation_contract_version", "profile", "served_model", "revision", "max_tokens",
            "max_model_len", "sampler", "chat_template_kwargs", "speculation", "vllm_version", "seed")
for t in traces:
    meta = t["metadata"]
    cfg = meta.get("generation_config", {})
    if not meta.get("token_replay", {}).get("completion_token_ids"):
        problems.append(f"{t['problem_id']}: missing token replay")
    missing = [k for k in required if k not in cfg]
    if missing:
        problems.append(f"{t['problem_id']}: contract fields missing {missing}")
    if cfg.get("sampler") != card["sampler"]:
        problems.append(f"{t['problem_id']}: sampler mismatch")
    if meta.get("scoring_status") not in ("ok", "prediction_parse_failed") and t.get("is_correct") is not None:
        problems.append(f"{t['problem_id']}: scoring_status {meta.get('scoring_status')}")
    if t.get("scoring_method") == "normalized_exact_numeric_fallback" and meta.get("scoring_status") != "prediction_parse_failed":
        problems.append(f"{t['problem_id']}: math_verify fallback ({meta.get('scoring_status')})")
if not conc or float(conc[-1][1]) < 1.0:
    problems.append(f"KV capacity does not fit one full-context request: {conc[-1:] or 'not reported'}")
receipt = dict(status="complete" if not problems else "failed", source=source, job=os.environ.get("JOB"),
               traces=len(traces), kv_cache_tokens=kv[-1:] , max_concurrency=conc[-1:],
               mean_acceptance_length=accept[-3:],
               completion_tokens=[t["metadata"].get("usage", {}).get("completion_tokens") for t in traces],
               terminations=[t["metadata"].get("termination") for t in traces],
               correct=[t.get("is_correct") for t in traces], problems=problems)
(out / "smoke_receipt.json").write_text(json.dumps(receipt, indent=1))
print(json.dumps(receipt))
sys.exit(0 if not problems else 3)
