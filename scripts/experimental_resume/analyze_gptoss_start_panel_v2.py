"""Descriptive three-model start-vote comparison for frozen discovery panel."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
PANEL = BASE / "GPTOSS_START_PANEL_v2.json"
RESULTS = BASE / "ratings-gptoss-start-panel-v2-9e81a6b6"
AGREEMENT = REPO / "report/experimental-resume-v1/FULL_PREFIX_NATIVE_VETO_AGREEMENT_v1.json"
OUT = REPO / "report/experimental-resume-v1/GPTOSS_FULLPREFIX_START_PANEL_AGREEMENT_v2.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed source: {path}")
    return value


def main():
    panel, agreement = sealed(PANEL), sealed(AGREEMENT)
    binding, summary = sealed(RESULTS / "BINDING.json"), sealed(RESULTS / "SUMMARY.json")
    batches = [sealed(p) for p in sorted((RESULTS / "batches").glob("[0-9][0-9][0-9][0-9].json"))]
    rated = [row for batch in batches for row in batch["records"]]
    if (panel["schema"] != "gptoss-start-panel-v2" or panel["rows"] != 27
            or binding["frame_sha256"] != panel["sha256"]
            or summary["binding_sha256"] != binding["sha256"]
            or summary["batch_sha256s"] != [b["sha256"] for b in batches]
            or [r["uid"] for r in rated] != [r["uid"] for r in panel["records"]]
            or summary["parsed_natural_stops"] != 27
            or agreement["frame_sha256"] != panel["source_frame_sha256"]):
        raise ValueError("GPT panel/Qwen audit binding or coverage changed")
    prior = {row["uid"]: row for row in agreement["records"]}
    counts = Counter()
    records = []
    for i, (source, result) in enumerate(zip(panel["records"], rated, strict=True)):
        item = prior[source["uid"]]
        if result["rating"] is None or result["finish_reason"] != "stop":
            raise ValueError("missing GPT panel vote")
        role = "qwen_approved_selected_start" if i < panel["selected_start_rows"] else "frozen_disagreement_or_negative_control"
        if role == "qwen_approved_selected_start" and item["qwen_status"] != "accepted":
            raise ValueError("selected start no longer Qwen-approved")
        vote = result["rating"]["start"]
        counts[role + "|" + source["transition"] + "|rows"] += 1
        counts[role + "|" + source["transition"] + "|gpt_true"] += vote
        counts[role + "|" + source["transition"] + "|qwen_native_gpt_joint_true"] += (
            item["qwen_status"] == "accepted" and item["native_status"] == "accepted" and vote)
        records.append({"uid": source["uid"], "family": source["family"],
                        "transition": source["transition"], "role": role,
                        "qwen_status": item["qwen_status"],
                        "native_status": item["native_status"],
                        "gpt_start": vote,
                        "gpt_raw_completion_sha256": hashlib.sha256(result["raw_completion"].encode()).hexdigest()})
    body = {"schema": "gptoss-fullprefix-start-panel-agreement-v2",
            "panel_sha256": panel["sha256"], "gpt_binding_sha256": binding["sha256"],
            "gpt_summary_sha256": summary["sha256"],
            "gpt_batch_sha256s": [b["sha256"] for b in batches],
            "qwen_native_agreement_sha256": agreement["sha256"],
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "counts": dict(counts), "records": records,
            "interpretation": "Discovery-only arm-blind third-family LLM measurement audit. GPT-OSS says true on two Qwen-agreed negative controls; none of these votes establishes human truth, transition control or an independently validated eligibility rule."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "counts": dict(counts)}), flush=True)


if __name__ == "__main__":
    main()
