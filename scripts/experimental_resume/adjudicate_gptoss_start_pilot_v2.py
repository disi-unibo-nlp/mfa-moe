"""Separate, sealed posthoc parse of immutable GPT-OSS pilot raw completions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gptoss_final_parser_v2 import parse_rating

BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
PILOT = BASE / "ratings-gptoss-start-pilot-v1-18a7f804-eda3cc45"
FRAME = BASE / "GPTOSS_START_PILOT_FRAME_v1.json"
OUT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1/GPTOSS_START_PILOT_POSTHOC_PARSE_v2.json")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed source: {path}")
    return value


def main():
    frame = sealed(FRAME)
    binding = sealed(PILOT / "BINDING.json")
    result = sealed(PILOT / "RESULTS.json")
    if (frame["schema"] != "gptoss-start-pilot-frame-v1"
            or binding["frame_sha256"] != frame["sha256"]
            or result["binding_sha256"] != binding["sha256"]
            or [r["uid"] for r in result["records"]] != [r["uid"] for r in frame["records"]]
            or len(result["records"]) != 4):
        raise ValueError("pilot source binding changed")
    rows = [{"uid": r["uid"], "finish_reason": r["finish_reason"],
             "old_parser": r["rating"], "v2_final_parser": parse_rating(r["raw_completion"]),
             "raw_completion_sha256": hashlib.sha256(r["raw_completion"].encode()).hexdigest()}
            for r in result["records"]]
    body = {"schema": "gptoss-start-pilot-posthoc-parse-v2",
            "frame_sha256": frame["sha256"], "binding_sha256": binding["sha256"],
            "result_sha256": result["sha256"],
            "parser_sha256": hashlib.sha256(Path(__file__).with_name("gptoss_final_parser_v2.py").read_bytes()).hexdigest(),
            "rows": rows,
            "parsed_natural_stops": sum(r["v2_final_parser"] is not None and
                                        r["finish_reason"] == "stop" for r in rows),
            "interpretation": "Versioned parse repair of saved raw output, not a new GPT-OSS model draw or human truth"}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "parsed_natural_stops": body["parsed_natural_stops"],
                      "ratings": [r["v2_final_parser"] for r in rows]}))


if __name__ == "__main__":
    main()
