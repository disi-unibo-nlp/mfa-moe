"""Bind a compiled methods snapshot; no scientific dispatch or inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from moe_exp.expert_memory_study.contracts import REPO, check_files, file_sha, load, require, save


def closeout(build_path, output, pages):
    import fitz
    build_path = Path(build_path)
    build = json.loads(build_path.read_text())
    require(build["state"] == "COMPILED_FULL_CLASS" and build["exit_code"] == 0 and build["only_cached"],
            "full offline CL build required")
    check_files(build["canonical_source_files"])
    require(file_sha(build["pdf_path"]) == build["pdf_sha256"], "PDF differs from its source snapshot")
    document = fitz.open(build["pdf_path"])
    text = "\n".join(p.get_text() for p in document)
    normal = re.sub(r"\s+", " ", text)
    required = ["Prospective comparison of accuracy and memory", "25%", "32,768", "40,960",
                "GPT-OSS", "Nemotron", "Qwen3.5", "Qwen3.6", "Gemma", "GLM", "Qwen3-30B"]
    require(all(term in normal for term in required), "prospective design or all-seven scope missing from PDF")
    outside = []
    for number, page in enumerate(document, 1):
        for block in page.get_text("blocks"):
            if block[0] < -1 or block[1] < -1 or block[2] > page.rect.width + 1 or block[3] > page.rect.height + 1:
                outside.append({"page": number, "bounds": list(block[:4])})
    require(not outside, "off-page text/image blocks require layout review")
    log = (build_path.parent / "output/main.log").read_text(errors="replace")
    require(not re.search(r"(?:Reference|Citation).*undefined|Missing character|^!", log, re.MULTILINE),
            "unresolved TeX references, glyphs or errors")
    visuals = {str(build_path.parent / "visual" / f"page-{p}.png"):
               file_sha(build_path.parent / "visual" / f"page-{p}.png") for p in pages}
    campaign = load(REPO / "report/expert-memory-study-v1/CAMPAIGN.json")
    for key in ("code_files", "scientific_files_preserved", "readiness_files_preserved"):
        check_files(campaign[key])
    return save(output, {"schema": "expert-memory-cl-paper-closeout-v1", "status": "COMPILED_METHODS_SNAPSHOT",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "build_receipt_path": str(build_path),
        "build_receipt_sha256": file_sha(build_path), "source_files": build["canonical_source_files"],
        "canonical_sources_match": True, "pdf_path": build["pdf_path"], "pdf_sha256": build["pdf_sha256"],
        "pages": len(document), "offline_exit_code": 0, "offline_seconds": build["elapsed_seconds"],
        "visually_checked_pages": pages, "rendered_page_files": visuals, "off_page_blocks": outside,
        "undefined_references_or_missing_glyphs": False, "campaign_sha256": campaign["sha256"],
        "scientific35_readiness29_preserved": True, "new_study_allocations": 0,
        "prior_cl_snapshot_preserved": "report/model-analysis-review-20261008/CL_MANUSCRIPT_BUILD-r3.json",
        "helper_sha256": file_sha(__file__),
        "limitations": ["Methods/implementation snapshot; all new GPU and accuracy measurements remain unmeasured.",
                        "Existing appendix and automatic verification status retain their observation cutoffs; this is not a new verification scientific closeout.",
                        "Working draft; inherited layout warnings and submission metadata remain to be resolved."]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visually-checked-pages", required=True)
    args = parser.parse_args()
    result = closeout(args.build, args.output, [int(x) for x in args.visually_checked_pages.split(",")])
    print(json.dumps({"status": result["status"], "pages": result["pages"], "sha256": result["sha256"]}))
