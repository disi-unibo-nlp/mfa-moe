"""Bind the observed retrospective CL snapshot, preserving earlier receipts."""
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
            "offline full CL build required")
    check_files(build["canonical_source_files"])
    require(file_sha(build["pdf_path"]) == build["pdf_sha256"], "PDF/source snapshot differs")
    document = fitz.open(build["pdf_path"])
    normal = re.sub(r"\s+", " ", "\n".join(p.get_text() for p in document))
    required = ["Prospective comparison of accuracy and memory", "Minimum held-out", "Family-balanced",
                "Qwen3.5 and Qwen3.6", "509", "32,768", "40,960", "25%", "unmeasured",
                "GPT-OSS", "Nemotron", "Gemma", "GLM", "Qwen3-30B"]
    require(all(term in normal for term in required), "observed analysis, limitations or seven-model scope absent")
    outside = []
    for number, page in enumerate(document, 1):
        for block in page.get_text("blocks"):
            if block[0] < -1 or block[1] < -1 or block[2] > page.rect.width+1 or block[3] > page.rect.height+1:
                outside.append({"page": number, "bounds": list(block[:4])})
    require(not outside, "off-page text/image blocks")
    log = (build_path.parent / "output/main.log").read_text(errors="replace")
    require(not re.search(r"(?:Reference|Citation).*undefined|Missing character|^!", log, re.MULTILINE),
            "unresolved TeX reference, glyph or error")
    campaign = load(REPO / "report/expert-memory-study-v1/CAMPAIGN.json")
    for key in ("code_files", "scientific_files_preserved", "readiness_files_preserved"):
        check_files(campaign[key])
    visuals = {str(build_path.parent / "visual" / f"page-{p}.png"):
               file_sha(build_path.parent / "visual" / f"page-{p}.png") for p in pages}
    root = REPO / "report/expert-memory-study-v1"
    fits = load(root / "SAVED_DATA_CLOSEOUT-r1.json")
    qualifications = load(root / "QUALIFICATION_SUBMISSIONS-r1.json")
    figures = load(root / "RETROSPECTIVE_PAPER_EXPORT-r1.json")
    check_files(figures["canonical_files"])
    return save(output, {"schema": "expert-memory-cl-paper-closeout-v2",
        "status": "COMPILED_RETROSPECTIVE_RESULTS_SNAPSHOT", "observed_utc": datetime.now(timezone.utc).isoformat(),
        "build_receipt_path": str(build_path), "build_receipt_sha256": file_sha(build_path),
        "source_files": build["canonical_source_files"], "canonical_sources_match": True,
        "pdf_path": build["pdf_path"], "pdf_sha256": build["pdf_sha256"], "pages": len(document),
        "offline_exit_code": 0, "offline_seconds": build["elapsed_seconds"],
        "visually_checked_pages": pages, "rendered_page_files": visuals,
        "off_page_blocks": outside, "undefined_references_or_missing_glyphs": False,
        "campaign_sha256": campaign["sha256"], "scientific35_readiness29_preserved": True,
        "saved_data_closeout_sha256": fits["sha256"], "retrospective_export_sha256": figures["sha256"],
        "qualification_submissions_sha256": qualifications["sha256"], "qualification_jobs": 7,
        "gpu_and_accuracy_results": "UNMEASURED", "older_paper_receipts_preserved": True,
        "failed_older_builder_attempt_preserved": "tools/tmp/expert-memory-cl-build-20261008-r5/BUILD.json",
        "helper_code_files": {str(Path(__file__).resolve()): file_sha(__file__)},
        "limitations": ["Observed coverage and sparse-label transitions do not establish accuracy under physical removal.",
                        "GPU qualification, realized memory and prospective accuracy remain unmeasured.",
                        "Existing verification appendix retains its source-bound observation cutoff; no new verification closeout.",
                        "Working draft with inherited layout warnings and submission metadata; not camera-ready."]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visually-checked-pages", required=True)
    args = parser.parse_args()
    result = closeout(args.build, args.output, [int(x) for x in args.visually_checked_pages.split(",")])
    print(json.dumps({"status": result["status"], "pages": result["pages"], "sha256": result["sha256"]}))
