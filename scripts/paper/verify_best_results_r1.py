"""Verify an editorial saved-result snapshot without refitting or dispatch."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/6a510c6db0d9b08aa438c142"
BUNDLE = PAPER / "generated/best_results_20261009"
SAVED = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(name):
    with (BUNDLE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_files(files):
    for path, expected in files.items():
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        if sha(path) != expected:
            raise ValueError(f"Bound file changed: {path}")


def verify(build_path, output):
    import fitz

    build = read(build_path)
    if build["state"] != "COMPILED_FULL_CLASS" or build["exit_code"] != 0 or not build["only_cached"]:
        raise ValueError("Successful offline full-class build required")
    check_files(build["canonical_source_files"])
    if sha(build["pdf_path"]) != build["pdf_sha256"]:
        raise ValueError("PDF differs from compiled snapshot")
    sources = read(BUNDLE / "SOURCES.json")
    for name, item in sources.items():
        if sha(ROOT / item["source"]) != item["sha256"] or sha(BUNDLE / name) != item["sha256"]:
            raise ValueError("Bundled measurements differ from original CSV")
    figures = read(BUNDLE / "FIGURES.json")
    for name, expected in figures["files"].items():
        if sha(BUNDLE / name) != expected:
            raise ValueError(f"Figure input/output changed: {name}")

    original_bindings = read(ROOT / "report/model-analysis-review-20261008/SOURCES.json")
    for filename in ["readout-results.v3.json", "anticipation-results.json"]:
        if sha(SAVED / filename) != original_bindings[str(SAVED / filename)]["sha256"]:
            raise ValueError("Original statistical result binding changed")
    original = read(SAVED / "readout-results.v3.json")["models"]
    readout_rows = rows("routing_readout.csv")
    if len(readout_rows) != 7:
        raise ValueError("All seven readout results required")
    for row in readout_rows:
        item = original[row["model"]]["judge_context"]
        assert row["status"] == item["status"] == "CLEAN"
        for column, key in [("questions", "n_questions"), ("families", "n_families"), ("sentences", "n_sentences")]:
            assert int(row[column]) == item[key]
        for prefix in ["routing", "routing_excess_over_lexical_null"]:
            assert float(row[prefix + "_delta_logloss"]) == item[prefix]["P"]
            assert [float(row[prefix + "_simultaneous_ci7_" + end]) for end in ["low", "high"]] == item[prefix]["simultaneous_ci7"]
    original = read(SAVED / "anticipation-results.json")["models"]
    anticipation_rows = rows("anticipation.csv")
    for row in anticipation_rows:
        item = original[row["model"]][row["mode"]]
        assert row["status"] == item["status"]
        assert int(row["questions"]) == item["questions"] and int(row["pairs"]) == item["pairs"]
        metric = item["routing"][row["feature"]]
        assert float(row["delta_logloss"]) == metric["P"]
        assert [float(row["simultaneous_ci6_" + end]) for end in ["low", "high"]] == metric["simultaneous_ci6"]

    export = read(ROOT / "report/expert-memory-study-v1/exhibits-r2/EXPORT.json")
    check_files(export["source_files"])
    check_files(export["outputs"])
    selections = {}
    selection_rows = rows("selection_coverage.csv")
    for row in selection_rows:
        model = row["model"]
        if model not in selections:
            selections[model] = read(ROOT / f"report/expert-memory-study-v1/{model}/SELECTION.json")
        item = next(f for f in selections[model]["folds"] if f["fold"] == int(row["fold"]))
        metric = item["held_out"][row["arm"]]
        assert float(row["minimum_class_coverage"]) == metric["minimum_class_coverage"]
        assert min(metric["class_mean_layer_coverage"]) == metric["minimum_class_coverage"]
    assert len(selection_rows) == 105 and len(selections) == 7
    readiness = read(ROOT / "report/research-campaign-v2-20261009-r1/READINESS.json")
    check_files(readiness["files"])

    doc = fitz.open(build["pdf_path"])
    page_text = [re.sub(r"\s+", " ", re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])", "",
                 unicodedata.normalize("NFKC", p.get_text()))) for p in doc]
    required = ["0.079", "1,486", "Qwen3.5 and Qwen3.6", "Held-out present-class",
                "Next-class loss", "Paired", "Saved-Data Details and Exploratory Associations",
                "not a confidence interval", "resident memory", "GPT-OSS", "Nemotron", "GLM", "Gemma", "Qwen3-30B"]
    combined = " ".join(page_text)
    missing = [term for term in required if term not in combined]
    if missing:
        raise ValueError(f"Missing PDF evidence/scope: {missing}")
    log = (build_path.parent / "output/main.log").read_text(errors="replace")
    if re.search(r"(?:Reference|Citation).*undefined|Missing character|^!", log, re.MULTILINE):
        raise ValueError("Unresolved reference, citation, glyph or TeX error")
    outside = []
    for number, page in enumerate(doc, 1):
        for block in page.get_text("blocks"):
            if block[0] < -1 or block[1] < -1 or block[2] > page.rect.width + 1 or block[3] > page.rect.height + 1:
                outside.append({"page": number, "bounds": list(block[:4])})
    if outside:
        raise ValueError(f"Off-page PDF blocks: {outside[:5]}")
    selected_pages = {1}
    for term in ["Held-out present-class", "Next-class loss", "Minimum held-out reasoning-class",
                 "Saved-Data Details and Exploratory Associations", "Adjusted expert-profile associations"]:
        hits = [i + 1 for i, txt in enumerate(page_text) if term in txt]
        if not hits:
            raise ValueError(f"Could not locate {term}")
        selected_pages.add(hits[0])
    visual = build_path.parent / "visual"
    visual.mkdir(exist_ok=True)
    rendered = {}
    for number in sorted(selected_pages):
        path = visual / f"page-{number}.png"
        doc[number - 1].get_pixmap(matrix=fitz.Matrix(1.35, 1.35)).save(path)
        rendered[str(path)] = sha(path)
    receipt = {"schema": "paper-best-results-validation-v1", "observed_utc": datetime.now(timezone.utc).isoformat(),
               "status": "NUMERIC_AND_PDF_CHECKS_PASS_VISUAL_REVIEW_PENDING", "pages": len(doc),
               "build_receipt": str(build_path), "build_receipt_sha256": sha(build_path),
               "pdf_path": build["pdf_path"], "pdf_sha256": build["pdf_sha256"],
               "readout_rows_matched_original": len(readout_rows),
               "anticipation_rows_matched_original": len(anticipation_rows),
               "selection_cells_matched_original": len(selection_rows),
               "all_seven_selection_models": True, "v2_source_bindings_unchanged": len(readiness["files"]),
               "scientific_files_preserved": readiness["preserved_scientific_files"],
               "operational_files_preserved": readiness["preserved_operational_files"],
               "figures_manifest_sha256": sha(BUNDLE / "FIGURES.json"),
               "off_page_blocks": outside, "undefined_references_or_missing_glyphs": False,
               "rendered_page_files": rendered, "rendered_pages": sorted(selected_pages),
               "new_scientific_computation": False, "scientific_dispatch": "NONE", "script_sha256": sha(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.build.resolve(), args.output.resolve())
