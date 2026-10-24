"""Read every ordered printed table cell against the accepted manuscript cells.

The tables of the manuscript folder are compared by table, model, condition and metric with the accepted revision
ACCEPTED_REVISION of its git history (presentation order may change, cell content may not). With --build the printed
PDF of a successful build (BUILD/BUILD.json) is then read cell by cell: coordinates, font faces (bold best estimates),
underline strokes (second best), column headers, and blank regions that must hold no measurement. Both checks also
show that swapped model values are rejected. This is an editorial check with no scientific recomputation; the PDF
readback needs pymupdf.

  python -m paper.verify_cl_printed_tables --paper PAPER_FOLDER [--build BUILD.json] --output READBACK.json
"""
import argparse
from collections import Counter
import copy
from decimal import Decimal
import json
from pathlib import Path
import re
import statistics
import subprocess
import unicodedata

from store import file_sha

ACCEPTED_REVISION = "4c9d7d44ac68bcaed2f1e300f6122a5f4c558b61"
AUTHORIZED_REMOVALS = {"tab:historical-interventions"}
CI = re.compile(r"\\(bestcid|secondcid|cid)\{([^}]+)\}\{([^}]+)\}\{([^}]+)\}")
TABLE = re.compile(r"\\begin\{(table|longtable)\}.*?\\end\{\1\}", re.S)
MODEL = re.compile(r"\\(?:modelname|benchmarkmodel)\{[^}]+\}\{([^}]+)\}")
CONDITION_TABLES = {"tab:selection-folds", "tab:next-state"}


def require(condition: bool, message: str) -> None:
    """Fail at the first missing or mismatched item, with its identity."""
    if not condition:
        raise ValueError(message)


def compact(text: str) -> str:
    """Normalize typography, preserving signs, digits and annotation content."""
    text = unicodedata.normalize("NFKC", text).replace("−", "-")
    return re.sub(r"\s+", "", text)


def latex_text(text: str) -> str:
    """Render the small table macro vocabulary without discarding measurements."""
    def interval(match):
        margin = max((match[3], match[4]), key=Decimal)
        return match[2] + "±" + margin
    text = CI.sub(interval, text)
    text = MODEL.sub(lambda match: match[1], text)
    text = re.sub(r"\\multicolumn\{\d+\}\{.*?\}\{(.*)\}", r"\1", text)
    text = re.sub(r"\\(?:quad|qquad|,|!|;|:)\s*", "", text)
    replacements = {
        r"\textemdash": "—",
        r"\dagger": "†",
        r"\pm": "±",
        r"\%": "%",
        r"\_": "_",
        r"\Delta": "∆",
        r"\to": "→",
        r"\uparrow": "↑",
        r"\ ": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[A-Za-z]+(?:\[[^]]*\])?", "", text)
    text = re.sub(r"\\\\\*?(?:\[[^]]*\])?", "", text)
    return compact(text.translate(str.maketrans("", "", "{}$^_")))


def source_signature(text: str) -> str:
    """Preserve original asymmetric CI arguments and rank macros in source checks."""
    return compact(text.replace("Nemotron-3.5 30B-A3B", "Nemotron-3.5-Lightning-30B-A3B"))


def parse_table(text: str, path: str) -> dict:
    """Extract all data rows, column headers and hierarchical row identities."""
    label = re.search(r"\\label\{(tab:[^}]+)\}", text)
    require(label is not None, "unlabelled populated table: " + path)
    label = label[1]
    body = re.split(r"\\begin\{(?:tabular|longtable)\}", text, maxsplit=1)[1]
    body = body[body.index(r"\toprule") + len(r"\toprule"):]
    rows, headers, events = [], [], []
    group, condition, started = "", "", False
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        model = MODEL.search(line)
        if model and line.startswith(r"\multicolumn"):
            group, condition, started = latex_text(model[0]), "", True
            events.append({"kind": "group", "text": latex_text(line)})
            continue
        if line.startswith(r"\multicolumn") and ("strength" in line or (label == "tab:benchmark-details" and "Continued" not in line)):
            condition = latex_text(line)
            events.append({"kind": "condition", "text": condition})
            continue
        if " & " not in line:
            continue
        cells = [re.sub(r"\\\\\*?(?:\[[^]]*\])?\s*$", "", cell).strip() for cell in re.split(r"\s*&\s*", line)]
        if model:
            started = True
        if not started:
            headers.append(cells)
            continue
        if label == "tab:whole-cells" and latex_text(cells[0]) in {"Promote", "Suppress"}:
            condition = latex_text(cells[0])
        identity = [group, condition, source_signature(cells[0])]
        if label in CONDITION_TABLES:
            identity.append(source_signature(cells[1]))
        row = {"identity": identity, "cells": cells, "text": "".join(latex_text(cell) for cell in cells)}
        rows.append(row)
        events.append({"kind": "row", "row": row})
    require(rows, "no populated rows: " + label)
    require(len({tuple(row["identity"]) for row in rows}) == len(rows), "duplicate row identity: " + label)
    return {
        "label": label,
        "path": path,
        "rows": rows,
        "headers": headers,
        "events": events,
    }


def collect_tables(paper: Path, revision: str | None = None) -> dict:
    """Read current or accepted tables, including embedded roster/status tables."""
    names = ["journal/setup.tex", "journal/appendices.tex"]
    if revision:
        names += subprocess.check_output(["git", "-C", str(paper), "ls-tree", "-r", "--name-only", revision, "journal/tables"], text=True).splitlines()
    else:
        names += [str(path.relative_to(paper)) for path in sorted((paper / "journal/tables").glob("*.tex"))]
    tables = {}
    for name in names:
        source = subprocess.check_output(["git", "-C", str(paper), "show", revision + ":" + name], text=True) if revision else (paper / name).read_text()
        for match in TABLE.finditer(source):
            table = parse_table(match[0], name)
            require(table["label"] not in tables, "duplicate table label: " + table["label"])
            tables[table["label"]] = table
    return tables


def compare_source_tables(current: dict, accepted: dict) -> dict:
    """Compare scientific row content by identity, allowing presentation reorder only."""
    removed = set(accepted) - set(current)
    require(removed <= AUTHORIZED_REMOVALS, "unauthorized missing tables: " + str(removed))
    require(not set(current) - set(accepted), "new tables require an accepted scientific source")
    result = {}
    for label, table in current.items():
        prior = accepted[label]
        expected = {tuple(row["identity"]): tuple(map(source_signature, row["cells"])) for row in prior["rows"]}
        actual = {tuple(row["identity"]): tuple(map(source_signature, row["cells"])) for row in table["rows"]}
        require(actual.keys() == expected.keys(), label + ": model/dataset/condition coverage differs")
        for identity, cells in actual.items():
            require(cells == expected[identity], label + "/" + str(identity) + ": ordered scientific cells or rank annotations differ")
        header = lambda value: Counter(tuple(map(source_signature, cells)) for cells in value)
        require(header(table["headers"]) == header(prior["headers"]), label + ": metric headers differ")
        result[label] = {"rows": len(actual), "cells": sum(len(row["cells"]) - 1 for row in table["rows"]), "source": table["path"]}
    return {"status": "pass", "accepted_revision": ACCEPTED_REVISION, "tables": result, "authorized_removed_tables": sorted(removed)}


def verify_declared_source(paper: Path) -> dict:
    """Verify all current scientific cells against the accepted manuscript.

    Args:
        paper: Active Overleaf checkout.
    Returns:
        Per-table counts and the pinned accepted revision.
    """
    current, accepted = collect_tables(paper), collect_tables(paper, ACCEPTED_REVISION)
    report = compare_source_tables(current, accepted)
    # Preserve all numbers and row labels while exchanging their metric cells.
    corrupted = copy.deepcopy(current)
    first, second = corrupted["tab:benchmarks"]["rows"][:2]
    first["cells"][1:], second["cells"][1:] = second["cells"][1:], first["cells"][1:]
    try:
        compare_source_tables(corrupted, accepted)
    except ValueError:
        report["swapped_row_negative_rejected"] = True
    else:
        raise ValueError("source audit accepted swapped model values")
    return report


def pdf_characters(document) -> tuple[str, list]:
    """Keep PDF reading order with per-character page, font and bounding box."""
    characters, coordinates = [], []
    for page_number, page in enumerate(document, 1):
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    for char in span["chars"]:
                        for value in compact(char["c"]):
                            characters.append(value)
                            coordinates.append({"page": page_number, "bbox": char["bbox"], "font": span["font"], "size": span["size"]})
    return "".join(characters), coordinates


def rank_evidence(cell: str, records: list, document) -> dict:
    """Check visible best/second marks on the estimate, including PDF strokes."""
    is_best = r"\bestcid" in cell or r"\mathbf" in cell or r"\textbf" in cell
    is_second = r"\secondcid" in cell or r"\underline" in cell
    if not (is_best or is_second):
        return {"expected": "plain"}
    match = CI.search(cell)
    mean = match[2] if match else re.search(r"[+-]?\d+(?:\.\d+)?", latex_text(cell))[0]
    estimate = records[:len(mean)]
    if is_best:
        require(all("bold" in record["font"].lower() or "cmb" in record["font"].lower() for character, record in zip(mean, estimate) if character.isdigit()), "best estimate is not printed bold: " + cell)
        return {"expected": "bold", "fonts": sorted({record["font"] for record in estimate})}
    page = document[estimate[0]["page"] - 1]
    left = min(record["bbox"][0] for record in estimate)
    right = max(record["bbox"][2] for record in estimate)
    top = min(record["bbox"][1] for record in estimate)
    bottom = max(record["bbox"][3] for record in estimate)
    strokes = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.height < 1 and abs(rect.x0-left) < 3 and abs(rect.x1-right) < 3 and top < rect.y0 < bottom + 3:
            strokes.append(list(rect))
    require(strokes, "second estimate lacks printed underline: " + cell)
    return {"expected": "underline", "strokes": strokes}


def table_caption_positions(tables: dict, document, auxiliary: str, text: str, coordinates: list) -> dict:
    """Locate actual caption lines while excluding in-text table references."""
    labels = dict(re.findall(r"\\newlabel\{(tab:[^}@]+)\}\{\{([^}]+)\}", auxiliary))
    starts = {}
    page_ranges = {}
    for index, record in enumerate(coordinates):
        page_ranges.setdefault(record["page"], [index, index])[1] = index + 1
    page_lines = [[compact(line) for line in page.get_text().splitlines()] for page in document]
    for label in tables:
        require(label in labels, "table missing from build auxiliary: " + label)
        needle = "Table" + labels[label]
        matches = []
        for page_number, lines in enumerate(page_lines, 1):
            if any(line == needle or line.startswith(needle + ":") for line in lines):
                begin, end = page_ranges[page_number]
                position = text.find(needle, begin, end)
                if position >= 0:
                    matches.append(position)
        require(matches, "printed caption not found: " + needle)
        starts[label] = matches[0]
    return starts


def read_printed_row(row: dict, found: int, text: str, coordinates: list, document, label: str) -> dict:
    """Read one complete row with spatial, rank and uncertainty evidence."""
    position, cells = found, []
    for column, source in enumerate(row["cells"]):
        rendered = latex_text(source)
        records = coordinates[position:position + len(rendered)]
        if not rendered:
            cells.append({"column": column, "source": source, "printed": "", "empty": True})
            continue
        require(records and len({record["page"] for record in records}) == 1, label + ": cell splits across pages")
        require(text[position:position+len(rendered)] == rendered, label + ": ordered column differs")
        cells.append({
            "column": column,
            "source": source,
            "printed": rendered,
            "page": records[0]["page"],
            "bbox": [
                min(record["bbox"][0] for record in records),
                min(record["bbox"][1] for record in records),
                max(record["bbox"][2] for record in records),
                max(record["bbox"][3] for record in records),
            ],
            "rank": rank_evidence(source, records, document),
        })
        position += len(rendered)
    populated = [cell for cell in cells if not cell.get("empty")]
    require(all(left["bbox"][0] < right["bbox"][0] for left, right in zip(populated, populated[1:])), label + ": columns are horizontally displaced")
    numeric = [cell for cell in populated[1:] if re.search(r"\d", cell["printed"])]
    if numeric:
        tops = [cell["bbox"][1] for cell in numeric]
        require(max(tops) - min(tops) < 4, label + ": numeric cells are vertically displaced")
    return {"identity": row["identity"], "cells": cells}


def read_printed_tables(tables: dict, document, auxiliary: str, cached_text=None) -> dict:
    """Match complete ordered rows inside their caption-delimited table region."""
    text, coordinates = cached_text if cached_text is not None else pdf_characters(document)
    starts = table_caption_positions(tables, document, auxiliary, text, coordinates)
    ordered = sorted(starts, key=starts.get)
    report = {}
    for index, label in enumerate(ordered):
        start = starts[label]
        # Floats may interrupt a longtable; ordered model/condition anchors
        # retain ownership through the continued body on following pages.
        stop = len(text)
        cursor, rows, groups = start, [], []
        checked_headers = []
        header_stop = text.find(tables[label]["rows"][0]["text"], start)
        require(header_stop >= 0, label + ": first printed data row differs")
        for header in tables[label]["headers"]:
            rendered = "".join(latex_text(cell) for cell in header)
            location = text.casefold().find(rendered.casefold(), start, header_stop)
            require(location >= 0, label + ": printed ordered metric header differs: " + rendered)
            checked_headers.append({"source": header, "printed": rendered, "page": coordinates[location]["page"]})
        for event in tables[label]["events"]:
            wanted = event["row"]["text"] if event["kind"] == "row" else event["text"]
            found = text.find(wanted, cursor, stop)
            require(found >= 0, label + ": printed " + event["kind"] + " missing or displaced: " + wanted)
            cursor = found + len(wanted)
            if event["kind"] != "row":
                groups.append({"kind": event["kind"], "text": wanted, "page": coordinates[found]["page"]})
                continue
            row = event["row"]
            rows.append(read_printed_row(row, found, text, coordinates, document, label))
        verify_empty_cells(rows, document, label)
        report[label] = {"source": tables[label]["path"], "rows": rows, "group_headings": groups, "column_headers": checked_headers}
    return report



def verify_empty_cells(rows: list, document, label: str) -> None:
    """Check that structurally empty cells contain no apparent PDF measurement."""
    import pymupdf
    centers = {}
    for row in rows:
        for cell in row["cells"]:
            if cell.get("bbox"):
                centers.setdefault(cell["column"], []).append((cell["bbox"][0] + cell["bbox"][2]) / 2)
    centers = {column: statistics.median(values) for column, values in centers.items()}
    for row in rows:
        visible = [cell for cell in row["cells"] if cell.get("bbox")]
        page_number = visible[0]["page"]
        top = min(cell["bbox"][1] for cell in visible)
        bottom = max(cell["bbox"][3] for cell in visible)
        for cell in row["cells"]:
            if not cell.get("empty"):
                continue
            column = cell["column"]
            require(column in centers and column - 1 in centers, label + ": no measured blank-column geometry")
            center, previous = centers[column], centers[column - 1]
            following = centers.get(column + 1, center + center - previous)
            left, right = (previous + center) / 2, (center + following) / 2
            previous_cell = next((value for value in visible if value["column"] == column - 1), None)
            next_cell = next((value for value in visible if value["column"] == column + 1), None)
            if previous_cell:
                left = max(left, previous_cell["bbox"][2] + 0.1)
            if next_cell:
                right = min(right, next_cell["bbox"][0] - 0.1)
            rectangle = pymupdf.Rect(left, top, right, bottom)
            apparent = compact(document[page_number - 1].get_text(clip=rectangle))
            require(not apparent, label + ": apparent content in blank column " + str(column) + ": " + apparent)
            cell.update({"page": page_number, "blank_region": list(rectangle), "blank_region_text": apparent})


def main() -> int:
    """Run current-source and rendered-table checks and write their bound receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "fail"}
    try:
        report["source_verification"] = verify_declared_source(args.paper)
        if args.build:
            import pymupdf
            build = json.loads(args.build.read_text())
            require(build["exit_code"] == 0, "build receipt is not successful")
            pdf = Path(build["pdf_path"])
            require(file_sha(pdf) == build["pdf_sha256"], "PDF differs from build receipt")
            snapshot = Path(build["source_snapshot"])
            for name, expected in build["source_hashes"].items():
                require(file_sha(snapshot / name) == expected, "build snapshot source differs: " + name)
            tables = collect_tables(snapshot)
            current_tables = collect_tables(args.paper)
            compare_source_tables(tables, current_tables)
            for label, table in tables.items():
                require(table["rows"] == current_tables[label]["rows"], label + ": printed build is stale relative to current table order or cells")
            document = pymupdf.open(pdf)
            auxiliary = (snapshot / ".paper-review/build/main.aux").read_text()
            cached_text = pdf_characters(document)
            report["tables"] = read_printed_tables(tables, document, auxiliary, cached_text)
            negative = copy.deepcopy({"tab:benchmarks": tables["tab:benchmarks"]})
            first, second = negative["tab:benchmarks"]["rows"][:2]
            first["cells"][1:], second["cells"][1:] = second["cells"][1:], first["cells"][1:]
            for row in (first, second):
                row["text"] = "".join(latex_text(cell) for cell in row["cells"])
            try:
                read_printed_tables(negative, document, auxiliary, cached_text)
            except ValueError:
                report["pdf_swapped_row_negative_rejected"] = True
            else:
                raise ValueError("PDF audit accepted swapped model values")
            report["pdf_sha256"] = build["pdf_sha256"]
            report["source_snapshot"] = str(snapshot)
            report["table_count"] = len(report["tables"])
            report["row_count"] = sum(len(table["rows"]) for table in report["tables"].values())
            report["cell_count"] = sum(len(row["cells"]) for table in report["tables"].values() for row in table["rows"])
        report["status"] = "pass"
    except (ValueError, OSError, KeyError) as error:
        report["error"] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"tables", "source_verification"}}, indent=2))
    return report["status"] != "pass"


if __name__ == "__main__":
    raise SystemExit(main())
