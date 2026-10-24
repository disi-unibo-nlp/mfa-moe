"""Check the cells and rank marks of the manuscript tables against the accepted revision and the saved estimates.

Reads only: no fitting, resampling or inference. Every table cell, found by table, model, condition and metric, must
equal the cell of the accepted manuscript revision (verify_cl_printed_tables.ACCEPTED_REVISION); the benchmark and routing
tables are also checked against benchmark_cells.csv and the saved routing estimates. The row order of a table may differ
from that revision, its numbers may not. --self-test also shows that the check rejects swapped rows, changed intervals
and wrong rank marks in copies of the tables held in memory, and that export_cl_exhibits, run on a temporary copy of the
manuscript, writes only below figures/journal/data and never a table. The printed PDF is read back by
verify_cl_printed_tables.

  python -m paper.verify_cl_tables [--paper PAPER_FOLDER] [--self-test]
"""
import argparse
import contextlib
import copy
import csv
from decimal import Decimal
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from unittest.mock import patch

from paper import cl_paths
from paper.cl_models import MODELS as PRESENTATION_MODELS
from paper.cl_paths import PAPER as DEFAULT_PAPER, REPO
from paper.verify_cl_printed_tables import (
    MODEL as TABLE_MODEL, collect_tables, compare_source_tables, latex_text, verify_declared_source)
from store import verified


def figure_root(paper):
    """Canonical figure sources and data in the selected manuscript checkout."""
    return paper / "figures/journal"


MODELS = ("gpt", "nemotron", "qwen35", "qwen36", "gemma", "glm", "qwen330b")
LFM_MODEL = "lfm25"
DISPLAY_MODELS = (*MODELS, LFM_MODEL)
DASH = r"\textemdash"
LABELS = dict(zip(("GPT-OSS-20B", "Nemotron-3.5-Lightning-30B-A3B",
                   "Qwen3.5-35B-A3B", "Qwen3.6-35B-A3B", "Gemma-4-26B-A4B",
                   "GLM-4.7-Flash (30B-A3B)", "Qwen3-30B-A3B"), MODELS))
LABELS["LFM2.5-24B-A2B"] = LFM_MODEL
LABELS["Nemotron-3.5 30B-A3B"] = "nemotron"
LABELS["Nemotron-3.5-Lightning 30B-A3B"] = "nemotron"
BENCHMARKS = ("aime24", "aime25", "amc23", "math500", "minerva", "olympiad")
BENCH_LABELS = dict(zip(("AIME24", "AIME25", "AMC23", "MATH-500", "Minerva Math", "OlympiadBench"), BENCHMARKS))
CI = re.compile(r"\\(bestcid|secondcid|cid)\{([^}]+)\}\{([^}]+)\}\{([^}]+)\}")
MODEL = re.compile(r"\\modelname\{[^}]+\}\{((?:\\shortstack\[l\]\{[^}]+\})|[^}]+)\}")


class TableMismatch(ValueError):
    pass


def require(condition, context):
    if not condition:
        raise TableMismatch(context)


def decimal(value):
    return Decimal(str(value).replace("\\,", ""))


def formatted(values, precision):
    return tuple(decimal(f"{float(value):.{precision}f}") for value in values)


def ci_cells(text):
    return [(match[1], tuple(decimal(match[i]) for i in (2, 3, 4))) for match in CI.finditer(text)]


def model_label(match):
    value = match[1]
    prefix = r"\shortstack[l]{"
    if value.startswith(prefix):
        value = value[len(prefix):-1].replace(r"\\", "-")
    return value


def model_rows(text):
    for line in text.splitlines():
        if line.startswith("\\modelname"):
            match = MODEL.search(line)
            require(match and model_label(match) in LABELS, "unregistered model identity: " + line)
            yield LABELS[model_label(match)], line.split(" & ")


def rank_mark(value, values, *, higher=True):
    ranks = sorted(set(values), reverse=higher)
    if value == ranks[0]:
        return "bestci"
    return "secondci" if len(ranks) > 1 and value == ranks[1] else "ci"


def signed_deviations(values, precision):
    """Round original deviations, never subtract separately rounded endpoints."""
    mean, low, high = map(decimal, values)
    require(low <= mean <= high, "saved interval does not bracket its estimate")
    errors = tuple(decimal(f"{value:.{precision}f}") for value in (mean - low, high - mean))
    return (formatted([mean], precision)[0], *errors)


def check_ci(text, values, precision, context, mark="ci"):
    require(ci_cells(text) == [(mark + "d", signed_deviations(values, precision))], context + ": mean/printed CI deviations or rank differs")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def clean_cell(text):
    return text.split(r"\\", 1)[0].strip()


def dash_cells(columns, context):
    """Cells without an estimate print a dash and nothing else."""
    require(all(clean_cell(cell) == DASH for cell in columns), context + ": a cell without an estimate holds a value")


ROUTING_STATES = ("activate:occupancy", "deactivate:occupancy", "activate:first", "deactivate:first")
ROUTING_ACCURACY = ("activate:correct:vs_native", "random_activate:correct:vs_native",
                    "deactivate:correct:vs_native", "random_deactivate:correct:vs_native")
ROUTING_TOKENS = ("activate:log_tokens:vs_native", "random_activate:log_tokens:vs_native",
                  "deactivate:log_tokens:vs_native", "random_deactivate:log_tokens:vs_native")
ROUTING_ARMS = (("native", "Native"), ("target_activate", "Targeted activation"),
                ("random_activate", "Random activation"), ("target_deactivate", "Targeted deactivation"),
                ("random_deactivate", "Random deactivation"))
STRONG_MODELS = ("gpt", "nemotron", "qwen330b", "gemma")
STATE_ORDER = ("Read", "Analyze", "Explore", "Plan", "Implement", "Verify", "Monitor")
PREREG = REPO / "report/final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json"


def symmetric_cell(text, mean, margin, digits, context):
    """A routing cell prints the saved mean and its one symmetric margin, both rounded once."""
    shown = decimal(f"{mean:.{digits}f}"), decimal(f"{margin:.{digits}f}")
    require(ci_cells(text) == [("cid", (shown[0], shown[1], shown[1]))], context + ": routing estimate or margin differs from routing_results.json")


def verify_routing(paper, texts):
    """Check the live routing tables and figure coordinates against routing_results.json."""
    data = figure_root(paper) / "data"
    record = json.loads((data / "routing_results.json").read_text())
    require(all(model in record for model in DISPLAY_MODELS) and "all" in record, "routing_results.json model roster differs")
    source = "figures/journal/data/routing_results.json"
    cells = ((ROUTING_STATES[0], 2), (ROUTING_STATES[1], 2), (ROUTING_STATES[2], 2), (ROUTING_STATES[3], 2),
             ("activate:correct", 2), ("activate:log_tokens", 1),
             ("activate:correct:vs_native", 2), ("activate:log_tokens:vs_native", 1))

    seen = []
    for model, columns in model_rows(texts["routing.tex"]):
        seen.append(model)
        require(len(columns) == 9, model + ": routing columns differ")
        for cell, (key, digits) in zip(columns[1:], cells):
            symmetric_cell(cell, record[model][key]["mean"], record[model][key]["margin"], digits, model + "/" + key)
    require(len(seen) == len(DISPLAY_MODELS) and set(seen) == set(DISPLAY_MODELS), "routing model coverage differs")
    pooled = [line for line in texts["routing.tex"].splitlines() if line.startswith("All models & ")]
    require(len(pooled) == 1, "routing pooled row missing or duplicated")
    columns = pooled[0].split(" & ")
    require(len(columns) == 9, "routing pooled columns differ")
    for cell, (key, digits) in zip(columns[1:], cells):
        symmetric_cell(cell, record["all"][key]["mean"], record["all"][key]["margin"], digits, "all/" + key)
    # Family counts live in the saved evidence; the author excludes count inventories from captions.
    require(record["lfm25"]["levels"]["native"]["cells"] == 112,
            "routing evidence lost the LFM2.5 family count")
    require("Dashes" not in texts["routing.tex"] and r"\textemdash" not in texts["routing.tex"], "routing table contains cells without an estimate")
    checked = [("routing.tex", (len(seen) + 1) * len(cells), source, {"rank_scope": "none; paired contrasts with model-specific simultaneous intervals"})]

    levels = {}
    model = None
    arms = iter(())
    for line in texts["routing_levels.tex"].splitlines():
        if line.startswith(r"\multicolumn{5}{l}{\modelname"):
            match = MODEL.search(line)
            require(match is not None and model_label(match) in LABELS, "routing-level model identity differs")
            model = LABELS[model_label(match)]
            require(model not in levels, "duplicated routing-level model " + model)
            levels[model] = []
            arms = iter(ROUTING_ARMS)
        elif model and " & " in line and not line.startswith("Arm & "):
            arm, label = next(arms, (None, None))
            columns = line.split(" & ")
            require(arm is not None and columns[0] == label and len(columns) == 5, model + ": routing-level arm identity or columns differ")
            levels[model].append(arm)
            values = record[model]["levels"][arm]
            for cell, (key, digits) in zip(columns[1:], (("accuracy", 2), ("generated_tokens", 0), ("reasoning_tokens", 0), ("natural_stop", 1))):
                symmetric_cell(cell, values[key][0], values[key][1], digits, model + "/" + arm + "/" + key)
    require(set(levels) == set(DISPLAY_MODELS) and all(tuple(v) == tuple(a for a, _ in ROUTING_ARMS) for v in levels.values()), "routing-level coverage differs")
    checked.append(("routing_levels.tex", len(levels) * len(ROUTING_ARMS) * 4, source, {"rank_scope": "none; arm levels within model"}))

    strong = []
    for model, columns in model_rows(texts["routing_strong.tex"]):
        strong.append(model)
        require(len(columns) == 7, model + ": strong-dose columns differ")
        for cell, key in zip(columns[1:3], ("occupancy", "first")):
            symmetric_cell(cell, *record["strong"][model][key], 2, model + "/strong/" + key)
        dash_cells(columns[3:], model + "/strong efficiency")
    require(len(strong) == len(STRONG_MODELS) and set(strong) == set(STRONG_MODELS), "strong-dose model coverage differs")
    pooled = [line for line in texts["routing_strong.tex"].splitlines() if line.startswith("All four models & ")]
    require(len(pooled) == 1, "strong-dose pooled row missing or duplicated")
    columns = pooled[0].split(" & ")
    require(len(columns) == 7, "strong-dose pooled columns differ")
    for cell, (key, digits) in zip(columns[1:], (("occupancy", 2), ("first", 2), ("accuracy_vs_random", 2), ("tokens_vs_random", 1),
                                                 ("accuracy_vs_native", 2), ("random_accuracy_vs_native", 2))):
        symmetric_cell(cell, *record["strong"]["all"][key], digits, "strong/all/" + key)
    checked.append(("routing_strong.tex", len(strong) * 2 + 6, source, {"rank_scope": "none; exploratory, pooled efficiency only"}))

    prereg = verified(PREREG)
    link = {(c["model"], c["class"]): c["tok_reason"] for c in prereg["cells"]}
    link.update({("pooled", c["class"]): c["tok_reason"] for c in prereg["pooled"]})
    rows = {model: columns for model, columns in model_rows(texts["efficiency_link.tex"])}
    pooled = [line for line in texts["efficiency_link.tex"].splitlines() if line.startswith("Eight-model mean & ")]
    require(len(pooled) == 1 and set(rows) == set(prereg["models"]), "efficiency-link model coverage differs")
    rows["pooled"] = pooled[0].split(" & ")
    for model, columns in rows.items():
        require(len(columns) == 8, model + ": efficiency-link columns differ")
        for state, cell in zip(STATE_ORDER, columns[1:]):
            value = link[model, state]
            mean, low, high = (100 * value[k] for k in ("beta", "ci_low", "ci_high"))
            require(ci_cells(cell) == [("cid", tuple(decimal(f"{x:.2f}") for x in (mean, mean - low, high - mean)))],
                    model + "/" + state + ": efficiency-link estimate or interval differs from the sealed receipt")
    checked.append(("efficiency_link.tex", len(rows) * len(STATE_ORDER), str(PREREG.relative_to(REPO)), {"rank_scope": "none; native association coefficients"}))

    # The routing figure plots the same saved values; routing.dat keeps six significant digits.
    table = [line.split() for line in (data / "routing.dat").read_text().splitlines()]
    heading, body = table[0], table[1:]
    require(len(body) == len(DISPLAY_MODELS) + 1, "routing.dat row count differs")
    plotted = ROUTING_STATES + ROUTING_ACCURACY + ROUTING_TOKENS
    require(heading == ["x"] + [name for key in plotted for name in (key.replace(":", "_"), key.replace(":", "_") + "_pm")], "routing.dat columns differ")
    for row, model in zip(body, (*PRESENTATION_MODELS, "all")):
        for i, key in enumerate(plotted):
            for offset, field in enumerate(("mean", "margin")):
                saved, shown = record[model][key][field], float(row[1 + 2 * i + offset])
                require(abs(saved - shown) <= 1e-5 * max(1.0, abs(saved)), model + "/" + key + ": routing.dat differs from routing_results.json")
    return checked


def exporter_writes_only_data(paper):
    """Run export_cl_exhibits on a temporary copy of the tables and data; only files below figures/journal/data may change."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        figures = root / "figures/journal"
        shutil.copytree(paper / "journal/tables", root / "journal/tables")
        shutil.copytree(figure_root(paper) / "data", figures / "data")
        for path in root.rglob("*"):
            if path.is_file():
                os.utime(path, ns=(0, 0))  # a file the exporter writes gets a later time
        spec = importlib.util.spec_from_file_location("export_cl_exhibits_under_test",
                                                      Path(__file__).with_name("export_cl_exhibits.py"))
        exporter = importlib.util.module_from_spec(spec)
        with patch.multiple(cl_paths, PAPER=root, FIGURES=figures, DATA=figures / "data"):
            spec.loader.exec_module(exporter)  # binds the temporary folders
        with contextlib.redirect_stdout(io.StringIO()):
            exporter.main()
        written = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.stat().st_mtime_ns != 0}
    elsewhere = sorted(name for name in written if not name.startswith("figures/journal/data/"))
    require(written and not elsewhere, "the exporter wrote outside figures/journal/data: " + ", ".join(elsewhere))


def verify_current_benchmarks(paper):
    """All eight models' benchmark cells against benchmark_cells.csv: accuracy, interval, rank mark and denominators."""
    tables = collect_tables(paper)
    records = [row for row in read_csv(figure_root(paper) / "data/benchmark_cells.csv") if row["corpus"] == "card-v3"]
    benchmark = {(row["model"], row["dataset"]): row for row in records}
    expected = {(model, dataset) for model in DISPLAY_MODELS for dataset in BENCHMARKS}
    require(len(records) == len(expected) and set(benchmark) == expected, "benchmark saved model/dataset roster differs")
    values = {dataset: [float(benchmark[model, dataset]["adjudicated_accuracy"]) for model in DISPLAY_MODELS] for dataset in BENCHMARKS}
    count = 0
    for label in ("tab:benchmarks", "tab:benchmark-details"):
        seen = set()
        for row in tables[label]["rows"]:
            columns = row["cells"]
            match = TABLE_MODEL.search(columns[0])
            require(match is not None and match[1] in LABELS, "unregistered benchmark model")
            model = LABELS[match[1]]
            if label == "tab:benchmarks":
                require(len(columns) == 7, "main benchmark width differs")
                assignments = zip(BENCHMARKS, columns[1:])
            else:
                dataset_label = row["identity"][1]
                dataset = next((key for name, key in BENCH_LABELS.items() if latex_text(name) == dataset_label), None)
                require(dataset is not None and len(columns) == 4, "detailed benchmark identity differs")
                source = benchmark[model, dataset]
                require(columns[1] == source["questions"] and columns[2] == source["adjudicated_scored"], model + "/" + dataset + ": denominator differs")
                assignments = [(dataset, columns[3])]
            for dataset, cell in assignments:
                require((model, dataset) not in seen, "duplicate benchmark identity")
                seen.add((model, dataset))
                source = benchmark[model, dataset]
                interval = [100 * float(source[key]) for key in ("adjudicated_accuracy", "adjudicated_ci_low", "adjudicated_ci_high")]
                mark = rank_mark(float(source["adjudicated_accuracy"]), values[dataset])
                check_ci(cell, interval, 1, label + "/" + model + "/" + dataset, mark)
                count += 1
        require(seen == expected, label + ": missing model/dataset cell")
    return {
        "source": "figures/journal/data/benchmark_cells.csv",
        "models": len(DISPLAY_MODELS),
        "datasets": len(BENCHMARKS),
        "ordered_accuracy_cells": count,
        "ordered_denominator_cells": len(expected) * 2,
    }


def verify(paper):
    """Every table cell against the accepted manuscript revision, and the benchmark and routing tables against the saved estimates."""
    result = verify_declared_source(paper)
    result["benchmark_saved_estimates"] = verify_current_benchmarks(paper)
    texts = {path.name: path.read_text() for path in (paper / "journal/tables").glob("*.tex")}
    result["routing_saved_estimates"] = [
        {"table": name, "ordered_cells": count, "source": source, **extra}
        for name, count, source, extra in verify_routing(paper, texts)
    ]
    style = re.sub(r"\s+", "", (paper / "journal/style.tex").read_text())
    require(r"\fp_compare:nTF{#1>#2}{\pm#1}{\pm#2}" in style, "CI printer no longer preserves maximum original margin")
    result["uncertainty_contract"] = "Both original deviations checked; printed plus/minus is their maximum."
    return result


def self_test(paper):
    """The source comparison rejects an altered interval and a wrong rank mark; the exporter writes only figure data."""
    original = collect_tables(paper)
    checks = []
    for name, alteration in (
        ("changed_interval_rejected", lambda cell: cell.replace("{13.0}", "{13.1}", 1)),
        ("wrong_rank_rejected", lambda cell: cell.replace("\\bestcid", "\\cid", 1)),
    ):
        candidate = copy.deepcopy(original)
        applied = False
        for row in candidate["tab:benchmarks"]["rows"]:
            for index, cell in enumerate(row["cells"]):
                updated = alteration(cell)
                if updated != cell:
                    row["cells"][index] = updated
                    applied = True
                    break
            if applied:
                break
        require(applied, "self-test mutation did not apply: " + name)
        try:
            compare_source_tables(candidate, original)
        except ValueError:
            checks.append(name)
        else:
            raise TableMismatch("invalid source passed: " + name)
    exporter_writes_only_data(paper)
    return {"status": "pass", "checks": ["model_row_swap_rejected", *checks, "export_writes_only_figure_data"],
            "filesystem_writes": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        report = verify(args.paper.resolve())
        if args.self_test:
            report["self_test"] = self_test(args.paper.resolve())
    except (TableMismatch, KeyError, ValueError, OSError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
