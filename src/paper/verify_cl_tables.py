"""Check the cells and rank marks of the manuscript tables against the accepted revision and the saved estimates.

Reads only: no fitting, resampling or inference. Every table cell, found by table, model, condition and metric, must
equal the cell of the accepted manuscript revision (verify_cl_printed_tables.ACCEPTED_REVISION); the benchmark and routing
tables are also checked against benchmark_cells.csv and the saved routing estimates, the readout table and figure labels
against routing_readout.csv, the next-state table against next_state.csv, and the occupancy and transition tables against
native_dynamics.json. The row order of a table may differ
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


def thousands(value):
    return f"{int(value):,}"


NEXT_STATE_COLUMNS = (("", "gain"), ("null_", "alignment gain"))


def gains_from(rows, prefix):
    """Each model's gain in nats (positive when routing lowers the loss) with its simultaneous limits."""
    gains = {}
    for model in DISPLAY_MODELS:
        r = rows[model]
        gains[model] = [-float(r[prefix + key]) for key in ("delta_logloss", "simultaneous_ci7_high", "simultaneous_ci7_low")]
    return gains


def verify_next_state(paper, texts):
    """The next-state table and the plotting data against next_state.csv: sizes, gains, intervals and rank marks."""
    data = figure_root(paper) / "data"
    rows = {row["model"]: row for row in read_csv(data / "next_state.csv")}
    require(set(rows) == set(DISPLAY_MODELS) and all(row["status"] == "CLEAN" for row in rows.values()),
            "next_state.csv model roster or status differs")
    gains = [gains_from(rows, prefix) for prefix, _ in NEXT_STATE_COLUMNS]
    seen = []
    for model, columns in model_rows(texts["next_state.tex"]):
        seen.append(model)
        require(len(columns) == 5, model + ": next-state columns differ")
        require([clean_cell(columns[1]), clean_cell(columns[2])] == [thousands(rows[model]["questions"]), thousands(rows[model]["pairs"])],
                model + ": next-state question or pair count differs from next_state.csv")
        for cell, (prefix, name), values in zip(columns[3:], NEXT_STATE_COLUMNS, gains):
            check_ci(cell, values[model], 5, model + "/next-state " + name, rank_mark(values[model][0], [v[0] for v in values.values()]))
    require(len(seen) == len(DISPLAY_MODELS) and set(seen) == set(DISPLAY_MODELS), "next-state model coverage differs")
    table = [line.split() for line in (data / "next_state.dat").read_text().splitlines()]
    heading, body = table[0], table[1:]
    require(heading == ["x", "gain", "gainminus", "gainplus", "gainlabely", "gainlabel",
                        "null", "nullminus", "nullplus", "nulllabely", "nulllabel"], "next_state.dat columns differ")
    require(len(body) == len(DISPLAY_MODELS), "next_state.dat row count differs")
    for row, model in zip(body, PRESENTATION_MODELS):
        for offset, (prefix, name) in zip((1, 6), NEXT_STATE_COLUMNS):
            mean, low, high = (float(rows[model][prefix + key]) for key in ("delta_logloss", "simultaneous_ci7_low", "simultaneous_ci7_high"))
            saved = [float(value) for value in row[offset:offset + 3]]
            require(all(abs(a - b) <= 1e-12 for a, b in zip(saved, (mean, mean - low, high - mean))), model + "/" + name + ": next_state.dat differs from next_state.csv")
            require(row[offset + 4] == f"{-mean:.3f}", model + "/" + name + ": value label differs from next_state.csv")
    return [("next_state.tex", len(DISPLAY_MODELS) * 4, "figures/journal/data/next_state.csv",
             {"rank_scope": "descriptive, within each gain column across the models"})]


def dynamics_text(panel, key, scale, digits):
    """A native-dynamics cell as the tables print it: mean with its symmetric margin, or the mean alone when no interval exists."""
    cell = panel["results"][key]
    mean = decimal(f"{scale * cell['estimate']:.{digits}f}")
    if cell["interval"] is None:
        return ("n/e", mean)
    low, high = (scale * x for x in cell["interval"])
    return ("cid", (mean, *formatted([max(scale * cell["estimate"] - low, high - scale * cell["estimate"])] * 2, digits)))


def check_dynamics_cell(text, panel, key, scale, digits, context):
    kind, value = dynamics_text(panel, key, scale, digits)
    if kind == "n/e":
        require(clean_cell(text).replace(" ", "") == f"${value}_{{\\mathrm{{n/e}}}}$", context + ": estimate without an interval is not printed as n/e")
    else:
        require(ci_cells(text) == [("cid", value)], context + ": estimate or margin differs from native_dynamics.json")


def verify_dynamics(paper, texts):
    """The occupancy and transition tables against native_dynamics.json: every estimate, margin and n/e mark."""
    panels = json.loads((figure_root(paper) / "data/native_dynamics.json").read_text())
    require(set(panels) == set(DISPLAY_MODELS), "native_dynamics.json model roster differs")
    seen = []
    for model, columns in model_rows(texts["occupancy.tex"]):
        seen.append(model)
        require(len(columns) == 8, model + ": occupancy columns differ")
        for state, cell in zip(STATE_ORDER, columns[1:]):
            check_dynamics_cell(cell, panels[model], state + ":occupancy", 100, 2, model + "/" + state)
    require(len(seen) == len(DISPLAY_MODELS) and set(seen) == set(DISPLAY_MODELS), "occupancy model coverage differs")
    checked = [("occupancy.tex", len(seen) * len(STATE_ORDER), "figures/journal/data/native_dynamics.json",
                {"rank_scope": "none; state shares within model"})]
    seen = []
    cells = (("label_coverage", 100), ("Analyze:next:Implement", 100), ("Implement:next:Verify", 100),
             ("Verify:next:Verify", 100), ("Verify:observed_dwell_tokens", 1))
    for model, columns in model_rows(texts["state_transitions.tex"]):
        seen.append(model)
        require(len(columns) == 7, model + ": state-transition columns differ")
        families = panels[model]["results"]["label_coverage"]["family_support"]
        require(clean_cell(columns[1]) == thousands(families), model + ": family count differs from native_dynamics.json")
        for cell, (key, scale) in zip(columns[2:], cells):
            check_dynamics_cell(cell, panels[model], key, scale, 2, model + "/" + key)
    require(len(seen) == len(DISPLAY_MODELS) and set(seen) == set(DISPLAY_MODELS), "state-transition model coverage differs")
    checked.append(("state_transitions.tex", len(seen) * (len(cells) + 1), "figures/journal/data/native_dynamics.json",
                    {"rank_scope": "none; transition probabilities within model"}))
    return checked


OCCUPANCY_ORDER = ("Read", "Analyze", "Explore", "Plan", "Implement", "Verify", "Monitor")


def verify_occupancy_data(paper):
    """occupancy.dat against native_dynamics.json: means, margins, whisker tops, value labels and leader lengths.

    The figure types no number: every label and leader comes from this file.
    """
    data = figure_root(paper) / "data"
    panels = json.loads((data / "native_dynamics.json").read_text())
    table = [line.split() for line in (data / "occupancy.dat").read_text().splitlines()]
    heading, body = table[0], table[1:]
    require(len(body) == len(PRESENTATION_MODELS), "occupancy.dat row count differs")
    require(not re.search(r"at \(axis cs:", (figure_root(paper) / "occupancy.tex").read_text()), "the occupancy figure types a coordinate")
    for row, model in zip(body, PRESENTATION_MODELS):
        values = dict(zip(heading, row))
        for state in OCCUPANCY_ORDER:
            cell = panels[model]["results"][state + ":occupancy"]
            mean, (low, high) = 100 * cell["estimate"], cell["interval"]
            name, initial = state.lower(), state[0].lower()
            top = mean + 100 * max(cell["estimate"] - low, high - cell["estimate"])
            where = model + "/" + state
            require(abs(float(values[name]) - mean) < 1e-9, where + ": occupancy mean differs from native_dynamics.json")
            require(abs(float(values[initial + "minus"]) - 100 * (cell["estimate"] - low)) < 1e-9
                    and abs(float(values[initial + "plus"]) - 100 * (high - cell["estimate"])) < 1e-9, where + ": occupancy margins differ")
            require(abs(float(values[name + "top"]) - top) < 1e-9, where + ": whisker top differs")
            require(values[name + "label"] == f"{mean:.1f}", where + ": occupancy label differs from native_dynamics.json")
            require(float(values[name + "lead"]) > 0 and float(values[name + "labely"]) > top, where + ": label does not lie above its whisker")
    return [("occupancy.dat (figure labels)", len(PRESENTATION_MODELS) * len(OCCUPANCY_ORDER), "figures/journal/data/native_dynamics.json",
             {"rank_scope": "none"})]


def verify_summary_data(paper):
    """state_transitions_summary.dat against native_dynamics.json: means, margins, labels and the unavailable-interval flags."""
    data = figure_root(paper) / "data"
    panels = json.loads((data / "native_dynamics.json").read_text())
    table = [line.split() for line in (data / "state_transitions_summary.dat").read_text().splitlines()]
    heading, body = table[0], table[1:]
    names = ("ai", "iv", "vv", "vd")
    keys = ("Analyze:next:Implement", "Implement:next:Verify", "Verify:next:Verify", "Verify:observed_dwell_tokens")
    scales = (100, 100, 100, 1)
    require(heading == ["x"] + [n + suffix for n in names for suffix in ("", "minus", "plus", "labely", "label", "missing")],
            "state_transitions_summary.dat columns differ")
    require(len(body) == len(PRESENTATION_MODELS), "state_transitions_summary.dat row count differs")
    for row, model in zip(body, PRESENTATION_MODELS):
        for i, (key, scale) in enumerate(zip(keys, scales)):
            mean, minus, plus, _, label, missing = row[1 + 6 * i:7 + 6 * i]
            cell = panels[model]["results"][key]
            require(abs(float(mean) - scale * cell["estimate"]) < 1e-9 * max(1, scale), model + "/" + key + ": summary mean differs")
            require(int(missing) == int(cell["interval"] is None), model + "/" + key + ": unavailable-interval flag differs")
            require(label == f"{scale * cell['estimate']:.1f}" + (r"$^{\dagger}$" if cell["interval"] is None else ""),
                    model + "/" + key + ": summary label differs")
            if cell["interval"] is not None:
                low, high = (scale * x for x in cell["interval"])
                require(abs(float(minus) - (scale * cell["estimate"] - low)) < 1e-9 * max(1, scale)
                        and abs(float(plus) - (high - scale * cell["estimate"])) < 1e-9 * max(1, scale),
                        model + "/" + key + ": summary margins differ")
    return [("state_transitions_summary.dat", len(PRESENTATION_MODELS) * len(names), "figures/journal/data/native_dynamics.json",
             {"rank_scope": "none"})]


NEXT_RESULTS = r"\subsection{Routing adds information about the next reasoning state}"
DECIMAL = re.compile(r"(?<![\d.])\d+\.\d+")
MODEL_VERSION = re.compile(r"(?:Qwen|LFM)\d+\.\d+")


def verify_next_state_prose(paper):
    """The next-state Results subsection against next_state.csv: its claims about intervals, ranges and every number."""
    data = figure_root(paper) / "data"
    rows = {row["model"]: row for row in read_csv(data / "next_state.csv")}
    gains = gains_from(rows, "")
    nulls = gains_from(rows, "null_")
    excluding = {model for model, (mean, low, high) in gains.items() if low > 0}
    require(len(excluding) == len(DISPLAY_MODELS) - 1 and set(DISPLAY_MODELS) - excluding == {LFM_MODEL},
            "next-state gain intervals no longer exclude zero in exactly seven models")
    require(all(low > 0 for _, low, _ in nulls.values()), "an alignment-gain interval contains zero")
    results = (paper / "journal/results.tex").read_text()
    text = results.split(NEXT_RESULTS)[1].split(r"\subsection")[0]
    text = "\n".join(line for line in text.splitlines() if not line.startswith("%"))
    cells = {values for _, values in ci_cells((paper / "journal/tables/next_state.tex").read_text())}
    shown = [values for _, values in ci_cells(text)]
    require(shown and set(shown) <= cells, "a number of the next-state Results is not a cell of the next-state table")
    lowest, highest = min(excluding, key=lambda m: gains[m][0]), max(excluding, key=lambda m: gains[m][0])
    for model in (lowest, highest, LFM_MODEL):
        require(formatted_cell(gains[model]) in shown, model + ": next-state Results omits its gain")
    require(formatted_cell(nulls[LFM_MODEL]) in shown, "next-state Results omits the alignment gain of LFM2.5")
    relative = [100 * g[0] / float(rows[m]["text_loss"]) for m, g in gains.items() if m in excluding]
    allowed = {f"{min(relative):.1f}", f"{max(relative):.1f}", f"{nulls[LFM_MODEL][1]:.4f}"}
    stray = [number for number in DECIMAL.findall(CI.sub("", MODEL_VERSION.sub("", text))) if number not in allowed]
    require(not stray, "next-state Results prints numbers that no receipt supports: " + ", ".join(stray))
    return [("results.tex (next-state subsection)", len(shown) + len(allowed), "figures/journal/data/next_state.csv",
             {"rank_scope": "none"})]


def formatted_cell(values):
    """The (mean, minus, plus) a CI macro prints for a gain with its simultaneous limits."""
    return signed_deviations(values, 5)


READOUT_COLUMNS = (("routing_", "RIG"), ("routing_excess_over_lexical_null_", "alignment gain"))
VALUE_NODE = re.compile(r"at \(axis cs:(\d+),([\d.]+)\) \{([\d.]+)\};")


def verify_readout(paper, texts):
    """The present-state table and the typed value labels of its figure against routing_readout.csv."""
    data = figure_root(paper) / "data"
    rows = {row["model"]: row for row in read_csv(data / "routing_readout.csv")}
    require(set(rows) == set(DISPLAY_MODELS) and all(row["status"] == "CLEAN" for row in rows.values()),
            "routing_readout.csv model roster or status differs")
    gains = [gains_from(rows, prefix) for prefix, _ in READOUT_COLUMNS]
    seen = []
    for model, columns in model_rows(texts["readout.tex"]):
        seen.append(model)
        require(len(columns) == 4, model + ": readout columns differ")
        require(clean_cell(columns[1]) == thousands(rows[model]["sentences"]), model + ": readout sentence count differs from routing_readout.csv")
        for cell, (prefix, name), values in zip(columns[2:], READOUT_COLUMNS, gains):
            check_ci(cell, values[model], 5, model + "/readout " + name, rank_mark(values[model][0], [v[0] for v in values.values()]))
    require(len(seen) == len(DISPLAY_MODELS) and set(seen) == set(DISPLAY_MODELS), "readout model coverage differs")
    blocks = (figure_root(paper) / "readout.tex").read_text().split("\\begin{axis}")[1:]
    require(len(blocks) == len(READOUT_COLUMNS), "readout figure panel count differs")
    for block, values in zip(blocks, gains):
        nodes = {int(x): (float(y), text) for x, y, text in VALUE_NODE.findall(block)}
        require(set(nodes) == set(range(1, len(PRESENTATION_MODELS) + 1)), "readout figure label missing")
        for x, model in enumerate(PRESENTATION_MODELS, 1):
            mean, low, high = values[model]
            require(nodes[x][1] == f"{mean:.3f}", model + ": readout figure label differs from routing_readout.csv")
            require(nodes[x][0] > mean + max(mean - low, high - mean), model + ": readout figure label lies inside its whisker")
    return [("readout.tex", len(DISPLAY_MODELS) * 3, "figures/journal/data/routing_readout.csv",
             {"rank_scope": "descriptive, within each gain column across the models"})]


SELECTION_NODE = re.compile(r"at \(axis cs:(\d+),([-\d.]+)\) \{([^}]+)\};")


def verify_selection_labels(paper):
    """The typed fold and mean labels of the selection figure against selection.dat."""
    table = [line.split() for line in (figure_root(paper) / "data/selection.dat").read_text().splitlines()]
    heading, body = table[0], table[1:]
    blocks = (figure_root(paper) / "selection.tex").read_text().split("\\begin{axis}")[1:]
    require(len(blocks) == len(body) == len(PRESENTATION_MODELS), "selection figure panel count differs")
    for block, row, model in zip(blocks, body, PRESENTATION_MODELS):
        values = dict(zip(heading, map(float, row)))
        expected = {x: values[f"f{x - 1}"] for x in range(1, 6)} | {6: values["mean"]}
        nodes = {int(x): text for x, _, text in SELECTION_NODE.findall(block)}
        require(set(nodes) == set(expected), model + ": selection figure label missing")
        for x, value in expected.items():
            require(nodes[x] == f"{value:.2f}", model + ": selection figure label differs from selection.dat")
    return [("selection.tex (figure labels)", len(body) * 6, "figures/journal/data/selection.dat", {"rank_scope": "none"})]


def verify(paper):
    """Every table cell against the accepted manuscript revision, and the benchmark and routing tables against the saved estimates."""
    result = verify_declared_source(paper)
    result["benchmark_saved_estimates"] = verify_current_benchmarks(paper)
    texts = {path.name: path.read_text() for path in (paper / "journal/tables").glob("*.tex")}
    result["saved_estimates"] = [
        {"table": name, "ordered_cells": count, "source": source, **extra}
        for name, count, source, extra in (*verify_routing(paper, texts), *verify_readout(paper, texts), *verify_selection_labels(paper), *verify_next_state(paper, texts),
                                           *verify_dynamics(paper, texts), *verify_occupancy_data(paper),
                                           *verify_summary_data(paper), *verify_next_state_prose(paper))
    ]
    style = re.sub(r"\s+", "", (paper / "journal/style.tex").read_text())
    require(r"\fp_compare:nTF{#1>#2}{\pm#1}{\pm#2}" in style, "CI printer does not preserve the maximum original margin")
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
    texts = {path.name: path.read_text() for path in (paper / "journal/tables").glob("*.tex")}
    for name, check, table in (("changed_readout_cell_rejected", verify_readout, "readout.tex"),
                               ("changed_next_state_cell_rejected", verify_next_state, "next_state.tex"),
                               ("changed_occupancy_cell_rejected", verify_dynamics, "occupancy.tex"),
                               ("changed_transition_cell_rejected", verify_dynamics, "state_transitions.tex")):
        changed = {**texts, table: re.sub(r"(\\(?:best|second)?cid\{)(\d)", lambda m: m[1] + str((int(m[2]) + 1) % 10), texts[table], count=1)}
        require(changed[table] != texts[table], "self-test mutation did not apply: " + name)
        try:
            check(paper, changed)
        except TableMismatch:
            checks.append(name)
        else:
            raise TableMismatch("invalid table passed: " + name)
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
