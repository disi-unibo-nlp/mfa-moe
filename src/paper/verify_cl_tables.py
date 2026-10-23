"""Check ordered manuscript table cells and rank marks against saved evidence.

This bounded, read-only check performs no fitting, resampling or inference.
Real estimates, single engineering fixtures and unavailable intervals have
separate metadata. --self-test also
rejects in-memory row swaps, changed intervals and wrong rank marks, and proves
that the exporter default cannot overwrite author-formatted tables or their
manifest. No source/data files are written by either check.
"""
import argparse
import contextlib
import csv
from decimal import Decimal
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
from paper.cl_paths import DATA as SUPPORT_DATA, FIGURES as SUPPORT_FIGURES, NOTES, PAPER as DEFAULT_PAPER  # noqa: E402


def figure_root(paper):
    """Canonical figure sources and data in the selected manuscript checkout."""
    return SUPPORT_FIGURES if paper == DEFAULT_PAPER else paper / "figures/journal"


def notes_root(paper):
    """Notes such as PLANNED_MODEL.json: the support folder for the manuscript."""
    return NOTES if paper == DEFAULT_PAPER else paper
MODELS = ("gpt", "nemotron", "qwen35", "qwen36", "gemma", "glm", "qwen330b")
PLANNED_MODEL = "lfm25"
DISPLAY_MODELS = (*MODELS, PLANNED_MODEL)
PENDING = r"\textemdash"
SOURCE_LABELS = dict(zip(("GPT-OSS", "Nemotron", "Qwen3.5", "Qwen3.6", "Gemma", "GLM", "Qwen3-30B"), MODELS))
LABELS = dict(zip(("GPT-OSS-20B", "Nemotron-3.5-Lightning-30B-A3B",
                   "Qwen3.5-35B-A3B", "Qwen3.6-35B-A3B", "Gemma-4-26B-A4B",
                   "GLM-4.7-Flash (30B-A3B)", "Qwen3-30B-A3B"), MODELS))
LABELS["LFM2.5-24B-A2B"] = PLANNED_MODEL
BENCHMARKS = ("aime24", "aime25", "amc23", "math500", "minerva", "olympiad")
BENCH_LABELS = dict(zip(("AIME24", "AIME25", "AMC23", "MATH-500", "Minerva Math", "OlympiadBench"), BENCHMARKS))
ARMS = ("state_balanced", "labelled_frequency", "random")
CI = re.compile(r"\\(bestcid|secondcid|cid)\{([^}]+)\}\{([^}]+)\}\{([^}]+)\}")
MODEL = re.compile(r"\\modelname\{[^}]+\}\{((?:\\shortstack\[l\]\{[^}]+\})|[^}]+)\}")
NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")
ROSTER_LABELS = LABELS



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


def displayed_ci_shape(text):
    cells = ci_cells(text)
    require(len(cells) == 1, "CI shape check requires one cell")
    _, (_, minus, plus) = cells[0]
    require(minus >= 0 and plus >= 0, "CI deviations must be nonnegative")
    # Source cells retain both original deviations. The author-approved
    # printer uses the larger one as the symmetric enclosing half-width.
    return "symmetric_pm"



def first_number(text):
    match = NUMBER.search(text)
    require(match is not None, "missing numeric cell: " + text)
    return decimal(match[0])


def scalar_rank(text):
    if "\\mathbf{" in text or "\\textbf{" in text:
        return "bestci"
    return "secondci" if "\\underline{" in text else "ci"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def inline_table(text, label):
    marker = "\\label{" + label + "}"
    require(text.count(marker) == 1, "inline table label missing or duplicated: " + label)
    anchor = text.index(marker)
    start = text.rfind("\\begin{table}", 0, anchor)
    stop = text.find("\\end{table}", anchor)
    require(start >= 0 and stop >= 0, "inline table boundary missing: " + label)
    return text[start:stop + len("\\end{table}")]


def clean_cell(text):
    return text.split(r"\\", 1)[0].strip()


def pending_cells(columns, context):
    require(all(clean_cell(cell) == PENDING for cell in columns), context + ": pending data were filled or concealed")


def verify_caption_only_notes(text, context):
    # Ordinary table content must end with its width wrapper. Semantic notes
    # belong in the caption, not a separate paragraph or second tabular.
    if r"\begin{table}" in text or r"\begin{table}[" in text:
        require(text.count(r"\begin{tabular}") == 1, context + ": extra table key or body returned")
        end = text.rfind(r"\end{adjustbox}")
        stop = text.rfind(r"\end{table}")
        require(end >= 0 and stop > end, context + ": table boundary missing")
        require(not text[end + len(r"\end{adjustbox}"):stop].strip(), context + ": explanatory material below table body")


def verify_table_width_policy(text, context):
    tabulars = text.count(r"\begin{tabular}")
    wrappers = re.findall(r"\\begin\{adjustbox\}\{([^}]+)\}", text)
    require(all(value == r"max width=\linewidth" for value in wrappers), context + ": adjustbox must cap width without enlargement")
    require(len(wrappers) == tabulars, context + ": ordinary tabular lacks its max-width wrapper")
    if r"\begin{longtable}" in text:
        require(not wrappers, context + ": longtable must remain breakable")
        require("Multipage longtable: retained outside adjustbox so page breaks remain available." in text, context + ": missing multipage width-policy explanation")
        require(r"\setlength{\tabcolsep}{2pt}" in text, context + ": compact multipage spacing differs")


def verify_inline(paper, inline_texts=None):
    source_texts = inline_texts or {}
    appendix = source_texts.get("appendices", (paper / "journal/appendices.tex").read_text())
    setup = source_texts.get("setup", (paper / "journal/setup.tex").read_text())
    prefix = inline_table(appendix, "tab:prefix-probes")
    probe_source = {}
    for line in (paper / "probe_grouped_summary_table.tex").read_text().splitlines():
        if line.startswith(("Qwen & ", "GPT-OSS & ", "Gemma & ")):
            columns = line.split(" & ")
            model = {"Qwen": "qwen35", "GPT-OSS": "gpt", "Gemma": "gemma"}[columns[0]]
            metrics = []
            for cell in columns[2:5]:
                values = [float(x) for x in NUMBER.findall(cell)]
                require(len(values) == 3, model + ": supplied probe mean/interval differs")
                metrics.append(values)
            probe_source[model] = metrics
    require(set(probe_source) == {"qwen35", "gpt", "gemma"}, "supplied probe model roster differs")
    seen = set()
    for model, columns in model_rows(prefix):
        require(model in probe_source and model not in seen and len(columns) == 4, "inline probe row/columns differ")
        seen.add(model)
        for i, cell in enumerate(columns[1:]):
            # Position AUROC measures a nuisance baseline, not reasoning quality.
            mark = "ci" if i == 1 else rank_mark(probe_source[model][i][0], [v[i][0] for v in probe_source.values()])
            check_ci(cell, probe_source[model][i], 3, model + "/prefix-" + ("probe", "position", "gain")[i], mark)
    require(seen == set(probe_source), "inline probe model coverage differs")
    require("n/e" not in prefix, "available probe intervals still described as unavailable")

    roster = inline_table(setup, "tab:models")
    coverage = (REPO / "paper/wiki/MODEL_COVERAGE.md").read_text()
    geometry = {}
    for line in coverage.splitlines():
        match = re.match(r"\|\s*\x60([^\x60]+)\x60[^|]*\|\s*(\d+)\s*\|\s*(\d+)\s*/\s*(\d+)\s*\|", line)
        if match and match[1] in MODELS:
            require(match[1] not in geometry, "duplicate captured model geometry in source catalogue")
            geometry[match[1]] = tuple(int(match[i]) for i in (2, 3, 4))
    require(set(geometry) == set(MODELS), "source catalogue geometry roster differs")
    fixtures = {r["model"]: r for r in json.loads((figure_root(paper) / "data/native_fixtures.json").read_text())}
    planned_source = json.loads((notes_root(paper) / "journal/PLANNED_MODEL.json").read_text())
    require(planned_source["model_id"] == "LiquidAI/LFM2.5-24B-A2B" and planned_source["measured_fixture"] is None and not planned_source["scientific_campaign_roster_modified"], "planned model source claims unperformed measurement or changed scientific enrollment")
    planned_config = planned_source["config"]
    planned_geometry = (planned_config["num_hidden_layers"] - planned_config["num_dense_layers"], planned_config["num_experts"], planned_config["num_experts_per_tok"])
    require(planned_source["routed_layers"] == planned_geometry[0], "planned routed-layer derivation differs")
    require(planned_source["official_config_url"] == "https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/resolve/main/config.json", "planned geometry lost its official source")
    seen = set()
    for line in roster.splitlines():
        if not line.startswith("\\modelname"):
            continue
        columns = line.split(" & ")
        match = MODEL.search(columns[0])
        require(match and model_label(match) in ROSTER_LABELS and len(columns) == 6, "inline model roster identity/columns differ")
        model = ROSTER_LABELS[model_label(match)]
        require(model not in seen, "duplicated inline model roster row")
        seen.add(model)
        if model == PLANNED_MODEL:
            require(tuple(int(first_number(cell)) for cell in columns[1:4]) == planned_geometry, "planned model advertised routing geometry differs")
            pending_cells(columns[4:], "planned native fixture")
            require("release configuration" in re.sub(r"\s+", " ", roster) and "pending measurements" in roster, "advertised planned geometry lacks caption qualification")
            continue
        require(tuple(int(first_number(cell)) for cell in columns[1:4]) == geometry[model], model + ": captured L/E/k differs")
        source = fixtures[model]
        require(columns[4] == f"{source['full_context']:,}", model + ": fixture context differs")
        require(first_number(columns[5]) == formatted([source["native_resident_bytes_aggregate"] / 2**30], 2)[0], model + ": native residence differs")
        require(not ci_cells(columns[5]) and scalar_rank(columns[5]) == "ci", "single fixture falsely presented as CI or performance rank")
    require(seen == set(DISPLAY_MODELS), "inline model roster omits a model")
    verify_caption_only_notes(roster, "inline model roster")
    require(not any(value in roster for value in ("MXFP4", "NVFP4", "GPTQ", "FP8", "BF16", "Weight format")), "weight-format column returned to the model roster")

    verify_table_width_policy(prefix, "inline prefix probes")
    verify_caption_only_notes(prefix, "inline prefix probes")
    verify_table_width_policy(roster, "inline model roster")
    protocol = inline_table(appendix, "tab:verification-qualification")
    verify_table_width_policy(protocol, "inline qualification")
    verify_caption_only_notes(protocol, "inline qualification")
    # This is the preserved scientific protocol, separate from the later
    # campaign's operational progress. Its generated table supplies typed status.
    status_source_path = paper / "archive/20261014-pre-journal/generated/verification_status.tex"
    expected = {}
    for line in status_source_path.read_text().splitlines():
        if " & " in line and line.split(" & ")[0] in SOURCE_LABELS:
            columns = line.split(" & ")
            model = SOURCE_LABELS[columns[0]]
            engineering = {"PASS": "Qualified"}[columns[1]]
            recognition = {"FAIL": "Criterion unmet", "Unmeasured": "Incomplete"}[columns[2]]
            control = columns[3].split("\\\\", 1)[0].strip()
            expected[model] = (engineering, recognition, control)
    require(set(expected) == set(MODELS), "preserved protocol status roster differs")
    seen = set()
    for model, columns in model_rows(protocol):
        require(model not in seen and len(columns) == 4, "inline protocol identity/columns differ")
        seen.add(model)
        actual = tuple(cell.split("\\\\", 1)[0].strip() for cell in columns[1:])
        require(actual == expected[model], model + ": engineering/recognition/control status differs")
    require(seen == set(MODELS), "inline protocol omits a model")
    return {
        "prefix_probes": {"source": "probe_grouped_summary_table.tex", "ordered_cells": 9,
                          "model_rows": 3, "available_intervals": 9,
                          "rank_scope": "probe AUROC and paired gain; position baseline unranked"},
        "model_roster": {"source": "paper/wiki/MODEL_COVERAGE.md", "ordered_geometry_cells": 21,
                         "model_rows": 8, "evaluated_model_rows": 7, "pending_model_rows": 1,
                         "advertised_geometry_cells": 3, "advertised_geometry_source": "journal/PLANNED_MODEL.json",
                         "pending_fixture_cells": 2,
                         "weight_formats_in_table": False,
                         "ordered_fixture_cells": 7,
                         "fixture_source": "figures/journal/data/native_fixtures.json",
                         "fixture_kind": "single_engineering_fixtures",
                         "fixture_rank_scope": "none; model-specific context profiles, no repeated-run mean/CI"},
        "verification_protocol": {"source": str(status_source_path.relative_to(paper)),
                                  "ordered_status_cells": 21, "model_rows": 7,
                                  "engineering_qualified": 7, "recognition_criterion_unmet": 5,
                                  "recognition_incomplete": 2, "targeted_control_unmeasured": 7},
    }


def verify_numeric_prose(paper, inline_texts=None):
    """Validate narrative uncertainty against raw estimates where retained.

    Historical engagement/semantic prose was supplied only at printed precision;
    those four displays retain that documented boundary, without invented digits.
    """
    data = figure_root(paper) / "data"
    supplied = inline_texts or {}
    results = supplied.get("results", (paper / "journal/results.tex").read_text())
    appendix = supplied.get("appendices", (paper / "journal/appendices.tex").read_text())
    readout = {r["model"]: r for r in read_csv(data / "routing_readout.csv")}
    dynamics = json.loads((data / "native_dynamics.json").read_text())
    anticipation = {(r["model"], r["mode"]): r for r in read_csv(data / "anticipation.csv")
                    if r["feature"] == "composition16" and r["status"] == "CLEAN"}
    specs = []
    for model in ("gpt", "gemma"):
        row = readout[model]
        specs.append(([-decimal(row[k]) for k in ("routing_delta_logloss", "routing_simultaneous_ci7_high", "routing_simultaneous_ci7_low")], 4, "narrative/RIG/" + model))
    state_examples = (("gpt", "Implement:next:Implement"), ("gemma", "Implement:next:Implement"),
                      ("gpt", "Implement:next:Verify"), ("qwen330b", "Implement:next:Verify"),
                      ("glm", "Verify:observed_dwell_tokens"), ("qwen36", "Verify:observed_dwell_tokens"))
    for model, key in state_examples:
        row = dynamics[model]["results"][key]
        scale = 1 if "dwell" in key else 100
        specs.append(([decimal(x) * scale for x in (row["estimate"], *row["interval"])], 2, "narrative/dynamics/" + model + "/" + key))
    for model in ("gpt", "qwen36"):
        row = anticipation[model, "next"]
        specs.append(([-decimal(row[k]) for k in ("delta_logloss", "simultaneous_ci6_high", "simultaneous_ci6_low")], 5, "narrative/ARG/" + model))
    specs.extend(((("0.1376", "0.0763", "0.1882"), 4, "narrative/historical-engagement/A"),
                  (("0.2940", "0.1277", "0.4761"), 4, "narrative/historical-engagement/B")))
    matches = list(CI.finditer(results))
    require(len(matches) == len(specs), "narrative results CI roster differs")
    for match, (values, precision, context) in zip(matches, specs):
        check_ci(match[0], values, precision, context)
    narrative = appendix.replace(inline_table(appendix, "tab:prefix-probes"), "")
    appendix_matches = list(CI.finditer(narrative))
    semantic = ((("5.50", "-4.50", "15.50"), 2, "narrative/historical-semantic/native"),
                (("11.00", "-0.50", "23.50"), 2, "narrative/historical-semantic/random"))
    require(len(appendix_matches) == len(semantic), "narrative appendix CI roster differs")
    for match, (values, precision, context) in zip(appendix_matches, semantic):
        check_ci(match[0], values, precision, context)
    all_matches = matches + appendix_matches
    return {"ordered_ci_displays": len(all_matches), "original_precision_available": 10,
            "historical_rounded_source_only": 4,
            "symmetric_pm": sum(displayed_ci_shape(match[0]) == "symmetric_pm" for match in all_matches),
            "asymmetric_signed": sum(displayed_ci_shape(match[0]) == "asymmetric_signed" for match in all_matches)}


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
    require(tuple(seen) == DISPLAY_MODELS, "routing model coverage or order differs")
    pooled = [line for line in texts["routing.tex"].splitlines() if line.startswith("All models & ")]
    require(len(pooled) == 1, "routing pooled row missing or duplicated")
    columns = pooled[0].split(" & ")
    require(len(columns) == 9, "routing pooled columns differ")
    for cell, (key, digits) in zip(columns[1:], cells):
        symmetric_cell(cell, record["all"][key]["mean"], record["all"][key]["margin"], digits, "all/" + key)
    require("LFM2.5 (112)" in texts["routing.tex"], "routing caption lost the LFM2.5 family count")
    require("Dashes" not in texts["routing.tex"] and r"\textemdash" not in texts["routing.tex"], "routing table contains pending cells")
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
    require(tuple(levels) == DISPLAY_MODELS and all(tuple(v) == tuple(a for a, _ in ROUTING_ARMS) for v in levels.values()), "routing-level coverage differs")
    checked.append(("routing_levels.tex", len(levels) * len(ROUTING_ARMS) * 4, source, {"rank_scope": "none; arm levels within model"}))

    strong = []
    for model, columns in model_rows(texts["routing_strong.tex"]):
        strong.append(model)
        require(len(columns) == 7, model + ": strong-dose columns differ")
        for cell, key in zip(columns[1:3], ("occupancy", "first")):
            symmetric_cell(cell, *record["strong"][model][key], 2, model + "/strong/" + key)
        pending_cells(columns[3:], model + "/strong efficiency")
    require(tuple(strong) == STRONG_MODELS, "strong-dose model coverage or order differs")
    pooled = [line for line in texts["routing_strong.tex"].splitlines() if line.startswith("All four models & ")]
    require(len(pooled) == 1, "strong-dose pooled row missing or duplicated")
    columns = pooled[0].split(" & ")
    require(len(columns) == 7, "strong-dose pooled columns differ")
    for cell, (key, digits) in zip(columns[1:], (("occupancy", 2), ("first", 2), ("accuracy_vs_random", 2), ("tokens_vs_random", 1),
                                                 ("accuracy_vs_native", 2), ("random_accuracy_vs_native", 2))):
        symmetric_cell(cell, *record["strong"]["all"][key], digits, "strong/all/" + key)
    checked.append(("routing_strong.tex", len(strong) * 2 + 6, source, {"rank_scope": "none; exploratory, pooled efficiency only"}))

    prereg = json.loads(PREREG.read_text())
    seal = prereg.pop("sha256")
    payload = json.dumps(prereg, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    require(hashlib.sha256(payload.encode()).hexdigest() == seal, "efficiency-link receipt seal differs")
    link = {(c["model"], c["class"]): c["tok_reason"] for c in prereg["cells"]}
    link.update({("pooled", c["class"]): c["tok_reason"] for c in prereg["pooled"]})
    rows = {model: columns for model, columns in model_rows(texts["efficiency_link.tex"])}
    pooled = [line for line in texts["efficiency_link.tex"].splitlines() if line.startswith("Eight-model mean & ")]
    require(len(pooled) == 1 and tuple(rows) == tuple(prereg["models"]), "efficiency-link model coverage differs")
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
    for row, model in zip(body, (*DISPLAY_MODELS, "all")):
        for i, key in enumerate(plotted):
            for offset, field in enumerate(("mean", "margin")):
                saved, shown = record[model][key][field], float(row[1 + 2 * i + offset])
                require(abs(saved - shown) <= 1e-5 * max(1.0, abs(saved)), model + "/" + key + ": routing.dat differs from routing_results.json")
    return checked


def verify(paper, texts=None, inline_texts=None):
    style = re.sub(r"\s+", "", (paper / "journal/style.tex").read_text())
    require(r"\fp_compare:nTF{#1>#2}{\pm#1}{\pm#2}" in style,
            "CI printer no longer uses the larger saved margin on both sides")
    tables = paper / "journal/tables"
    data = figure_root(paper) / "data"
    if texts is None:
        texts = {path.name: path.read_text() for path in tables.glob("*.tex")}
    metadata = {}
    pending_metadata = {}

    def record(name, count, source, kind="saved_estimates", **extra):
        metadata[name] = {"ordered_cells": count, "source": source, "evidence_kind": kind, **extra}

    def rows(name):
        require(name in texts, "missing table: " + name)
        all_rows = list(model_rows(texts[name]))
        planned = [columns for model, columns in all_rows if model == PLANNED_MODEL]
        widths = {"benchmarks.tex": 7, "benchmark_details.tex": 5,
                  "readout.tex": 3, "occupancy.tex": 8,
                  "state_transitions.tex": 7, "selection.tex": 5,
                  "selection_folds.tex": 6}
        if name in widths:
            expected = 6 if name == "benchmark_details.tex" else 1
            require(len(planned) == expected, name + ": pending model row coverage differs")
            for columns in planned:
                require(len(columns) == widths[name], name + ": pending row width differs")
                pending_cells(columns[2:] if name == "benchmark_details.tex" else columns[1:], name)
            if name == "benchmark_details.tex":
                require([columns[1] for columns in planned] == list(BENCH_LABELS), name + ": pending benchmark identities differ")
            pending_metadata[name] = {"model": PLANNED_MODEL, "rows": len(planned),
                                      "cells": sum(len(columns) - (2 if name == "benchmark_details.tex" else 1) for columns in planned),
                                      "measured_results": False}
        else:
            require(not planned, name + ": planned model was added to a historical observed-only comparison")
        return [(model, columns) for model, columns in all_rows if model != PLANNED_MODEL]

    benchmark = {(r["model"], r["dataset"]): r for r in read_csv(data / "benchmark_cells.csv") if r["corpus"] == "card-v3"}
    require(len(benchmark) == 42, "benchmark source roster differs")
    benchmark_values = {b: [float(benchmark[m, b]["adjudicated_accuracy"]) for m in MODELS] for b in BENCHMARKS}
    seen = set()
    count = 0
    for model, columns in rows("benchmarks.tex"):
        require(model not in seen, "duplicate benchmark row: " + model)
        seen.add(model)
        require(len(columns) == 7, "benchmark columns differ: " + model)
        for cell, benchmark_name in zip(columns[1:], BENCHMARKS):
            source = benchmark[model, benchmark_name]
            values = tuple(100 * float(source[k]) for k in ("adjudicated_accuracy", "adjudicated_ci_low", "adjudicated_ci_high"))
            rank = rank_mark(float(source["adjudicated_accuracy"]), benchmark_values[benchmark_name])
            check_ci(cell, values, 1, model + "/" + benchmark_name, rank)
            count += 1
    require(seen == set(MODELS), "benchmark model coverage differs")
    headers = [line.split(" & ") for line in texts["benchmarks.tex"].splitlines() if line.startswith("Model & ")]
    require(len(headers) == 1 and len(headers[0]) == 7, "benchmark header shape differs")
    expected_headers = ("AIME24", "AIME25", "AMC23", "MATH-500", "Minerva", "Olympiad")
    require(tuple(cell.split("\\\\", 1)[0].strip() for cell in headers[0][1:]) == expected_headers, "benchmark header identities differ")
    require("Minerva Math" in texts["benchmarks.tex"] and "OlympiadBench" in texts["benchmarks.tex"], "benchmark caption key missing")
    require(r"\modelname{nvidia}{Nemotron-3.5-Lightning-30B-A3B}" in texts["benchmarks.tex"], "benchmark Nemotron identity must occupy one line")
    require("American Invitational" not in texts["benchmarks.tex"], "rejected expanded AIME header returned")
    record("benchmarks.tex", count, "figures/journal/data/benchmark_cells.csv", rank_scope="unrounded means within benchmark; ties share rank")

    identities = set()
    for model, columns in rows("benchmark_details.tex"):
        require(len(columns) == 5 and columns[1] in BENCH_LABELS, "benchmark-detail columns or benchmark label differ")
        b = BENCH_LABELS[columns[1]]
        require((model, b) not in identities, "duplicate detailed benchmark row")
        identities.add((model, b))
        source = benchmark[model, b]
        require(columns[2] == source["questions"] and columns[3] == source["adjudicated_scored"], model + "/" + b + ": denominator differs")
        values = tuple(100 * float(source[k]) for k in ("adjudicated_accuracy", "adjudicated_ci_low", "adjudicated_ci_high"))
        rank = rank_mark(float(source["adjudicated_accuracy"]), benchmark_values[b])
        check_ci(columns[4], values, 1, model + "/" + b, rank)
    require(identities == set(benchmark), "detailed benchmark coverage differs")
    record("benchmark_details.tex", len(identities), "figures/journal/data/benchmark_cells.csv", rank_scope="same as main benchmark table")

    readout = {r["model"]: r for r in read_csv(data / "routing_readout.csv")}
    seen = set()
    for model, columns in rows("readout.tex"):
        require(model not in seen, "duplicate readout row")
        seen.add(model)
        source = readout[model]
        require(columns[1] == f"{int(source['sentences']):,}", model + ": readout sentence count differs")
        values = tuple(-float(source[k]) for k in ("routing_delta_logloss", "routing_simultaneous_ci7_high", "routing_simultaneous_ci7_low"))
        rank = rank_mark(values[0], [-float(r["routing_delta_logloss"]) for r in readout.values()])
        check_ci(columns[2], values, 5, model + "/readout", rank)
    require(seen == set(MODELS), "readout model coverage differs")
    record("readout.tex", len(seen), "figures/journal/data/routing_readout.csv", rank_scope="positive RIG = baseline minus augmented; descriptive gains, model-specific CI correction")

    next_state = {(r["model"], r["mode"]): r for r in read_csv(data / "anticipation.csv") if r["feature"] == "composition16" and r["status"] == "CLEAN" and r["mode"] in ("next", "next-contained")}
    seen = set()
    for model, columns in rows("next_state.tex"):
        mode = {"None": "next", "32": "next-contained"}.get(columns[1])
        require((model, mode) in next_state and (model, mode) not in seen, "next-state identity differs")
        seen.add((model, mode))
        source = next_state[model, mode]
        require(columns[2] == source["questions"] and columns[3] == f"{int(source['pairs']):,}", model + "/" + mode + ": population differs")
        values = tuple(-float(source[k]) for k in ("delta_logloss", "simultaneous_ci6_high", "simultaneous_ci6_low"))
        rank = rank_mark(values[0], [-float(r["delta_logloss"]) for (m, md), r in next_state.items() if md == mode])
        check_ci(columns[4], values, 5, model + "/" + mode, rank)
    require(seen == set(next_state), "next-state coverage differs")
    record("next_state.tex", len(seen), "figures/journal/data/anticipation.csv", rank_scope="positive ARG = baseline minus augmented; gains by minimum source length, supported populations differ")

    dynamics = json.loads((data / "native_dynamics.json").read_text())
    states = ("Read", "Analyze", "Explore", "Plan", "Implement", "Verify", "Monitor")
    seen = set()
    for model, columns in rows("occupancy.tex"):
        require(model not in seen and len(columns) == 8, "occupancy row identity/columns differ")
        seen.add(model)
        for state, cell in zip(states, columns[1:]):
            source = dynamics[model]["results"][state + ":occupancy"]
            check_ci(cell, [100 * x for x in (source["estimate"], *source["interval"])], 2, model + "/" + state)
            require(source["family_support"] == (509 if model == "qwen36" else 1486), "occupancy support differs from shared note")
    require(seen == set(MODELS), "occupancy model/state coverage differs")
    require("all states use 1,486 families per model, except Qwen3.6 (509)" in texts["occupancy.tex"], "occupancy family support note missing")
    occupancy_headers = [line for line in texts["occupancy.tex"].splitlines() if line.startswith("Model & ")]
    require(occupancy_headers == [r"Model & \textrm{Re} & \textrm{An} & \textrm{Ex} & \textrm{Pl} & \textrm{Im} & \textrm{Ve} & \textrm{Mo} \\"] , "occupancy state column identities differ")
    record("occupancy.tex", 49, "figures/journal/data/native_dynamics.json", rank_scope="none; descriptive state endpoints")

    transition_keys = ("Analyze:next:Implement", "Implement:next:Verify", "Verify:next:Verify", "Verify:observed_dwell_tokens")
    seen = set()
    unavailable = 0
    for model, columns in rows("state_transitions.tex"):
        require(model not in seen and len(columns) == 7, "transition row identity/columns differ")
        seen.add(model)
        coverage = dynamics[model]["results"]["label_coverage"]
        require(columns[1] == f"{coverage['family_support']:,}", model + ": label-coverage family support differs")
        check_ci(columns[2], [100 * x for x in (coverage["estimate"], *coverage["interval"])], 2, model + "/label_coverage")
        for i, (cell, key) in enumerate(zip(columns[3:], transition_keys)):
            source = dynamics[model]["results"][key]
            scale = 100 if i < 3 else 1
            if source["interval"] is None:
                require(first_number(cell) == formatted([source["estimate"] * scale], 2)[0] and "n/e" in cell and not ci_cells(cell), model + "/" + key + ": unavailable CI was filled or mean differs")
                unavailable += 1
            else:
                check_ci(cell, [scale * x for x in (source["estimate"], *source["interval"])], 2, model + "/" + key)
    require(seen == set(MODELS), "transition model coverage differs")
    record("state_transitions.tex", len(seen) * 5, "figures/journal/data/native_dynamics.json", unavailable_intervals=unavailable, rank_scope="none; transition and dwell are descriptive")

    selection = {(r["model"], int(r["fold"]), r["arm"]): r for r in read_csv(data / "selection_coverage.csv")}
    require(len(selection) == 105, "selection source coverage differs")
    means = {m: {arm: sum(100 * float(selection[m, f, arm]["minimum_class_coverage"]) for f in range(5)) / 5 for arm in ARMS} for m in MODELS}
    seen = set()
    for model, columns in rows("selection.tex"):
        require(model not in seen and len(columns) == 5, "selection row identity/columns differ")
        seen.add(model)
        for arm, cell in zip(ARMS, columns[1:4]):
            require(first_number(cell) == formatted([means[model][arm]], 2)[0], model + "/" + arm + ": selection mean differs")
            require(scalar_rank(cell) == rank_mark(means[model][arm], means[model].values()), model + "/" + arm + ": selection rank differs")
            require("n/e" in cell and not ci_cells(cell), "selection interval falsely filled")
        delta = means[model]["state_balanced"] - means[model]["labelled_frequency"]
        require(first_number(columns[4]) == formatted([delta], 2)[0] and "n/e" in columns[4] and not ci_cells(columns[4]), model + ": selection contrast or unavailable interval differs")
    require(seen == set(MODELS), "selection model coverage differs")
    record("selection.tex", len(seen) * 4, "figures/journal/data/selection_coverage.csv", unavailable_intervals=len(seen) * 4, rank_scope="selector means within model; overlapping training-fold means have no saved CI")

    seen = set()
    for model, columns in rows("selection_folds.tex"):
        fold = int(columns[1]) - 1
        require((model, fold) not in seen and len(columns) == 6, "per-fold selection identity/columns differ")
        seen.add((model, fold))
        values = [100 * float(selection[model, fold, arm]["minimum_class_coverage"]) for arm in ARMS]
        require(columns[2] == selection[model, fold, "state_balanced"]["test_families"], "per-fold family count differs")
        for value, cell in zip(values, columns[3:]):
            require(first_number(cell) == formatted([value], 3)[0] and scalar_rank(cell) == rank_mark(value, values), model + "/fold" + str(fold + 1) + ": value or rank differs")
    require(seen == {(m, f) for m in MODELS for f in range(5)}, "per-fold selection coverage differs")
    record("selection_folds.tex", len(seen) * 3, "figures/journal/data/selection_coverage.csv", "fold_diagnostics", rank_scope="within model and fold; no CIs invented")

    historical = {}
    for line in (paper / "guiding_weighting_table.tex").read_text().splitlines():
        if line.startswith(("Qwen ", "GPT-OSS ")):
            columns = line.split(" & ")
            model = "qwen35" if columns[0].startswith("Qwen") else "gpt"
            policy = columns[0].split(" ", 1)[1].replace("paper", "strong")
            values = [float(x) for x in re.findall(r"[+-]?\d+\.\d+", " & ".join(columns[1:]))]
            historical[model, policy] = (values[:3], values[3:])
    model = None
    seen = set()
    for line in texts["interventions.tex"].splitlines():
        match = MODEL.search(line)
        if match:
            model = LABELS[model_label(match)]
        elif model and line.startswith(("Fixed", "Margin", "Strong")):
            columns = line.split(" & ")
            policy = columns[0].lower().replace("$", "")
            identity = (model, policy)
            require(identity in historical and identity not in seen, "historical policy identity differs")
            seen.add(identity)
            for i in (0, 1):
                values = historical[identity][i]
                rank = rank_mark(values[0], [v[i][0] for (m, p), v in historical.items() if m == model])
                check_ci(columns[i + 1], values, 2, str(identity) + "/weighting" + str(i), rank)
    require(seen == set(historical), "historical policy coverage differs")
    record("interventions.tex", len(seen) * 2, "guiding_weighting_table.tex", rank_scope="declared-final supplied means within model and weighting; execution profiles differ")

    for name, count, source, extra in verify_routing(paper, texts):
        record(name, count, source, **extra)
    require(set(texts) == set(metadata), "unregistered table sources: " + str(sorted(set(texts) - set(metadata))))
    for name, text in texts.items():
        require(not re.search(r"(?<!\\)\$?\[[+-]?\d", text), name + ": bracketed numeric interval returned")
        require("Fold range" not in text, name + ": removed extrema column returned")
        require(not re.search(r"\\(?:bestci|secondci|ci)\{", text), name + ": rounded-endpoint CI macro returned")
        require("\\begin{tabular*}" not in text and "\\extracolsep{\\fill}" not in text, name + ": rejected forced full-width table returned")
        verify_table_width_policy(text, name)
        verify_caption_only_notes(text, name)
    inline = verify_inline(paper, inline_texts)
    prose = verify_numeric_prose(paper, inline_texts)
    return {"numeric_prose": prose, "pending_model": {"identity": "LiquidAI/LFM2.5-24B-A2B", "tables": pending_metadata, "scientific_enrollment_changed": False},
            "uncertainty_contract": "original estimate-to-endpoint deviations retained; printed plus/minus half-width is their maximum, a symmetric envelope of the saved nominal 95% interval",
            "historical_table_ci_precision": "35 prefix-probe/global-steering cells retain supplied rounded-source precision",
"status": "pass", "tables": metadata,
            "inline_tables": inline,
            "ordered_populated_cells": inline["prefix_probes"]["ordered_cells"] + inline["model_roster"]["ordered_fixture_cells"] + sum(v["ordered_cells"] for v in metadata.values()),
            "real_and_diagnostic_cells": inline["prefix_probes"]["ordered_cells"] + inline["model_roster"]["ordered_fixture_cells"] + sum(v["ordered_cells"] for v in metadata.values()),
            "new_fitting": False, "new_resampling": False, "new_inference": False}


def self_test(paper):
    texts = {path.name: path.read_text() for path in (paper / "journal/tables").glob("*.tex")}
    checks = []
    lines = texts["benchmarks.tex"].splitlines()
    eligible = [i for i, line in enumerate(lines) if line.startswith("\\modelname")]
    first, second = eligible[:2]
    wrong = lines.copy()
    a, b = wrong[first].split(" & "), wrong[second].split(" & ")
    wrong[first] = a[0] + " & " + " & ".join(b[1:])
    wrong[second] = b[0] + " & " + " & ".join(a[1:])
    first_ci = CI.search(texts["benchmarks.tex"])
    require(first_ci is not None, "no benchmark CI for the regression check")
    changed_bound = texts["benchmarks.tex"][:first_ci.start(3)] + str(decimal(first_ci[3]) + 1) + texts["benchmarks.tex"][first_ci.end(3):]
    swapped_headers = texts["benchmarks.tex"].replace("AIME24", "AIMETEMP", 1).replace("AIME25", "AIME24", 1).replace("AIMETEMP", "AIME25", 1)
    unavailable = re.search(r"\$(\d+\.\d+)_\{\\mathrm\{n/e\}\}\$", texts["state_transitions.tex"])
    require(unavailable is not None, "no unavailable transition interval for the regression check")
    filled_gap = texts["state_transitions.tex"][:unavailable.start()] + "\\cid{" + unavailable[1] + "}{1.00}{1.00}" + texts["state_transitions.tex"][unavailable.end():]
    # New presentation transforms must preserve the source interval shape.
    midpoint_width = (decimal(first_ci[4]) + decimal(first_ci[3])) / 2
    symmetry_replacement = "\\" + first_ci[1] + "{" + first_ci[2] + "}{" + str(midpoint_width) + "}{" + str(midpoint_width) + "}"
    symmetrized = texts["benchmarks.tex"][:first_ci.start()] + symmetry_replacement + texts["benchmarks.tex"][first_ci.end():]
    gain = CI.search(texts["readout.tex"])
    require(gain is not None, "no RIG cell for the transform regression check")
    wrong_gain_interval = "\\" + gain[1] + "{" + gain[2] + "}{" + gain[4] + "}{" + gain[3] + "}"
    inverted_gain = texts["readout.tex"][:gain.start()] + wrong_gain_interval + texts["readout.tex"][gain.end():]
    # A symmetric original dynamics interval must not acquire asymmetry from
    # rounding its estimate and endpoints independently (the R1 review failure).
    symmetric_display = r"\cid{17.27}{4.27}{4.27}"
    require(symmetric_display in texts["state_transitions.tex"], "saved symmetric transition example missing")
    require(displayed_ci_shape(symmetric_display) == "symmetric_pm", "symmetric CI display did not select plus/minus")
    require(displayed_ci_shape(first_ci[0]) == "symmetric_pm", "asymmetric source CI did not select the enclosing plus/minus display")
    false_asymmetry = texts["state_transitions.tex"].replace(symmetric_display, r"\cid{17.27}{4.26}{4.27}", 1)
    rounded_endpoint_error = texts["benchmarks.tex"].replace(r"\cid{76.0}{11.4}{10.0}", r"\cid{76.0}{11.3}{10.0}", 1)
    mutations = {
        "model_row_swap_rejected": {**texts, "benchmarks.tex": "\n".join(wrong)},
        "incomplete_model_identity_rejected": {**texts, "benchmarks.tex": texts["benchmarks.tex"].replace("Gemma-4-26B-A4B", "Gemma", 1)},
        "table_enlargement_rejected": {**texts, "benchmarks.tex": texts["benchmarks.tex"].replace("max width=\\linewidth", "width=\\linewidth", 1)},
        "benchmark_header_swap_rejected": {**texts, "benchmarks.tex": swapped_headers},
        "changed_ci_rejected": {**texts, "benchmarks.tex": changed_bound},
        "silently_symmetrized_ci_rejected": {**texts, "benchmarks.tex": symmetrized},
        "symmetric_original_false_asymmetry_rejected": {**texts, "state_transitions.tex": false_asymmetry},
        "errors_from_rounded_endpoints_rejected": {**texts, "benchmarks.tex": rounded_endpoint_error},
        "swapped_signed_deviations_rejected": {**texts, "readout.tex": inverted_gain},
        "transposed_state_header_swap_rejected": {**texts, "occupancy.tex": texts["occupancy.tex"].replace(r"Model & \textrm{Re} & \textrm{An}", r"Model & \textrm{An} & \textrm{Re}", 1)},
        "forced_table_stretch_rejected": {**texts, "selection.tex": texts["selection.tex"].replace("\\begin{tabular}", "\\begin{tabular*}", 1)},
        "wrong_best_mark_rejected": {**texts, "benchmarks.tex": texts["benchmarks.tex"].replace("\\bestcid", "\\cid", 1)},
        "invented_missing_ci_rejected": {**texts, "state_transitions.tex": filled_gap},
    }
    routing_first = CI.search(texts["routing.tex"])
    require(routing_first is not None, "no routing CI for the regression check")
    routing_changed = texts["routing.tex"][:routing_first.start(2)] + str(decimal(routing_first[2]) + Decimal("0.01")) + texts["routing.tex"][routing_first.end(2):]
    routing_lines = texts["routing.tex"].splitlines()
    routing_rows = [i for i, line in enumerate(routing_lines) if line.startswith("\\modelname")]
    routing_lines[routing_rows[0]], routing_lines[routing_rows[1]] = routing_lines[routing_rows[1]], routing_lines[routing_rows[0]]
    routing_swapped = "\n".join(routing_lines)
    level_first = CI.search(texts["routing_levels.tex"])
    require(level_first is not None, "no routing-level CI for the regression check")
    levels_changed = texts["routing_levels.tex"][:level_first.start(3)] + str(decimal(level_first[3]) + Decimal("0.01")) + texts["routing_levels.tex"][level_first.end(3):]
    pending_line = next(line for line in texts["benchmarks.tex"].splitlines() if line.startswith(r"\modelname{liquid}"))
    filled_pending = pending_line.replace(PENDING, r"\cid{80.0}{1.0}{1.0}", 1)
    mutations.update({
        "invented_planned_benchmark_rejected": {**texts, "benchmarks.tex": texts["benchmarks.tex"].replace(pending_line, filled_pending, 1)},
        "missing_planned_row_rejected": {**texts, "benchmarks.tex": texts["benchmarks.tex"].replace(pending_line, "", 1)},
        "table_sublegend_rejected": {**texts, "readout.tex": texts["readout.tex"].replace(r"\end{adjustbox}", r"\end{adjustbox}\par Small sub-legend.", 1)},
        "wrapped_nemotron_benchmark_rejected": {**texts, "benchmarks.tex": texts["benchmarks.tex"].replace(r"\modelname{nvidia}{Nemotron-3.5-Lightning-30B-A3B}", r"\modelname{nvidia}{\shortstack[l]{Nemotron-3.5-Lightning\\30B-A3B}}", 1)},
        "changed_routing_estimate_rejected": {**texts, "routing.tex": routing_changed},
        "routing_row_swap_rejected": {**texts, "routing.tex": routing_swapped},
        "changed_routing_level_rejected": {**texts, "routing_levels.tex": levels_changed},
    })
    for name, candidate in mutations.items():
        require(candidate != texts, "self-test mutation did not apply: " + name)
        try:
            verify(paper, candidate)
        except TableMismatch:
            checks.append(name)
        else:
            raise TableMismatch("invalid table passed self-test: " + name)
    appendix = (paper / "journal/appendices.tex").read_text()
    prefix = inline_table(appendix, "tab:prefix-probes")
    baseline_row = next(line for line in prefix.splitlines() if line.startswith("\\modelname"))
    baseline_cell = baseline_row.split(" & ")[2]
    baseline_ci = CI.search(baseline_cell)
    require(baseline_ci is not None, "no saved baseline CI for the regression check")
    altered_cell = baseline_cell[:baseline_ci.start(3)] + str(decimal(baseline_ci[3]) + Decimal("0.001")) + baseline_cell[baseline_ci.end(3):]
    altered_appendix = appendix.replace(baseline_cell, altered_cell, 1)
    try:
        verify(paper, texts, {"appendices": altered_appendix})
    except TableMismatch:
        checks.append("changed_prefix_baseline_bound_rejected")
    else:
        raise TableMismatch("changed prefix-position baseline interval passed self-test")
    narrative = (paper / "journal/results.tex").read_text()
    changed_narrative = narrative.replace(r"\cid{17.27}{4.27}{4.27}", r"\cid{17.27}{4.26}{4.27}", 1)
    require(changed_narrative != narrative, "narrative CI corruption did not apply")
    try:
        verify(paper, texts, {"results": changed_narrative})
    except TableMismatch:
        checks.append("narrative_false_asymmetry_rejected")
    else:
        raise TableMismatch("narrative interval shape corruption passed self-test")
    setup = (paper / "journal/setup.tex").read_text()
    roster = inline_table(setup, "tab:models")
    fixture_line = next(line for line in roster.splitlines() if line.startswith("\\modelname"))
    fixture_columns = fixture_line.split(" & ")
    changed_columns = fixture_columns.copy()
    changed_columns[4] = f"{int(fixture_columns[4].replace(',', '')) + 1:,}"
    changed_fixture_setup = setup.replace(fixture_line, " & ".join(changed_columns), 1)
    try:
        verify(paper, texts, {"setup": changed_fixture_setup})
    except TableMismatch:
        checks.append("changed_merged_fixture_context_rejected")
    else:
        raise TableMismatch("changed merged engineering-fixture context passed self-test")
    planned_fixture = next(line for line in roster.splitlines() if line.startswith(r"\modelname{liquid}"))
    filled_fixture = planned_fixture.replace(" & " + PENDING, " & 131,072", 1)
    try:
        verify(paper, texts, {"setup": setup.replace(planned_fixture, filled_fixture, 1)})
    except TableMismatch:
        checks.append("advertised_context_as_measured_fixture_rejected")
    else:
        raise TableMismatch("advertised context misrepresented as measured fixture passed self-test")
    sys.dont_write_bytecode = True
    module_path = Path(__file__).with_name("export_cl_exhibits.py")
    spec = importlib.util.spec_from_file_location("cl_export_guard_test", module_path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    exporter.PAPER = paper
    exporter.ROOT = figure_root(paper)
    exporter.DATA = exporter.ROOT / "data"
    exporter.TABLES = paper / "journal/tables"
    writes = []

    def capture_write(path, content, *args, **kwargs):
        writes.append(path)
        return len(content)

    with patch.object(Path, "write_text", capture_write), patch.object(Path, "write_bytes", capture_write), patch.object(Path, "mkdir"), contextlib.redirect_stdout(io.StringIO()):
        exporter.export_main_data()
    require(writes and all(path.parent == exporter.DATA for path in writes), "default export tried to overwrite tables, manifest or unrelated files")
    checks.append("default_export_preserves_tables_and_manifest")
    return {"status": "pass", "checks": checks, "filesystem_writes": False}


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
