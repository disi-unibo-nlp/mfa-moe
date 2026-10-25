"""Write the data behind the paper figures from the bundled copies of completed estimates.

Nothing is fitted, resampled or run here. The bundled copies live in figures/journal/data (readout, selection and
next-state CSVs, native dynamics, benchmark cells). A plain run turns them into the plotting files readout.dat,
selection.dat, selection_summary.json, next_state.dat, occupancy.dat and state_transitions_*.dat, and rewrites the model
rows of the tables readout, next_state, occupancy and state_transitions from the same copies. Their captions, headers and
every other table stay as the manuscript has them; `verify_cl_tables` checks all tables against the same copies and that a
rerun reproduces the manuscript byte for byte. Before writing it also checks that the native-memory fixtures add up over workers and that every model has 4,647
benchmark attempts. The study is the one store.STUDY names (STUDY_DATA_ROOT).

  python -m paper.export_cl_exhibits                 plotting files from the bundled copies
  python -m paper.export_cl_exhibits --capture-lfm   first add the LFM2.5 rows from the study
  python -m paper.export_cl_exhibits --capture-next-state   first copy the eight next-state estimates from the study
  python -m paper.export_cl_exhibits --capture-qwen36-panel   first replace Qwen3.6's native-dynamics panel by the
                                                     one the study rebuilt on all its question families
  python -m paper.export_cl_exhibits --capture-pairs VARIANT   first replace every native-dynamics panel by the one
                                                     completed with the next-sentence labels of VARIANT
  python -m paper.export_cl_exhibits --capture-existing   first copy the seven-model estimates from the saved reports
"""
import argparse
import csv
import json
import shutil

from paper.cl_models import MODELS, SEVEN_MODELS
from paper.cl_paths import DATA, PAPER, REPO, SAVED_REPORTS
from paper.export_routing_results import PAPER_CLASSES
from store import BENCHMARKS, FAST, STUDY, file_sha, verified


def read_csv(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def source(path):
    """How SOURCES.json names a file: its path below the folder that holds report/, and its hash."""
    root = REPO if path.is_relative_to(REPO) else STUDY.parents[1]
    return {"path": str(path.relative_to(root)), "sha256": file_sha(path)}


QWEN36_PANEL = "transition-pairs/qwen36/NATIVE_DYNAMICS_SUMMARY.original-sample.json"


def study_panel(path):
    """A native-dynamics panel the study published: its path and its content, after the receipt hash is checked."""
    return path, verified(path)


def qwen36_panel():
    """Qwen3.6's panel on all 1,486 question families; the capture's saved sentence table covers 509 of them."""
    path, panel = study_panel(STUDY / QWEN36_PANEL)
    families = {r["family_support"] for k, r in panel["results"].items() if k.endswith(":occupancy")}
    if panel["panel"] != "native-dynamics:qwen36" or families != {1486}:
        raise ValueError("Qwen3.6 panel does not cover 1,486 families")
    return path, panel


def capture():
    DATA.mkdir(parents=True, exist_ok=True)
    sources = {}
    saved, review = SAVED_REPORTS / "generated/best_results_20261009", REPO / "report/model-analysis-review-20261008"
    for origin in [saved / name for name in ("routing_readout.csv", "selection_coverage.csv")] + [
            review / name for name in ("benchmark_cells.csv", "expert_usage_correctness_mean_all.csv")]:
        shutil.copyfile(origin, DATA / origin.name)
        sources[origin.name] = source(origin)
    imported = REPO / "report/research-campaign-v3-20261014-r6/EVIDENCE_IMPORT.json"
    bindings = {r["path"]: r["sha256"] for r in json.loads(imported.read_text())["references"]}
    summaries = {}
    for model in SEVEN_MODELS:
        if model == "qwen36":
            path, summaries[model] = qwen36_panel()
        else:
            path = REPO / "report/research-campaign-v3-20261014-r5-r2" / model / "NATIVE_DYNAMICS_SUMMARY.json"
            if bindings[str(path)] != file_sha(path):
                raise ValueError("imported dynamics changed: " + model)
            summaries[model] = json.loads(path.read_text())
        sources[model + ":dynamics"] = source(path)
    (DATA / "native_dynamics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fixture = REPO / "report/research-campaign-v3-20261014-r6/PARTIAL_VALIDATION_EVIDENCE.json"
    records = json.loads(fixture.read_text())["models"]
    if {r["model"] for r in records} != set(SEVEN_MODELS):
        raise ValueError("native fixture roster changed")
    (DATA / "native_fixtures.json").write_text(json.dumps(records, indent=2) + "\n")
    sources["native_fixtures"] = source(fixture)
    (DATA / "SOURCES.json").write_text(json.dumps({"sources": sources, "import_binding": source(imported),
                                                   "new_measurement": False}, indent=2) + "\n")


NEXT_STATE_LABEL_GAP = .006  # axis units between the end of a whisker and its value label
NEXT_STATE_CLEAR = .024     # axis units from the top of one label to the top of a label that must clear it
LFM_ROOT = STUDY / "lfm25"
LFM_BULK = FAST / STUDY.name / "lfm25"


def lfm_benchmark_cells():
    """card-v3 cells for LFM2.5 with the source adjudication rule: math_verify True stays correct, a capped
    answer is incorrect, and a finished rejected answer takes its J1 answer-equivalence verdict. Intervals
    use the same 1,000-replicate question bootstrap (seed 0) as the seven-model cells."""
    import numpy as np
    items = {}
    for line in (LFM_BULK / "j1/new_lfm25_items.jsonl").open():
        r = json.loads(line)
        items[r["item_id"]] = (r["dataset"], r["problem_id"], r["sample_id"])
    verdicts = {}
    for line in (LFM_BULK / "j1/new_lfm25_verdicts.jsonl").open():
        r = json.loads(line)
        verdicts[items[r["item_id"]]] = r["verdict"]
    if len(verdicts) != len(items):
        raise ValueError("J1 verdicts incomplete for LFM2.5")
    generation = LFM_BULK / "generation/LiquidAI--LFM2.5-24B-A2B"
    cells, sources = [], {}
    for dataset in BENCHMARKS:
        path = generation / dataset / "traces.jsonl"
        sources["lfm25:generation:" + dataset] = {"path": str(path), "sha256": file_sha(path)}
        rows = []
        for line in path.open():
            r = json.loads(line)
            meta = r["metadata"]
            termination = meta.get("termination") or meta.get("finish_reason")
            strict = r.get("is_correct")
            if termination == "length" and not strict:
                adjudicated = 0.
            elif strict:
                adjudicated = 1.
            else:
                verdict = verdicts.get((dataset, r["problem_id"], r.get("sample_id")))
                adjudicated = {"EQUIVALENT": 1., "NOT_EQUIVALENT": 0.}.get(verdict, float("nan"))
            tokens = len((meta.get("token_replay") or {}).get("completion_token_ids") or [])
            rows.append({"question": dataset + "|" + (r.get("source_problem_id") or r["problem_id"]),
                         "strict": float(bool(strict)), "adjudicated": adjudicated, "capped": termination == "length",
                         "tokens": tokens})
        adjudicated = np.array([r["adjudicated"] for r in rows])
        scored = ~np.isnan(adjudicated)
        questions = sorted({r["question"] for r in rows})
        index = {q: i for i, q in enumerate(questions)}
        group = np.array([index[r["question"]] for r in rows])
        sums = np.bincount(group[scored], weights=adjudicated[scored], minlength=len(questions))
        counts = np.bincount(group[scored], minlength=len(questions)).astype(float)
        draw = np.random.default_rng(0).integers(0, len(questions), size=(1000, len(questions)))
        low, high = np.quantile(sums[draw].sum(1) / counts[draw].sum(1), [.025, .975])
        tokens = np.array([r["tokens"] for r in rows])
        strict_correct = sum(r["strict"] for r in rows)
        cells.append({"model": "lfm25", "dataset": dataset, "corpus": "card-v3", "attempts": len(rows),
            "questions": len(questions), "capped": int(sum(r["capped"] for r in rows)),
            "completion_tokens_sum": int(tokens.sum()), "completion_tokens_median": float(np.median(tokens)),
            "completion_tokens_mean": float(tokens.mean()), "strict_scored": len(rows),
            "strict_correct": int(strict_correct), "strict_unknown": 0,
            "strict_accuracy": strict_correct / len(rows),
            "adjudicated_scored": int(scored.sum()), "adjudicated_correct": int(adjudicated[scored].sum()),
            "adjudicated_unknown": int((~scored).sum()), "adjudicated_accuracy": float(adjudicated[scored].mean()),
            "adjudicated_ci_low": float(low), "adjudicated_ci_high": float(high)})
    path = LFM_BULK / "j1/new_lfm25_verdicts.jsonl"
    sources["lfm25:j1_verdicts"] = {"path": str(path), "sha256": file_sha(path)}
    return cells, sources


def replace_lfm_rows(path, rows):
    """Drop any earlier lfm25 rows, append the given ones, and keep the file's own line ending."""
    text = path.read_bytes().decode()
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = [l for l in text.splitlines() if not l.startswith("lfm25,")]
    path.write_bytes((newline.join(lines + rows) + newline).encode())


def capture_lfm_rows():
    """Append the LFM2.5 rows to the bundled copies; the seven-model rows stay byte-identical."""
    sources = json.loads((DATA / "SOURCES.json").read_text())
    readout = LFM_ROOT / "READOUT_RESULTS.json"
    result = verified(readout)["models"]["lfm25"]["judge_context"]
    if result["status"] != "CLEAN" or (result["n_questions"], result["n_families"]) != (522, 509):
        raise ValueError("LFM2.5 readout scope differs")
    row = ["lfm25", "CLEAN", "522", "509", str(result["n_sentences"]), repr(result["routing"]["P"]),
           *map(repr, result["routing"]["simultaneous_ci7"]), repr(result["routing_excess_over_lexical_null"]["P"]),
           *map(repr, result["routing_excess_over_lexical_null"]["simultaneous_ci7"])]
    selection = verified(LFM_ROOT / "SELECTION_COVERAGE.json")
    if selection["keep"] != 48 or selection["experts"] != 64:
        raise ValueError("LFM2.5 retention budget differs")
    dynamics = LFM_ROOT / "NATIVE_DYNAMICS_SUMMARY.json"
    summaries = json.loads((DATA / "native_dynamics.json").read_text())
    summaries["lfm25"] = json.loads(dynamics.read_text())
    cells, generation_sources = lfm_benchmark_cells()
    replace_lfm_rows(DATA / "routing_readout.csv", [",".join(row)])
    held_rows = []
    for fold in selection["folds"]:
        for arm in ("labelled_frequency", "random", "state_balanced"):
            held = fold["held_out"][arm]
            held_rows.append(",".join(["lfm25", str(fold["fold"]), arm, repr(float(held["minimum_class_coverage"])),
                                       str(int(fold["test_families"])), str(held["minimum_complete"] in (True, "true"))]))
    replace_lfm_rows(DATA / "selection_coverage.csv", held_rows)
    (DATA / "native_dynamics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (DATA / "benchmark_cells.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, kept = reader.fieldnames, [r for r in reader if r["model"] != "lfm25"]
    with (DATA / "benchmark_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept + [{k: c.get(k, "") for k in fields} for c in cells])
    for name, path in (("lfm25:readout", readout), ("lfm25:selection", LFM_ROOT / "SELECTION_COVERAGE.json"),
                       ("lfm25:dynamics", dynamics)):
        sources["sources"][name] = source(path)
    sources["sources"].update(generation_sources)
    (DATA / "SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n")


def capture_qwen36_panel():
    """Replace Qwen3.6's panel in the bundled native dynamics by the study's, which covers all its families."""
    path, panel = qwen36_panel()
    summaries = json.loads((DATA / "native_dynamics.json").read_text())
    summaries["qwen36"] = panel
    (DATA / "native_dynamics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    sources = json.loads((DATA / "SOURCES.json").read_text())
    sources["sources"]["qwen36:dynamics"] = source(path)
    (DATA / "SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n")


def capture_pairs(variant):
    """Replace every model's panel in the bundled native dynamics by the one completed with the next-sentence labels.

    The labels of `variant` (the folder name of a source-class rule, `routing.next_sentence_labels variant`) enter only
    the next-state estimates; every panel is the study's own, so its simultaneous critical value covers all its columns.
    """
    joined = verified(STUDY / f"driver/TRANSITION_PAIRS.{variant}.json")
    if set(joined["models"]) != set(MODELS):
        raise ValueError("transition-pair roster differs")
    summaries = json.loads((DATA / "native_dynamics.json").read_text())
    sources = json.loads((DATA / "SOURCES.json").read_text())
    for model in MODELS:
        path, panel = study_panel(STUDY / "transition-pairs" / model / variant / "NATIVE_DYNAMICS_SUMMARY.json")
        if panel["panel"] != "native-dynamics:" + model or panel["sha256"] != joined["models"][model]["panel_sha256"]:
            raise ValueError("completed panel differs from the joined receipt: " + model)
        if panel["results"]["label_coverage"]["family_support"] != 1486:
            raise ValueError("completed panel does not cover 1,486 families: " + model)
        summaries[model] = panel
        sources["sources"][model + ":dynamics"] = source(path)
    sources["sources"]["transition_pairs"] = source(STUDY / f"driver/TRANSITION_PAIRS.{variant}.json")
    (DATA / "native_dynamics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (DATA / "SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n")


NEXT_STATE_DEFINITION = ("gain = held-out cross-entropy of the smaller model minus the larger one (nats per adjacent pair, "
                         "questions weighted equally); routing_gain.P is routing minus text, so negative values favour routing")
NEXT_STATE_COLUMNS = ("model", "status", "questions", "families", "pairs", "labelled_sentences", "source_tokens",
                      "markov_loss", "text_loss", "routing_loss")
INTERVALS = ("ci95_low", "ci95_high", "simultaneous_ci7_low", "simultaneous_ci7_high")


def next_state_row(model, result):
    """One model's receipt as a CSV row: sizes, the three cross-entropies, and the two gains with their intervals."""
    analysis = result["readout_analysis"]
    if result["status"] != "CLEAN" or analysis["status"] != "CLEAN" or result["definition"] != NEXT_STATE_DEFINITION:
        raise ValueError("next-state receipt differs: " + model)
    loss = result["cross_entropy"]
    row = {"model": model, "status": result["status"], "questions": analysis["n_questions"],
           "families": analysis["n_families"], "pairs": result["pairs"], "labelled_sentences": result["labelled_sentences"],
           "source_tokens": result["source_tokens"], "markov_loss": loss["markov"], "text_loss": loss["text"],
           "routing_loss": loss["routing"]}
    for prefix, gain in (("", result["routing_gain"]), ("null_", result["routing_excess_over_null"])):
        low, high = gain["ci95"]
        simultaneous = gain["simultaneous_ci7"]
        if not simultaneous[0] <= low <= gain["P"] <= high <= simultaneous[1]:
            raise ValueError("next-state interval mismatch: " + model)
        row[prefix + "delta_logloss"] = gain["P"]
        row.update(zip((prefix + name for name in INTERVALS), (low, high, *simultaneous)))
    if abs(row["delta_logloss"] - (loss["routing"] - loss["text"])) > 1e-9:
        raise ValueError("next-state gain is not routing minus text: " + model)
    return row


def capture_next_state():
    """Copy the eight next-state estimates of the study into next_state.csv and name the receipt in SOURCES.json."""
    path = STUDY / "driver/NEXT_STATE.json"
    receipt = verified(path)
    if set(receipt["models"]) != set(MODELS) or set(receipt["results"]) != set(MODELS):
        raise ValueError("next-state roster differs")
    rows = [next_state_row(model, receipt["results"][model]) for model in MODELS]
    with (DATA / "next_state.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sources = json.loads((DATA / "SOURCES.json").read_text())
    sources["sources"].pop("anticipation.csv", None)
    sources["sources"]["next_state"] = source(path)
    (DATA / "SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n")


def dat(name, headings, rows):
    (DATA / name).write_text(" ".join(headings) + "\n" + "\n".join(
        " ".join(format(v, ".16g") if isinstance(v, float) else str(v) for v in row)
        for row in rows) + "\n")


def readout_data():
    """Present-state cross-entropy change relative to the text baseline, and its change over the lexical null."""
    routing = {r["model"]: r for r in read_csv("routing_readout.csv")}
    if set(routing) != set(MODELS):
        raise ValueError("readout roster mismatch")
    coords = []
    for i, m in enumerate(MODELS):
        r = routing[m]
        if r["status"] != "CLEAN" or (r["questions"], r["families"]) != ("522", "509"):
            raise ValueError("readout scope mismatch")
        values = []
        for prefix in ("routing", "routing_excess_over_lexical_null"):
            mean = float(r[prefix + "_delta_logloss"])
            low = float(r[prefix + "_simultaneous_ci7_low"])
            high = float(r[prefix + "_simultaneous_ci7_high"])
            if not low <= mean <= high:
                raise ValueError("readout interval mismatch")
            values.extend((mean, mean - low, high - mean))
        coords.append((i + 1, *values))
    dat("readout.dat", ("x", "text", "textminus", "textplus", "null", "nullminus", "nullplus"), coords)


def selection_data():
    """Minimum held-out class coverage of the three selectors, per model and fold."""
    lookup = {(r["model"], int(r["fold"]), r["arm"]): r for r in read_csv("selection_coverage.csv")}
    if len(lookup) != 15 * len(MODELS):
        raise ValueError("selection cells missing")
    summary, coords = [], []
    for i, model in enumerate(MODELS):
        if sum(int(lookup[model, f, "state_balanced"]["test_families"]) for f in range(5)) != 1486:
            raise ValueError("family population mismatch")
        vals = {}
        for arm in ("state_balanced", "labelled_frequency", "random"):
            records = [lookup[model, f, arm] for f in range(5)]
            if any(r["minimum_complete"] != "True" for r in records):
                raise ValueError("missing class support")
            vals[arm] = [100 * float(r["minimum_class_coverage"]) for r in records]
        delta = [a - b for a, b in zip(vals["state_balanced"], vals["labelled_frequency"])]
        mu = sum(delta) / 5
        coords.append((i + 1, mu, mu - min(delta), max(delta) - mu, *delta))
        summary.append({"model": model, "means_percent": {a: sum(v) / len(v) for a, v in vals.items()},
                        "paired_fold_differences_pp": delta})
    dat("selection.dat", ("x", "mean", "minus", "plus", "f0", "f1", "f2", "f3", "f4"), coords)
    (DATA / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def next_state_data():
    """Next-state cross-entropy change of routing over text, and over the lexical null, with the label of each bar.

    The columns keep the loss sign of the receipts (routing minus baseline, negative when routing helps), and the
    bars hang from zero. A label sits below the end of its whisker; one that would touch the label of a neighbouring
    column hangs lower on a gray leader. Label text, height and leader length are written here, not typed.
    """
    rows = {r["model"]: r for r in read_csv("next_state.csv")}
    if set(rows) != set(MODELS):
        raise ValueError("next-state roster mismatch")
    columns = {}
    for prefix in ("", "null_"):
        values = {}
        for model in MODELS:
            r = rows[model]
            if r["status"] != "CLEAN":
                raise ValueError("next-state status mismatch")
            mean = float(r[prefix + "delta_logloss"])
            low, high = float(r[prefix + "simultaneous_ci7_low"]), float(r[prefix + "simultaneous_ci7_high"])
            if not low <= mean <= high:
                raise ValueError("next-state interval mismatch")
            values[model] = (mean, mean - low, high - mean, mean - max(mean - low, high - mean))
        depths = hanging_labels({m: -v[3] for m, v in values.items()}, {m: i for i, m in enumerate(MODELS)})
        columns[prefix] = [(*values[m][:3], values[m][3], -depths[m], depths[m] + values[m][3] if depths[m] + values[m][3] > NEXT_STATE_LABEL_GAP + 1e-9 else 0.,
                            signed_label(values[m][0])) for m in MODELS]
    dat("next_state.dat", ("x", "gain", "gainminus", "gainplus", "gainend", "gainlabely", "gainlead", "gainlabel",
                           "null", "nullminus", "nullplus", "nullend", "nulllabely", "nulllead", "nulllabel"),
        [(i + 1, *columns[""][i], *columns["null_"][i]) for i in range(len(MODELS))])


def hanging_labels(ends, column):
    """Depth of the label under each whisker end (a depth below zero): the shallowest first, each at the gap below its
    end unless a neighbouring column's label is closer than NEXT_STATE_CLEAR, which pushes it below that label."""
    placed = {}
    for key in sorted(ends, key=ends.get):
        depth = ends[key] + NEXT_STATE_LABEL_GAP
        for other, other_depth in placed.items():
            if abs(column[other] - column[key]) <= 1 and depth < other_depth + NEXT_STATE_CLEAR - 1e-12:
                depth = other_depth + NEXT_STATE_CLEAR
        placed[key] = depth
    return placed


def signed_label(value):
    """A value label with three decimals and a typeset minus sign."""
    return ("$-$" if value < 0 else "") + format(abs(value), ".3f")


# Panels of the transition summary: column prefix, panel of native_dynamics.json, unit scale and the top of the axis.
SUMMARY_PANELS = (("ai", "Analyze:next:Implement", 100, 52), ("iv", "Implement:next:Verify", 100, 52),
                  ("vv", "Verify:next:Verify", 100, 52), ("vd", "Verify:observed_dwell_tokens", 1, 28))
SUMMARY_LABEL_GAP = .04  # of the axis height between the top of a whisker and its value label
DAGGER = r"$^{\dagger}$"


def summary_data(dynamics):
    """Transition and dwell means with their margins, value labels and a flag for each interval that is unavailable.

    A model whose interval is unavailable keeps its mean, gets no whisker and a dagger on its label.
    """
    coords = []
    for i, model in enumerate(MODELS):
        row = [i + 1]
        for name, key, scale, top in SUMMARY_PANELS:
            cell = dynamics[model]["results"][key]
            mean = scale * cell["estimate"]
            missing = cell["interval"] is None
            low, high = (float("nan"),) * 2 if missing else (scale * x for x in cell["interval"])
            margin = 0. if missing else max(mean - low, high - mean)
            row.extend((mean, mean - low, high - mean, mean + margin + SUMMARY_LABEL_GAP * top,
                        format(mean, ".1f") + (DAGGER if missing else ""), int(missing)))
        coords.append(tuple(row))
    headings = ["x"] + [name + suffix for name, *_ in SUMMARY_PANELS
                        for suffix in ("", "minus", "plus", "labely", "label", "missing")]
    dat("state_transitions_summary.dat", headings, coords)


def dynamics_data():
    """State-transition matrices and state occupancies of the native dynamics."""
    dynamics = json.loads((DATA / "native_dynamics.json").read_text())
    # Matrix coordinates are copied by named state, never relabeled in place.
    for model in MODELS:
        matrix = []
        for y, state in enumerate(PAPER_CLASSES, 1):
            row = [dynamics[model]["results"][state + ":next:" + destination] for destination in PAPER_CLASSES]
            # An undefined estimate stays missing (nan), never zero, and is not summed.
            defined = [r["estimate"] for r in row if r["estimate"] is not None]
            if len(defined) == len(PAPER_CLASSES) and abs(sum(defined) - 1) > 1e-10:
                raise ValueError("transition row does not sum to one")
            if sum(defined) > 1 + 1e-10:
                raise ValueError("transition row exceeds one")
            for x, r in enumerate(row, 1):
                matrix.append((x, y, float("nan") if r["estimate"] is None else r["estimate"], int(r["interval"] is None)))
        dat("state_transitions_" + model + ".dat", ("to", "from", "probability", "missing"), matrix)
    summary_data(dynamics)
    order = ("Analyze", "Implement", "Verify", "Monitor", "Read", "Explore", "Plan")
    coords = []
    for i, m in enumerate(MODELS):
        vals, tops = [], {}
        for c in order:
            r = dynamics[m]["results"][c + ":occupancy"]
            mu, low, high = r["estimate"], *r["interval"]
            vals.extend((100 * mu, 100 * (mu - low), 100 * (high - mu)))
            tops[c] = (100 * (mu + max(mu - low, high - mu)), PAPER_CLASSES.index(c) + 1, 100 * mu)
        for c, (end, label) in zip(order, occupancy_labels(tops, order)):
            vals.extend((tops[c][0], end - tops[c][0], end + OCCUPANCY_LABEL_GAP, label))
        coords.append((i + 1, *vals))
    headings = [h for c in order for h in (c.lower(), c[0].lower() + "minus", c[0].lower() + "plus")]
    headings += [c.lower() + suffix for c in order for suffix in ("top", "lead", "labely", "label")]
    dat("occupancy.dat", ("x", *headings), coords)


# Value labels of the occupancy panels, in units of their 0-60 axis. A label sits OCCUPANCY_LEAD above the top of its
# whisker; when the label of a neighbouring column (one position left or right) is already placed lower, it is raised by
# OCCUPANCY_CLEAR above that one's leader, the height of a label.
OCCUPANCY_LEAD = 7.8 / 7
OCCUPANCY_CLEAR = 51 / 7
OCCUPANCY_LABEL_GAP = .6


def occupancy_labels(tops, order):
    """Leader end and label text of each state of one panel, in `order`: tops maps state to (whisker top, column, mean)."""
    placed, ends = {}, {}
    for state in sorted(tops, key=lambda c: tops[c][0]):
        top, column, mean = tops[state]
        end = top + OCCUPANCY_LEAD
        for other, other_end in placed.items():
            if abs(other - column) <= 1 and end < other_end + OCCUPANCY_CLEAR - 1e-9:
                end = other_end + OCCUPANCY_CLEAR
        placed[column], ends[state] = end, (end, format(mean, ".1f"))
    return [ends[c] for c in order]


TABLE_NAMES = {
    "gpt": r"\modelname{openai}{GPT-OSS-20B}",
    "lfm25": r"\modelname{liquid}{LFM2.5-24B-A2B}",
    "gemma": r"\modelname{gemma}{Gemma-4-26B-A4B}",
    "nemotron": r"\modelname{nvidia}{Nemotron-3.5 30B-A3B}",
    "glm": r"\modelname{zhipu}{GLM-4.7-Flash (30B-A3B)}",
    "qwen330b": r"\modelname{qwen}{Qwen3-30B-A3B}",
    "qwen35": r"\modelname{qwen}{Qwen3.5-35B-A3B}",
    "qwen36": r"\modelname{qwen}{Qwen3.6-35B-A3B}",
}
STATES = ("Read", "Analyze", "Explore", "Plan", "Implement", "Verify", "Monitor")


def cell(mean, low, high, digits, mark="cid", symmetric=False):
    """A table cell with its mean and the two margins to the interval limits (the larger twice when symmetric)."""
    minus, plus = mean - low, high - mean
    if symmetric:
        minus = plus = max(minus, plus)
    return "\\%s{%s}{%s}{%s}" % (mark, *(f"{x:.{digits}f}" for x in (mean, minus, plus)))


def marked(values, *, lowest=False):
    """The cell macro of each model: the best and second-best distinct values get bold and underline."""
    ranks = sorted(set(v[0] for v in values.values()), reverse=not lowest)
    mark = {ranks[0]: "bestcid"}
    if len(ranks) > 1:
        mark[ranks[1]] = "secondcid"
    return {model: mark.get(v[0], "cid") for model, v in values.items()}


def thousands(value):
    return f"{int(value):,}"


def rewrite_rows(name, rows):
    """Replace the model rows of a table, written in the order of `rows`; everything else stays as it is."""
    path = PAPER / "journal/tables" / name
    lines = path.read_text().split("\n")
    first = next(i for i, line in enumerate(lines) if line.startswith("\\modelname"))
    last = first + len(rows)
    if len(rows) != len(MODELS) or not all(line.startswith("\\modelname") for line in lines[first:last]) \
            or lines[last].startswith("\\modelname"):
        raise ValueError(name + ": the rows to replace differ from the model rows")
    lines[first:last] = [" & ".join([TABLE_NAMES[model], *cells]) + " \\\\" for model, cells in rows]
    text = "\n".join(lines)
    if text != path.read_text():
        path.write_text(text)


def gain_cells(rows, prefix, *, flip):
    """(mean, low, high) of each model's gain with its simultaneous limits; `flip` turns a loss change into a reduction."""
    out = {}
    for model in MODELS:
        r = rows[model]
        mean, low, high = (float(r[prefix + key]) for key in ("delta_logloss", "simultaneous_ci7_low", "simultaneous_ci7_high"))
        out[model] = (-mean, -high, -low) if flip else (mean, low, high)
    return out


def readout_table():
    """Rig and alignment gain: reductions of the text baseline's and of the lexical null's loss."""
    rows = {r["model"]: r for r in read_csv("routing_readout.csv")}
    columns = [gain_cells(rows, prefix, flip=True) for prefix in ("routing_", "routing_excess_over_lexical_null_")]
    marks = [marked(c) for c in columns]
    rewrite_rows("readout.tex", [(m, [thousands(rows[m]["sentences"]), *(cell(*c[m], 5, k[m]) for c, k in zip(columns, marks))])
                                 for m in MODELS])


def next_state_table():
    """Next-state routing gain and alignment contrast, as the receipts sign them: negative when routing helps."""
    rows = {r["model"]: r for r in read_csv("next_state.csv")}
    columns = [gain_cells(rows, prefix, flip=False) for prefix in ("", "null_")]
    marks = [marked(c, lowest=True) for c in columns]
    rewrite_rows("next_state.tex", [(m, [thousands(rows[m]["questions"]), thousands(rows[m]["pairs"]),
                                         *(cell(*c[m], 5, k[m]) for c, k in zip(columns, marks))]) for m in MODELS])


def dynamics_cell(panel, key, scale, mark="cid"):
    """An estimate of the native-dynamics panel with its symmetric margin, or with n/e when it has no interval."""
    r = panel["results"][key]
    mean = scale * r["estimate"]
    if r["interval"] is None:
        return "$%.2f_{\\mathrm{n/e}}$" % mean
    low, high = (scale * x for x in r["interval"])
    return cell(mean, low, high, 2, mark, symmetric=True)


def dynamics_tables():
    """Occupancy of the seven states, and annotation coverage with three transitions and the Verify dwell."""
    dynamics = json.loads((DATA / "native_dynamics.json").read_text())
    rewrite_rows("occupancy.tex", [(m, [dynamics_cell(dynamics[m], state + ":occupancy", 100) for state in STATES]) for m in MODELS])
    cells = (("label_coverage", 100), ("Analyze:next:Implement", 100), ("Implement:next:Verify", 100),
             ("Verify:next:Verify", 100), ("Verify:observed_dwell_tokens", 1))
    rewrite_rows("state_transitions.tex",
                 [(m, [thousands(dynamics[m]["results"]["label_coverage"]["family_support"]),
                       *(dynamics_cell(dynamics[m], key, scale) for key, scale in cells)]) for m in MODELS])


def check_inputs():
    """The seven-model captures each have a native-memory fixture that adds up over workers; every model has 4,647 attempts."""
    fixtures = {r["model"]: r for r in json.loads((DATA / "native_fixtures.json").read_text())}
    for m in SEVEN_MODELS:
        if sum(fixtures[m]["native_resident_bytes_by_worker"]) != fixtures[m]["native_resident_bytes_aggregate"]:
            raise ValueError("worker aggregation differs")
    cells = {(r["model"], r["dataset"]): r for r in read_csv("benchmark_cells.csv") if r["corpus"] == "card-v3"}
    for m in MODELS:
        if sum(int(cells[m, b]["attempts"]) for b in BENCHMARKS) != 4647:
            raise ValueError("benchmark attempt population differs")


def main(capture_existing=False, capture_lfm=False, capture_next=False, capture_qwen36=False, pairs_variant=None):
    if capture_existing:
        capture()
    if capture_lfm:
        capture_lfm_rows()
    if capture_qwen36:
        capture_qwen36_panel()
    if pairs_variant:
        capture_pairs(pairs_variant)
    if capture_next:
        capture_next_state()
    check_inputs()
    readout_data()
    selection_data()
    next_state_data()
    dynamics_data()
    readout_table()
    next_state_table()
    dynamics_tables()
    print(json.dumps({"models": len(MODELS), "selection_cells": len(MODELS) * 15, "native_dynamics_models": len(MODELS),
                      "next_state_models": len(MODELS), "new_inference": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture-existing", action="store_true")
    parser.add_argument("--capture-lfm", action="store_true", help="append the LFM2.5 rows to the bundled copies")
    parser.add_argument("--capture-next-state", action="store_true", help="copy the eight next-state estimates")
    parser.add_argument("--capture-qwen36-panel", action="store_true",
                        help="replace Qwen3.6's native-dynamics panel by the one covering all its families")
    parser.add_argument("--capture-pairs", metavar="VARIANT",
                        help="replace every native-dynamics panel by the one completed with the next-sentence labels of VARIANT")
    args = parser.parse_args()
    main(args.capture_existing, args.capture_lfm, args.capture_next_state, args.capture_qwen36_panel, args.capture_pairs)
