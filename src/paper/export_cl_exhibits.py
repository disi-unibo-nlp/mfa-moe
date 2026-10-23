"""Write the data behind the paper figures from the bundled copies of completed estimates.

Nothing is fitted, resampled or run here. The bundled copies live in figures/journal/data (readout, selection and
anticipation CSVs, native dynamics, benchmark cells). A plain run turns them into the plotting files readout.dat,
selection.dat, selection_summary.json, next_state.dat, occupancy.dat and state_transitions_*.dat. The tables of the
manuscript are edited by hand and are never written here; `verify_cl_tables` checks them against the same copies.
Before writing it also checks that the native-memory fixtures add up over workers and that every model has 4,647
benchmark attempts.

  python -m paper.export_cl_exhibits                 plotting files from the bundled copies
  python -m paper.export_cl_exhibits --capture-lfm   first add the LFM2.5 rows from final-routing-20261015-r1
  python -m paper.export_cl_exhibits --capture-existing   first freeze the original seven-model copies (done once)
"""
import argparse
import csv
import json
import shutil

from paper.cl_paths import DATA, LEGACY, REPO
from paper.export_routing_results import CLASSES
from store import BENCHMARKS, FAST, MODELS, file_sha, verified


def read_csv(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def source(path):
    """How SOURCES.json names a file: its path below the repository and its hash."""
    return {"path": str(path.relative_to(REPO)), "sha256": file_sha(path)}


def capture():
    DATA.mkdir(parents=True, exist_ok=True)
    sources = {}
    older, review = LEGACY / "generated/best_results_20261009", REPO / "report/model-analysis-review-20261008"
    for origin in [older / name for name in ("routing_readout.csv", "selection_coverage.csv", "anticipation.csv")] + [
            review / name for name in ("benchmark_cells.csv", "expert_usage_correctness_mean_all.csv")]:
        shutil.copyfile(origin, DATA / origin.name)
        sources[origin.name] = source(origin)
    imported = REPO / "report/research-campaign-v3-20261014-r6/EVIDENCE_IMPORT.json"
    bindings = {r["path"]: r["sha256"] for r in json.loads(imported.read_text())["references"]}
    summaries = {}
    for model in MODELS[:7]:
        path = REPO / "report/research-campaign-v3-20261014-r5-r2" / model / "NATIVE_DYNAMICS_SUMMARY.json"
        if bindings[str(path)] != file_sha(path):
            raise ValueError("imported dynamics changed: " + model)
        summaries[model] = json.loads(path.read_text())
        sources[model + ":dynamics"] = source(path)
    (DATA / "native_dynamics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fixture = REPO / "report/research-campaign-v3-20261014-r6/PARTIAL_VALIDATION_EVIDENCE.json"
    records = json.loads(fixture.read_text())["models"]
    if {r["model"] for r in records} != set(MODELS[:7]):
        raise ValueError("native fixture roster changed")
    (DATA / "native_fixtures.json").write_text(json.dumps(records, indent=2) + "\n")
    sources["native_fixtures"] = source(fixture)
    (DATA / "SOURCES.json").write_text(json.dumps({"sources": sources, "import_binding": source(imported),
                                                   "new_measurement": False}, indent=2) + "\n")


LFM_ROOT = REPO / "report/final-routing-20261015-r1/lfm25"
LFM_BULK = FAST / "final-routing-20261015-r1/lfm25"


def lfm_benchmark_cells():
    """card-v3 cells for LFM2.5 with the source adjudication rule: math_verify True stays correct, a capped
    answer is incorrect, and a finished rejected answer takes its J1 answer-equivalence verdict. Intervals
    repeat the original 1,000-replicate question bootstrap with seed 0."""
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


def capture_lfm():
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
    """Next-state cross-entropy change (adjacent and contained windows) for the two models that have it."""
    lookup = {(r["model"], r["mode"]): r for r in read_csv("anticipation.csv") if r["feature"] == "composition16"}
    coords = []
    for i, m in enumerate(("gpt", "qwen36")):
        vals = []
        for mode in ("next", "next-contained"):
            mu, low, high = (float(lookup[m, mode][k]) for k in ("delta_logloss", "simultaneous_ci6_low", "simultaneous_ci6_high"))
            if not low <= mu <= high:
                raise ValueError("next-state interval mismatch")
            vals.extend((mu, mu - low, high - mu))
        coords.append((i + 1, *vals))
    dat("next_state.dat", ("x", "adjacent", "adjminus", "adjplus", "contained", "conminus", "conplus"), coords)


def dynamics_data():
    """State-transition matrices and state occupancies of the native dynamics."""
    dynamics = json.loads((DATA / "native_dynamics.json").read_text())
    # Matrix coordinates are copied by named state, never relabeled in place.
    for model in MODELS:
        matrix, absent = [], []
        for y, state in enumerate(CLASSES, 1):
            row = [dynamics[model]["results"][state + ":next:" + destination] for destination in CLASSES]
            # An undefined estimate stays missing (nan), never zero, and is not summed.
            defined = [r["estimate"] for r in row if r["estimate"] is not None]
            if len(defined) == len(CLASSES) and abs(sum(defined) - 1) > 1e-10:
                raise ValueError("transition row does not sum to one")
            if sum(defined) > 1 + 1e-10:
                raise ValueError("transition row exceeds one")
            for x, r in enumerate(row, 1):
                matrix.append((x, y, float("nan") if r["estimate"] is None else r["estimate"], int(r["interval"] is None)))
                if r["interval"] is None:
                    absent.append((x + .32, y - .32))
        dat("state_transitions_" + model + ".dat", ("to", "from", "probability", "missing"), matrix)
        if absent:
            dat("state_transitions_missing_" + model + ".dat", ("to", "from"), absent)
    order = ("Analyze", "Implement", "Verify", "Monitor", "Read", "Explore", "Plan")
    coords = []
    for i, m in enumerate(MODELS):
        vals = []
        for c in order:
            r = dynamics[m]["results"][c + ":occupancy"]
            mu, low, high = r["estimate"], *r["interval"]
            vals.extend((100 * mu, 100 * (mu - low), 100 * (high - mu)))
        coords.append((i + 1, *vals))
    headings = [h for c in order for h in (c.lower(), c[0].lower() + "minus", c[0].lower() + "plus")]
    dat("occupancy.dat", ("x", *headings), coords)


def check_inputs():
    """Native-memory fixtures (seven original models) add up over workers; every model has 4,647 benchmark attempts."""
    for r in json.loads((DATA / "native_fixtures.json").read_text()):
        if sum(r["native_resident_bytes_by_worker"]) != r["native_resident_bytes_aggregate"]:
            raise ValueError("worker aggregation differs")
    cells = {(r["model"], r["dataset"]): r for r in read_csv("benchmark_cells.csv") if r["corpus"] == "card-v3"}
    for m in MODELS:
        if sum(int(cells[m, b]["attempts"]) for b in BENCHMARKS) != 4647:
            raise ValueError("benchmark attempt population differs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture-existing", action="store_true")
    parser.add_argument("--capture-lfm", action="store_true", help="append the LFM2.5 rows to the bundled copies")
    args = parser.parse_args()
    if args.capture_existing:
        capture()
    if args.capture_lfm:
        capture_lfm()
    readout_data()
    selection_data()
    next_state_data()
    dynamics_data()
    check_inputs()
    print(json.dumps({"models": len(MODELS), "selection_cells": len(MODELS) * 15, "native_dynamics_models": len(MODELS),
                      "next_state_models": 2, "new_inference": False}))
