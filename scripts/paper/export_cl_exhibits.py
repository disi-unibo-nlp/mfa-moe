"""Export compact coordinates from completed estimates, preserving paper tables.

This exporter performs no fitting, annotation, resampling or model execution.
--capture-existing freezes compact source copies and checks the existing import
bindings. Subsequent runs read only the bundled copies, including all original
confidence endpoints. The default writes data derivatives only: the author-edited
journal tables and exhibit manifest remain untouched. Run verify_cl_tables.py to
check those tables against the copied data. The historical table renderer is
available only through the explicit --legacy-table-format option; it is not the
current manuscript style. --refresh-manifest explicitly updates exhibit hashes.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper/6a510c6db0d9b08aa438c142"
ROOT = PAPER / "figures/journal"
DATA = ROOT / "data"
TABLES = PAPER / "journal/tables"
MODELS = ("gpt", "nemotron", "qwen35", "qwen36", "gemma", "glm", "qwen330b", "lfm25")
CLASSES = ("Read", "Analyze", "Explore", "Plan", "Implement", "Verify", "Monitor")
NAMES = ("GPT-OSS", "Nemotron", "Qwen3.5", "Qwen3.6", "Gemma", "GLM", "Qwen3-30B", "LFM2.5")
BRANDS = ("openai", "nvidia", "qwen", "qwen", "gemma", "zhipu", "qwen", "liquid")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def label(model):
    i = MODELS.index(model)
    return "\\modelname{" + BRANDS[i] + "}{" + NAMES[i] + "}"


def capture():
    DATA.mkdir(parents=True, exist_ok=True)
    sources = {}
    older = PAPER / "generated/best_results_20261009"
    for name in ("routing_readout.csv", "selection_coverage.csv", "anticipation.csv"):
        origin = older / name
        shutil.copyfile(origin, DATA / name)
        sources[name] = {"path": str(origin.relative_to(REPO)), "sha256": sha(origin)}
    for name in ("benchmark_cells.csv", "expert_usage_correctness_mean_all.csv"):
        origin = REPO / "report/model-analysis-review-20261008" / name
        shutil.copyfile(origin, DATA / name)
        sources[name] = {"path": str(origin.relative_to(REPO)), "sha256": sha(origin)}
    imported = REPO / "report/research-campaign-v3-20261014-r6/EVIDENCE_IMPORT.json"
    bindings = {r["path"]: r["sha256"] for r in json.loads(imported.read_text())["references"]}
    summaries = {}
    for model in MODELS[:7]:
        path = REPO / "report/research-campaign-v3-20261014-r5-r2" / model / "NATIVE_DYNAMICS_SUMMARY.json"
        if bindings[str(path)] != sha(path):
            raise ValueError("imported dynamics changed: " + model)
        summaries[model] = json.loads(path.read_text())
        sources[model + ":dynamics"] = {"path": str(path.relative_to(REPO)), "sha256": sha(path)}
    (DATA / "native_dynamics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fixture = REPO / "report/research-campaign-v3-20261014-r6/PARTIAL_VALIDATION_EVIDENCE.json"
    records = json.loads(fixture.read_text())["models"]
    if {r["model"] for r in records} != set(MODELS[:7]):
        raise ValueError("native fixture roster changed")
    (DATA / "native_fixtures.json").write_text(json.dumps(records, indent=2) + "\n")
    sources["native_fixtures"] = {"path": str(fixture.relative_to(REPO)), "sha256": sha(fixture)}
    (DATA / "SOURCES.json").write_text(json.dumps({"sources": sources,
        "import_binding": {"path": str(imported.relative_to(REPO)), "sha256": sha(imported)},
        "new_measurement": False}, indent=2) + "\n")


LFM_ROOT = REPO / "report/final-routing-20261015-r1/lfm25"
LFM_BULK = Path("/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs/final-routing-20261015-r1/lfm25")


def sealed(path):
    value = json.loads(path.read_text())
    seal = value.pop("sha256")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if hashlib.sha256(payload.encode()).hexdigest() != seal:
        raise ValueError("receipt seal differs: " + str(path))
    return value


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
    for dataset in ("aime24", "aime25", "amc23", "math500", "minerva", "olympiad"):
        path = generation / dataset / "traces.jsonl"
        sources["lfm25:generation:" + dataset] = {"path": str(path), "sha256": sha(path)}
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
        cells.append({"model": "lfm25", "dataset": dataset, "corpus": "card-v3", "attempts": len(rows),
            "questions": len(questions), "capped": int(sum(r["capped"] for r in rows)),
            "completion_tokens_sum": int(tokens.sum()), "completion_tokens_median": float(np.median(tokens)),
            "completion_tokens_mean": float(tokens.mean()), "strict_scored": len(rows),
            "strict_correct": int(sum(r["strict"] for r in rows)), "strict_unknown": 0,
            "strict_accuracy": sum(r["strict"] for r in rows) / len(rows),
            "adjudicated_scored": int(scored.sum()), "adjudicated_correct": int(adjudicated[scored].sum()),
            "adjudicated_unknown": int((~scored).sum()), "adjudicated_accuracy": float(adjudicated[scored].mean()),
            "adjudicated_ci_low": float(low), "adjudicated_ci_high": float(high)})
    path = LFM_BULK / "j1/new_lfm25_verdicts.jsonl"
    sources["lfm25:j1_verdicts"] = {"path": str(path), "sha256": sha(path)}
    return cells, sources


def capture_lfm():
    """Append the LFM2.5 rows to the bundled copies; the seven-model rows stay byte-identical."""
    sources = json.loads((DATA / "SOURCES.json").read_text())
    readout = LFM_ROOT / "READOUT_RESULTS.json"
    result = sealed(readout)["models"]["lfm25"]["judge_context"]
    if result["status"] != "CLEAN" or (result["n_questions"], result["n_families"]) != (522, 509):
        raise ValueError("LFM2.5 readout scope differs")
    row = ["lfm25", "CLEAN", "522", "509", str(result["n_sentences"]), repr(result["routing"]["P"]),
           *map(repr, result["routing"]["simultaneous_ci7"]), repr(result["routing_excess_over_lexical_null"]["P"]),
           *map(repr, result["routing_excess_over_lexical_null"]["simultaneous_ci7"])]
    selection = sealed(LFM_ROOT / "SELECTION_COVERAGE.json")
    if selection["keep"] != 48 or selection["experts"] != 64:
        raise ValueError("LFM2.5 retention budget differs")
    dynamics = LFM_ROOT / "NATIVE_DYNAMICS_SUMMARY.json"
    summaries = json.loads((DATA / "native_dynamics.json").read_text())
    summaries["lfm25"] = json.loads(dynamics.read_text())
    cells, generation_sources = lfm_benchmark_cells()
    with (DATA / "routing_readout.csv").open() as stream:
        lines = [l for l in stream.read().splitlines() if not l.startswith("lfm25,")]
    (DATA / "routing_readout.csv").write_text("\n".join(lines + [",".join(row)]) + "\n")
    with (DATA / "selection_coverage.csv").open() as stream:
        lines = [l for l in stream.read().splitlines() if not l.startswith("lfm25,")]
    for fold in selection["folds"]:
        for arm in ("labelled_frequency", "random", "state_balanced"):
            held = fold["held_out"][arm]
            lines.append(",".join(["lfm25", str(fold["fold"]), arm, repr(float(held["minimum_class_coverage"])),
                                   str(int(fold["test_families"])), str(held["minimum_complete"] in (True, "true"))]))
    (DATA / "selection_coverage.csv").write_text("\n".join(lines) + "\n")
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
        sources["sources"][name] = {"path": str(path.relative_to(REPO)), "sha256": sha(path)}
    sources["sources"].update(generation_sources)
    (DATA / "SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n")


def dat(name, headings, rows):
    (DATA / name).write_text(" ".join(headings) + "\n" + "\n".join(
        " ".join(format(v, ".16g") if isinstance(v, float) else str(v) for v in row)
        for row in rows) + "\n")


def legacy_table(name, columns, header, rows, caption, key, *, long=False):
    # Native SPSD benchmark / SYNCO diagnostic ancestry: booktabs, grouped
    # columns, compact model icons, readable type and captions above for CL.
    spec = "@{\\extracolsep{\\fill}}" + columns + "@{}"
    if long:
        text = "\\begingroup\\small\n\\begin{longtable}{" + columns + "}\n"
        text += "\\caption{" + caption + "}\\label{" + key + "}\\\\\n"
        text += "\\toprule\n" + header + " \\\\\n\\midrule\\endfirsthead\n"
        text += "\\multicolumn{" + str(len(columns)) + "}{l}{\\textit{Continued}}\\\\\n\\toprule\n"
        text += header + " \\\\\n\\midrule\\endhead\n\\bottomrule\\endfoot\n"
        text += "\n".join(rows) + "\n\\end{longtable}\n\\endgroup\n"
    else:
        placement = 'hb' if name == 'readout.tex' else 'tb'
        text = "\\begin{table}[" + placement + "]\n\\centering\\small\n\\caption{" + caption + "}\n\\label{" + key + "}\n"
        text += "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular*}{\\linewidth}{" + spec + "}\n\\toprule\n"
        text += header + " \\\\\n\\midrule\n" + "\n".join(rows)
        text += "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    (TABLES / name).write_text("% Adapted from SPSD tab:main-results and SYNCO tab:mainRes.\n" + text)


def preserve_formatted_tables(*args, **kwargs):
    """Leave author-edited table structure, captions, CI notation and ranks intact."""


def export_main_data(*, legacy_table_format=False, refresh_manifest=False):
    table = legacy_table if legacy_table_format else preserve_formatted_tables
    if legacy_table_format:
        TABLES.mkdir(parents=True, exist_ok=True)
    routing = {r["model"]: r for r in read_csv("routing_readout.csv")}
    if set(routing) != set(MODELS):
        raise ValueError("readout roster mismatch")
    coords, tabrows = [], []
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
        tabrows.append(label(m) + " & " + f"{int(r['sentences']):,}" + " & $" +
            f"{values[0]:.5f}" + "$ & $[" + f"{values[0]-values[1]:.5f},{values[0]+values[2]:.5f}" + "]$ \\\\ ")
    dat("readout.dat", ("x", "text", "textminus", "textplus", "null", "nullminus", "nullplus"), coords)
    table("readout.tex", "lrrr", "Model & Sentences & Loss change & 95\\% interval", tabrows,
          "Present-state cross-entropy change (nats per sentence) relative to the text baseline. Each model uses 522 questions in 509 families. Intervals retain the original model-specific seven-comparison family-bootstrap correction.", "tab:readout-detail")

    coverage = read_csv("selection_coverage.csv")
    lookup = {(r["model"], int(r["fold"]), r["arm"]): r for r in coverage}
    if len(lookup) != 15 * len(MODELS):
        raise ValueError("selection cells missing")
    summary, coords, tabrows = [], [], []
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
        means = {a: sum(v) / len(v) for a, v in vals.items()}
        mu = sum(delta) / 5
        coords.append((i + 1, mu, mu - min(delta), max(delta) - mu, *delta))
        best = max(("state_balanced", "labelled_frequency"), key=means.get)
        cells = []
        for arm in ("state_balanced", "labelled_frequency", "random"):
            value = f"{means[arm]:.2f}"
            cells.append("\\textbf{" + value + "}" if arm == best else value)
        tabrows.append(label(model) + " & " + " & ".join(cells) + f" & ${mu:+.2f}$ & $[{min(delta):+.2f},{max(delta):+.2f}]$ \\\\ ")
        summary.append({"model": model, "means_percent": means, "paired_fold_differences_pp": delta})
    dat("selection.dat", ("x", "mean", "minus", "plus", "f0", "f1", "f2", "f3", "f4"), coords)
    (DATA / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table("selection.tex", "lrrrrr", "Model & Balanced & Frequency & Random & $\\Delta$ & Fold range", tabrows,
          "Minimum held-out native selected-slot coverage across reasoning classes (\\%) at 75\\% expert retention. Entries average the five family folds. $\\Delta$ is balanced minus frequency in percentage points; its range spans all five paired folds. Fold variation is descriptive. Bold marks the higher mean of the two learned selectors.", "tab:selection")
    detail = []
    for m in MODELS:
        for f in range(5):
            r = lookup[m, f, "state_balanced"]
            values = [100 * float(lookup[m, f, a]["minimum_class_coverage"]) for a in ("state_balanced", "labelled_frequency", "random")]
            detail.append(label(m) + f" & {f+1} & {r['test_families']} & " + " & ".join(f"{v:.3f}" for v in values) + " \\\\ ")
    table("selection_folds.tex", "lrrrrr", "Model & Fold & Families & Balanced & Frequency & Random", detail,
          "Minimum-class native selected-slot coverage (\\%) by family fold at 75\\% retention. Test families are disjoint; training sets overlap.", "tab:selection-folds", long=True)

    adjacent = [r for r in read_csv("anticipation.csv") if r["feature"] == "composition16" and r["mode"] in ("next", "next-contained")]
    lookup = {(r["model"], r["mode"]): r for r in adjacent}
    coords, tabrows = [], []
    for i, m in enumerate(("gpt", "qwen36")):
        vals = []
        for mode in ("next", "next-contained"):
            r = lookup[m, mode]
            mu, low, high = (float(r[k]) for k in ("delta_logloss", "simultaneous_ci6_low", "simultaneous_ci6_high"))
            if not low <= mu <= high:
                raise ValueError("next-state interval mismatch")
            vals.extend((mu, mu - low, high - mu))
            tabrows.append(label(m) + " & " + ("Adjacent" if mode == "next" else "Contained") + f" & {int(r['questions']):,} & {int(r['pairs']):,} & ${mu:.5f}$ & $[{low:.5f},{high:.5f}]$ \\\\ ")
        coords.append((i + 1, *vals))
    dat("next_state.dat", ("x", "adjacent", "adjminus", "adjplus", "contained", "conminus", "conplus"), coords)
    table("next_state.tex", "llrrrr", "Model & Window & Questions & Pairs & Change & 95\\% interval", tabrows,
          "Next-state cross-entropy changes relative to the past-text/history baseline (nats per adjacent pair). Intervals retain the original six-comparison family-bootstrap correction. Contained windows stay inside source sentences of at least 32 tokens.", "tab:next-state")

    dynamics = json.loads((DATA / "native_dynamics.json").read_text())
    # Matrix coordinates are copied by named state, never relabeled in place.
    for model in MODELS:
        matrix, absent = [], []
        for y, state in enumerate(CLASSES, 1):
            row_total = 0.0
            for x, destination in enumerate(CLASSES, 1):
                result = dynamics[model]["results"][state + ":next:" + destination]
                mean = result["estimate"]
                missing = int(result["interval"] is None)
                row_total += mean
                matrix.append((x, y, mean, missing))
                if missing:
                    absent.append((x + .32, y - .32))
            if abs(row_total - 1) > 1e-10:
                raise ValueError("transition row does not sum to one")
        dat("state_transitions_" + model + ".dat", ("to", "from", "probability", "missing"), matrix)
        if absent:
            dat("state_transitions_missing_" + model + ".dat", ("to", "from"), absent)
    coords, detail, scope = [], [], []
    for i, m in enumerate(MODELS):
        d = dynamics[m]
        results = d["results"]
        coverage = results["label_coverage"]
        scope.append(label(m) + f" & {coverage['family_support']:,} & ${100*coverage['estimate']:.2f}$ & $[{100*coverage['interval'][0]:.2f},{100*coverage['interval'][1]:.2f}]$ \\\\ ")
        vals = []
        for c in ("Analyze", "Implement", "Verify", "Monitor", "Read", "Explore", "Plan"):
            r = results[c + ":occupancy"]
            mu, low, high = r["estimate"], *r["interval"]
            vals.extend((100 * mu, 100 * (mu - low), 100 * (high - mu)))
        coords.append((i + 1, *vals))
        for c in CLASSES:
            r = results[c + ":occupancy"]
            detail.append(label(m) + " & " + c + f" & {r['family_support']:,} & ${100*r['estimate']:.2f}$ & $[{100*r['interval'][0]:.2f},{100*r['interval'][1]:.2f}]$ \\\\ ")
    dat("occupancy.dat", ("x", "analyze", "aminus", "aplus", "implement", "iminus", "iplus", "verify", "vminus", "vplus", "monitor", "mminus", "mplus", "read", "rminus", "rplus", "explore", "eminus", "eplus", "plan", "pminus", "pplus"), coords)
    table("label_coverage.tex", "lrrr", "Model & Families & Label coverage (\\%) & 95\\% interval", scope,
          "Labelled tokens as a fraction of completion tokens in the native-state summaries. Benchmarks and supported families receive equal weight. Intervals are simultaneous 95\\% whole-family-bootstrap intervals within each original model-specific dynamics panel. Qwen3.6 uses the smaller common cohort.", "tab:label-coverage")
    table("occupancy.tex", "llrrr", "Model & State & Families & Occupancy (\\%) & 95\\% interval", detail,
          "All seven state occupancies among observed labelled tokens. Intervals retain the 10,000 whole-family resamples and the simultaneous correction within each model's native-dynamics panel. They do not jointly cover cross-model differences.", "tab:occupancy-detail", long=True)

    fixtures = {r["model"]: r for r in json.loads((DATA / "native_fixtures.json").read_text())}
    tabrows = []
    for m in MODELS[:7]:  # native-memory fixtures exist for the seven original models only
        r = fixtures[m]
        if sum(r["native_resident_bytes_by_worker"]) != r["native_resident_bytes_aggregate"]:
            raise ValueError("worker aggregation differs")
        tabrows.append(label(m) + f" & {r['full_context']:,} & {r['native_resident_bytes_aggregate']/2**30:.2f}" + " \\\\ ")
    table("native_memory.tex", "lrr", "Model & Context capacity (tokens) & Native residence (GiB)", tabrows,
          "Measured native process residence summed over two GPU workers, using the recorded fixed 8-GiB KV-cache setting. These are model-specific full-context fixtures, not pruning savings or repeated-run means.", "tab:native-memory")

    cells = {(r["model"], r["dataset"]): r for r in read_csv("benchmark_cells.csv") if r["corpus"] == "card-v3"}
    tabrows = []
    benchmarks = ("aime24", "aime25", "amc23", "math500", "minerva", "olympiad")
    for m in MODELS:
        vals = [cells[m, b] for b in benchmarks]
        if sum(int(r["attempts"]) for r in vals) != 4647:
            raise ValueError("benchmark attempt population differs")
        tabrows.append(label(m) + " & " + " & ".join(f"{100*float(r['adjudicated_accuracy']):.1f}" for r in vals) + f" & {sum(int(r['adjudicated_unknown']) for r in vals)} \\\\ ")
    table("benchmarks.tex", "lrrrrrrr", "Model & A24 & A25 & AMC & MATH & Minerva & Olympiad & $U$", tabrows,
          "Native mathematical accuracy (\\%) under each model's card profile. A24, A25, AMC and MATH denote AIME24, AIME25, AMC23 and MATH-500. Each model has 4,647 attempts on 1,547 questions; $U$ counts unresolved answers. Scores condition on adjudicated attempts and model-specific decoding budgets.", "tab:benchmarks")
    details = []
    for m in MODELS:
        for b in benchmarks:
            r = cells[m, b]
            details.append(label(m) + " & " + {"aime24":"AIME24", "aime25":"AIME25", "amc23":"AMC23", "math500":"MATH-500", "minerva":"Minerva", "olympiad":"Olympiad"}[b] + f" & {r['questions']} & {r['adjudicated_scored']} & {100*float(r['adjudicated_accuracy']):.1f} & $[{100*float(r['adjudicated_ci_low']):.1f},{100*float(r['adjudicated_ci_high']):.1f}]$ \\\\ ")
    table("benchmark_details.tex", "llrrrr", "Model & Benchmark & Questions & Scored & Acc. & 95\\% interval", details,
          "Native card-profile accuracy (\\%) and the original question-bootstrap intervals. Repeated attempts remain grouped by source question. Scored counts retain unresolved-answer exclusions.", "tab:benchmark-details", long=True)

    if refresh_manifest:
        files = [p for folder in (DATA, TABLES) for p in folder.glob("*") if p.is_file()]
        (ROOT / "EXHIBITS.json").write_text(json.dumps({"exporter_sha256": sha(Path(__file__)),
            "files": {str(p.relative_to(PAPER)): sha(p) for p in files},
            "new_inference": False,
            "observed_models": list(MODELS),
            "planned_models": ["LiquidAI/LFM2.5-24B-A2B"],
            "planned_model_source_sha256": sha(PAPER / "journal/PLANNED_MODEL.json"),
            "table_mode": "historical_renderer" if legacy_table_format else "author_formatted_preserved",
            "mock_preview_files": [str(p.relative_to(PAPER)) for p in files if "routing_mock" in p.name],
            "uncertainty": "original confidence endpoints copied; saved selection folds have no estimated CI"},
            indent=2) + "\n")
    print(json.dumps({"models": len(MODELS), "selection_cells": len(MODELS) * 15, "native_dynamics_models": len(MODELS),
                      "native_fixtures": 7, "next_state_models": 2, "new_inference": False,
                      "table_mode": "historical_renderer" if legacy_table_format else "author_formatted_preserved",
                      "manifest_refreshed": refresh_manifest}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-existing", action="store_true")
    parser.add_argument("--capture-lfm", action="store_true", help="append the LFM2.5 rows to the bundled copies")
    parser.add_argument("--legacy-table-format", action="store_true",
                        help="explicitly overwrite journal tables with the historical renderer")
    parser.add_argument("--refresh-manifest", action="store_true",
                        help="explicitly refresh exhibit hashes after the export")
    args = parser.parse_args()
    if args.capture_existing:
        capture()
    if args.capture_lfm:
        capture_lfm()
    export_main_data(legacy_table_format=args.legacy_table_format,
                     refresh_manifest=args.refresh_manifest)
