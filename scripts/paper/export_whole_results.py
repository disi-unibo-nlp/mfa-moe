"""Export the whole-reasoning strength-sweep results into the journal manuscript.

Reads one sealed receipt, EFFECTS_WHOLE.json (src/moe_exp/study/whole_effects.py::combine; no
fitting or resampling here) and writes, relative to --out (default: the paper root):
  journal/tables/whole.tex                 primary contrasts at full strength (8 models, pooled row)
  journal/tables/whole_cells.tex           per-cell length and accuracy effects (appendix)
  journal/tables/whole_occupancy.tex       per-cell response of the steered state (appendix)
  figures/journal/whole_dose.tex           dose-response figure
  figures/journal/data/whole_dose.dat      coordinates of the figure
  figures/journal/data/whole_results.json  every displayed number with its receipt key

Token contrasts are a benefit on log(1+reasoning tokens), positive meaning fewer tokens than the
comparator. They are shown as the percent change 100*expm1(-benefit), so the interval ends swap;
accuracy is shown in percentage points. Intervals are the receipt's simultaneous 95% envelopes
(over the outcome bounds when the receipt has them), displayed as mean +- the larger one-sided
margin.
"""
from pathlib import Path
import argparse
import json
import math
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_routing_results import (  # noqa: E402
    DRIVER, MODELS, PAPER, REPO, TABLE_NAMES, TICKS, cid, sha, verified)

DEFAULT_EFFECTS = REPO / "report/whole-reasoning-20261021-r11/analysis-combined/EFFECTS_WHOLE.json"
PREREG = DRIVER / "EFFICIENCY_LINK_PREREG.json"
CLASSES = ("Read", "Analyze", "Explore", "Plan", "Implement", "Verify", "Monitor")
STATE = dict(zip(CLASSES, ("Re", "An", "Ex", "Pl", "Im", "Ve", "Mo")))
OPERATIONS = ("promote", "suppress")
DOSES = ("quarter", "half", "main")  # figure order, weakest first
DOSE_LABELS = {"quarter": "Quarter", "half": "Half", "main": "Full"}
# Three tints of one hue: the bars are an ordered strength series, not categories.
COLORS = ("gomblueone!30", "gomblueone!65", "gomblueone")
SHIFTS = (-8, 0, 8)
DASH = r"\textemdash"
# Plain model names for captions: the table name without icon, parameter suffix or brand macro.
PLAIN = {m: re.fullmatch(r"\\modelname\{[^}]*\}\{(.*)\}", name).group(1).split(" (")[0]
         for m, name in TABLE_NAMES.items()}
COMPARISONS = (("primary", "Targeted $-$ random"), ("target_native", "Targeted $-$ native"),
               ("random_native", "Random $-$ native"))
STATE_KEY = ("Re, An, Ex, Pl, Im, Ve and Mo abbreviate Read, Analyze, Explore, Plan, Implement, "
             "Verify and Monitor.")
UNITS = ("Tokens give the percent change in reasoning tokens, with negative values meaning fewer "
         "tokens, and accuracy differences are in percentage points")
SIMULTANEOUS = ("Intervals are simultaneous over {} and cover the outcome bounds of unfinished "
                "continuations.")


# --- receipt access and display units --------------------------------------------------------

def interval(v):
    """The simultaneous envelope over the outcome bounds when the receipt has one."""
    return v.get("bound_simultaneous", v["simultaneous"])


def usable(v):
    """Contrasts the receipt left unqualified have no estimate or interval and stay blank."""
    return bool(v) and "estimate" in v and "simultaneous" in v


def scaled(v, kind):
    """Mean and interval ends of a contrast in display units.

    A token benefit b is the percent change 100*expm1(-b) in reasoning tokens, so its ends swap;
    anything else (accuracy, occupancy) is a proportion shown in percentage points.
    """
    if kind == "tokens":
        mean = 100 * math.expm1(-v["estimate"])
        low, high = sorted(100 * math.expm1(-x) for x in interval(v))
    else:
        mean = 100 * v["estimate"]
        low, high = (100 * x for x in interval(v))
    return mean, low, high


def display(record, key, v, kind):
    """Mean and symmetric margin (larger one-sided distance to the interval ends) of one contrast,
    logged under its receipt key."""
    mean, low, high = scaled(v, kind)
    margin = max(mean - low, high - mean, 0.)
    record[key] = {"mean": mean, "margin": margin, "interval_display": [low, high],
                   "estimate": v["estimate"], "interval": interval(v), "status": v.get("status"),
                   "interval_key": "bound_simultaneous" if "bound_simultaneous" in v
                   else "simultaneous", "point_identified": v.get("point_identified")}
    return mean, margin


def entry(record, key, v, kind, digits):
    """One table cell; a dash when the receipt has no qualified contrast."""
    if not usable(v):
        return DASH
    mean, margin = display(record, key, v, kind)
    return cid(round(mean, digits) + 0., margin, digits)  # + 0. turns -0.0 into 0.0


def names(models):
    """'A', 'A and B' or 'A, B and C'."""
    plain = [PLAIN[m] for m in models]
    return plain[0] if len(plain) == 1 else ", ".join(plain[:-1]) + " and " + plain[-1]


def base_doses(effects):
    """Each model's base strength: the one release that holds its native rows."""
    base = {}
    for level in effects["levels"]:
        if level["arm"] != "native":
            continue
        if base.setdefault(level["model"], level["dose"]) != level["dose"]:
            raise ValueError("a model has native rows at several strengths: " + level["model"])
    return base


def directed(model, operation, class_name, prereg):
    """Replicates whole.directions: promote classes with a negative frozen beta, suppress with a
    positive one (a model outside the preregistration uses its pooled signs)."""
    rows = (prereg["pooled"] if model not in prereg["models"]
            else [c for c in prereg["cells"] if c["model"] == model])
    sign = {c["class"]: c["tok_reason"]["sign"] for c in rows}[class_name]
    return sign == ("-" if operation == "promote" else "+")


def cell_keys(effects, base, prereg):
    """Steered cells (operation, class) of each model at its base strength.

    A sweep release runs only efficiency-directed cells, so every cell the receipt holds at
    another strength must be directed under the frozen signs.
    """
    cells = {}
    for key in effects["cell_effects"]:
        model, dose, operation, class_name = key.split("|")
        if dose == base.get(model):
            cells.setdefault(model, []).append((operation, class_name))
        elif not directed(model, operation, class_name, prereg):
            raise ValueError("sweep cell is not efficiency-directed under the frozen signs: " + key)
    return {m: sorted(c, key=lambda cell: (OPERATIONS.index(cell[0]), CLASSES.index(cell[1])))
            for m, c in cells.items()}


def caption(*sentences):
    """One sentence per source line, separated by percent lines; empty sentences are dropped."""
    return "\n%\n".join(s for s in sentences if s)


def block(lines):
    """Rows of one model block; every row but the last forbids a page break after it."""
    ends = [" \\\\*"] * (len(lines) - 1) + [" \\\\"]
    return "\n".join(line + end for line, end in zip(lines, ends))


def heading(model, dose, columns):
    """Full-width model row naming the strength it is shown at."""
    return ("\\multicolumn{" + str(columns) + "}{l}{" + TABLE_NAMES[model] + ", "
            + DOSE_LABELS[dose].lower() + " strength}")


# --- tables ----------------------------------------------------------------------------------

def primary_table(effects, record, source, sweep_models):
    """Primary contrasts at full strength, with the two native companions."""
    groups = {"primary": effects["primary"],
              **{k: effects["companions"][k] for k in ("target_native", "random_native")}}
    rows = []
    for model in (*MODELS, "pooled"):
        values = []
        for comparison, _ in COMPARISONS:
            prefix = "primary" if comparison == "primary" else "companions/" + comparison
            for metric in ("tokens", "accuracy"):
                values.append(entry(record, f"{prefix}/{model}:{metric}",
                                    groups[comparison].get(f"{model}:{metric}"), metric,
                                    1 if metric == "tokens" else 2))
        rows.append(("\\midrule\n" if model == "pooled" else "")
                    + TABLE_NAMES.get(model, "All models") + " & " + " & ".join(values) + " \\\\")
    top = " & ".join("\\multicolumn{2}{c}{" + label + "}" for _, label in COMPARISONS)
    body = "\n".join(rows)
    cap = caption(
        "Paired effects of routing edits at full strength on reasoning tokens and strict accuracy, "
        "with family-bootstrap simultaneous 95\\% confidence intervals as subscripts.",
        "Targeted edits promote or suppress the efficiency-directed states of each model, random "
        "edits use matched control expert sets, and native routing is unedited; each column pair "
        "compares the first arm with the second.",
        UNITS + ".",
        SIMULTANEOUS.format("the rows and both outcomes of each comparison"),
        f"{names(sweep_models)} ran full strength in the strength sweep." if sweep_models else "")
    return rf"""% Source: {source} via scripts/paper/export_whole_results.py.
\begin{{table}}[tb]
\centering\small
\caption{{{cap}}}
\label{{tab:whole}}
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{adjustbox}}{{max width=\linewidth}}
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
 & {top} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}
Model & Tokens (\%) & Acc.\ (pp) & Tokens (\%) & Acc.\ (pp) & Tokens (\%) & Acc.\ (pp) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table}}
"""


def cap_rate(effects, model, dose, operation):
    """Share (%) of a targeted arm that reaches the length limit, among completed continuations."""
    for level in effects["levels"]:
        if (level["model"], level["dose"], level["arm"]) == (model, dose, "target_" + operation):
            return None if level["cap_rate"] is None else 100 * level["cap_rate"]
    return None


def cells_table(effects, record, source, base, cells, prereg):
    """Appendix longtable: tokens, accuracy and length-limit rate of every steered cell."""
    top = (" & \\multicolumn{2}{c}{Targeted $-$ random} & \\multicolumn{2}{c}{Targeted $-$ native}"
           " & \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    columns = r"Cell & Tokens (\%) & Acc.\ (pp) & Tokens (\%) & Acc.\ (pp) & Limit (\%) \\"
    blocks = []
    for model in MODELS:
        if model not in cells:
            continue
        dose = base[model]
        lines = [heading(model, dose, 6)]
        for operation in OPERATIONS:
            members = [c for o, c in cells[model] if o == operation]
            if not members:
                continue
            # The receipt keeps the length-limit rate per arm, so it is one value per operation.
            limit = cap_rate(effects, model, dose, operation)
            if limit is not None:
                record[f"levels/{model}|{dose}|target_{operation}/cap_rate"] = {"value": limit}
            lines.append("\\textit{" + operation.capitalize() + "} & & & & & "
                         + (f"{limit:.1f}" if limit is not None else DASH))
            for class_name in members:
                key = f"{model}|{dose}|{operation}|{class_name}"
                values = []
                for comparison in ("target_random", "target_native"):
                    for metric in ("tokens", "accuracy"):
                        values.append(entry(
                            record, f"cell_effects/{key}/{comparison}:{metric}",
                            effects["cell_effects"][key].get(f"{comparison}:{metric}"), metric,
                            1 if metric == "tokens" else 2))
                mark = "$^{\\dagger}$" if directed(model, operation, class_name, prereg) else ""
                lines.append("\\quad \\textrm{" + STATE[class_name] + "}" + mark + " & "
                             + " & ".join(values) + " & ")
        blocks.append(block(lines))
    half = [m for m in MODELS if m in base and base[m] != "main"]
    body = "\n\\midrule\n".join(blocks)
    cap = caption(
        "Effects of every steered cell at each model's base strength, with family-bootstrap "
        "simultaneous 95\\% confidence intervals as subscripts.",
        STATE_KEY,
        "A dagger marks the efficiency-directed cells that enter the primary contrast.",
        UNITS + ", for targeted edits against matched random edits and against native routing.",
        "Limit gives the share of the targeted continuations of each operation that reach the "
        "length limit.",
        SIMULTANEOUS.format("the six contrasts of each cell"),
        f"Each model is shown at its base strength, which is full except for {names(half)} at half."
        if half else "Each model is shown at its base strength, which is full.")
    return rf"""% Multipage longtable: retained outside adjustbox so page breaks remain available.
% Source: {source} (cell_effects, levels) via scripts/paper/export_whole_results.py.
\begingroup\small
\setlength{{\tabcolsep}}{{2pt}}
\begin{{longtable}}{{lccccc}}
\caption{{{cap}}}\label{{tab:whole-cells}}\\
\toprule
{top}
{columns}
\midrule\endfirsthead
\multicolumn{{6}}{{l}}{{\textit{{Continued}}}}\\
\toprule
{top}
{columns}
\midrule\endhead
\bottomrule\endfoot
{body}
\end{{longtable}}
\endgroup
"""


def occupancy_table(effects, record, source, base, cells):
    """Appendix table: response of the steered state's own occupancy to targeted vs random edits."""
    blocks = []
    for model in MODELS:
        if model not in cells:
            continue
        dose = base[model]
        lines = [heading(model, dose, 8)]
        for operation in OPERATIONS:
            present = {c for o, c in cells[model] if o == operation}
            values = []
            for class_name in CLASSES:
                key = f"{model}|{dose}|{operation}|{class_name}"
                column = "target_random:response:" + class_name
                v = effects["mechanism"].get(key, {}).get(column)
                values.append(entry(record, f"mechanism/{key}/{column}", v, "points", 2)
                              if class_name in present else DASH)
            lines.append(operation.capitalize() + " & " + " & ".join(values))
        blocks.append(block(lines))
    states = " & ".join("\\textrm{" + STATE[c] + "}" for c in CLASSES)
    body = "\n\\midrule\n".join(blocks)
    cap = caption(
        "Response of the steered state's own occupancy to targeted edits relative to matched "
        "random edits at each model's base strength, in percentage points of the generation "
        "allowance, with family-bootstrap simultaneous 95\\% confidence intervals as subscripts.",
        STATE_KEY,
        "Columns give the steered state, and positive values are changes in the intended "
        "direction, an increase after promotion and a decrease after suppression.",
        SIMULTANEOUS.format("the contrasts of each cell"))
    return rf"""% Source: {source} (mechanism) via scripts/paper/export_whole_results.py.
\begin{{table}}[tb]
\centering\small
\caption{{{cap}}}
\label{{tab:whole-occupancy}}
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{adjustbox}}{{max width=\linewidth}}
\begin{{tabular}}{{@{{}}lccccccc@{{}}}}
\toprule
Operation & {states} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table}}
"""


# --- dose-response figure --------------------------------------------------------------------

def nice_axis(entries, height, digits):
    """Y limits and ticks that leave room for the rotated value label beyond every whisker end.

    entries are (mean, margin) pairs. A label is about 3 pt per character plus padding and sits
    beyond the whisker end on the side of the bar's sign.
    """
    ends = [(m, m - e, m + e) for m, e in entries]
    low = min([0.] + [lo for _, lo, _ in ends])
    high = max([0.] + [hi for _, _, hi in ends])

    def reach(positive):
        return max([2.9 * len(f"{m:.{digits}f}") + 3 for m, _, _ in ends if (m >= 0) == positive]
                   or [0.])

    top, bottom = reach(True), reach(False)
    if high == low:
        high = low + 1
    scale = (high - low) / (1 - (top + bottom) / height)  # data units across the whole axis
    raw_low, raw_high = low - bottom * scale / height, high + top * scale / height
    for step in (s * 10. ** k for k in range(-3, 5) for s in (1, 2, 2.5, 5)):
        first, last = math.floor(raw_low / step + 1e-9), math.ceil(raw_high / step - 1e-9)
        if last - first <= 6:
            break
    ticks = [format(round(i * step, 6) + 0., "g") for i in range(first, last + 1)]
    return (ticks[0], ticks[-1]), ",".join(ticks)


def axis_options(p, panel, height, gap, panels, series):
    """pgfplots options of one stacked panel."""
    name, title, digits, _ = panel
    entries = [v for dose in DOSES for v in series[name][dose] if v is not None]
    limits, ticks = nice_axis(entries, height, digits)
    return ("\\begin{axis}[journalaxis,scale only axis,width=352pt,height=" + str(height) + "pt,"
            "anchor=south west,at={(0pt," + str((panels - 1 - p) * (height + gap)) + "pt)},"
            "ybar=0pt,bar width=7pt,xmin=.5,xmax=9.5,enlarge x limits=false,"
            "ymin=" + limits[0] + ",ymax=" + limits[1] + ",ytick={" + ticks + "},"
            "xtick={1,2,3,4,5,6,7,8,9},xticklabels={" + TICKS + "},"
            "xticklabel style={rotate=0,anchor=north,align=center,"
            "font=\\fontsize{7}{8}\\selectfont},"
            + ("" if p == panels - 1 else "xticklabels=\\empty,") +
            "extra y ticks={0},extra y tick style={grid=major,"
            "major grid style={draw=gomaxis,line width=.45pt}},"
            "title={\\textbf{" + title + "}},title style={font=\\fontsize{8}{9}\\selectfont}]")


def dose_figure(series, panels, out):
    """Native GoM grouped bars in stacked panels, three strengths per model group.

    Missing model-strength combinations have no bar; value labels sit beyond each whisker (or the
    bar end where there is no interval).
    """
    lines = ["% Native GoM signed grouped-bar recipe (plot-gom-signed-bars.tex) in stacked "
             "MoM-style panels.",
             "% Generated by scripts/paper/export_whole_results.py from "
             "figures/journal/data/whole_dose.dat.",
             "\\begin{tikzpicture}"]
    for p, panel in enumerate(panels):
        name, _, digits, has_error = panel
        lines.append(axis_options(p, panel, 72, 32, len(panels), series))
        for k, dose in enumerate(DOSES):
            column = f"{name}_{dose}"
            options = ("fill=" + COLORS[k] + ",draw=black,line width=.35pt,bar shift="
                       + str(SHIFTS[k]) + "pt" + (",journalerror" if has_error else ""))
            table = "x=x,y=" + column + (",y error=" + column + "_pm" if has_error else "")
            lines.append("\\addplot[" + options + "] table[" + table
                         + "]{figures/journal/data/whole_dose.dat};")
            for x, value in enumerate(series[name][dose], 1):
                if value is None:
                    continue
                mean, margin = value
                end = mean + margin if mean >= 0 else mean - margin
                anchor = "west" if mean >= 0 else "east"
                text = f"{round(mean, digits) + 0.:.{digits}f}"
                lines.append("\\node[rotate=90,anchor=" + anchor + ",inner sep=1pt,"
                             "font=\\fontsize{5.6}{6}\\selectfont] at ([xshift=" + str(SHIFTS[k])
                             + "pt]axis cs:" + str(x) + "," + format(end, ".4g") + ") {$"
                             + text + "$};")
        lines.append("\\end{axis}")
    keys = "\\quad ".join("\\statekey{" + COLORS[k] + "}{" + DOSE_LABELS[d] + " strength}"
                          for k, d in enumerate(DOSES))
    lines.append("\\node[journallegend,anchor=north,yshift=-4pt,font=\\fontsize{7.5}{9}"
                 "\\selectfont] at (176pt,0pt |- current bounding box.south) {" + keys + "};")
    lines.append("\\end{tikzpicture}")
    out.write_text("\n".join(lines) + "\n")


def dose_series(effects, record, continuations, prereg):
    """Per panel and strength, nine (mean, margin) pairs (eight models, pooled); None if absent."""
    contrasts = effects["dose_response"]["contrasts"]
    series = {"tok": {}, "acc": {}, "cap": {}}
    # The length-limit share uses the same cells at every strength: the efficiency-directed ones,
    # the only cells a sweep strength runs. It is descriptive, over completed targeted rows.
    targeted = continuations[continuations.arm.isin(("target_promote", "target_suppress"))
                             & (continuations.outcome_status == "observed")]
    targeted = targeted[[directed(m, o, c, prereg) for m, o, c in
                         zip(targeted.model, targeted.operation, targeted["class"])]]
    for dose in DOSES:
        for name, metric in (("tok", "tokens"), ("acc", "accuracy")):
            row = []
            for group in (*MODELS, "pooled"):
                key = f"{group}@{dose}:{metric}"
                row.append(display(record, "dose_response/contrasts/" + key, contrasts[key], metric)
                           if usable(contrasts.get(key)) else None)
            series[name][dose] = row
        # The pooled value is the mean over the models that ran that strength.
        row = []
        for model in MODELS:
            rows = targeted[(targeted.model == model) & (targeted.dose == dose)]
            if rows.empty:
                row.append(None)
                continue
            value = 100 * float((rows.finish_reason == "length").mean())
            record[f"continuations/{model}|{dose}|directed-target/cap_rate"] = {
                "value": value, "completed_rows": len(rows)}
            row.append((value, 0.))
        present = [v[0] for v in row if v]
        pooled = sum(present) / len(present) if present else None
        record[f"continuations/pooled|{dose}|directed-target/cap_rate"] = {"value": pooled,
                                                                            "models": len(present)}
        series["cap"][dose] = row + [(pooled, 0.) if present else None]
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS,
                        help="sealed EFFECTS_WHOLE.json")
    parser.add_argument("--out", type=Path, default=PAPER,
                        help="root to write under (default: the paper)")
    args = parser.parse_args()
    effects, prereg = verified(args.effects), verified(PREREG)
    if not set(effects["models"]) <= set(MODELS):
        raise ValueError("whole receipt has a model outside the study roster")
    base = base_doses(effects)
    cells = cell_keys(effects, base, prereg)
    sweep_models = [m for m in MODELS if base.get(m, "main") != "main" and any(
        lv["model"] == m and lv["dose"] == "main" for lv in effects["levels"])]
    try:
        source = str(args.effects.resolve().relative_to(REPO))
    except ValueError:
        source = str(args.effects.resolve())
    tables, data = args.out / "journal/tables", args.out / "figures/journal/data"
    for folder in (tables, data):
        folder.mkdir(parents=True, exist_ok=True)

    record = {}
    (tables / "whole.tex").write_text(primary_table(effects, record, source, sweep_models))
    (tables / "whole_cells.tex").write_text(
        cells_table(effects, record, source, base, cells, prereg))
    (tables / "whole_occupancy.tex").write_text(
        occupancy_table(effects, record, source, base, cells))

    import pandas as pd
    continuations = pd.read_parquet(args.effects.parent / "continuations.parquet",
                                    columns=["model", "dose", "arm", "operation", "class",
                                             "outcome_status", "finish_reason"])
    series = dose_series(effects, record, continuations, prereg)
    panels = (("tok", r"(a) Reasoning-token change, targeted $-$ random (\%)", 1, True),
              ("acc", "(b) Accuracy change, targeted $-$ random (pp)", 1, True),
              ("cap", r"(c) Targeted continuations at the length limit (\%)", 1, False))
    nan = float("nan")
    columns = ["x"]
    for name, _, _, has_error in panels:
        for dose in DOSES:
            columns += [f"{name}_{dose}"] + ([f"{name}_{dose}_pm"] if has_error else [])
    rows = []
    for x in range(9):
        row = [x + 1]
        for name, _, _, has_error in panels:
            for dose in DOSES:
                mean, margin = series[name][dose][x] or (nan, nan)
                row += [mean, margin] if has_error else [mean]
        rows.append(" ".join(format(v, ".6g") for v in row))
    (data / "whole_dose.dat").write_text(" ".join(columns) + "\n" + "\n".join(rows) + "\n")
    dose_figure(series, panels, args.out / "figures/journal/whole_dose.tex")

    summary = {
        "receipt": {"path": source, "file_sha256": sha(args.effects),
                    "seal_sha256": effects["sha256"], "bootstrap_draws": effects["bootstrap_draws"],
                    "seed": effects["seed"], "models": effects["models"], "base_dose": base},
        "prereg": {"path": str(PREREG.relative_to(REPO)), "seal_sha256": prereg["sha256"]},
        "units": {"tokens": "percent change in reasoning tokens, 100*expm1(-estimate)",
                  "accuracy": "percentage points, 100*estimate",
                  "mechanism": "percentage points of the generation allowance, 100*estimate",
                  "cap_rate": ("percent of completed targeted continuations of the efficiency-"
                               "directed cells at the length limit, from continuations.parquet"),
                  "margin": "larger one-sided distance from the mean to the interval ends"},
    }
    (data / "whole_results.json").write_text(
        json.dumps({**summary, "values": record}, indent=1, default=float) + "\n")
    print(json.dumps({"models": len(effects["models"]), "values": len(record), "tables": 3,
                      "figure": 1, "out": str(args.out)}))


if __name__ == "__main__":
    main()
