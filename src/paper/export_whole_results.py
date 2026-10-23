"""Write the whole-reasoning tables and figure data of the paper from the sealed EFFECTS_WHOLE.json receipt.

Reads one receipt (made by steering.whole_effects; no fitting or resampling here). Command:
python -m paper.export_whole_results [--effects EFFECTS_WHOLE.json] [--out DIR]. It writes, below --out (default: the
paper folder):
  journal/tables/whole.tex                 primary contrasts at full strength (8 models, pooled row)
  journal/tables/whole_cells.tex           per-cell length and accuracy effects (appendix)
  journal/tables/whole_occupancy.tex       per-cell response of the steered state (appendix)
  journal/tables/whole_levels.tex          levels of every arm at every strength (appendix)
  figures/journal/whole_dose.tex           dose-response figure
  figures/journal/data/whole_dose.dat      coordinates of the figure
  figures/journal/whole_states.tex         per-state effect heatmaps at each model's base strength
  figures/journal/data/whole_states_tokens.dat, whole_states_accuracy.dat   cells of the heatmaps
  figures/journal/data/whole_results.json  every displayed number with its receipt key

Token contrasts are a benefit on log(1+reasoning tokens), positive meaning fewer tokens than the
comparator. They are shown as the percent change 100*expm1(-benefit), so the interval ends swap;
accuracy is shown in percentage points. Intervals are the receipt's simultaneous 95% envelopes
(over the outcome bounds when the receipt has them), displayed as mean +- the larger one-sided
margin.
The heatmaps use the same units. Their colors clip at STATE_CLIP, a number is bold when its
interval excludes zero, a cell is outlined when its state is efficiency-directed, and a cell is
empty when the model did not run that state.
"""
from pathlib import Path
import argparse
import json
import math
import re

from paper.cl_paths import PAPER, REPO
from paper.export_routing_results import CLASSES, DRIVER, TABLE_NAMES, TICKS, cid
from store import MODELS, file_sha, verified

DEFAULT_EFFECTS = REPO / "report/whole-reasoning-20261021-r20/analysis-combined/EFFECTS_WHOLE.json"  # the final receipt
PREREG = DRIVER / "EFFICIENCY_LINK_PREREG.json"
STATE = dict(zip(CLASSES, ("Re", "An", "Ex", "Pl", "Im", "Ve", "Mo")))
OPERATIONS = ("promote", "suppress")
DOSES = ("quarter", "half", "main")  # figure order, weakest first
DOSE_LABELS = {"quarter": "Quarter", "half": "Half", "main": "Full"}
# Three tints of one hue: the bars are an ordered strength series, not categories.
COLORS = ("gomblueone!30", "gomblueone!65", "gomblueone")
SHIFTS = (-8, 0, 8)
# Axis caps of the strength figure: an interval that leaves the panel is cut at its edge (the
# tables give every interval), so a few unknown outcomes cannot flatten the bars.
CAPS = {"tok": (-100., 200.), "acc": (-40., 25.), "cap": (0., 100.)}
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
LEVEL_ARMS = (("native", "Native"), ("native_repeat", "Native repeat"),
              ("target_promote", "Targeted promote"), ("random_promote", "Random promote"),
              ("target_suppress", "Targeted suppress"), ("random_suppress", "Random suppress"))
# Heatmap columns: the promote block, then the suppress block, each in the paper's state order.
STATE_COLUMNS = tuple((o, c) for o in OPERATIONS for c in CLASSES)
STATE_CLIP = {"tokens": 100., "accuracy": 30.}  # symmetric color limits, in display units
STATE_DIGITS = {"tokens": 0, "accuracy": 1}
STATE_PANELS = (("tokens", r"(a) Reasoning-token change, targeted $-$ random (\%)"),
                ("accuracy", "(b) Accuracy change, targeted $-$ random (pp)"))
CELL_W, CELL_H, BLOCK_GAP, PANEL_GAP = 20, 18, 8, 46  # pt
# Ends of the diverging scale of state_transitions.tex: gomblueone below zero, F00000 above.
SCALE_ENDS = {"low": (24, 118, 210), "high": (240, 0, 0)}
# One \plotmodel label per model, in paper order (the pooled "All models" tick is dropped).
PLOT_LABELS = re.findall(r"\{\\plotmodel\{[^}]*\}\{[^}]*\}\{[^}]*\}\}", TICKS)


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
    positive one (a model outside the link receipt uses its pooled signs)."""
    rows = (prereg["pooled"] if model not in prereg["models"]
            else [c for c in prereg["cells"] if c["model"] == model])
    sign = {c["class"]: c["tok_reason"]["sign"] for c in rows}[class_name]
    return sign == ("-" if operation == "promote" else "+")


def cell_keys(effects, base, prereg, run_signs):
    """Steered cells (operation, class) of each model at its base strength.

    A sweep release runs only the cells directed under the signs it ran with (run_signs, the
    seven-model preregistration), so every cell the receipt holds at another strength must be
    directed under those; under the eight-model link LFM2.5's sweep holds a few more.
    """
    cells = {}
    for key in effects["cell_effects"]:
        model, dose, operation, class_name = key.split("|")
        if dose == base.get(model):
            cells.setdefault(model, []).append((operation, class_name))
        elif not directed(model, operation, class_name, run_signs):
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


def level_values(record, model, level):
    """Display cells of one arm's levels, logged under their receipt keys."""
    key = f"levels/{model}|{level['dose']}|{level['arm']}/"
    check = level["manipulation_check"]
    record[key + "completed"] = {"value": level["complete"]}
    record[key + "planned"] = {"value": level["planned"]}
    cells = [f"{level['complete']}/{level['planned']}"]
    for field, scale, digits in (("cap_rate", 100, 1), ("repetition_16gram", 100, 1),
                                 ("mean_total_tokens", 1, 0)):
        value = level[field]
        if value is not None:
            record[key + field] = {"value": scale * value}
        cells.append(DASH if value is None else f"{scale * value:.{digits}f}")
    # Native arms apply no edit, and an arm without active router positions has nothing to check.
    if level["arm"].startswith("native") or not check["active_layer_positions"]:
        return cells + [DASH]
    changed = 100 * check["changed_membership_fraction"]
    record[key + "changed_membership_fraction"] = {
        "value": changed, "active_layer_positions": check["active_layer_positions"]}
    return cells + [f"{changed:.1f}"]


def levels_table(effects, record, source):
    """Appendix longtable: descriptive levels of every arm at every strength a model ran."""
    by_key = {(lv["model"], lv["dose"], lv["arm"]): lv for lv in effects["levels"]}
    columns = r"Arm & Done & Limit (\%) & Repeat (\%) & Tokens & Changed (\%) \\"
    blocks = []
    for model in MODELS:
        parts = []
        for dose in DOSES:
            arms = [(arm, label) for arm, label in LEVEL_ARMS if (model, dose, arm) in by_key]
            if not arms:
                continue
            # The model's name row stays with its first strength, and each strength stays whole.
            lines = [] if parts else ["\\multicolumn{6}{l}{" + TABLE_NAMES[model] + "}"]
            lines.append("\\multicolumn{6}{l}{\\quad\\textit{" + DOSE_LABELS[dose] + " strength}}")
            lines += ["\\qquad " + label + " & "
                      + " & ".join(level_values(record, model, by_key[model, dose, arm]))
                      for arm, label in arms]
            parts.append(block(lines))
        if parts:
            blocks.append("\n".join(parts))
    body = "\n\\midrule\n".join(blocks)
    cap = caption(
        "Descriptive levels of every arm at each strength it ran, without intervals.",
        "Quarter, half and full give the strength of the routing edit, and the native arms of a "
        "model belong to its base strength.",
        "Done counts the continuations that finished out of those scheduled.",
        "Limit is the share of finished continuations that reach the length limit, Repeat is the "
        "share of 16-token windows that repeat an earlier window of the same continuation, and "
        "Tokens is the mean number of generated tokens.",
        "Changed is the share of active router positions at which the edit changed the selected "
        "experts, and native arms have none.")
    return rf"""% Multipage longtable: retained outside adjustbox so page breaks remain available.
% Source: {source} (levels) via scripts/paper/export_whole_results.py.
\begingroup\small
\setlength{{\tabcolsep}}{{2pt}}
\begin{{longtable}}{{lccccc}}
\caption{{{cap}}}\label{{tab:whole-levels}}\\
\toprule
{columns}
\midrule\endfirsthead
\multicolumn{{6}}{{l}}{{\textit{{Continued}}}}\\
\toprule
{columns}
\midrule\endhead
\bottomrule\endfoot
{body}
\end{{longtable}}
\endgroup
"""


# --- dose-response figure --------------------------------------------------------------------

def clamp(value, cap):
    return min(max(value, cap[0]), cap[1])


def nice_axis(entries, height, digits, cap):
    """Y limits and ticks that leave room for the rotated value label beyond every whisker end.

    entries are (mean, margin) pairs. A label is about 3 pt per character plus padding and sits
    beyond the whisker end on the side of the bar's sign; whisker ends beyond the cap count as
    at it.
    """
    ends = [(m, clamp(m - e, cap), clamp(m + e, cap)) for m, e in entries]
    low = min([0.] + [lo for _, lo, _ in ends])
    high = max([0.] + [hi for _, _, hi in ends])

    def reach(positive):
        return max([2.9 * len(f"{m:.{digits}f}") + 3 for m, _, _ in ends if (m >= 0) == positive]
                   or [0.])

    # A label whose whisker reaches the cap sits inside the panel, so it needs no room beyond it.
    top = 0. if high >= cap[1] else reach(True)
    bottom = 0. if low <= cap[0] else reach(False)
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
    limits, ticks = nice_axis(entries, height, digits, CAPS[name])
    # The printed endpoint can differ slightly from a cap after data rounding.
    # Padding also keeps the physical error-bar cap clear of the panel stroke.
    panel_padding = {"tok": ("-120", "220"), "acc": ("-45", "45")}
    if name in panel_padding:
        limits = panel_padding[name]
    return tuple(map(float, limits)), (
        "\\begin{axis}[journalaxis,scale only axis,width=352pt,height=" + str(height) + "pt,"
            "anchor=south west,at={(0pt," + str((panels - 1 - p) * (height + gap)) + "pt)},"
            "ybar=0pt,bar width=7pt,xmin=.5,xmax=9.5,enlarge x limits=false,"
            "clip=true,clip marker paths=true,"
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
        limits, options = axis_options(p, panel, 72, 32, len(panels), series)
        lines.append(options)
        per_point = (limits[1] - limits[0]) / 72  # data units per pt of panel height
        for k, dose in enumerate(DOSES):
            column = f"{name}_{dose}"
            options = ("fill=" + COLORS[k] + ",draw=black,line width=.35pt,bar shift="
                       + str(SHIFTS[k]) + "pt" + (",journalerror" if has_error else ""))
            table = "x=x,y=" + column + (",y error plus=" + column + "_p,y error minus=" + column
                                         + "_m" if has_error else "")
            lines.append("\\addplot[" + options + "] table[" + table
                         + "]{figures/journal/data/whole_dose.dat};")
            for x, value in enumerate(series[name][dose], 1):
                if value is None:
                    continue
                mean, margin = value
                raw = mean + margin if mean >= 0 else mean - margin
                end = clamp(raw, CAPS[name])
                text = f"{round(mean, digits) + 0.:.{digits}f}"
                # Beyond the whisker end, or just inside the panel edge on a background patch when
                # the interval is cut at the cap or the label would cross the edge.
                reach = (2.9 * len(text) + 3) * per_point
                tip = end + (reach if mean >= 0 else -reach)
                cut = end != raw or not limits[0] <= tip <= limits[1]
                end = clamp(end, limits)
                outward = "west" if mean >= 0 else "east"
                anchor = {"west": "east", "east": "west"}[outward] if cut else outward
                lines.append("\\node[rotate=90,anchor=" + anchor + ",inner sep=1pt,"
                             + ("fill=journalplotbackground," if cut else "")
                             + "font=\\fontsize{5.6}{6}\\selectfont] at ([xshift=" + str(SHIFTS[k])
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


# --- per-state heatmaps ----------------------------------------------------------------------

def state_grid(effects, record, base, cells, prereg, models, metric):
    """Targeted-minus-random contrast of every model by state column, at each model's base strength.

    A column the model did not run has no entry; one it ran without a qualified contrast is
    "unqualified". Significant means the simultaneous interval excludes zero.
    """
    grid = {}
    for row, model in enumerate(models, 1):
        for col, (operation, class_name) in enumerate(STATE_COLUMNS, 1):
            if (operation, class_name) not in cells.get(model, ()):
                continue
            key = f"{model}|{base[model]}|{operation}|{class_name}"
            v = effects["cell_effects"][key].get("target_random:" + metric)
            directed_cell = directed(model, operation, class_name, prereg)
            cell = {"x": col, "y": row, "directed": directed_cell, "mean": None,
                    "significant": False}
            if usable(v):
                display(record, f"cell_effects/{key}/target_random:{metric}", v, metric)
                cell["mean"], low, high = scaled(v, metric)
                cell["significant"] = low > 0 or high < 0
            state = record.setdefault("states/" + key, {"directed": directed_cell})
            state[metric + "_significant"] = cell["significant"]
            grid[row, col] = cell
    return grid


def state_data(grid, models, metric):
    """Rows of the heatmap's data file, row-major as a matrix plot needs; nan marks empty cells."""
    clip = STATE_CLIP[metric]
    lines = ["x y meta value significant directed"]
    for row in range(1, len(models) + 1):
        for col in range(1, len(STATE_COLUMNS) + 1):
            cell = grid.get((row, col))
            if cell is None or cell["mean"] is None:
                lines.append(f"{col} {row} nan nan 0 {int(cell is not None and cell['directed'])}")
                continue
            mean = cell["mean"]
            lines.append(f"{col} {row} {max(-clip, min(clip, mean)):.6g} {mean:.6g} "
                         f"{int(cell['significant'])} {int(cell['directed'])}")
    return "\n".join(lines) + "\n"


def dark_fill(mean, clip):
    """True where the cell color is dark enough for white text (luminance below 140 of 255)."""
    t = min(abs(mean) / clip, 1.)
    end = SCALE_ENDS["low" if mean < 0 else "high"]
    red, green, blue = (255 + t * (c - 255) for c in end)
    return 0.299 * red + 0.587 * green + 0.114 * blue < 140


def state_label(cell, metric):
    """Cell text: the rounded mean, bold when the interval excludes zero; a dash if unqualified."""
    if cell["mean"] is None:
        return "\\textcolor{black!55}{" + DASH + "}"
    digits = STATE_DIGITS[metric]
    # Text mode keeps one font for both weights; an en dash serves as the minus sign.
    text = f"{round(cell['mean'], digits) + 0.:.{digits}f}".replace("-", "--")
    return "\\textbf{" + text + "}" if cell["significant"] else text


def scale_bar(x, middle, clip, metric):
    """TikZ lines of one panel's vertical scale (as in state_transitions.tex): blue below zero,
    red above, ticks at zero, half and full clip, on a gray backdrop; x is its left edge and
    middle the height of its center, in pt."""
    half = 50  # pt from the center to either end
    bar = f"({x}pt,{middle - half:g}pt) rectangle ({x + 7}pt,{middle + half:g}pt)"
    lines = [f"\\fill[gray!12] ({x - 5}pt,{middle - 60:g}pt) rectangle "
             f"({x + 42}pt,{middle + 60:g}pt);",
             f"\\shade[bottom color=gomblueone,top color=white] ({x}pt,{middle - half:g}pt) "
             f"rectangle ({x + 7}pt,{middle:g}pt);",
             f"\\shade[bottom color=white,top color=transitionlow] ({x}pt,{middle:g}pt) "
             f"rectangle ({x + 7}pt,{middle + half:g}pt);",
             f"\\draw[draw=gomaxis,line width=.45pt] {bar};"]
    for fraction in (-1, -.5, 0, .5, 1):
        lines.append("\\node[anchor=west,inner sep=.1pt,font=\\fontsize{7}{8}\\selectfont] at "
                     f"({x + 10}pt,{middle + half * fraction:g}pt) {{${fraction * clip:g}$}};")
    unit = "Change (\\%)" if metric == "tokens" else "Change (pp)"
    lines.append("\\node[rotate=90,inner sep=.1pt,font=\\fontsize{8}{9}\\selectfont] at "
                 f"({x + 37}pt,{middle:g}pt) {{{unit}}};")
    return lines


def states_figure(grids, models, out):
    """Per-state heatmaps in the native recipe of state_transitions.tex.

    Two stacked panels (tokens, accuracy), each with a promote block beside a suppress block of the
    seven states, models as rows and a vertical diverging scale. Cells are matrix plot cells on the
    beige carrier, so cells a model did not run stay empty; numbers are set as one node per cell.
    """
    rows = len(models)
    height, block_width = rows * CELL_H, len(CLASSES) * CELL_W
    width = 2 * block_width + BLOCK_GAP
    labels = ",".join(PLOT_LABELS[MODELS.index(m)] for m in models)
    states = ",".join(STATE[c] for c in CLASSES)
    lines = ["% Native PORTS temperature-heatmap cell grid (plot-ports-temperature-heatmap.tex), "
             "adapted as in state_transitions.tex:",
             "% matrix plot* cells, blue/white/red scale centered at zero, beige carrier.",
             "% Generated by scripts/paper/export_whole_results.py from "
             "figures/journal/data/whole_states_*.dat.",
             "\\begin{tikzpicture}",
             "\\definecolor{transitionlow}{HTML}{F00000}",
             "\\pgfplotsset{colormap={journalstatesmap}{color(0pt)=(gomblueone);"
             "color(500pt)=(white);color(1000pt)=(transitionlow)},",
             "  statepanel/.style={journalaxis,scale only axis,width=" + str(block_width)
             + "pt,height=" + str(height) + "pt,anchor=south west,",
             "    ymin=.5,ymax=" + str(rows + .5) + ",y dir=reverse,enlarge x limits=false,"
             "xmajorgrids=false,ymajorgrids=false,",
             "    xticklabels={" + states + "},xticklabel style={rotate=0,anchor=north,"
             "font=\\fontsize{7}{8}\\selectfont},",
             "    yticklabel style={align=center,font=\\fontsize{7}{8}\\selectfont},",
             "    tick style={draw=none},colormap name=journalstatesmap}}",
             "\\tikzset{statecell/.style={inner sep=0pt,minimum width=" + str(CELL_W - 2)
             + "pt,minimum height=" + str(CELL_H - 2) + "pt,font=\\fontsize{6}{7}\\selectfont},",
             "  directedcell/.style={draw=black,line width=.6pt},lightlabel/.style={text=white}}"]
    for p, (metric, title) in enumerate(STATE_PANELS):
        clip, bottom = STATE_CLIP[metric], (len(STATE_PANELS) - 1 - p) * (height + PANEL_GAP)
        for k, operation in enumerate(OPERATIONS):
            first = k * len(CLASSES) + 1
            ticks = ",".join(str(first + i) for i in range(len(CLASSES)))
            left = k * (block_width + BLOCK_GAP)
            lines.append(
                "\\begin{axis}[statepanel,at={(" + str(left) + "pt," + str(bottom) + "pt)},"
                "xmin=" + str(first - .5) + ",xmax=" + str(first + len(CLASSES) - .5)
                + ",xtick={" + ticks + "},point meta min=-" + format(clip, "g")
                + ",point meta max=" + format(clip, "g") + ","
                + ("ytick={" + ",".join(map(str, range(1, rows + 1))) + "},yticklabels={" + labels
                   + "}," if k == 0 else "ytick=\\empty,")
                + "]")
            lines.append("\\addplot[matrix plot*,mesh/cols=" + str(len(STATE_COLUMNS))
                         + ",point meta=explicit,draw=black!25,line width=.2pt]\n"
                         "table[x=x,y=y,meta=meta]{figures/journal/data/whole_states_" + metric
                         + ".dat};")
            for (row, col), cell in grids[metric].items():
                if not first <= col < first + len(CLASSES):
                    continue
                options = "statecell" + (",directedcell" if cell["directed"] else "")
                if cell["mean"] is not None and dark_fill(cell["mean"], clip):
                    options += ",lightlabel"
                lines.append("\\node[" + options + "] at (axis cs:" + f"{col},{row}) {{"
                             + state_label(cell, metric) + "};")
            lines.append("\\end{axis}")
            lines.append("\\node[anchor=base,font=\\fontsize{7.5}{8.5}\\selectfont] at ("
                         + format(left + block_width / 2, "g") + "pt," + str(bottom + height + 6)
                         + "pt) {\\textbf{" + operation.capitalize() + "}};")
        lines += scale_bar(width + 12, bottom + height / 2, clip, metric)
        lines.append("\\node[anchor=base,font=\\fontsize{8}{9}\\selectfont] at ("
                     + format(width / 2, "g") + "pt," + str(bottom + height + 22) + "pt) {\\textbf{"
                     + title + "}};")
    keys = ("\\textbf{12} Interval excludes zero\\quad "
            "\\tikz[baseline=-.4ex]\\draw[line width=.6pt] (0,0) rectangle (.3,.22);\\,"
            " Efficiency-directed state\\quad "
            "\\tikz[baseline=-.4ex]\\draw[draw=black!25,line width=.2pt,fill=gombackground] "
            "(0,0) rectangle (.3,.22);\\, State not run")
    lines.append("\\node[journallegend,anchor=north,yshift=-4pt,font=\\fontsize{7.5}{9}"
                 "\\selectfont] at (" + format(width / 2, "g")
                 + "pt,0pt |- current bounding box.south) {" + keys + "};")
    lines.append("\\end{tikzpicture}")
    out.write_text("\n".join(lines) + "\n")


def dose_data(series, panels, path):
    """The coordinate file of the dose figure; whiskers are cut at the panel caps, as TeX cannot draw far-off ends."""
    nan = float("nan")
    columns = ["x"]
    for name, _, _, has_error in panels:
        for dose in DOSES:
            columns += [f"{name}_{dose}"] + ([f"{name}_{dose}_p", f"{name}_{dose}_m"] if has_error else [])
    rows = []
    for x in range(9):
        row = [x + 1]
        for name, _, _, has_error in panels:
            for dose in DOSES:
                mean, margin = series[name][dose][x] or (nan, nan)
                low, high = CAPS[name]
                cut = [min(margin, max(high - mean, 0.)), min(margin, max(mean - low, 0.))]
                row += [mean, *cut] if has_error else [mean]
        rows.append(" ".join(format(v, ".6g") for v in row))
    path.write_text(" ".join(columns) + "\n" + "\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS, help="sealed EFFECTS_WHOLE.json")
    parser.add_argument("--out", type=Path, default=PAPER, help="folder laid out like the paper (default: the paper)")
    args = parser.parse_args()
    effects, run_signs = verified(args.effects), verified(PREREG)
    if not set(effects["models"]) <= set(MODELS):
        raise ValueError("whole receipt has a model outside the study roster")
    # The signs the analysis used (receipts before r19 record none: the preregistration).
    link_path = REPO / effects["efficiency_link"]["path"] if "efficiency_link" in effects else PREREG
    prereg = verified(link_path)
    if "efficiency_link" in effects and prereg["sha256"] != effects["efficiency_link"]["sha256"]:
        raise ValueError("efficiency-link receipt differs from the one the analysis used")
    base = base_doses(effects)
    cells = cell_keys(effects, base, prereg, run_signs)
    sweep_models = [m for m in MODELS if base.get(m, "main") != "main" and any(
        lv["model"] == m and lv["dose"] == "main" for lv in effects["levels"])]
    try:
        source = str(args.effects.resolve().relative_to(REPO))
    except ValueError:
        source = str(args.effects.resolve())
    tables, figures = args.out / "journal/tables", args.out / "figures/journal"
    data = figures / "data"
    for folder in (tables, data):
        folder.mkdir(parents=True, exist_ok=True)

    record = {}
    (tables / "whole.tex").write_text(primary_table(effects, record, source, sweep_models))
    (tables / "whole_cells.tex").write_text(cells_table(effects, record, source, base, cells, prereg))
    (tables / "whole_occupancy.tex").write_text(occupancy_table(effects, record, source, base, cells))
    (tables / "whole_levels.tex").write_text(levels_table(effects, record, source))

    import pandas as pd
    continuations = pd.read_parquet(args.effects.parent / "continuations.parquet",
                                    columns=["model", "dose", "arm", "operation", "class",
                                             "outcome_status", "finish_reason"])
    series = dose_series(effects, record, continuations, prereg)
    panels = (("tok", r"(a) Reasoning-token change, targeted $-$ random (\%)", 1, True),
              ("acc", "(b) Accuracy change, targeted $-$ random (pp)", 1, True),
              ("cap", r"(c) Targeted continuations at the length limit (\%)", 1, False))
    dose_data(series, panels, data / "whole_dose.dat")
    dose_figure(series, panels, figures / "whole_dose.tex")

    models = [m for m in MODELS if m in cells]
    grids = {metric: state_grid(effects, record, base, cells, prereg, models, metric)
             for metric, _ in STATE_PANELS}
    for metric, _ in STATE_PANELS:
        (data / f"whole_states_{metric}.dat").write_text(state_data(grids[metric], models, metric))
    states_figure(grids, models, figures / "whole_states.tex")

    summary = {
        "receipt": {"path": source, "file_sha256": file_sha(args.effects),
                    "seal_sha256": effects["sha256"], "bootstrap_draws": effects["bootstrap_draws"],
                    "seed": effects["seed"], "models": effects["models"], "base_dose": base},
        "prereg": {"path": str(link_path.relative_to(REPO)), "seal_sha256": prereg["sha256"]},
        "units": {"tokens": "percent change in reasoning tokens, 100*expm1(-estimate)",
                  "accuracy": "percentage points, 100*estimate",
                  "mechanism": "percentage points of the generation allowance, 100*estimate",
                  "cap_rate": ("percent of completed targeted continuations of the efficiency-"
                               "directed cells at the length limit, from continuations.parquet"),
                  "margin": "larger one-sided distance from the mean to the interval ends"},
        "states": {"rows": models, "columns": [list(c) for c in STATE_COLUMNS],
                   "color_clip": STATE_CLIP,
                   "significant": "the simultaneous interval (bound_simultaneous when present) "
                                  "excludes zero",
                   "directed": "efficiency-directed under the efficiency-link signs (prereg)"},
    }
    (data / "whole_results.json").write_text(
        json.dumps({**summary, "values": record}, indent=1, default=float) + "\n")
    print(json.dumps({"models": len(effects["models"]), "values": len(record), "tables": 4,
                      "figures": 2, "out": str(args.out)}))


if __name__ == "__main__":
    main()
