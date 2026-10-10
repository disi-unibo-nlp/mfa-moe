"""Observed class dynamics: no transitions or durations across unlabelled gaps."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
from moe_exp.correlation_pipeline.spans import reasoning_ranges, sentence_spans, trace_digest
from .common import CLASSES, keys, termination


def saved_layout(trace):
    """Same ownership rule as spans.token_layout; uses exact saved offsets only."""
    units = sentence_spans(trace)
    replay = trace.metadata.get("token_replay")
    if not replay or "completion_offsets" not in replay:
        return dict(units=units, unit_tokens=[None] * len(units),
                    reasoning_tokens=None, token_count=None)
    offsets = replay["completion_offsets"]
    if len(offsets) != len(replay["completion_token_ids"]):
        raise ValueError("Saved completion IDs/offsets disagree")
    ranges = reasoning_ranges(trace)
    owners, reasoning, index = [[] for _ in units], [], 0
    for token, (left, right) in enumerate(offsets):
        if not 0 <= left <= right <= len(trace.cot_text):
            raise ValueError("Invalid completion offset")
        if right <= left or not units or not any(right > a and left < b for a, b in ranges):
            continue
        reasoning.append(token)
        while index + 1 < len(units) and units[index]["end"] <= left:
            index += 1
        candidates, j = [], max(0, index - 1)
        while j < len(units) and units[j]["start"] < right:
            candidates.append((max(0, min(right, units[j]["end"]) - max(left, units[j]["start"])), -j))
            j += 1
        overlap, negative = max(candidates, default=(0, -index))
        owner = -negative if overlap else max(0, index - int(units[index]["start"] >= right))
        owners[owner].append(token)
    return dict(units=units, unit_tokens=owners, reasoning_tokens=reasoning, token_count=len(offsets))


def sentence_table(trace, annotation=None, layout=None):
    layout = layout or saved_layout(trace)
    units, labeled = layout["units"], {}
    annotation = annotation or trace.metadata.get("reasoning_annotation")
    if annotation is not None:
        if annotation.get("schema_version") != 1 or annotation.get("trace_sha256") != trace_digest(trace):
            raise ValueError("Stale/unsupported annotation")
        for key, value in (("source_model", trace.model_id), ("dataset", trace.dataset),
                           ("problem_id", trace.problem_id), ("sample_id", trace.sample_id)):
            if key in annotation and annotation[key] != value:
                raise ValueError("Annotation identity mismatch")
        if annotation.get("sentence_selection") != trace.metadata.get("sentence_selection"):
            raise ValueError("Annotation selection mismatch")
        for unit in annotation.get("units", []):
            i = unit.get("index")
            if not isinstance(i, int) or not 0 <= i < len(units) or i in labeled:
                raise ValueError("Invalid/duplicate annotation index")
            if any(unit.get(k) != v for k, v in units[i].items()):
                raise ValueError("Annotation offsets/text mismatch")
            label = unit.get("label")
            if label is not None and (label not in CLASSES or unit.get("status") == "unknown"):
                raise ValueError("Invalid class")
            labeled[i] = label
    ranges, rows, shared = reasoning_ranges(trace), [], keys(trace)
    segments = [next(j for j, (a, b) in enumerate(ranges) if a <= u["start"] < b) for u in units]
    for i, unit in enumerate(units):
        previous = labeled.get(i - 1) if i and segments[i] == segments[i - 1] else None
        following = labeled.get(i + 1) if i + 1 < len(units) and segments[i] == segments[i + 1] else None
        tokens = layout["unit_tokens"][i]
        rows.append({**shared, "sentence_index": i, "segment": segments[i],
                     **{k: unit[k] for k in ("start", "end", "text")},
                     "label": labeled.get(i), "previous_class": previous, "next_class": following,
                     "token_indices": tokens, "tokens": len(tokens) if tokens is not None else None,
                     "token_start": min(tokens) if tokens else None,
                     "token_end": max(tokens) + 1 if tokens else None, "termination": termination(trace)})
    return rows


def class_dynamics(sentences):
    tables = {name: [] for name in ("transitions", "runs", "motifs", "returns")}
    groups = defaultdict(list)
    for row in sentences:
        groups[row["trace_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: r["sentence_index"])
        run, history, last = None, [], {}
        for i, row in enumerate(rows):
            label, prev = row["label"], rows[i - 1] if i else None
            contiguous = bool(prev and prev["sentence_index"] + 1 == row["sentence_index"]
                              and prev["segment"] == row["segment"])
            shared = {k: row[k] for k in ("trace_id", "model", "dataset", "question_id")}
            if not contiguous or not label or not prev["label"]:
                history, last = [], {}
            if run is not None and (not contiguous or label != run["label"]):
                event = label if contiguous and label else "gap"
                run.update(exit_event=event, right_censored=event == "gap")
                tables["runs"].append(run)
                run = None
            if label is None:
                continue
            if contiguous and prev["label"]:
                tables["transitions"].append({**shared, "source": prev["label"], "destination": label,
                    "previous_class": prev["previous_class"], "sentence_index": prev["sentence_index"]})
            if run is None:
                entry = (i == 0 and row["sentence_index"] == 0) or bool(
                    contiguous and prev["label"] and prev["label"] != label)
                run = {**shared, "label": label, "start_sentence": row["sentence_index"],
                       "entry_observed": entry, "sentences": 0, "tokens": 0}
                if label in last:
                    old = last[label]
                    tables["returns"].append({**shared, "label": label,
                        "return_sentences": row["sentence_index"] - old["sentence_index"],
                        "return_tokens": row["token_start"] - old["token_start"]
                        if row["token_start"] is not None and old["token_start"] is not None else None})
                last[label] = row
            run["sentences"] += 1
            run["tokens"] = run["tokens"] + row["tokens"] if (
                run["tokens"] is not None and row["tokens"] is not None) else None
            row["run_age_sentences"] = run["sentences"] if run["entry_observed"] else None
            row["run_age_tokens"] = run["tokens"] if run["entry_observed"] else None
            history.append(label)
            for n in (2, 3):
                if len(history) >= n:
                    tables["motifs"].append({**shared, "pattern": history[-n:]})
            history = history[-3:]
        if run:
            event = rows[-1]["termination"]
            run.update(exit_event=event, right_censored=event != "natural_termination")
            tables["runs"].append(run)
    for run in tables["runs"]:
        run["duration_sentences"] = run["sentences"] if run["entry_observed"] else None
        run["duration_tokens"] = run["tokens"] if run["entry_observed"] else None
    return tables


def summaries(tables, min_problems=10):
    sources, counts, support = Counter(), Counter(), defaultdict(set)
    conditioned, condition_totals = Counter(), Counter()
    for row in tables["transitions"]:
        key = (row["model"], row["dataset"], row["source"])
        sources[key] += 1
        counts[(*key, row["destination"])] += 1
        support[(*key, row["destination"])].add(row["question_id"])
        if row["previous_class"]:
            c = (row["model"], row["dataset"], row["previous_class"], row["source"])
            conditioned[(*c, row["destination"])] += 1
            condition_totals[c] += 1
    rates = []
    for model, dataset in sorted({(r["model"], r["dataset"]) for r in tables["runs"]}):
        for source in CLASSES:
            key = (model, dataset, source)
            n = sources[key]
            probs = [counts[(*key, dest)] / n for dest in CLASSES] if n else []
            entropy = -sum(p * math.log(p) for p in probs if p) if n else None
            for dest in CLASSES:
                problems = len(support[(*key, dest)])
                rates.append(dict(model=model, dataset=dataset, source=source, destination=dest,
                    count=counts[(*key, dest)], source_count=n, rate=counts[(*key, dest)] / n if n else None,
                    transition_entropy=entropy, stay_rate=counts[(*key, source)] / n if n else None,
                    exit_rate=1 - counts[(*key, source)] / n if n else None, problems=problems,
                    status="descriptive" if problems >= min_problems else "insufficient_support"))
    previous = [dict(zip(("model", "dataset", "previous_class", "source", "destination"), key),
                     count=n, rate=n / condition_totals[key[:-1]]) for key, n in sorted(conditioned.items())]
    risk, exits = Counter(), Counter()
    for run in tables["runs"]:
        if not run["entry_observed"]:
            continue
        key = (run["model"], run["dataset"], run["label"])
        for age in range(1, run["sentences"] + 1 - int(run["right_censored"])):
            risk[(*key, age)] += 1
        if not run["right_censored"]:
            exits[(*key, run["sentences"], run["exit_event"])] += 1
    hazards = [dict(model=m, dataset=d, source=c, age_sentences=age, destination=dest,
                    at_risk=n, events=exits[(m, d, c, age, dest)], hazard=exits[(m, d, c, age, dest)] / n)
               for (m, d, c, age), n in sorted(risk.items())
               for dest in (*CLASSES, "natural_termination") if dest != c]
    mc, ms = Counter(), defaultdict(set)
    for row in tables["motifs"]:
        key = (row["model"], row["dataset"], tuple(row["pattern"]))
        mc[key] += 1
        ms[key].add(row["question_id"])
    motifs = [dict(model=m, dataset=d, pattern=list(p), count=n, problems=len(ms[(m, d, p)]),
                   status="descriptive" if len(ms[(m, d, p)]) >= min_problems else "insufficient_support")
              for (m, d, p), n in sorted(mc.items())]
    return dict(transition_rates=rates, previous_conditioned=previous, exit_hazards=hazards, motif_counts=motifs)
