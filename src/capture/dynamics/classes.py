"""Sentence tables of a trace: the token ids each sentence owns, its class label and its neighbours' labels."""
from __future__ import annotations
from labeling.spans import reasoning_ranges, sentence_spans, trace_digest
from capture.dynamics.common import keys, termination
from store import CLASSES


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
