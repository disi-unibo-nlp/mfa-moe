"""Original full-A labels/offsets, independent of frozen causal enrollment."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .artifacts import npz
from .contracts import CLASSES, digest, file_sha, load, publish, require
from .inputs import BASE, counts_for, read_source


def token_ownership(trace, labels, expected_digest, expected_counts):
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.spans import trace_digest
    trace = TraceRecord(**trace)
    require(trace_digest(trace) == expected_digest, "mapping trace identity differs")
    layout = saved_layout(trace)
    require(layout["token_count"] is not None, "mapping lacks original token offsets")
    units, owners = layout["units"], layout["unit_tokens"]
    accepted, conflicts = {}, set()
    audit = dict(offset_mismatch=0, out_of_range=0, unknown_label=0, conflicting=0)
    for i, start, end, label in labels:
        if not 0 <= i < len(units):
            audit["out_of_range"] += 1
            continue
        if (start, end) != (units[i]["start"], units[i]["end"]):
            audit["offset_mismatch"] += 1
            continue
        if label not in CLASSES:
            audit["unknown_label"] += 1
            continue
        if i in accepted and accepted[i] != label:
            conflicts.add(i)
        accepted[i] = label
    for i in conflicts:
        accepted.pop(i, None)
    audit["conflicting"] = len(conflicts)
    classes = [np.array(sorted(t for i, c in accepted.items() if c == name
                               for t in owners[i]), dtype=np.int64) for name in CLASSES]
    reasoning = np.asarray(layout["reasoning_tokens"], dtype=np.int64)
    all_labelled = np.sort(np.concatenate(classes))
    require(len(all_labelled) == len(set(all_labelled)), "duplicate labelled token ownership")
    counts = [len(t) for t in classes] + [len(reasoning), len(all_labelled)]
    require(counts == list(map(int, expected_counts)), "full-A mapping differs from original nine class counts")
    return classes, reasoning, audit


def mapping(campaign, model):
    root = Path(campaign["root"])
    sources = load(root/model/"SOURCES.json")["records"]
    _, tokens = counts_for(campaign, model, sources)
    labels = defaultdict(list)
    files = {}
    for path in sorted((BASE/"campaign-v3/labels"/model).glob("part-*/annotations.json")):
        files[str(path)] = file_sha(path)
        for row in json.loads(path.read_text()):
            i = row["identity"]
            key = (i["dataset"], i["problem_id"], int(i["sample_id"]), i["trace_sha256"])
            labels[key].append((int(i["sentence_index"]), int(i["start"]), int(i["end"]), row.get("label")))
    artifacts = []
    for n, source in enumerate(sources):
        aid = source["attempt_id"]
        key = (source["benchmark"], source["problem_id"], source["sample_id"], source["trace_sha256"])
        classes, reasoning, audit = token_ownership(read_source(source["source_location"]), labels[key],
                                                  source["trace_sha256"], tokens[n])
        path = root/model/"token_ownership"/(aid+".npz")
        artifact = npz(path, reasoning=reasoning, **{str(c): t for c, t in enumerate(classes)})
        artifacts.append({"attempt_id": aid, "artifact": artifact, "audit": audit,
                          "counts": list(map(int, tokens[n]))})
    return publish(root/model/"TOKEN_OWNERSHIP.json", {
        "status": "complete", "records": artifacts, "original_label_files": files,
        "sources_sha256": load(root/model/"SOURCES.json")["sha256"],
        "full_A_attempts": len(sources), "enrollment_changed": False,
        "causal_cohort_sha256": load(root/"COHORTS.json")["sha256"],
        "prefix_roster_sha256": load(root/model/"PREFIXES.json")["sha256"]})


def class_tokens(campaign, model, source, reasoning):
    from .contracts import check_files
    path = Path(campaign["root"])/model/"token_ownership"/(source["attempt_id"]+".npz")
    # Manifest digests are verified once by each worker; token agreement also
    # binds ownership to the exact native capture at every attempt.
    with np.load(path, allow_pickle=False) as z:
        require(np.array_equal(z["reasoning"], reasoning), "native reasoning ownership differs")
        return [z[str(c)].copy() for c in range(7)]


def validate_mapping(campaign, model):
    from .contracts import check_files
    manifest = load(Path(campaign["root"])/model/"TOKEN_OWNERSHIP.json")
    check_files(manifest["original_label_files"])
    check_files({r["artifact"]["path"]: r["artifact"]["sha256"] for r in manifest["records"]})
