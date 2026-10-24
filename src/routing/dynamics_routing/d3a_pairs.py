"""Judge label records of the cohort-A sentences: text, label and span of each labelled sentence.

A record is keyed by (dataset, problem_id, sample_id, trace_sha256, sentence_index), the identity the label files of
the capture use. `load_unit_texts` reads the label parts of one model and checks that each record's unit agrees with
its identity.
"""
from __future__ import annotations

import json

from store import ANALYSIS

from .common import CLASSES  # noqa: F401  (read as `P.CLASSES` by routing.next_state_anticipation)

LABELS_ROOT = ANALYSIS / "campaign-v3/labels"


def load_unit_texts(model: str, wanted_keys: set[tuple]) -> dict[tuple, tuple]:
    """(dataset, problem_id, sample_id, trace_sha256, sentence_index) -> (text, label, start, end)."""
    out: dict[tuple, tuple] = {}
    for part in sorted((LABELS_ROOT / model).glob("part-*")):
        f = part / "annotations.json"
        if not f.exists():
            continue
        for rec in json.loads(f.read_text()):
            ident, unit = rec["identity"], rec["unit"]
            key = (ident["dataset"], ident["problem_id"], int(ident["sample_id"]), ident["trace_sha256"])
            if key not in wanted_keys:
                continue
            k5 = key + (int(ident["sentence_index"]),)
            if k5 in out:
                raise ValueError(f"duplicate label record {k5}")
            if int(unit["index"]) != int(ident["sentence_index"]) or unit["start"] != ident["start"] \
                    or unit["end"] != ident["end"]:
                raise ValueError(f"unit/identity disagree for {k5}")
            out[k5] = (unit["text"], rec.get("label"), int(ident["start"]), int(ident["end"]))
    return out
