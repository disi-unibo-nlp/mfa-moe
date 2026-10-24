"""Loaders for the saved cohort tables: attempts (without outcome columns), sentences and class histograms."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import RESULTS, V3R2, cohort_dir, keep_mask
from .features import NC, SentenceTable


@dataclass
class Cohort:
    model: str
    cohort: str
    attempts: pd.DataFrame          # outcome-free attempt table, row i = attempt index i
    sent: pd.DataFrame              # sentence rows (sorted by attempt, sentence)
    table: SentenceTable
    hist: dict                      # first16 / last16 histograms and their row keys
    shares: np.ndarray              # [A, 7] class shares of labelled sentences
    n_labelled: np.ndarray


def load_attempts(model: str, cohort: str) -> pd.DataFrame:
    """Attempts table of the capture restricted to the analysed questions, WITHOUT outcome columns."""
    att = pd.read_parquet(V3R2 / model / cohort / "attempts.parquet")
    att = att[keep_mask(model, att["question"].to_numpy())]
    att = att.sort_values("attempt_id").reset_index(drop=True)
    keep = ["attempt_id", "question", "dataset", "problem_id", "source_problem_id", "sample_id",
            "trace_sha256", "completion_tokens", "n_reasoning", "capped", "termination",
            "labelled_units", "labelled_tokens", "sentence_units"]
    return att[[c for c in keep if c in att.columns]]


def load_cohort(model: str, cohort: str, root: Path | None = None) -> Cohort:
    d = cohort_dir(model, cohort, root or RESULTS)
    att = load_attempts(model, cohort)
    sent = pd.read_parquet(d / "sentences.parquet")
    z = np.load(d / "hist.npz")
    h = z["h"]
    n, layers, experts = h.shape
    a_index = pd.Index(att["attempt_id"]).get_indexer(sent["attempt_id"])
    if (a_index < 0).any():
        raise ValueError("sentence rows of attempts absent from the attempt table")
    table = SentenceTable(h.reshape(n, -1).astype(np.float32), sent["cls"].to_numpy(np.int64),
                          a_index.astype(np.int64), sent["abs_bin"].to_numpy(np.int64),
                          _rel_bin(sent["rel_pos"].to_numpy()), sent["n_tokens"].to_numpy(np.int64),
                          layers, experts, len(att))
    counts = np.zeros((len(att), NC))
    np.add.at(counts, (a_index, sent["cls"].to_numpy(np.int64)), 1.0)
    n_lab = counts.sum(1)
    shares = counts / np.maximum(n_lab, 1)[:, None]
    hist = {k: z[k] for k in ("first16", "first16_rows", "last16", "last16_rows")}
    return Cohort(model, cohort, att, sent, table, hist, shares, n_lab)


def _rel_bin(rel: np.ndarray) -> np.ndarray:
    from .common import rel_bin
    return rel_bin(rel).astype(np.int64)
