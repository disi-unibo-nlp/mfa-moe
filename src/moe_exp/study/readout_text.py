"""Hashed text features for P-A1 (deterministic; no fitted state until `TextTransform.fit`).

One shared hash space of 2^18 buckets. Word tokens (`\\w+` runs and single punctuation marks, lower-cased)
give unigrams and bigrams; the whitespace-normalised lower-cased text gives character 3-, 4- and 5-grams.
The bucket is crc32(family-prefixed n-gram utf-8) mod 2^18. Term weights: 1 + log(count) (sublinear tf).
Fold-fitted: document frequencies over the TRAINING rows (idf = ln((1 + n) / (1 + df)) + 1, columns with
training df < 2 dropped), then one L2 normalisation of each row and a fixed block scale.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

import numpy as np
from scipy import sparse

N_BUCKETS = 1 << 18
MIN_DF = 2
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
SPACE_RE = re.compile(r"\s+")


def _bucket(prefix: str, gram: str) -> int:
    return zlib.crc32((prefix + gram).encode("utf-8")) & (N_BUCKETS - 1)


def ngrams(text: str) -> list[int]:
    """Bucket ids (with repetition) of the word 1-2-grams and character 3-5-grams of one sentence."""
    low = text.lower()
    words = TOKEN_RE.findall(low)
    ids = [_bucket("w1|", w) for w in words]
    ids += [_bucket("w2|", words[i] + " " + words[i + 1]) for i in range(len(words) - 1)]
    norm = SPACE_RE.sub(" ", low).strip()
    n = len(norm)
    for size in (3, 4, 5):
        prefix = f"c{size}|"
        ids += [_bucket(prefix, norm[i:i + size]) for i in range(n - size + 1)]
    return ids


def hash_counts(texts) -> sparse.csr_matrix:
    """Raw n-gram counts [n_texts, 2^18] (float32 CSR, duplicate buckets summed)."""
    rows, cols = [], []
    for i, t in enumerate(texts):
        ids = ngrams(t)
        rows.append(np.full(len(ids), i, dtype=np.int64))
        cols.append(np.asarray(ids, dtype=np.int64))
    n = len(rows)
    if n == 0:
        return sparse.csr_matrix((0, N_BUCKETS), dtype=np.float32)
    r, c = np.concatenate(rows), np.concatenate(cols)
    m = sparse.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, N_BUCKETS)).tocsr()
    m.sum_duplicates()
    m.sort_indices()
    return m


def _row_ids(x: sparse.csr_matrix) -> np.ndarray:
    return np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))


@dataclass
class TextTransform:
    """Fold-fitted tf-idf transform: `fit` on training rows, `transform` any rows."""
    idf: np.ndarray            # [2^18], 0 for dropped columns
    keep: np.ndarray           # bool [2^18]
    remap: np.ndarray          # compact column id or -1
    n_active: int
    scale: float

    @classmethod
    def fit(cls, counts: sparse.csr_matrix, train: np.ndarray, scale: float = 1.0) -> TextTransform:
        x = counts[np.asarray(train)]
        df = np.bincount(x.indices, minlength=N_BUCKETS).astype(np.float64)
        n = x.shape[0]
        keep = df >= MIN_DF
        idf = np.where(keep, np.log((1.0 + n) / (1.0 + df)) + 1.0, 0.0)
        remap = np.full(N_BUCKETS, -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        return cls(idf, keep, remap, int(keep.sum()), float(scale))

    def transform(self, counts: sparse.csr_matrix, rows: np.ndarray) -> sparse.csr_matrix:
        """tf-idf rows of `counts[rows]` restricted to the active columns, L2-normalised, times `scale`."""
        x = counts[np.asarray(rows)]
        ok = self.keep[x.indices]
        rid = _row_ids(x)[ok]
        cols = self.remap[x.indices[ok]]
        data = (1.0 + np.log(x.data[ok].astype(np.float64))) * self.idf[x.indices[ok]]
        norm = np.sqrt(np.bincount(rid, weights=data * data, minlength=x.shape[0]))
        data = data / np.where(norm[rid] > 0, norm[rid], 1.0) * self.scale
        indptr = np.r_[0, np.cumsum(np.bincount(rid, minlength=x.shape[0]))]
        return sparse.csr_matrix((data, cols.astype(np.int32), indptr.astype(np.int64)),
                                 shape=(x.shape[0], self.n_active))
