"""Focused fold-only lexical-reference invariance checks."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'r3e_clean_v2'))
from dynrt.d3c_features import Tokens, fold_shuffled_labels


def make(cls, tokid, pos, att):
    n = len(cls)
    ids = np.zeros((n, 2, 2), np.uint8)
    sent = np.arange(n, dtype=np.int64)
    return Tokens(ids, np.asarray(cls, np.int64), np.asarray(att, np.int64), sent,
                  np.asarray(tokid, np.int64), np.asarray(pos, np.int64),
                  np.asarray(att, np.int64), np.asarray(cls, np.int64),
                  np.ones(n, np.int64), int(max(att) + 1), 32)


def main():
    fit = make([0, 1, 2, 0, 1, 2, 6, 6],
               [10, 10, 11, 11, 10, 11, 10, 11],
               [0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 2, 2])
    member = np.asarray([[1, 1, 0]], float)
    a = fold_shuffled_labels(fit, fit, member, n_shuf=5)
    other = make([0, 1, 2, 0, 1, 2, 3, 4], fit.tokid, fit.pos, fit.att)
    b = fold_shuffled_labels(other, other, member, n_shuf=5)
    assert all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))
    assert all(set(x.tolist()) <= {0, 1, 2} for x in a[0])
    ext = make([6, 6, 6], [10, 999, 999], [0, 1, 7], [0, 1, 2])
    p = fold_shuffled_labels(fit, ext, member, n_shuf=5)
    ext_changed = make([3, 4, 5], ext.tokid, ext.pos, ext.att)
    q = fold_shuffled_labels(fit, ext_changed, member, n_shuf=5)
    assert all(np.array_equal(x, y) for x, y in zip(p[0], q[0]))
    assert all(set(x.tolist()) <= {0, 1, 2} for x in p[0])
    print('fold-only pseudo-class invariance PASS')


if __name__ == '__main__':
    main()
