"""Blinded sizing and shared-family paired inference; unknowns remain bounded."""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import beta

from .contracts import require, seal
from .protocol import ARMS, MODELS, SEED


def sample_size(discordant, n=32):
    require(n == 32 and len(discordant) == 21 and
            all(type(d) is int and 0 <= d <= n for d in discordant),
            "complete blinded 32-family/21-contrast discordance required")
    upper = [1.0 if d == n else float(beta.ppf(1 - .05 / 21, d + 1, n - d)) for d in discordant]
    q = max(upper)
    size = 32 * math.ceil(3.038**2 * q / (.05**2 * 32))
    require(640 <= size <= 3712, "sample-rule implementation differs")
    return seal({"schema": "expert-memory-final-size-v1", "discordant": discordant,
                 "pilot_families": n, "simultaneous_upper": upper, "q_upper": q,
                 "N": size, "direction_or_arm_accuracy_used": False,
                 "planned_generations_before_alias": size * 28,
                 "half_width": .05, "method": "simultaneous one-sided Clopper-Pearson"})


def blinded_discordance(scores):
    scores = np.asarray(scores, dtype=float)
    require(scores.shape == (32, 7, 4), "pilot assignment grid incomplete")
    require(bool((np.isnan(scores) | (scores == 0) | (scores == 1)).all()), "nonbinary pilot score")
    # Any unresolved pair is conservatively discordant for sizing.
    return [int((np.isnan(scores[:, m, 1]) | np.isnan(scores[:, m, c]) |
                 (scores[:, m, 1] != scores[:, m, c])).sum())
            for m in range(7) for c in (0, 2, 3)]


def paired_analysis(scores, *, draws=50000):
    """[family,model,arm]; signs state-balanced minus each control."""
    scores = np.asarray(scores, dtype=float)
    require(scores.ndim == 3 and scores.shape[1:] == (7, 4) and len(scores) >= 2,
            "all-seven paired family grid required")
    require(bool((np.isnan(scores) | (scores == 0) | (scores == 1)).all()), "nonbinary score")
    require(draws >= 1000, "too few shared-family replicates")
    lo, hi = np.nan_to_num(scores, nan=0.), np.nan_to_num(scores, nan=1.)
    lower = np.stack([lo[:, m, 1] - hi[:, m, c] for m in range(7) for c in (0, 2, 3)], axis=1)
    upper = np.stack([hi[:, m, 1] - lo[:, m, c] for m in range(7) for c in (0, 2, 3)], axis=1)
    rng = np.random.default_rng(SEED)
    boot_l, boot_u = [], []
    for start in range(0, draws, 256):
        indices = rng.integers(len(scores), size=(min(256, draws - start), len(scores)))
        boot_l.append(lower[indices].mean(1))
        boot_u.append(upper[indices].mean(1))
    boot_l, boot_u = np.concatenate(boot_l), np.concatenate(boot_u)
    tail = .05 / (2 * 21)
    rows = []
    for i, (m, c) in enumerate((m, c) for m in range(7) for c in (0, 2, 3)):
        degenerate = bool(np.ptp(boot_l[:, i]) == 0 or np.ptp(boot_u[:, i]) == 0)
        interval = None if degenerate else [float(np.quantile(boot_l[:, i], tail)),
                                            float(np.quantile(boot_u[:, i], 1 - tail))]
        rows.append({"model": MODELS[m], "control": ARMS[c], "families": len(scores),
                     "identified_difference": [float(lower[:, i].mean()), float(upper[:, i].mean())],
                     "simultaneous_interval": interval,
                     "uncertainty_status": "UNAVAILABLE_DEGENERATE" if degenerate else "MEASURED",
                     "unknown_pairs": int((np.isnan(scores[:, m, 1]) | np.isnan(scores[:, m, c])).sum())})
    return seal({"schema": "expert-memory-accuracy-analysis-v1", "contrasts": rows,
                 "draws": draws, "shared_family_indices": True, "multiplicity": "Bonferroni 21 two-sided intervals",
                 "accuracy_retention_claim": False, "random_scope": "conditional on frozen masks"})
