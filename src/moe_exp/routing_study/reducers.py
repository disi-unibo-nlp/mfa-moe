"""Bounded trailing-window reduction of native routing signals.

All entropies/JSD use natural logarithms. Drift compares the first and second
halves of the same trailing window. Shuffles use only that window's available
past, so no future token can alter a saved feature.
"""
from __future__ import annotations
import numpy as np
DEFAULT_CONFIG = {"windows": [256], "stride": 64, "lags": [1, 16], "seed": 42, "shuffle_replicates": 0}


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    return -(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum(axis=-1)


def jsd(p, q):
    middle = (p + q) / 2
    return float(entropy(middle) - (entropy(p) + entropy(q)) / 2)


def probabilities(scores):
    scores = np.asarray(scores, dtype=np.float64)
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def overlap(ids, lag, width):
    """Mean intersection/k, using all available predecessors of window tokens."""
    right = np.arange(max(len(ids) - width, lag), len(ids))
    if len(right) == 0:
        return None
    return float((ids[right, :, None] == ids[right - lag, None, :]).any(axis=-1).mean())


class RoutingReducer:
    """One layer/contiguous segment; O((max window + max lag) * experts) state.

    push accepts arbitrary chunk sizes and returns newly completed windows.
    Outputs are handed to the caller, never retained by the reducer.
    """
    def __init__(self, num_experts, top_k, *, layer, model, config=None, semantics=None,
                 start_token=0):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.windows = tuple(sorted(set(self.config["windows"])))
        self.lags = tuple(sorted(set(self.config["lags"])))
        self.stride = self.config["stride"]
        if not self.windows or min(self.windows) < 2 or self.stride < 1 or min(self.lags) < 1:
            raise ValueError("Invalid windows/stride/lags")
        if any(w % self.stride for w in self.windows):
            raise ValueError("Windows must be multiples of stride")
        if not 1 <= top_k <= num_experts:
            raise ValueError("Invalid top-k")
        self.e, self.k, self.layer, self.model = num_experts, top_k, layer, model
        self.semantics = semantics or {}
        self.limit = max(self.windows) + max(self.lags)
        self.start, self.count = start_token, 0
        self.buffers = {}
        self.availability = None

    @property
    def retained_tokens(self):
        return len(self.buffers.get("ids", []))

    def push(self, ids, *, weights=None, full_probabilities=None, selection_scores=None):
        ids = np.asarray(ids)
        if ids.ndim != 2 or ids.shape[1] != self.k or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Expected integer [tokens, top_k] native selections")
        if np.any(ids < 0) or np.any(ids >= self.e) or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
            raise ValueError("Invalid/duplicate expert IDs")
        n = len(ids)
        incoming = {"ids": ids.astype(np.int32)}
        if full_probabilities is not None:
            p = np.asarray(full_probabilities, dtype=float)
            if p.shape != (n, self.e) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
                raise ValueError("Invalid full-router probability distribution")
            incoming["p"] = p
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if w.shape != ids.shape or not np.isfinite(w).all() or (w < 0).any() or (w.sum(1) <= 0).any():
                raise ValueError("Invalid executed mixture weights")
            mixture = np.zeros((n, self.e), dtype=float)
            np.put_along_axis(mixture, ids, w / w.sum(1, keepdims=True), axis=1)
            incoming["mixture"] = mixture
        if selection_scores is not None and self.semantics.get("deterministic_topk"):
            scores = np.asarray(selection_scores, dtype=float)
            if scores.shape != (n, self.e) or not np.isfinite(scores).all():
                raise ValueError("Invalid native selection scores")
            if self.k < self.e:
                selected_min = np.take_along_axis(scores, ids, axis=1).min(1)
                other = scores.copy()
                np.put_along_axis(other, ids, -np.inf, axis=1)
                gaps = selected_min - other.max(1)
                if (gaps < -1e-6).any():
                    raise ValueError("Native selection is incompatible with deterministic top-k scores")
                incoming["gaps"] = np.maximum(gaps, 0)
        available = tuple(sorted(incoming))
        if self.availability is not None and available != self.availability:
            raise ValueError("Signal availability changed within a segment")
        self.availability = available
        outputs, offset = [], 0
        while offset < n:
            length = min(n - offset, self.stride - self.count % self.stride)
            for name, values in incoming.items():
                old = self.buffers.get(name)
                new = values[offset:offset + length].copy()
                self.buffers[name] = new if old is None else np.concatenate((old, new))[-self.limit:]
            offset += length
            self.count += length
            if self.count % self.stride == 0:
                for width in self.windows:
                    if self.count >= width:
                        outputs.append(self._window(width))
        return outputs

    def _window(self, width):
        b, end = self.buffers, self.start + self.count
        # Independent seed at each endpoint makes chunking and prefix extension invariant.
        rng = np.random.default_rng([self.config["seed"], self.layer, end, width])
        row = dict(model=self.model, layer=self.layer, start_token=end - width,
                   end_token=end, window=width, token_count=width)
        counts = np.bincount(b["ids"][-width:].reshape(-1), minlength=self.e)
        row["expert_rates"] = {str(i): float(c / width) for i, c in enumerate(counts) if c}
        row["selection_entropy"] = float(entropy(counts / counts.sum()))
        row["boundary_gaps"] = b["gaps"][-width:].tolist() if "gaps" in b else None
        row["boundary_gap_mean"] = float(b["gaps"][-width:].mean()) if "gaps" in b else None
        row["weak_boundary_fraction"] = None  # fitted inside each training fold
        row["weak_boundary_status"] = "requires_training_fold_threshold" if "gaps" in b else "unavailable"
        for signal in ("p", "mixture"):
            prefix = "full" if signal == "p" else "mixture"
            for name in ("local_entropy", "marginal_entropy", "entropy_difference", "jsd", "shuffled_jsd"):
                row[prefix + "_" + name] = None
            if signal in b:
                values = b[signal][-width:]
                local, marginal = float(entropy(values).mean()), float(entropy(values.mean(0)))
                row.update({prefix + "_local_entropy": local, prefix + "_marginal_entropy": marginal,
                            prefix + "_entropy_difference": max(0., marginal - local),
                            prefix + "_jsd": jsd(values[:width // 2].mean(0), values[width // 2:].mean(0))})
                shuffles = [values[rng.permutation(width)] for _ in range(self.config["shuffle_replicates"])]
                row[prefix + "_shuffled_jsd"] = float(np.mean([
                    jsd(v[:width // 2].mean(0), v[width // 2:].mean(0)) for v in shuffles])) if shuffles else None
                if signal == "mixture":
                    row["mixture_effective_experts"] = float(np.exp(marginal))
        row.setdefault("mixture_effective_experts", None)
        for lag in self.lags:
            pool = b["ids"][-min(width + lag, self.count):]
            raw = overlap(pool, lag, width)
            c = np.bincount(pool.reshape(-1), minlength=self.e).astype(float)
            # Exact mean under uniform permutation of token routing sets, without replacement.
            chance = float(np.sum(c * (c - 1)) / (len(pool) * (len(pool) - 1) * self.k)) if len(pool) > 1 else None
            row[f"overlap_lag{lag}"] = raw
            row[f"overlap_corrected_lag{lag}"] = (raw - self.k / self.e) / (1 - self.k / self.e) if (
                raw is not None and self.k < self.e) else None
            row[f"overlap_shuffled_lag{lag}"] = chance if raw is not None else None
        row["set_turnover"] = 1 - row["overlap_lag1"] if row.get("overlap_lag1") is not None else None
        row["full_distribution_status"] = "available" if "p" in b else "unavailable"
        row["executed_mixture_status"] = "available" if "mixture" in b else "unavailable"
        row["selection_score_status"] = "available" if "gaps" in b else "unavailable"
        return row

