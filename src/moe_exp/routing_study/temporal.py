"""Causal per-layer gates using the pilot's trailing-window reducer."""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from .common import digest
from .reducers import RoutingReducer


def weak_fraction(window, threshold):
    if "boundary_gaps" in window:
        return float(np.mean(np.asarray(window["boundary_gaps"]) <= threshold))
    histogram = window["gap_histogram"]
    total = sum(n for _,n in histogram)
    if total != 256:
        raise ValueError("Weak-boundary window must contain exactly 256 observations")
    return sum(n for value,n in histogram if value <= threshold) / total


def histogram_quantile(histograms, q):
    from collections import Counter
    counts = Counter()
    for histogram in histograms:
        for value,count in histogram:
            if not np.isfinite(value) or value < 0 or type(count) is not int or count <= 0:
                raise ValueError("Invalid unique-token gap histogram")
            counts[value] += count
    if not counts or not 0 <= q <= 1:
        raise ValueError("No native gap observations")
    ordered = sorted(counts)
    cumulative = np.cumsum([counts[v] for v in ordered])
    rank = q*(int(cumulative[-1])-1)
    lo,hi = int(np.floor(rank)), int(np.ceil(rank))
    left = ordered[int(np.searchsorted(cumulative,lo,side="right"))]
    right = ordered[int(np.searchsorted(cumulative,hi,side="right"))]
    return float(left+(right-left)*(rank-lo))


def fires(kind, window, thresholds):
    if window is None:
        return False
    if kind == "persistent":
        return (window["overlap_lag16"] > thresholds["overlap_q75"]
                and window["mixture_jsd"] < thresholds["jsd_q25"])
    if kind == "unstable":
        weak = weak_fraction(window, thresholds["gap_q10"])
        return window["set_turnover"] > thresholds["turnover_q75"] and weak > thresholds["weak_q75"]
    raise ValueError("Unknown temporal hypothesis")


@lru_cache(maxsize=65536)
def uniform(key, layer, position, family):
    return int(digest([key, layer, position, family])[:13], 16) / 16**13


class Gate:
    """Decide before token t using [t-256,t); never append recomputation twice."""
    def __init__(self, key, layer, kind="always", timing=None):
        self.key, self.layer, self.kind = key, layer, kind
        self.timing = timing or {}
        self.active_until = self.cooldown_until = 0
        self.decisions = {}

    def decide(self, position, window):
        if position in self.decisions:
            return self.decisions[position]
        if position < 0:
            raise ValueError("Negative reasoning position")
        if self.kind == "always":
            active = True
        elif position < 256 or window is None:
            active = False
        elif position < self.active_until:
            active = True
        elif position < self.cooldown_until or position % 64:
            active = False
        else:
            layer = self.timing["layers"][str(self.layer)]
            if self.kind in ("persistent", "unstable"):
                active = fires(self.kind, window, layer)
            else:
                control, family = self.kind.split("_", 1)
                spec = layer["controls"][self.kind]
                probability = spec["probability"]
                if control == "position":
                    probability = min(1., probability * spec["profile"].get(str(position // 64), 0.))
                active = uniform(self.key, self.layer, position, family) < probability
            if active:
                self.active_until, self.cooldown_until = position + 64, position + 128
        self.decisions[position] = active
        return active


class Features:
    def __init__(self, layer):
        self.reducer = RoutingReducer(256, 8, layer=layer, model="qwen36-fp8",
            semantics={"deterministic_topk": True})
        self.last = None

    def push(self, ids, weights, scores):
        completed = self.reducer.push(np.asarray(ids, dtype=np.int32).reshape(1, 8),
            weights=np.asarray(weights).reshape(1, 8),
            selection_scores=np.asarray(scores).reshape(1, 256))
        if completed:
            self.last = completed[-1]
        return completed


def simulate(key, layer, windows, length, kind, timing):
    gate = Gate(key, layer, kind, timing)
    by_end = {w["end_token"]: w for w in windows}
    active = 0
    decisions = {}
    for pos in range(256, length, 64):
        selected = gate.decide(pos, by_end.get(pos))
        decisions[pos] = selected
        active += min(64, length - pos) * selected
    return active, decisions


def fit_timing(samples, calibration_binding):
    """Pooled native-token gap quantiles, pooled trailing-window quantiles, per layer.

    Input samples contain unique, contiguous reasoning positions, never overlapping
    window copies of a gap. Controls fit deterministic schedules on calibration only.
    """
    result = {"calibration": calibration_binding, "layers": {},
              "aggregation": "per_layer_pooled_unique_reasoning_tokens_and_windows",
              "window": 256, "stride": 64, "duration": 64, "cooldown": 64}
    for layer in range(40):
        entries = [s["layers"][str(layer)] for s in samples]
        gap_parts = [e["gap_histogram"] for e in entries if e["gap_histogram"]]
        windows = [w for e in entries for w in e["windows"]]
        if not gap_parts or not windows:
            raise ValueError(f"Insufficient native history for layer {layer}")
        q10 = histogram_quantile(gap_parts, .1)
        thresholds = dict(gap_q10=q10,
            overlap_q75=float(np.quantile([w["overlap_lag16"] for w in windows], .75)),
            jsd_q25=float(np.quantile([w["mixture_jsd"] for w in windows], .25)),
            turnover_q75=float(np.quantile([w["set_turnover"] for w in windows], .75)),
            weak_q75=float(np.quantile([weak_fraction(w, q10) for w in windows], .75)), controls={})
        result["layers"][str(layer)] = thresholds
        for family in ("persistent", "unstable"):
            target, lengths, positions, triggered = 0, 0, {}, {}
            for sample, entry in zip(samples, entries):
                length = entry["reasoning_tokens"]
                n, decisions = simulate(sample["id"], layer, entry["windows"], length, family, result)
                target += n
                lengths += length
                for pos, active in decisions.items():
                    block = str(pos // 64)
                    positions[block] = positions.get(block, 0) + 1
                    triggered[block] = triggered.get(block, 0) + int(active)
            profile = {b: triggered[b] / n for b, n in positions.items()}
            for control in ("random", "position"):
                name = control + "_" + family
                spec = dict(probability=0., profile=profile if control == "position" else {},
                            nominal_rate=target / lengths if lengths else 0.)
                thresholds["controls"][name] = spec
                if target == 0:
                    spec["calibration_realized_rate"] = spec["calibration_rate_error"] = 0.
                    continue
                lo, hi, best = 0., (1. if control == "random" else
                                    1. / min((p for p in profile.values() if p > 0), default=1.)), None
                # Search a prespecified finite schedule; record discreteness mismatch.
                for probability in np.linspace(lo, hi, 257):
                    spec["probability"] = float(probability)
                    actual = sum(simulate(s["id"], layer, e["windows"], e["reasoning_tokens"],
                                          name, result)[0] for s, e in zip(samples, entries))
                    candidate = (abs(actual - target), float(probability), actual)
                    best = min(best, candidate) if best is not None else candidate
                spec["probability"] = best[1]
                spec["calibration_realized_rate"] = best[2] / lengths if lengths else 0.
                spec["calibration_rate_error"] = best[0] / lengths if lengths else 0.
    return result
