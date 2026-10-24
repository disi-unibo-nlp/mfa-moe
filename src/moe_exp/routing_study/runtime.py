"""Request isolation and compact native/actual dispatch diagnostics."""
from __future__ import annotations
from collections import Counter
import hashlib
import numpy as np
from .calibration import MARKERS
from .common import CHECKPOINT, seal
from .temporal import Features, Gate


def applies(policy, layer):
    action = policy["action"]
    if action == "native" or action == "identity" and policy["strength"] == 0:
        return False
    if action == "temperature":
        return True
    if action == "identity":
        return any(policy["parameters"].get("layers", {}).get(str(layer), {}).get("scores", []))
    if action == "margin":
        return str(layer) in policy["parameters"].get("layers", {})
    return any(l == layer for l, _ in policy["targets"])


class TokenHistory:
    """Actual token text, absolute positions, and observed marker completions.

    ASCII marker occurrences are detected across decoded token pieces. The event
    aligns to the token completing the marker, including multi-token markers.
    No opening/closing marker is synthesized for capped traces.
    """
    def __init__(self, prompt_ids, decode_piece):
        self.prompt_ids, self.decode_piece = list(prompt_ids), decode_piece
        self.tokens, self.layout = [], []
        self.tail, self.open, self.closed = "", False, False
        self.reasoning_count = 0
        self.occurrences = Counter({m: 0 for m in MARKERS})

    def observe(self, position, token):
        if position < len(self.tokens):
            if self.tokens[position] != token:
                raise ValueError("Recomputation token changed")
            return self.layout[position]
        if position != len(self.tokens):
            raise ValueError("Noncontiguous request token history")
        if position < len(self.prompt_ids) and token != self.prompt_ids[position]:
            raise ValueError("Rendered prompt changed")
        piece = self.decode_piece(token)
        text = self.tail + piece
        events = sorted((end, marker) for marker in MARKERS
                        for end in range(len(self.tail)+1, len(text)+1) if text[:end].endswith(marker))
        counts = {m: sum(marker == m for _,marker in events) for m in MARKERS}
        self.occurrences.update(counts)
        for _,marker in events:
            if marker == "<think>" and not self.closed:
                self.open = True
            elif marker == "</think>":
                self.open = False
                # Quoted markers in user/system prompts must not permanently close
                # the assistant's subsequent reasoning segment.
                if position >= len(self.prompt_ids):
                    self.closed = True
        generated = position >= len(self.prompt_ids)
        reasoning = generated and self.open and not counts["<think>"] and not self.closed
        rpos = self.reasoning_count if reasoning else None
        self.reasoning_count += int(reasoning)
        value = dict(position=position, token=token, generated=generated, reasoning=rpos,
                     markers={m: bool(counts[m]) for m in MARKERS})
        self.tokens.append(token)
        self.layout.append(value)
        self.tail = text[-max(map(len, MARKERS)):]
        return value


class LayerHistory:
    def __init__(self, key, layer, policy):
        self.layer, self.policy = layer, policy
        self.features = Features(layer)
        self.gate = Gate(key, layer, policy["gate"], policy["parameters"].get("timing"))
        self.seen, self.decisions, self.dispatched = {}, {}, set()
        self.counts, self.reasoning_counts = np.zeros(256, dtype=np.int64), np.zeros(256, dtype=np.int64)
        self.marker_tokens = Counter({m: 0 for m in MARKERS})
        self.marker_experts = {m: np.zeros(256, dtype=np.int64) for m in MARKERS}
        self.gap_histogram, self.windows = Counter(), []
        self.probability_gap_sum = 0.
        self.diagnostics = Counter(tokens=0, reasoning_tokens=0, active=0, changed_sets=0,
                                   changed_weights=0, recomputations=0, dispatch_calls=0)

    def plan(self, layout, ids, weights, scores):
        position = layout["position"]
        ids, weights, scores = np.asarray(ids), np.asarray(weights), np.asarray(scores)
        if (ids.shape != (8,) or len(set(ids.tolist())) != 8 or (ids < 0).any() or (ids >= 256).any()
                or weights.shape != (8,) or not np.isfinite(weights).all() or (weights < 0).any()
                or weights.sum() <= 0 or scores.shape != (256,) or not np.isfinite(scores).all()):
            raise ValueError("Invalid native dispatch signals")
        signature = hashlib.sha256(ids.astype(np.int32).tobytes() + weights.astype(np.float32).tobytes()
                                   + scores.astype(np.float32).tobytes()).hexdigest()
        if position in self.seen:
            if self.seen[position] != signature:
                raise ValueError("Native routing changed during recomputation")
            self.diagnostics["recomputations"] += 1
            return self.decisions[position]
        self.seen[position] = signature
        self.counts[ids] += 1
        self.diagnostics["tokens"] += 1
        for marker, present in layout["markers"].items():
            if present:
                self.marker_tokens[marker] += 1
                self.marker_experts[marker][ids] += 1
        rpos = layout["reasoning"]
        scope = self.policy["token_scope"]
        active = False
        if rpos is not None:
            if rpos != self.diagnostics["reasoning_tokens"]:
                raise ValueError("Reasoning stream gap")
            active = self.gate.decide(rpos, self.features.last)
            # Window is updated only after the decision for this position.
            windows = self.features.push(ids, weights, scores)
            self.windows.extend({**{k: w[k] for k in ("start_token", "end_token", "overlap_lag16", "set_turnover", "mixture_jsd")},
                "gap_histogram": sorted(Counter(w["boundary_gaps"]).items())} for w in windows)
            other = scores.copy()
            other[ids] = -np.inf
            gap = scores[ids].min() - other.max()
            if gap < -1e-5:
                raise ValueError("Native selections disagree with deterministic top-k scores")
            self.gap_histogram[float(max(0., gap))] += 1
            exp = np.exp(scores.astype(float) - scores.max())
            probs = exp / exp.sum()
            ordered = np.sort(probs)
            self.probability_gap_sum += float(ordered[-8] - ordered[-9])
            self.reasoning_counts[ids] += 1
            self.diagnostics["reasoning_tokens"] += 1
        elif scope == "historical_all_tokens":
            if self.policy["gate"] != "always":
                raise ValueError("Historical scope is a separately named static mode")
            active = True
        active = active and applies(self.policy, self.layer)
        self.decisions[position] = active
        return active

    def dispatch(self, layout, native_ids, native_weights, actual_ids, actual_weights, active):
        position = layout["position"]
        self.diagnostics["dispatch_calls"] += 1
        if position in self.dispatched:
            return
        self.dispatched.add(position)
        if position not in self.seen:
            raise RuntimeError("Dispatch bypassed native selection capture")
        native_ids, actual_ids = np.asarray(native_ids), np.asarray(actual_ids)
        native_weights, actual_weights = np.asarray(native_weights), np.asarray(actual_weights)
        if (len(set(actual_ids.tolist())) != 8 or (actual_ids < 0).any() or (actual_ids >= 256).any()
                or not np.isfinite(actual_weights).all() or (actual_weights < 0).any()
                or actual_weights.sum() <= 0):
            raise ValueError("Invalid actual dispatch")
        if active and self.policy["action"] in ("steer_promote", "steer_suppress"):
            targets = [e for layer,e in self.policy["targets"] if layer == self.layer]
            included = [e in actual_ids for e in targets]
            if ((self.policy["action"] == "steer_promote" and not all(included))
                    or (self.policy["action"] == "steer_suppress" and any(included))):
                raise RuntimeError("Steer target selection failed at actual dispatch (finite precision or unsupported selector)")
        changed_set = not np.array_equal(np.sort(native_ids), np.sort(actual_ids))
        if self.policy["action"] in ("rice", "temperature") and not np.array_equal(native_ids, actual_ids):
            raise RuntimeError("Weight-only policy changed selected IDs")
        unchanged = not active or self.policy["action"] == "native" or (
            self.policy["action"] == "identity" and self.policy["strength"] == 0)
        if unchanged and (not np.array_equal(native_ids, actual_ids)
                          or not np.array_equal(native_weights, actual_weights)):
            raise RuntimeError("Native/zero-strength/inactive routing changed")
        changed_weights = changed_set or not np.array_equal(
            native_weights[np.argsort(native_ids)], actual_weights[np.argsort(actual_ids)])
        self.diagnostics.update(active=int(active), changed_sets=int(changed_set),
                                changed_weights=int(changed_weights))

    def summary(self, full=False):
        if set(self.seen) != self.dispatched:
            raise RuntimeError("Selected tokens did not reach verified dispatch")
        result = dict(diagnostics=dict(self.diagnostics),
                      native_sha256=hashlib.sha256("".join(self.seen[p] for p in sorted(self.seen)).encode()).hexdigest(),
                      decisions_sha256=hashlib.sha256(bytes(self.decisions[p] for p in sorted(self.decisions))).hexdigest())
        if full:
            result.update(all_tokens=self.diagnostics["tokens"], all_counts=self.counts.tolist(),
                reasoning_tokens=self.diagnostics["reasoning_tokens"], reasoning_counts=self.reasoning_counts.tolist(),
                marker_tokens=dict(self.marker_tokens), marker_experts={m: a.tolist() for m, a in self.marker_experts.items()},
                gap_histogram=sorted(self.gap_histogram.items()), windows=self.windows,
                mean_probability_gap=self.probability_gap_sum / self.diagnostics["reasoning_tokens"] if self.gap_histogram else None)
        return result


class RequestHistory:
    def __init__(self, spec, policy, decode_piece):
        self.spec, self.policy = spec, policy
        self.tokens = TokenHistory(spec["prompt_token_ids"], decode_piece)
        self.layers = {l: LayerHistory(spec["id"], l, policy) for l in range(40)}

    def summary(self, full=False):
        return {str(l): state.summary(full) for l, state in self.layers.items()}

    def calibration_sample(self, study_sha256, backend):
        value = dict(id=self.spec["id"], problem_key=self.spec["problem_key"],
            is_correct=self.spec["is_correct"], checkpoint=CHECKPOINT, study_sha256=study_sha256,
            capture="verified_modular_dispatch", infrastructure_failure=False, backend=backend, code=self.policy["code"],
            marker_occurrences=dict(self.tokens.occurrences), layers=self.summary(full=True))
        return seal(value)
