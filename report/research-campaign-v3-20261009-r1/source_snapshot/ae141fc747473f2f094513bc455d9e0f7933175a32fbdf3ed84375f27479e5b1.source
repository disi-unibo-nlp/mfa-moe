"""Process-local vLLM adapters preserving each existing native selector.

This module makes no backend qualification claim. Observed callback coverage,
same-state no-op parity, native k and tensor-parallel agreement are GPU gates.
"""
from __future__ import annotations

import inspect
import re

from .contracts import digest, require

ROUTER_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:mlp|moe|mixer|block_sparse_moe)\.experts$")


class PredictionClock:
    def __init__(self, prompt, prefix, closure):
        require(prompt and closure and all(s for s in closure), "prompt and native closure sequences required")
        self.prompt, self.prefix = list(prompt), list(prefix)
        self.saved = self.prompt + self.prefix
        self.closure = [tuple(s) for s in closure]
        self.tail, self.closed, self.last = [], False, -1
        self.contexts = []

    def observe(self, ids, positions):
        # Qwen3.5/3.6 text M-RoPE emits identical temporal/height/width
        # coordinates. Normalize only verified text axes; unequal multimodal
        # coordinates cannot define this study's absolute prediction clock.
        if positions and isinstance(positions[0], (list, tuple)):
            require(len(positions) in (3,4) and
                    all(isinstance(axis,(list,tuple)) and len(axis)==len(ids) and
                        all(type(p) is int and p>=0 for p in axis) for axis in positions) and
                    all(list(axis)==list(positions[0]) for axis in positions),
                    "non-text or malformed multi-axis token positions")
            positions=list(positions[0])
        require(len(ids) == len(positions), "token/position shape differs")
        require(all(type(p) is int and p>=0 for p in positions), "invalid absolute token positions")
        contexts = []
        for token, position in zip(ids, positions):
            require(position == self.last + 1, "noncontiguous request: padding/cache/neighboring request")
            self.last = position
            require(position >= len(self.saved) or token == self.saved[position], "runtime changed frozen input")
            if position >= len(self.prompt):
                self.tail.append(token)
                self.tail = self.tail[-max(map(len, self.closure)):]
                if any(tuple(self.tail[-len(s):]) == s for s in self.closure):
                    self.closed = True
            contexts.append({"output_position": position - len(self.prompt) + 1,
                             "reasoning": not self.closed})
        self.contexts = contexts
        return contexts


def discover_routers(model):
    found = {}
    for name, module in model.named_modules():
        match = ROUTER_NAME.search(name)
        router = getattr(module, "router", None)
        if match and router is not None and hasattr(router, "_compute_routing"):
            layer = int(match[1])
            require(layer not in found, "multiple routed expert modules for one decoder layer")
            quant = getattr(getattr(module, "routed_experts", None), "quant_method", None)
            found[layer] = (name, router, bool(getattr(quant, "is_monolithic", False)))
    return found


class NativeCapture:
    def __init__(self, model, contract, prompt, prefix, closure, *, action=None, capture_all=False, audit_noop=False):
        import torch
        self.torch = torch
        self.model, self.contract = model, contract
        self.clock = PredictionClock(prompt, prefix, closure)
        self.origin = len(prefix)
        self.action = action
        self.capture_all = capture_all
        self.audit_noop = audit_noop
        self.records, self.calibration_rows, self.parity = {}, {}, {}
        self.originals, self.handle = [], None
        routers = discover_routers(model)
        require(len(routers) == contract["expected_routed_layers"], "routed-layer count differs from checkpoint")
        self.geometry = []
        preflight = []
        for layer, (name, router, monolithic) in sorted(routers.items()):
            require(not monolithic, "monolithic kernel bypasses native callback: " + name)
            require(router.top_k == contract["native_k"] and router.global_num_experts == contract["experts"] and
                    getattr(router, "eplb_state", None) is None, "native geometry or expert identity differs")
            self.geometry.append({"layer": layer, "name": name, "router": type(router).__name__,
                                  "native_k": router.top_k, "experts": router.global_num_experts,
                                  "quant_method":type(getattr(getattr(model.get_submodule(name),"routed_experts",None),"quant_method",None)).__name__})
            if not capture_all and (action is None or layer != action["layer"]):
                continue
            original = router._compute_routing
            signature = inspect.signature(original)
            require("router_logits" in signature.parameters, "native router omits post-gate logits")
            preflight.append((layer, router, original, signature))
        require(preflight, "no requested native routing callback")
        if action is not None:
            self.validate_action(action)
        try:
            for layer, router, original, signature in preflight:
                self.originals.append((router, original))
                self.records[layer] = []
                self.calibration_rows[layer] = []
                self.parity[layer] = {"calls": 0, "evaluated_rows": 0, "ids_equal": True, "weights_equal": True}
                router._compute_routing = self._wrap(layer, original, signature)
            self.handle = model.register_forward_pre_hook(self._before, with_kwargs=True)
        except BaseException:
            self.restore()
            raise

    def validate_action(self, action):
        import math
        require(set(action) == {"layer", "expert", "bias", "start"} and
                type(action["layer"]) is int and type(action["expert"]) is int and
                0 <= action["expert"] < self.contract["experts"] and
                type(action["start"]) is int and action["start"] >= 0 and
                math.isfinite(action["bias"]) and action["bias"] >= 0,
                "invalid singleton logit action")

    def restore(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        for router, original in self.originals:
            router._compute_routing = original

    def _before(self, module, args, kwargs):
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        ids, positions = bound.arguments.get("input_ids"), bound.arguments.get("positions")
        require(ids is not None and positions is not None, "missing native token attribution")
        self.clock.observe(ids.detach().cpu().tolist(), positions.detach().cpu().tolist())

    def _wrap(self, layer, original, signature):
        torch = self.torch
        def compute(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            logits = bound.arguments["router_logits"]
            require(logits.ndim == 2 and logits.shape[-1] == self.contract["experts"] and
                    len(logits) == len(self.clock.contexts), "post-gate rows/experts differ")
            native_w, native_i = original(*args, **kwargs)
            require(native_i.shape == native_w.shape == (len(logits), self.contract["native_k"]), "native k changed")
            w, ids = native_w, native_i
            contexts = self.clock.contexts
            active = [bool(self.action and self.action["bias"] > 0 and layer == self.action["layer"] and c["reasoning"] and
                          self.action["start"] <= c["output_position"] < self.action["start"] + 256)
                      for c in contexts]
            if any(active) and self.action["bias"] != 0:
                changed = logits.clone()
                mask = torch.tensor(active, dtype=torch.bool, device=logits.device)
                changed[mask, self.action["expert"]] += self.action["bias"]
                bound.arguments["router_logits"] = changed
                changed_w, changed_i = original(*bound.args, **bound.kwargs)
                w = torch.where(mask[:, None], changed_w, native_w)
                ids = torch.where(mask[:, None], changed_i, native_i)
            else:
                # A second native evaluation at IDENTICAL hidden states audits
                # selector repeatability; the original tensors remain dispatched.
                if self.audit_noop:
                    other_w, other_i = original(*args, **kwargs)
                    self.parity[layer]["ids_equal"] &= bool(torch.equal(other_i, native_i))
                    self.parity[layer]["weights_equal"] &= bool(torch.equal(other_w, native_w))
                    self.parity[layer]["evaluated_rows"] += len(logits)
            self.parity[layer]["calls"] += 1
            keep = [j for j, c in enumerate(contexts) if c["output_position"] >= self.origin]
            if keep:
                index = torch.tensor(keep, device=logits.device)
                self.records[layer].append((
                    [contexts[j]["output_position"] for j in keep], [active[j] for j in keep],
                    native_i[index].detach().to(torch.int16), native_w[index].detach().float(),
                    ids[index].detach().to(torch.int16), w[index].detach().float()))
                # Calibration evidence is bounded and discovery/engineering only.
                if sum(len(v["router_logits"]) for v in self.calibration_rows[layer]) < 64:
                    native_arguments = signature.bind(*args, **kwargs).arguments
                    remaining = 64 - sum(len(v["router_logits"]) for v in self.calibration_rows[layer])
                    take = index[:remaining]
                    saved = {name: value[take].detach().clone() if isinstance(value, torch.Tensor) and
                             value.ndim > 0 and value.shape[0] == len(logits) else value
                             for name, value in native_arguments.items()}
                    self.calibration_rows[layer].append(saved)
            return w, ids
        return compute

    def set_action(self, action):
        self.validate_action(action)
        require(action["layer"] in self.records and action["start"] >= self.clock.last - len(self.clock.prompt) + 2,
                "action changes a past prediction or uninstrumented layer")
        require(self.action is None or self.action.get("bias", 0) == 0, "second pulse forbidden")
        self.action = dict(action)

    def finish(self):
        torch = self.torch
        self.restore()
        captured = {}
        for value in self.parity.values():
            value["status"] = "MEASURED" if value["evaluated_rows"] else "NOT_MEASURED"
        for layer, blocks in self.records.items():
            require(blocks, "native callback never captured output rows")
            positions = sum((b[0] for b in blocks), [])
            require(positions == list(range(self.origin, self.origin + len(positions))), "missing/duplicated prediction row")
            ni, nw, ai, aw = [torch.cat([b[j] for b in blocks]).cpu() for j in (2, 3, 4, 5)]
            require(bool(torch.isfinite(nw).all() and torch.isfinite(aw).all()) and
                    bool((nw >= 0).all() and (aw >= 0).all()), "invalid dispatched weight")
            require(bool((nw.sum(-1) > 0).all() and (aw.sum(-1) > 0).all()) and
                    bool((ni >= 0).all() and (ni < self.contract["experts"]).all() and
                         (ai >= 0).all() and (ai < self.contract["experts"]).all()),
                    "invalid expert identity or zero dispatched mass")
            require(bool((ni.sort(-1).values.diff(dim=-1) != 0).all() and
                         (ai.sort(-1).values.diff(dim=-1) != 0).all()), "duplicate native expert IDs")
            captured[layer] = {"positions": positions, "active": sum((b[1] for b in blocks), []),
                "native_ids": ni.tolist(), "native_weights": nw.tolist(),
                "executed_ids": ai.tolist(), "executed_weights": aw.tolist()}
        return {"geometry": self.geometry, "capture": captured, "same_state_noop": self.parity,
                "closed": self.clock.closed, "prediction_cursor": self.clock.last,
                "capture_sha256": digest(captured)}


class VerificationWorker:
    """vLLM extension RPCs, scoped to a single allocated request."""
    def verification_begin(self, contract, prompt, prefix, closure, action=None, capture_all=False, audit_noop=False):
        require(getattr(self, "_verification_capture", None) is None, "request already active")
        config = self.vllm_config
        require(config.model_config.enforce_eager and config.scheduler_config.max_num_seqs == 1 and
                not config.cache_config.enable_prefix_caching and config.speculative_config is None and
                not config.parallel_config.enable_expert_parallel and not config.parallel_config.enable_eplb,
                "unsupported request concurrency/cache/speculation/expert remapping")
        self._verification_capture = NativeCapture(self.model_runner.get_model(), contract,
            prompt, prefix, closure, action=action, capture_all=capture_all, audit_noop=audit_noop)
        return {"status": "ATTACHED", "geometry": self._verification_capture.geometry}

    def verification_action(self, action):
        self._verification_capture.set_action(action)
        return {"status": "ACKNOWLEDGED", "action": action}

    def verification_bank(self, uid, family, source_sha256):
        capture = self._verification_capture
        require(capture is not None and capture.action is not None and capture.action["bias"] == 0. and
                len(capture.records) == 1, "bank requires one inactive native discovery layer")
        layer = capture.action["layer"]
        bank = getattr(self, "_verification_bank", [])
        require(len(bank) < 32 and family not in {b["family"] for b in bank}, "calibration bank limit or duplicate family")
        original = next(original for router, original in capture.originals)
        require(capture.calibration_rows[layer], "no native calibration rows")
        bank.append({"uid": uid, "family": family, "source_sha256": source_sha256,
                     "layer": layer, "original": original, "arguments": capture.calibration_rows[layer]})
        self._verification_bank = bank
        return {"family": family, "rows": sum(len(v["router_logits"]) for v in bank[-1]["arguments"])}

    def verification_policy(self, contract, nomination_sha256, target, layer):
        import inspect
        import torch
        from .nomination import calibration
        from .contracts import seal
        bank = getattr(self, "_verification_bank", [])
        require(10 <= len(bank) <= 32 and all(b["layer"] == layer for b in bank), "calibration needs ten distinct discovery families")
        summaries = {}
        cache = {}
        score_scales = []
        for cell in bank:
            for values in cell["arguments"]:
                score_scales.append(float(values["router_logits"].float().std(-1).median()))
        scale = max(sorted(score_scales)[len(score_scales)//2], 1e-6)
        def evaluate(expert, bias):
            key = (expert, bias)
            if key in cache:
                return cache[key]
            families = []
            for cell in bank:
                native_counts = dose_counts = count = 0
                displacement = 0.
                for values in cell["arguments"]:
                    native_w, native_i = cell["original"](**values)
                    changed = dict(values)
                    changed["router_logits"] = values["router_logits"].clone()
                    changed["router_logits"][:, expert] += bias
                    action_w, action_i = cell["original"](**changed)
                    n = len(native_i); count += n
                    native_counts += int((native_i == expert).any(-1).sum())
                    dose_counts += int((action_i == expert).any(-1).sum())
                    # Align by expert ID before comparing dispatched weights.
                    native_dense = torch.zeros((n, contract["experts"]), device=native_i.device)
                    action_dense = torch.zeros_like(native_dense)
                    native_dense.scatter_(1, native_i.long(), native_w.float())
                    action_dense.scatter_(1, action_i.long(), action_w.float())
                    displacement += float(((action_dense-native_dense).abs().sum(-1) /
                                           native_dense.abs().sum(-1).clamp_min(1e-12)).sum())
                families.append((native_counts/count, (dose_counts-native_counts)/count, displacement/count))
            result = {"native_exposure": sum(v[0] for v in families)/len(families),
                      "increment": sum(v[1] for v in families)/len(families),
                      "relative_weight_displacement": sum(v[2] for v in families)/len(families)}
            cache[key] = result
            return result
        target_fit = calibration(lambda b: evaluate(target, b)["increment"], scale)
        pool, biases = [], {}
        reasons = []
        if target_fit["status"] == "PASS":
            target_stats = evaluate(target, target_fit["selected_bias"])
            for expert in range(contract["experts"]):
                if expert == target or abs(evaluate(expert, 0.)["native_exposure"] - target_stats["native_exposure"]) > .025:
                    continue
                fit = calibration(lambda b: evaluate(expert, b)["increment"], scale)
                if fit["status"] != "PASS":
                    continue
                stats = evaluate(expert, fit["selected_bias"])
                mismatch = abs(stats["relative_weight_displacement"] - target_stats["relative_weight_displacement"]) / max(target_stats["relative_weight_displacement"], 1e-12)
                if mismatch <= .20:
                    pool.append({"expert": expert, "bias": fit["selected_bias"], **stats, "displacement_mismatch": mismatch})
                    biases[str(expert)] = fit["selected_bias"]
            if not pool:
                reasons.append("NO_OPERATOR_MATCHED_RANDOM_POOL")
        else:
            target_stats = None
            reasons.append("TARGET_DOSE_UNQUALIFIED")
        receipt = seal({"schema": "verification-policy-v3", "status": "FAIL" if reasons else "PASS",
            "model_sha256": contract["sha256"], "nomination_sha256": nomination_sha256,
            "layer": layer, "target": target, "target_bias": target_fit["selected_bias"],
            "target_calibration": target_fit, "target_statistics": target_stats,
            "random_biases": biases, "matched_pool": pool, "failure_reasons": reasons,
            "calibration_families": [{k: b[k] for k in ("uid", "family", "source_sha256")} for b in bank],
            "operator": "native_logit_selection_bias", "causal_outcomes_used": False})
        self._verification_bank = []
        return receipt

    def verification_end(self):
        if getattr(self, "_verification_capture", None) is None:
            return {"status": "ALREADY_DETACHED"}
        try:
            return self._verification_capture.finish()
        finally:
            self._verification_capture = None
