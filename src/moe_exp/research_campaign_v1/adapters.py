"""Signed empirical expert-set pulses, native selectors and bounded captures."""
from __future__ import annotations

import inspect
import math
from .contracts import digest, require
from moe_exp.verification_study.adapters import PredictionClock, discover_routers


def validate_action(action, experts, layers=None):
    if action is None:
        return
    require(set(action) == {"layer", "experts", "bias", "start", "pulse_tokens"}, "invalid action schema")
    ids = action["experts"]
    require(type(action["layer"]) is int and (layers is None or action["layer"] in layers), "invalid physical layer")
    require(isinstance(ids, list) and ids == sorted(set(ids)) and ids and
            all(type(e) is int and 0 <= e < experts for e in ids), "invalid expert set")
    require(type(action["bias"]) in (int, float) and math.isfinite(action["bias"]), "invalid signed offset")
    require(type(action["start"]) is int and action["start"] >= 0 and
            type(action["pulse_tokens"]) is int and action["pulse_tokens"] > 0, "invalid pulse interval")


def pulse_active(action, position, reasoning):
    return bool(action and action["bias"] != 0 and reasoning and
                action["start"] <= position < action["start"] + action["pulse_tokens"])


class SignedCapture:
    def __init__(self, model, contract, prompt, prefix, closure, action, layers, window, audit=False):
        import torch
        self.clock = PredictionClock(prompt, prefix, closure)
        self.origin, self.action, self.window = len(prefix), action, window
        self.originals, self.records, self.parity, self.handle = [], {}, {}, None
        routers = discover_routers(model)
        require(len(routers) == contract["expected_routed_layers"], "native routed geometry mismatch")
        validate_action(action, contract["experts"], routers)
        requested = set(layers) | ({action["layer"]} if action else set())
        require(requested <= set(routers) and requested, "capture physical layer unavailable")
        try:
            for l in sorted(requested):
                name, router, monolithic = routers[l]
                require(not monolithic and router.top_k == contract["native_k"] and
                        router.global_num_experts == contract["experts"] and
                        getattr(router, "eplb_state", None) is None, "native selector not qualified")
                original = router._compute_routing
                signature = inspect.signature(original)
                require("router_logits" in signature.parameters, "router score access unavailable")
                self.originals.append((router, original))
                self.records[l] = []
                self.parity[l] = {"rows": 0, "noop_equal": True, "native_k": router.top_k}
                def wrap(*args, _l=l, _original=original, _signature=signature, **kwargs):
                    bound = _signature.bind(*args, **kwargs)
                    logits = bound.arguments["router_logits"]
                    contexts = self.clock.contexts
                    require(len(contexts) == len(logits), "router rows and token clock differ")
                    nw, ni = _original(*args, **kwargs)
                    active = [bool(action and _l == action["layer"] and pulse_active(
                        action, c["output_position"], c["reasoning"])) for c in contexts]
                    if any(active):
                        changed = logits.clone()
                        row = torch.tensor([j for j, a in enumerate(active) if a], device=logits.device)
                        col = torch.tensor(action["experts"], device=logits.device)
                        changed[row[:, None], col[None, :]] += action["bias"]
                        bound.arguments["router_logits"] = changed
                        aw, ai = _original(*bound.args, **bound.kwargs)
                        m = torch.tensor(active, device=logits.device)[:, None]
                        w, ids = torch.where(m, aw, nw), torch.where(m, ai, ni)
                    else:
                        w, ids = nw, ni
                    require(ids.shape == ni.shape and ids.shape[-1] == contract["native_k"], "native k changed")
                    require(bool(torch.isfinite(w).all()) and bool((ids.sort(-1).values.diff(dim=-1) != 0).all()), "invalid routing")
                    if audit:
                        ow, oi = _original(*args, **kwargs)
                        self.parity[_l]["noop_equal"] &= bool(torch.equal(oi, ni) and torch.equal(ow, nw))
                    self.parity[_l]["rows"] += len(ids)
                    keep = [j for j, c in enumerate(contexts) if self.origin <= c["output_position"] < self.origin + window]
                    if keep:
                        ix = torch.tensor(keep, device=logits.device)
                        self.records[_l].append({"positions": [contexts[j]["output_position"] for j in keep],
                            "active": [active[j] for j in keep], "native_ids": ni[ix].detach().to(torch.int16).cpu().tolist(),
                            "native_weights": nw[ix].detach().float().cpu().tolist(),
                            "executed_ids": ids[ix].detach().to(torch.int16).cpu().tolist(),
                            "executed_weights": w[ix].detach().float().cpu().tolist()})
                    return w, ids
                router._compute_routing = wrap
            def before(module, args, kwargs):
                bound = inspect.signature(module.forward).bind(*args, **kwargs)
                self.clock.observe(bound.arguments["input_ids"].detach().cpu().tolist(),
                                   bound.arguments["positions"].detach().cpu().tolist())
            self.handle = model.register_forward_pre_hook(before, with_kwargs=True)
        except BaseException:
            self.restore()
            raise

    def restore(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        for router, original in self.originals:
            router._compute_routing = original

    def finish(self):
        self.restore()
        result = {}
        for layer, blocks in self.records.items():
            merged = {k: sum((b[k] for b in blocks), []) for k in (
                "positions", "active", "native_ids", "native_weights", "executed_ids", "executed_weights")}
            require(merged["positions"] == list(range(self.origin, self.origin + len(merged["positions"]))), "token capture gap")
            result[str(layer)] = merged
        return {"capture": result, "parity": self.parity, "closed": self.clock.closed,
                "capture_sha256": digest(result)}


class CampaignWorker:
    def campaign_begin(self, **kwargs):
        require(getattr(self, "_campaign_capture", None) is None, "request already instrumented")
        cfg = self.vllm_config
        require(cfg.scheduler_config.max_num_seqs == 1 and cfg.speculative_config is None and
                cfg.model_config.enforce_eager and not cfg.cache_config.enable_prefix_caching, "unsupported execution concurrency")
        self._campaign_capture = SignedCapture(self.model_runner.get_model(), **kwargs)
        return {"status": "attached"}

    def campaign_end(self):
        require(self._campaign_capture is not None, "capture missing")
        try:
            return self._campaign_capture.finish()
        finally:
            self._campaign_capture = None
