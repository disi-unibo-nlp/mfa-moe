"""Private vLLM 0.29 loader. Construction precedes allocation; no checkpoint copies.

The original router sees original IDs. Only the expert executor sees compact IDs.
Unsupported monolithic kernels fail before scientific generation.
"""
from __future__ import annotations

from copy import copy
import importlib
import inspect
import json
from pathlib import Path
import re

from .contracts import check_files, load, require

LAYER = re.compile(r"(?:layers|blocks)\.(\d+)\.")
EXPERT = re.compile(r"(\.experts\.)(\d+)(\.)")
MODEL_MODULES = ("qwen3_moe", "qwen3_next", "qwen3_5", "gpt_oss", "gemma4",
                 "glm4_moe", "glm4_moe_lite", "nemotron_h")
_ACTIVE = None
_REGISTERED = False


def layer_id(name):
    found = LAYER.search(name)
    require(found is not None, f"cannot bind routed tensor/layer: {name}")
    return int(found.group(1))


def mapping(keep, experts):
    require(keep == sorted(set(keep)) and keep and all(type(e) is int and 0 <= e < experts for e in keep),
            "keep-set is not a sorted bijective subset")
    return {e: i for i, e in enumerate(keep)}


def tensor_action(name, shape, masks, experts):
    """All weight, bias, packed-zero and scale suffixes follow the same ID map."""
    if ".experts." not in name:
        return name, None
    layer = layer_id(name)
    require(str(layer) in masks, f"expert tensor outside routed layer roster: {name}")
    keep = masks[str(layer)]
    ids = mapping(keep, experts)
    numbered = EXPERT.search(name)
    if numbered:
        original = int(numbered.group(2))
        require(0 <= original < experts, "checkpoint expert outside native range")
        if original not in ids:
            return None, None
        return name[:numbered.start(2)] + str(ids[original]) + name[numbered.end(2):], None
    require(len(shape) > 0 and shape[0] == experts, f"unclassified fused expert metadata: {name}/{shape}")
    return name, keep


def weight_iterator(checkpoint, masks, experts, *, compact=True, only_layer=None, ledger=None, native_layer_limit=None):
    import torch
    from safetensors import safe_open
    root = Path(checkpoint)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard = {}
    for name, shard in sorted(index.items()):
        if only_layer is not None and (LAYER.search(name) is None or layer_id(name) != only_layer or ".experts." not in name):
            continue
        require(Path(shard).name == shard, "checkpoint shard escapes pinned directory")
        by_shard.setdefault(shard, []).append(name)
    for shard, names in by_shard.items():
        with safe_open(root / shard, framework="pt", device="cpu") as f:
            for name in names:
                if name.startswith("mtp.") or (".experts." in name and native_layer_limit is not None
                                               and layer_id(name) >= native_layer_limit):
                    if ledger is not None:
                        ledger[name] = {"status": "NATIVE_DRAFT_SKIPPED"}
                    continue
                view = f.get_slice(name)
                shape = view.get_shape()
                target, keep = tensor_action(name, shape, masks, experts) if compact else (name, None)
                if target is None:
                    if ledger is not None:
                        ledger[name] = {"status": "OMITTED", "shape": shape}
                    continue
                if keep is not None:
                    # Read only retained first-axis slices, including packed scales.
                    weight = torch.cat([view[e:e + 1] for e in keep], dim=0)
                else:
                    weight = f.get_tensor(name)
                if ledger is not None:
                    ledger[name] = {"status": "LOADED", "target": target,
                                    "shape": shape, "loaded_shape": list(weight.shape)}
                yield target, weight


class RouterMask:
    def __init__(self, router, keep, experts, *, compact):
        self.router, self.original = router, router._compute_routing
        self.signature = inspect.signature(self.original)
        self.keep, self.experts, self.compact = keep, experts, compact
        self.calls, self.rows, self.removed_dispatch = 0, 0, 0
        self.mode = "masked"
        router._compute_routing = self.compute

    def native(self, bound, *, masked):
        import torch
        logits = bound.arguments["router_logits"]
        require(logits.shape[-1] == self.experts, "router dimension changed")
        prior = getattr(self.router, "e_score_correction_bias", None)
        if masked and len(self.keep) < self.experts:
            available = torch.zeros(self.experts, dtype=torch.bool, device=logits.device)
            available[self.keep] = True
            bound.arguments["router_logits"] = logits.masked_fill(~available, -float("inf"))
            if prior is not None:
                # Sigmoid(-inf)=0, but a correction bias can still select a removed ID.
                self.router.e_score_correction_bias = prior.masked_fill(~available, -float("inf"))
        try:
            return self.original(*bound.args, **bound.kwargs)
        finally:
            if prior is not None:
                self.router.e_score_correction_bias = prior

    def compute(self, *args, **kwargs):
        import torch
        bound = self.signature.bind(*args, **kwargs)
        weights, ids = self.native(bound, masked=self.mode != "all_keep")
        require(weights.shape == ids.shape and ids.ndim == 2 and ids.shape[1] == self.router.top_k,
                "native selected-slot geometry changed")
        require(bool(torch.isfinite(weights).all()), "nonfinite routing weights")
        if self.mode == "all_keep":
            require(not self.compact or len(self.keep) == self.experts,
                    "all-keep dispatch into compact subset forbidden")
            return weights, ids
        remap = torch.full((self.experts,), -1, dtype=torch.int64, device=ids.device)
        remap[self.keep] = torch.arange(len(self.keep), device=ids.device)
        mapped = remap[ids.long()]
        bad = int((mapped < 0).sum().item())
        self.removed_dispatch += bad
        require(bad == 0, "removed expert dispatched")
        require(bool((ids.sort(-1).values.diff(dim=-1) != 0).all()), "duplicate selected experts")
        self.calls += 1
        self.rows += len(ids)
        return weights, mapped.to(ids.dtype) if self.compact else ids

    def restore(self):
        self.router._compute_routing = self.original


class ConstructionHooks:
    def __init__(self, config):
        self.config = config
        self.originals, self.layers = [], {}
        self.enabled = True

    def install(self):
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.fused_moe import layer
        p = get_current_vllm_config().parallel_config
        require(not p.enable_expert_parallel and not p.enable_eplb and not p.enable_dbo,
                "expert parallelism/load balancing/DBO unsupported")
        original = layer.FusedMoEFactory
        signature = inspect.signature(original)

        def factory(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            lid = layer_id(a["prefix"])
            require(str(lid) in self.config["masks"] and lid not in self.layers,
                    "routed-layer binding missing or duplicate")
            E = a["num_experts"]
            require(E == self.config["experts"] and a["num_redundant_experts"] == 0 and
                    not a["fuse_shared_experts"] and a["zero_expert_type"] is None,
                    "unsupported routed geometry/shared fusion")
            keep = self.config["masks"][str(lid)]
            mapping(keep, E)
            original_args = dict(a)
            # Run vLLM's exact router constructor with original geometry.
            if a["router"] is None:
                a["router"] = layer.create_fused_moe_router(
                    top_k=a["top_k"], global_num_experts=E, eplb_state=None,
                    renormalize=a["renormalize"], use_grouped_topk=a["use_grouped_topk"],
                    num_expert_group=a["num_expert_group"], topk_group=a["topk_group"],
                    custom_routing_function=a["custom_routing_function"], scoring_func=a["scoring_func"],
                    routed_scaling_factor=1.0 if a["apply_routed_scale_to_output"] else a["routed_scaling_factor"],
                    e_score_correction_bias=a["e_score_correction_bias"], num_fused_shared_experts=0,
                    zero_expert_type=None, num_logical_experts=E, hash_indices_table=a["hash_indices_table"])
            if self.config["compact"]:
                a["num_experts"] = len(keep)
            runner = original(*bound.args, **bound.kwargs)
            route = RouterMask(runner.router, keep, E, compact=self.config["compact"])
            self.layers[lid] = {"runner": runner, "routing": route, "native_args": original_args,
                                "factory": original}
            return runner

        modules = [layer, importlib.import_module("vllm.model_executor.layers.fused_moe")]
        modules += [importlib.import_module("vllm.model_executor.models." + name) for name in MODEL_MODULES]
        for module in modules:
            if getattr(module, "FusedMoEFactory", None) is original:
                self.originals.append((module, original))
                module.FusedMoEFactory = factory
        return self

    def restore(self):
        for module, original in self.originals:
            module.FusedMoEFactory = original
        self.originals.clear()


def register_loader():
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader("expert_memory_v1")
    class ExpertMemoryLoader(DefaultModelLoader):
        def __init__(self, load_config):
            require(_ACTIVE is not None, "explicit memory worker must install hooks before loading")
            compatible = copy(load_config)
            compatible.load_format = "safetensors"
            compatible.model_loader_extra_config = {"enable_weights_track": True}
            super().__init__(compatible)

        def get_all_weights(self, model_config, model):
            require(not getattr(model, "secondary_weights", ()), "secondary checkpoint not qualified")
            return weight_iterator(model_config.model, _ACTIVE.config["masks"], _ACTIVE.config["experts"],
                                   compact=_ACTIVE.config["compact"], ledger=_ACTIVE.config["load_ledger"],
                                   native_layer_limit=_ACTIVE.config["native_layer_limit"])

        def load_weights(self, model, model_config):
            # GPT's native fused MXFP4 loader reshapes using its config expert count.
            # Replace only loader-side config references, after router construction.
            replaced = []
            if _ACTIVE.config["model"] == "gpt" and _ACTIVE.config["compact"]:
                K = len(next(iter(_ACTIVE.config["masks"].values())))
                require(all(len(x) == K for x in _ACTIVE.config["masks"].values()), "GPT nonuniform loader count")
                for module in model.modules():
                    cfg = getattr(module, "config", None)
                    if cfg is not None and hasattr(cfg, "num_local_experts"):
                        temporary = copy(cfg)
                        temporary.num_local_experts = K
                        replaced.append((module, cfg))
                        module.config = temporary
            try:
                super().load_weights(model, model_config)
            finally:
                for module, cfg in replaced:
                    module.config = cfg

    _REGISTERED = True


def activate(config):
    global _ACTIVE
    require(_ACTIVE is None, "neighboring loader session still active")
    check_files(config["runtime_files"])
    check_files(config["model_files"])
    _ACTIVE = ConstructionHooks({**config, "load_ledger": {}}).install()
    register_loader()
    return _ACTIVE


def deactivate():
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.restore()
        _ACTIVE = None
