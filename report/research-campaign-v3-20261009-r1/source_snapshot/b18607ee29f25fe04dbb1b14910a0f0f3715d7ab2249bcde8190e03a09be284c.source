"""Variable-count GPT MXFP4 loader, without padding or resident removed tensors."""
from __future__ import annotations

from copy import copy
from types import MethodType
from .contracts import check_files, require
from moe_exp.expert_memory_study import loader as base

_ACTIVE = None
_REGISTERED = False


def gpt_loader_count(name, masks, native_count):
    return len(masks[str(base.layer_id(name))]) if ".experts." in name else native_count


def activate(config):
    global _ACTIVE
    require(_ACTIVE is None, "concurrent loader scope")
    check_files(config["runtime_files"])
    check_files(config["model_files"])
    _ACTIVE = base.ConstructionHooks({**config, "load_ledger": {}}).install()
    register_loader()
    return _ACTIVE


def deactivate():
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.restore()
        _ACTIVE = None


def register_loader():
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader("research_memory_v2")
    class ResearchLoader(DefaultModelLoader):
        def __init__(self, load_config):
            require(_ACTIVE is not None, "private worker did not install construction hooks")
            compatible = copy(load_config)
            compatible.load_format = "safetensors"
            compatible.model_loader_extra_config = {"enable_weights_track": True}
            super().__init__(compatible)

        def get_all_weights(self, model_config, model):
            require(not getattr(model, "secondary_weights", ()), "secondary checkpoint unsupported")
            c = _ACTIVE.config
            return base.weight_iterator(model_config.model, c["masks"], c["experts"], compact=c["compact"],
                ledger=c["load_ledger"], native_layer_limit=c["native_layer_limit"])

        def load_weights(self, model, model_config):
            c = _ACTIVE.config
            replaced = []
            if c["model"] == "gpt" and c["compact"]:
                # vLLM 0.29 caches num_local_experts once per loader invocation.
                # Invoke its ORIGINAL MXFP4 loader per retained tensor, with the
                # corresponding native physical layer's count. Packed tensors
                # and scales follow the same retained-ID mapping.
                for module in model.modules():
                    if not hasattr(module, "_load_weights_mxfp4"):
                        continue
                    original = module.load_weights
                    cfg = module.config
                    def variable_count(self, weights, _original=original, _cfg=cfg):
                        loaded = set()
                        try:
                            for name, tensor in weights:
                                temp = copy(_cfg)
                                temp.num_local_experts = gpt_loader_count(name, c["masks"], c["experts"])
                                self.config = temp
                                loaded.update(_original(iter(((name, tensor),))))
                        finally:
                            self.config = _cfg
                        return loaded
                    replaced.append((module, original))
                    module.load_weights = MethodType(variable_count, module)
            try:
                super().load_weights(model, model_config)
            finally:
                for module, original in replaced:
                    module.load_weights = original
    _REGISTERED = True


def worker_class():
    from moe_exp.expert_memory_study.worker import get_worker_class
    from .adapters import CampaignWorker
    from .measurement import MeasurementWorker
    parent = get_worker_class()
    class ResearchWorker(parent, CampaignWorker, MeasurementWorker):
        def load_model(self, *, load_dummy_weights=False):
            original_activate, original_deactivate = base.activate, base.deactivate
            base.activate, base.deactivate = activate, deactivate
            try:
                super().load_model(load_dummy_weights=load_dummy_weights)
            finally:
                base.activate, base.deactivate = original_activate, original_deactivate
    return ResearchWorker


def __getattr__(name):
    if name == "ResearchWorker":
        return worker_class()
    raise AttributeError(name)
