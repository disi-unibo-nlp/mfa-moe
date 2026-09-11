"""Loaded lazily by vLLM's registry inside each model worker."""

from vllm.model_executor.models import mixtral

from moe_exp.moe_guiding.config import RoutingConfig
from moe_exp.moe_guiding.integration import routing_construction, validate_vllm_config


class MoEGuidingMixtralForCausalLM(mixtral.MixtralForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        payload = (vllm_config.additional_config or {}).get("moe_guiding", {})
        config = RoutingConfig(**payload)
        layers = validate_vllm_config(vllm_config, config)
        with routing_construction(mixtral, config, layers) as routers:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.moe_guiding_config = config
        self.moe_guiding_layers = layers
        self.moe_guiding_routers = routers
