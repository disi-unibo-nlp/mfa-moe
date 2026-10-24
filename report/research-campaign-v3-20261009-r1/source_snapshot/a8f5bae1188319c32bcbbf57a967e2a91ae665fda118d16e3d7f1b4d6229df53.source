"""Checkpoint-specific geometry and offline execution bindings."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from moe_exp.correlation_pipeline.model_profiles import CARD_PROFILES
from .contracts import file_sha, require, seal

PROJECT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe")
REPO = PROJECT / "repo"
MODEL_KEYS = ("gpt", "nemotron", "qwen35", "qwen36", "gemma", "glm", "qwen330b")
GEOMETRY = {
    "gpt": (24, 32, 4), "nemotron": (23, 128, 6),
    "qwen35": (40, 256, 8), "qwen36": (40, 256, 8),
    "gemma": (30, 128, 8), "glm": (46, 64, 4), "qwen330b": (48, 128, 8),
}


def checkpoint_path(key):
    require(key in MODEL_KEYS, "unknown model")
    card = CARD_PROFILES[key]
    return Path(card["weights"]) if card["weights"] else PROJECT / "cache/hf/hub" / (
        "models--" + card["model"].replace("/", "--")) / "snapshots" / card["revision"]


def model_contract(key):
    card = deepcopy(CARD_PROFILES[key])
    layers, experts, k = GEOMETRY[key]
    path = checkpoint_path(key)
    bindings = {str(path / name): file_sha(path / name) for name in
                ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json") if (path / name).is_file()}
    require((path / "config.json").is_file(), "local pinned checkpoint unavailable: " + str(path))
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    return seal({"schema": "verification-model-v3", "key": key, "checkpoint": card["model"],
        "revision": card["revision"], "local_path": str(path), "config_files": bindings,
        "family": text.get("model_type", config.get("model_type")),
        "expected_routed_layers": layers, "experts": experts, "native_k": k,
        "routed_layer_map": "BOUND_BY_WORKER_QUALIFICATION", "shared_paths": "NATIVE_UNCHANGED",
        "sampler": {n: v for n, v in card["sampler"].items() if v is not None},
        "template_kwargs": card["template_kwargs"], "reasoning_effort": card["reasoning_effort"],
        "reasoning_parser": card["parser"], "native_context_limit": card["max_model_len"],
        "attention": card["attention"], "language_model_only": card["language_model_only"],
        "extra_args": card["extra_args"], "speculation": "DISABLED_FOR_TIMING_QUALIFICATION",
        "engine_changes": ["single request", "eager", "no prefix cache", "no speculation"],
        "weight_semantics": "native dispatched weights; do not assume unit mass"})


def engine_kwargs(contract, *, context=32768):
    require(context <= contract["native_context_limit"], "context exceeds checkpoint limit")
    extra = contract["extra_args"]
    mapped = {}
    names = {"--linear-backend": "linear_backend", "--mamba-backend": "mamba_backend",
             "--mamba-cache-mode": "mamba_cache_mode"}
    for i in range(0, len(extra), 2):
        require(extra[i] in names, "unmapped native engine option")
        mapped[names[extra[i]]] = extra[i + 1]
    return {"model": contract["local_path"], "tokenizer": contract["local_path"],
        "tensor_parallel_size": 2, "dtype": "bfloat16", "kv_cache_dtype": "bfloat16",
        "max_model_len": context, "max_num_seqs": 1, "enforce_eager": True,
        "enable_prefix_caching": False, "gpu_memory_utilization": .80,
        "max_num_batched_tokens": 2048, "language_model_only": contract["language_model_only"],
        "async_scheduling": False, "generation_config": "vllm",
        **({"moe_backend": "marlin"} if contract["key"] in ("gpt", "gemma", "nemotron") else {}),
        **({"attention_config": {"backend": contract["attention"]}} if contract["attention"] else {}),
        **mapped}
