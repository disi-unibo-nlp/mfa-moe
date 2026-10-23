"""Offline vLLM engine settings of the study profiles and the reasoning-closure marker.

Vendored verbatim from `verification_study.registry.engine_kwargs` and `verification_study.backend.closure_sequences`.
"""
from __future__ import annotations

from moe_exp.receipts import require


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


def closure_sequences(tokenizer, key):
    text = "<|channel|>final<|message|>" if key == "gpt" else "<channel|>" if key == "gemma" else "</think>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    require(ids and tokenizer.decode(ids, skip_special_tokens=False) == text,
            "pinned tokenizer cannot reproduce native closure marker")
    return [ids]
