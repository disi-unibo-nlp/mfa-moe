"""GPT-OSS replay without the native MXFP4 Triton kernels."""

import torch


def dequantized_mxfp4_kwargs() -> dict:
    """Expand the checkpoint's MXFP4 values to BF16, offloading excess weights.

    This preserves the quantized checkpoint values; it does not recover the
    original pre-quantization model. Eager experts avoid the native Triton
    compiler failures observed on the RTX 5090 with Transformers 5.5.
    """
    from transformers import Mxfp4Config

    if not torch.cuda.is_available():
        raise RuntimeError("MXFP4 BF16 replay requires a CUDA GPU with CPU offload")
    from .replay_attention import register_replay_attention

    free_bytes, _ = torch.cuda.mem_get_info()
    return {
        "quantization_config": Mxfp4Config(dequantize=True),
        "experts_implementation": "eager",
        "attn_implementation": register_replay_attention(),
        # Retain conservative headroom for activations and offloaded layers.
        "max_memory": {0: int(free_bytes * 0.45), "cpu": "80GiB"},
    }
