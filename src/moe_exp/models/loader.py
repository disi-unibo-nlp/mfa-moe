from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rich.console import Console

console = Console()

QUANTIZATION_CHOICES = ["none", "bnb-4bit", "bnb-8bit"]


def _make_bnb_config(quantization: str) -> BitsAndBytesConfig | None:
    if quantization == "bnb-4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    if quantization == "bnb-8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _model_already_quantized(model_id: str, trust_remote_code: bool) -> bool:
    """Return True if the model config already contains a quantization_config."""
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return getattr(cfg, "quantization_config", None) is not None


def load_model_and_tokenizer(
    model_id: str,
    device: str = "cuda",
    trust_remote_code: bool = False,
    offload_folder: str = "offload",
    quantization: str = "none",
) -> tuple:
    """Load a HuggingFace causal-LM model and tokenizer.

    - Uses bfloat16 precision by default.
    - Supports bitsandbytes 4-bit / 8-bit quantization.
    - Skips user quantization when the model is already pre-quantized (e.g. FP8).
    - Uses device_map='auto' for GPU; falls back to CPU if device='cpu'.
    - Sets padding_side='left' for correct decoder-only batched generation.
    """
    console.print(f"[bold blue]Loading tokenizer:[/] {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    # Decoder-only models must pad on the left so generated tokens are
    # contiguous from the right side of the sequence.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = _make_bnb_config(quantization)
    if bnb_config and _model_already_quantized(model_id, trust_remote_code):
        console.print(
            f"[bold red]Warning:[/] Model is already pre-quantized. "
            f"Ignoring --quantization {quantization}; loading with native config."
        )
        bnb_config = None
    elif bnb_config:
        console.print(f"[bold yellow]Quantization:[/] {quantization}")

    if device == "cpu" and bnb_config is not None:
        console.print(
            "[bold red]Warning:[/] bitsandbytes quantization requires CUDA. "
            "Ignoring quantization on CPU."
        )
        bnb_config = None

    console.print(f"[bold blue]Loading model:[/] {model_id}")
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            offload_folder=offload_folder,
            trust_remote_code=trust_remote_code,
        )
        model = model.to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder=offload_folder,
            trust_remote_code=trust_remote_code,
            quantization_config=bnb_config,
        )

    model.eval()
    console.print("[green]Model loaded.[/]")
    return model, tokenizer
