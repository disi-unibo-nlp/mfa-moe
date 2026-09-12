from __future__ import annotations

import importlib
import os

import torch
from rich.console import Console

console = Console()

QUANTIZATION_CHOICES = ["none", "bnb-4bit", "bnb-8bit", "unsloth-4bit"]


def _is_conditional_generation_config(config: object) -> bool:
    architectures = tuple(getattr(config, "architectures", ()) or ())
    return any(architecture.endswith("ForConditionalGeneration") for architecture in architectures)


def _make_bnb_config(quantization: str) -> object | None:
    from transformers import BitsAndBytesConfig

    if quantization == "bnb-4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    if quantization == "bnb-8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _load_unsloth_model_and_tokenizer(model_id: str) -> tuple:
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_VENDORED_FLA", "1")
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    if not any(os.environ.get(name) for name in ("LOGNAME", "USER", "LNAME", "USERNAME")):
        os.environ["USER"] = f"uid-{os.getuid()}"
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    compile_location = os.environ.setdefault(
        "UNSLOTH_COMPILE_LOCATION",
        os.path.join(cache_home, "unsloth_compiled_cache"),
    )
    torchinductor_cache = os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join(cache_home, "torchinductor"),
    )
    os.makedirs(compile_location, exist_ok=True)
    os.makedirs(torchinductor_cache, exist_ok=True)

    # Torch 2.11 can leave compiler modules partially initialized when Unsloth
    # probes Inductor, causing duplicate mega-cache registrations on retry.
    importlib.import_module("torch._inductor.async_compile")

    try:
        from unsloth import FastModel
    except ImportError as error:
        raise RuntimeError(
            "--quantization unsloth-4bit requires the optional 'unsloth' dependency"
        ) from error

    console.print(f"[bold blue]Loading Unsloth model:[/] {model_id}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_id,
        max_seq_length=32768,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        text_only=True,
    )
    model.eval()
    model.config.use_cache = False
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    console.print("[green]Unsloth model loaded.[/]")
    return model, tokenizer


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
    if quantization == "unsloth-4bit":
        if device == "cpu":
            raise ValueError("Unsloth 4-bit loading requires a CUDA device")
        return _load_unsloth_model_and_tokenizer(model_id)

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

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

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_class = (
        AutoModelForImageTextToText
        if _is_conditional_generation_config(config)
        else AutoModelForCausalLM
    )

    bnb_config = _make_bnb_config(quantization)
    if bnb_config and getattr(config, "quantization_config", None) is not None:
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

    console.print(f"[bold blue]Loading model:[/] {model_id} with {model_class.__name__}")
    if device == "cpu":
        model = model_class.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            offload_folder=offload_folder,
            trust_remote_code=trust_remote_code,
        )
        model = model.to("cpu")
    else:
        quantization_kwargs = {"quantization_config": bnb_config} if bnb_config is not None else {}
        model = model_class.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder=offload_folder,
            trust_remote_code=trust_remote_code,
            **quantization_kwargs,
        )

    model.eval()
    console.print("[green]Model loaded.[/]")
    return model, tokenizer
