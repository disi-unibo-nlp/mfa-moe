"""Load a Hugging Face checkpoint in bfloat16 for the teacher-forced router capture (the weights stay as shipped)."""
from __future__ import annotations

from pathlib import Path

import torch


def load_model_and_tokenizer(
    model_id: str,
    revision: str,
    local_files_only: bool,
    offload_folder: str,
) -> tuple:
    """Load the pinned snapshot of a causal-LM checkpoint and its tokenizer.

    - device_map='auto' spreads the model over the visible GPUs; padding_side='left'.
    - the snapshot is resolved once, so the model and the tokenizer come from the same files; a local
      directory must be named after the revision.
    """
    if not Path(model_id).is_dir():
        from huggingface_hub import snapshot_download
        model_id = snapshot_download(model_id, revision=revision, local_files_only=local_files_only)
    elif Path(model_id).name != revision:
        raise ValueError("Local checkpoint directory does not match requested revision")

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    # Decoder-only models must pad on the left so generated tokens are
    # contiguous from the right side of the sequence.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    architectures = AutoConfig.from_pretrained(model_id, trust_remote_code=False).architectures or ()
    model_class = (
        AutoModelForImageTextToText
        if any(name.endswith("ForConditionalGeneration") for name in architectures)
        else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder=offload_folder,
        trust_remote_code=False,
    )
    model.eval()
    return model, tokenizer
