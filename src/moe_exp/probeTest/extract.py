"""Teacher-forced boundary activation extraction for the gold corpus."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from moe_exp.probeTest.data import GoldResponse, label_counts, load_gold_responses


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
BOUNDARY_DEFINITION = "hidden state immediately preceding the first token of the gold unit"


@dataclass(frozen=True)
class PreparedResponse:
    input_ids: tuple[int, ...]
    boundary_positions: tuple[int, ...]
    response_token_indices: tuple[int, ...]
    prompt_tokens: int
    response_tokens: int


def _flat_token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected a single tokenized sequence")
        value = value[0]
    return [int(token_id) for token_id in value]


def _offset_pairs(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and value[0] and isinstance(value[0][0], list):
        if len(value) != 1:
            raise ValueError("Expected offsets for a single tokenized sequence")
        value = value[0]
    return [(int(start), int(end)) for start, end in value]


def token_containing_char(offsets: Sequence[tuple[int, int]], char_position: int) -> int:
    """Return the first real token whose character span reaches ``char_position``.

    Byte-level tokenizers can include preceding whitespace in the same token as
    the first characters of a sentence.  Looking at ``end > char_position``
    therefore preserves that token rather than incorrectly skipping it.
    """
    for token_index, (start, end) in enumerate(offsets):
        if end <= start:  # tokenizer-inserted special token
            continue
        if end > char_position:
            return token_index
    raise ValueError(f"No response token covers or follows character {char_position}")


def prepare_response(
    tokenizer: Any,
    response: GoldResponse,
    system_prompt: str | None = None,
) -> PreparedResponse:
    """Build prompt+gold-response token IDs and causal pre-unit positions."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": response.instruction})

    try:
        prompt_encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = _flat_token_ids(prompt_encoded)
    except (AttributeError, ValueError, TypeError):
        fallback = ""
        if system_prompt:
            fallback += f"System: {system_prompt}\n\n"
        fallback += f"User: {response.instruction}\n\nAssistant:"
        prompt_ids = _flat_token_ids(tokenizer(fallback, add_special_tokens=True))

    encoded_response = tokenizer(
        response.response_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    response_ids = _flat_token_ids(encoded_response)
    if "offset_mapping" not in encoded_response:
        raise ValueError(
            "The tokenizer did not return offset_mapping; use a fast tokenizer so gold "
            "sentence boundaries can be aligned exactly"
        )
    offsets = _offset_pairs(encoded_response["offset_mapping"])
    if len(response_ids) != len(offsets):
        raise ValueError(
            f"Token/offset length mismatch: {len(response_ids)} IDs vs {len(offsets)} offsets"
        )

    response_token_indices = tuple(
        token_containing_char(offsets, unit.char_start) for unit in response.units
    )
    boundary_positions = tuple(
        len(prompt_ids) + token_index - 1 for token_index in response_token_indices
    )
    if not prompt_ids or min(boundary_positions) < 0:
        raise ValueError(f"{response.response_id}: no prompt token precedes the first gold unit")
    if any(left > right for left, right in zip(boundary_positions, boundary_positions[1:])):
        raise ValueError(f"{response.response_id}: token boundaries are not monotonic")

    return PreparedResponse(
        input_ids=tuple(prompt_ids + response_ids),
        boundary_positions=boundary_positions,
        response_token_indices=response_token_indices,
        prompt_tokens=len(prompt_ids),
        response_tokens=len(response_ids),
    )


def _json_dump_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _tensor_save_atomic(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def _response_digest(response: GoldResponse) -> str:
    digest = hashlib.sha256()
    digest.update(response.response_id.encode())
    digest.update(b"\0")
    digest.update(response.instruction.encode())
    digest.update(b"\0")
    digest.update(response.response_text.encode())
    for unit in response.units:
        digest.update(b"\0")
        digest.update(unit.label.encode())
        digest.update(b"\0")
        digest.update(unit.text.encode())
    return digest.hexdigest()


def _make_quantization_config(name: str) -> Any | None:
    if name == "none":
        return None
    from transformers import BitsAndBytesConfig

    if name == "bnb-4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    if name == "bnb-8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unknown quantization mode: {name}")


def load_model_and_tokenizer(
    model_id: str,
    *,
    revision: str = "main",
    quantization: str = "bnb-4bit",
    trust_remote_code: bool = False,
    offload_dir: Path | None = None,
) -> tuple[Any, Any]:
    """Load Qwen3.6 as an image-text model, using text-only inputs."""
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    logger.info("Loading tokenizer %s (revision=%s)", model_id, revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for exact character-to-token alignment")

    config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    architectures = tuple(getattr(config, "architectures", ()) or ())
    is_conditional_generation = any(
        architecture.endswith("ForConditionalGeneration") for architecture in architectures
    )
    model_class = AutoModelForImageTextToText if is_conditional_generation else AutoModelForCausalLM

    quantization_config = _make_quantization_config(quantization)
    if getattr(config, "quantization_config", None) is not None:
        if quantization_config is not None:
            logger.warning(
                "Model config already contains quantization settings; ignoring --quantization %s",
                quantization,
            )
        quantization_config = None

    if offload_dir is not None:
        offload_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Loading %s with %s (quantization=%s)",
        model_id,
        model_class.__name__,
        quantization,
    )
    model = model_class.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder=str(offload_dir) if offload_dir is not None else None,
        quantization_config=quantization_config,
    )
    model.eval()
    return model, tokenizer


def _input_device(model: Any) -> torch.device:
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is not None and weight.device.type != "meta":
        return weight.device
    return next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")


def extract_boundary_activations(model: Any, prepared: PreparedResponse) -> torch.Tensor:
    """Run one causal forward and select only gold pre-unit states.

    Returns a CPU float32 tensor shaped ``(units, hidden_states, hidden_size)``.
    ``hidden_states`` includes the embedding output and every transformer layer,
    exactly as returned by ``output_hidden_states=True`` in the ACL reference.
    """
    device = _input_device(model)
    input_ids = torch.tensor([prepared.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    forward_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "use_cache": False,
        "return_dict": True,
    }
    signature = inspect.signature(model.forward)
    if "logits_to_keep" in signature.parameters:
        # Qwen3.6 has a 248k vocabulary.  Computing logits for every gold token
        # would dominate memory while being irrelevant to the hidden-state probe.
        forward_kwargs["logits_to_keep"] = 1

    with torch.inference_mode():
        outputs = model(**forward_kwargs)
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError("Model forward returned no hidden_states")

    expected_tokens = len(prepared.input_ids)
    selected_layers: list[torch.Tensor] = []
    for layer_index, hidden in enumerate(hidden_states):
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Unexpected hidden-state shape at index {layer_index}: {tuple(hidden.shape)}; "
                f"expected (1, {expected_tokens}, hidden_size)"
            )
        positions = torch.tensor(prepared.boundary_positions, device=hidden.device)
        selected = hidden[0].index_select(0, positions).to(device="cpu", dtype=torch.float32)
        selected_layers.append(selected)

    activations = torch.stack(selected_layers, dim=1).contiguous()
    del outputs, hidden_states, selected_layers, input_ids, attention_mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return activations


def _expected_shard_metadata(
    response: GoldResponse,
    prepared: PreparedResponse,
    *,
    model_id: str,
    revision: str,
    quantization: str,
    system_prompt: str | None,
    include_think_boundary_units: bool,
    shard_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "response_id": response.response_id,
        "response_sha256": _response_digest(response),
        "model_id": model_id,
        "model_revision": revision,
        "quantization": quantization,
        "system_prompt": system_prompt,
        "include_think_boundary_units": include_think_boundary_units,
        "boundary_definition": BOUNDARY_DEFINITION,
        "activation_file": shard_name,
        "n_units": len(response.units),
        "prompt_tokens": prepared.prompt_tokens,
        "response_tokens": prepared.response_tokens,
        "total_tokens": len(prepared.input_ids),
        "labels": [unit.label for unit in response.units],
        "paragraph_labels": [unit.paragraph_label for unit in response.units],
        "texts": [unit.text for unit in response.units],
        "char_spans": [[unit.char_start, unit.char_end] for unit in response.units],
        "response_token_indices": list(prepared.response_token_indices),
        "boundary_token_positions": list(prepared.boundary_positions),
    }


def _reusable_shard(
    activation_path: Path,
    metadata_path: Path,
    expected: dict[str, Any],
) -> dict[str, Any] | None:
    if not activation_path.is_file() or not metadata_path.is_file():
        return None
    with metadata_path.open(encoding="utf-8") as handle:
        actual = json.load(handle)
    keys = (
        "schema_version",
        "response_id",
        "response_sha256",
        "model_id",
        "model_revision",
        "quantization",
        "system_prompt",
        "include_think_boundary_units",
        "boundary_definition",
        "n_units",
    )
    mismatches = [key for key in keys if actual.get(key) != expected.get(key)]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            f"Existing shard {activation_path} is incompatible ({joined}). "
            "Use a fresh output directory for a different extraction configuration."
        )
    tensor = torch.load(activation_path, map_location="cpu", weights_only=True)
    shape = actual.get("activation_shape")
    if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != shape:
        raise ValueError(f"Existing shard failed shape validation: {activation_path}")
    return actual


def extract_gold_corpus(
    *,
    dataset_dir: Path,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = "main",
    quantization: str = "bnb-4bit",
    trust_remote_code: bool = False,
    system_prompt: str | None = None,
    include_think_boundary_units: bool = False,
    max_documents: int | None = None,
) -> Path:
    """Extract resumable per-response activation shards and return the manifest."""
    all_responses = load_gold_responses(dataset_dir, include_think_boundary_units=True)
    filtered_responses = load_gold_responses(dataset_dir, include_think_boundary_units=False)
    if max_documents is not None:
        if max_documents < 1:
            raise ValueError("max_documents must be positive")
        all_responses = all_responses[:max_documents]
        filtered_responses = filtered_responses[:max_documents]
    released_units = sum(len(response.units) for response in all_responses)
    responses = all_responses if include_think_boundary_units else filtered_responses
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    model: Any | None = None
    tokenizer: Any | None = None
    completed: list[dict[str, Any]] = []
    for response_index, response in enumerate(responses):
        # Tokenization is required to validate the resume signature.  Delay the
        # 35B model load until the first shard that actually needs computation.
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
            if not getattr(tokenizer, "is_fast", False):
                raise ValueError("A fast tokenizer is required for exact boundary alignment")
        prepared = prepare_response(tokenizer, response, system_prompt)
        stem = f"{response_index + 1:02d}-{response.response_id}"
        activation_path = shard_dir / f"{stem}.pt"
        metadata_path = shard_dir / f"{stem}.json"
        expected = _expected_shard_metadata(
            response,
            prepared,
            model_id=model_id,
            revision=revision,
            quantization=quantization,
            system_prompt=system_prompt,
            include_think_boundary_units=include_think_boundary_units,
            shard_name=activation_path.name,
        )
        reused = _reusable_shard(activation_path, metadata_path, expected)
        if reused is not None:
            logger.info(
                "[%d/%d] Reusing %s (%d units)",
                response_index + 1,
                len(responses),
                response.response_id,
                len(response.units),
            )
            completed.append(reused)
            continue

        if model is None:
            # Release the tokenizer instance and reload it together with the
            # model so both are guaranteed to use the same revision.
            model, tokenizer = load_model_and_tokenizer(
                model_id,
                revision=revision,
                quantization=quantization,
                trust_remote_code=trust_remote_code,
                offload_dir=output_dir / "offload",
            )
            prepared = prepare_response(tokenizer, response, system_prompt)
            expected = _expected_shard_metadata(
                response,
                prepared,
                model_id=model_id,
                revision=revision,
                quantization=quantization,
                system_prompt=system_prompt,
                include_think_boundary_units=include_think_boundary_units,
                shard_name=activation_path.name,
            )

        logger.info(
            "[%d/%d] Forward %s: %d tokens, %d gold units",
            response_index + 1,
            len(responses),
            response.response_id,
            len(prepared.input_ids),
            len(response.units),
        )
        activations = extract_boundary_activations(model, prepared)
        metadata = dict(expected)
        metadata.update(
            {
                "activation_shape": list(activations.shape),
                "activation_dtype": str(activations.dtype).removeprefix("torch."),
                "num_hidden_states": int(activations.shape[1]),
                "hidden_size": int(activations.shape[2]),
            }
        )
        _tensor_save_atomic(activations, activation_path)
        _json_dump_atomic(metadata, metadata_path)
        completed.append(metadata)
        del activations

        partial_manifest = _manifest_payload(
            dataset_dir=dataset_dir,
            model_id=model_id,
            revision=revision,
            quantization=quantization,
            system_prompt=system_prompt,
            include_think_boundary_units=include_think_boundary_units,
            released_units=released_units,
            responses=responses,
            completed=completed,
            status="partial",
        )
        _json_dump_atomic(partial_manifest, output_dir / "manifest.partial.json")

    manifest = _manifest_payload(
        dataset_dir=dataset_dir,
        model_id=model_id,
        revision=revision,
        quantization=quantization,
        system_prompt=system_prompt,
        include_think_boundary_units=include_think_boundary_units,
        released_units=released_units,
        responses=responses,
        completed=completed,
        status="complete",
    )
    manifest_path = output_dir / "manifest.json"
    _json_dump_atomic(manifest, manifest_path)
    return manifest_path


def _manifest_payload(
    *,
    dataset_dir: Path,
    model_id: str,
    revision: str,
    quantization: str,
    system_prompt: str | None,
    include_think_boundary_units: bool,
    released_units: int,
    responses: Iterable[GoldResponse],
    completed: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    response_list = list(responses)
    counts = label_counts(response_list)
    return {
        "schema_version": 1,
        "status": status,
        "dataset_dir": str(dataset_dir.resolve()),
        "model_id": model_id,
        "model_revision": revision,
        "quantization": quantization,
        "system_prompt": system_prompt,
        "include_think_boundary_units": include_think_boundary_units,
        "source_responses": "released DeepSeek-R1 gold traces (teacher forcing; no regeneration)",
        "boundary_definition": BOUNDARY_DEFINITION,
        "paper_reported_sentences": 3087,
        "released_units_before_filtering": released_units,
        "excluded_think_boundary_units": (
            0 if include_think_boundary_units else released_units - sum(counts.values())
        ),
        "loaded_documents": len(response_list),
        "loaded_units": sum(len(response.units) for response in response_list),
        "label_counts": counts,
        "completed_documents": len(completed),
        "shards": completed,
    }
