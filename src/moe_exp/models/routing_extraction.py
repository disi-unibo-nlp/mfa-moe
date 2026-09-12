import argparse
import hashlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from moe_exp.models.inference import extract_logs_single_pass
from moe_exp.models.loader import QUANTIZATION_CHOICES, load_model_and_tokenizer
from moe_exp.schemas import TraceRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _save_tensor_atomic(tensor: torch.Tensor, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def _tensor_audit(
    tensor: torch.Tensor,
    *,
    persisted_dtype: torch.dtype,
    path: Path | None,
) -> dict[str, Any]:
    element_count = int(tensor.numel())
    projected_bytes = element_count * torch.empty((), dtype=persisted_dtype).element_size()
    return {
        "shape": list(tensor.shape),
        "source_dtype": str(tensor.dtype).removeprefix("torch."),
        "persisted_dtype": str(persisted_dtype).removeprefix("torch."),
        "element_count": element_count,
        "projected_serialized_payload_bytes": projected_bytes,
        "persisted": path is not None,
        "serialized_bytes": path.stat().st_size if path is not None and path.is_file() else 0,
    }


def compute_selected_experts(router_logits: torch.Tensor, top_k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Derive selected experts and their weights from router logits.

    Args:
        router_logits: (num_layers, seq_len, num_experts)
        top_k: number of experts selected per token (OLMoE uses top-8 of 64)

    Returns:
        selected_experts: (num_layers, seq_len, top_k) - indices of selected experts
        expert_weights: (num_layers, seq_len, top_k) - normalized weights for each
    """
    # Softmax over experts to get routing probabilities
    probs = F.softmax(router_logits, dim=-1)
    # Select top-k experts per token per layer
    weights, indices = torch.topk(probs, k=top_k, dim=-1)
    # Normalize weights to sum to 1 over the selected experts
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return indices, weights


def process_file(
    input_path: Path,
    model_id: str,
    output_path: Path,
    limit: int | None = None,
    top_k: int | None = None,
    extract_hidden_states: bool = False,
    quantization: str = "none",
    model: Any | None = None,
    tokenizer: Any | None = None,
    strict_top_k: bool = False,
    layer_indices: list[int] | None = None,
    save_expert_weights: bool = True,
    feature_reducer: Callable[
        [torch.Tensor, torch.Tensor | None, torch.Tensor, list[int] | None],
        dict[str, Any],
    ]
    | None = None,
    feature_schema_version: int | None = None,
    feature_config: dict[str, Any] | None = None,
    save_raw_tensors: bool = True,
    view_reducer: Callable[..., dict[str, Any]] | None = None,
    view_config: dict[str, Any] | None = None,
    trace_annotations: dict[str, dict[str, Any]] | None = None,
):
    """
    Run the offline extraction loop over traces to compute routing dynamics.
    Saves per-trace tensors: router_logits, selected_experts, expert_weights,
    and optionally hidden_states for linear probing.

    top_k: number of experts selected per token. When None, it is read from the
    model config (num_experts_per_tok), so it is correct for any MoE model
    (OLMoE=8, Qwen1.5-MoE=4, …). An explicit value overrides the config.
    layer_indices: optional original layer indices to retain in every saved
    tensor. The forward pass still computes all layers, but unselected tensors
    are discarded before CPU storage and serialization.
    save_expert_weights: whether to serialize the normalized top-k routing
    weights. Selected expert IDs are saved regardless.
    feature_reducer: optional callback that reduces raw tensors to compact,
    JSON-serializable per-trace features before the tensors are released.
    feature_schema_version: required with feature_reducer and included in the
    content-addressed checkpoint.
    feature_config: JSON-serializable reducer settings that must invalidate a
    checkpoint when changed (for example, a geometry sampling limit).
    save_raw_tensors: whether to persist router logits and hidden states. Set
    this to False only with feature_reducer; selected expert IDs remain saved.
    """
    if not save_raw_tensors and feature_reducer is None:
        raise ValueError("save_raw_tensors=False requires a feature_reducer")
    if feature_reducer is not None and feature_schema_version is None:
        raise ValueError("feature_schema_version is required with feature_reducer")
    if view_reducer is not None and feature_reducer is None:
        raise ValueError("view_reducer requires the compact feature_reducer")
    traces: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    if limit is not None:
        traces = traces[:limit]

    if not traces:
        raise RuntimeError(
            f"Input {input_path} contains zero traces; refusing to load {model_id}."
        )

    if (model is None) != (tokenizer is None):
        raise ValueError("model and tokenizer must be provided together")
    if model is None:
        logger.info(f"Loading model {model_id}")
        model, tokenizer = load_model_and_tokenizer(model_id, quantization=quantization)
    else:
        logger.info("Using preloaded model %s", model_id)

    if layer_indices is not None:
        layer_indices = sorted(set(layer_indices))
        if not layer_indices or layer_indices[0] < 0:
            raise ValueError("layer_indices must contain non-negative layer indices")

    from moe_exp.models.router_adapters import (
        ROUTER_CAPTURE_VERSION, SIGMOID_ROUTERS, available_router_layers,
        configured_top_k, model_family,
    )
    config_top_k = configured_top_k(model.config)
    native_routing = available_router_layers(model) is not None
    if top_k is None:
        if config_top_k is None:
            raise ValueError(
                f"Could not infer top_k from {model_id} config "
                "(no num_experts_per_tok); pass --top-k explicitly."
            )
        top_k = int(config_top_k)
        logger.info(f"Using top_k={top_k} from model config")
    elif config_top_k is not None and top_k != config_top_k:
        if strict_top_k:
            raise ValueError(
                f"Requested top_k={top_k} differs from model config "
                f"num_experts_per_tok={config_top_k}"
            )
        logger.warning(
            f"Requested top_k={top_k} differs from model config "
            f"num_experts_per_tok={config_top_k}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_dir = output_path.parent / "tensors"
    tensor_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Processing {len(traces)} traces for offline routing metrics computation...")
    tmp_output_path = output_path.with_name(f".{output_path.name}.tmp")
    n_with_routing = 0
    with open(tmp_output_path, "w", encoding="utf-8") as out_f:
        for trace_dict in tqdm(traces, desc="Extracting Routing"):
            trace = TraceRecord(**trace_dict)
            if native_routing:
                trace.metadata["router_distribution"] = (
                    "normalized_sigmoid" if model_family(model) in SIGMOID_ROUTERS else "softmax"
                )
            if trace_annotations is not None:
                from moe_exp.correlation_pipeline.spans import validate_annotation
                annotation = trace_annotations.get(trace.problem_id)
                if annotation is None:
                    raise ValueError(f"Missing reasoning annotation for {trace.problem_id}")
                validate_annotation(trace, annotation)
                trace.metadata["reasoning_annotation"] = annotation
            
            if not trace.cot_text.strip():
                out_f.write(trace.model_dump_json() + "\n")
                continue
                
            safe_problem_id = trace.problem_id.replace("/", "_").replace("\\", "_")
            trace_id = f"{trace.dataset}_{safe_problem_id}"
            logits_path = tensor_dir / f"{trace_id}_logits.pt"
            hidden_path = tensor_dir / f"{trace_id}_hidden.pt"
            experts_path = tensor_dir / f"{trace_id}_experts.pt"
            weights_path = tensor_dir / f"{trace_id}_weights.pt"
            checkpoint_path = tensor_dir / f"{trace_id}_extraction.json"
            trace_digest = hashlib.sha256(
                json.dumps(
                    {
                        "prompt": trace.prompt,
                        "system_prompt": trace.system_prompt,
                        "generation_messages": trace.generation_messages,
                        "cot_text": trace.cot_text,
                        **({"token_replay": trace.metadata["token_replay"]}
                           if "token_replay" in trace.metadata else {}),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            expected_checkpoint = {
                "trace_sha256": trace_digest,
                "model_id": model_id,
                "quantization": quantization,
                "top_k": top_k,
                "hidden_states": extract_hidden_states,
                "layer_indices": layer_indices,
                "expert_weights": save_expert_weights,
            }
            if native_routing:
                expected_checkpoint["router_capture_version"] = ROUTER_CAPTURE_VERSION
            if feature_reducer is not None:
                expected_checkpoint.update(
                    {
                        "feature_schema_version": feature_schema_version,
                        "feature_config": feature_config or {},
                        "save_raw_tensors": save_raw_tensors,
                    }
                )
            if view_reducer is not None:
                expected_checkpoint["view_config"] = view_config or {}
                expected_checkpoint["annotation_sha256"] = hashlib.sha256(json.dumps(
                    trace.metadata.get("reasoning_annotation"), sort_keys=True,
                ).encode()).hexdigest()
            checkpoint = None
            if checkpoint_path.is_file():
                try:
                    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    checkpoint = None
            required_paths = [experts_path]
            if save_raw_tensors:
                required_paths.append(logits_path)
            if save_expert_weights:
                required_paths.append(weights_path)
            if extract_hidden_states and save_raw_tensors:
                required_paths.append(hidden_path)
            checkpoint_config_matches = checkpoint == expected_checkpoint
            if feature_reducer is not None and isinstance(checkpoint, dict):
                checkpoint_config_matches = checkpoint.get("config") == expected_checkpoint
            checkpoint_has_features = feature_reducer is None or (
                isinstance(checkpoint, dict)
                and isinstance(checkpoint.get("correlation_features"), dict)
                and isinstance(checkpoint.get("correlation_storage"), dict)
                and (view_reducer is None or isinstance(checkpoint.get("correlation_views"), dict))
            )
            if (
                checkpoint_config_matches
                and checkpoint_has_features
                and all(path.is_file() for path in required_paths)
            ):
                if not save_raw_tensors:
                    logits_path.unlink(missing_ok=True)
                    hidden_path.unlink(missing_ok=True)
                if not save_expert_weights:
                    weights_path.unlink(missing_ok=True)
                trace.model_logs.router_logits = (
                    logits_path.as_posix() if save_raw_tensors else None
                )
                trace.model_logs.selected_experts = experts_path.as_posix()
                trace.model_logs.expert_weights = (
                    weights_path.as_posix() if save_expert_weights else None
                )
                trace.model_logs.hidden_states = (
                    hidden_path.as_posix()
                    if extract_hidden_states and save_raw_tensors
                    else None
                )
                trace.model_logs.layer_indices = layer_indices
                if feature_reducer is not None:
                    trace.metadata["correlation_features"] = checkpoint["correlation_features"]
                    trace.metadata["correlation_storage"] = checkpoint["correlation_storage"]
                    if view_reducer is not None:
                        trace.metadata["correlation_views"] = checkpoint["correlation_views"]
                n_with_routing += 1
                out_f.write(trace.model_dump_json() + "\n")
                continue

            routing_details = {} if native_routing else None
            extra_args = {}
            if routing_details is not None:
                extra_args["routing_details"] = routing_details
            if "token_replay" in trace.metadata:
                extra_args["token_replay"] = trace.metadata["token_replay"]
            extracted = extract_logs_single_pass(
                model=model,
                tokenizer=tokenizer,
                problem=trace.prompt,
                cot_text=trace.cot_text,
                system_prompt=trace.system_prompt,
                messages=trace.generation_messages,
                extract_hidden_states=extract_hidden_states,
                layer_indices=layer_indices,
                **extra_args,
            )
            if extract_hidden_states:
                assert isinstance(extracted, tuple)
                router_logits, hidden_states = extracted
            else:
                assert isinstance(extracted, torch.Tensor)
                router_logits = extracted
                hidden_states = None

            if layer_indices is not None and router_logits.shape[0] != len(layer_indices):
                # Current extraction filters before moving activations to CPU.
                # Accept full-layer tensors too for custom extractors and tests.
                invalid = [index for index in layer_indices if index >= router_logits.shape[0]]
                if invalid:
                    raise ValueError(
                        f"Requested layer indices {invalid} are unavailable in an extractor "
                        f"output with {router_logits.shape[0]} layers"
                    )
                selection = torch.tensor(layer_indices, dtype=torch.long)
                router_logits = router_logits.index_select(0, selection)
                if hidden_states is not None:
                    hidden_states = hidden_states.index_select(0, selection)
            
            # router_logits is (num_layers, seq_len, num_experts)
            if router_logits.numel() > 0:
                trace.model_logs.layer_indices = layer_indices

                # Compute selected experts. Only materialize normalized weights
                # when this run is configured to persist them.
                if routing_details is not None:
                    selected = routing_details["selected_experts"]
                    weights = routing_details["expert_weights"] if save_expert_weights else None
                    if selected.shape[-1] != top_k:
                        raise ValueError("Requested top_k differs from the captured native routing")
                    trace.metadata["router_distribution"] = routing_details["router_distribution"]
                elif save_expert_weights:
                    selected, weights = compute_selected_experts(router_logits, top_k=top_k)
                else:
                    selected = torch.topk(router_logits, k=top_k, dim=-1).indices
                    weights = None

                if save_raw_tensors:
                    # Store POSIX-style paths so the JSONL stays portable across
                    # Windows development and Linux Docker/SLURM analysis.
                    _save_tensor_atomic(router_logits.to(torch.float32), logits_path)
                    trace.model_logs.router_logits = logits_path.as_posix()
                    if hidden_states is not None and hidden_states.numel() > 0:
                        _save_tensor_atomic(hidden_states.to(torch.bfloat16), hidden_path)
                        trace.model_logs.hidden_states = hidden_path.as_posix()
                else:
                    trace.model_logs.router_logits = None
                    trace.model_logs.hidden_states = None

                _save_tensor_atomic(selected.to(torch.int16), experts_path)
                trace.model_logs.selected_experts = experts_path.as_posix()

                if save_expert_weights:
                    assert weights is not None
                    _save_tensor_atomic(weights.to(torch.float16), weights_path)
                    trace.model_logs.expert_weights = weights_path.as_posix()
                else:
                    trace.model_logs.expert_weights = None

                correlation_features = None
                if feature_reducer is not None:
                    values = feature_reducer(
                        router_logits,
                        hidden_states,
                        selected,
                        layer_indices,
                    )
                    correlation_features = {
                        "schema_version": feature_schema_version,
                        "config": feature_config or {},
                        "token_count": int(router_logits.shape[1]),
                        "layer_indices": layer_indices,
                        "num_layers": int(router_logits.shape[0]),
                        "num_experts": int(router_logits.shape[2]),
                        "top_k": int(top_k),
                        "values": values,
                    }
                    trace.metadata["correlation_features"] = correlation_features
                correlation_views = None
                if view_reducer is not None:
                    correlation_views = view_reducer(
                        trace, router_logits, hidden_states, selected, layer_indices,
                    )
                    trace.metadata["correlation_views"] = correlation_views

                tensor_audit = {
                    "router_logits": _tensor_audit(
                        router_logits,
                        persisted_dtype=torch.float32,
                        path=logits_path if save_raw_tensors else None,
                    ),
                    "selected_experts": _tensor_audit(
                        selected,
                        persisted_dtype=torch.int16,
                        path=experts_path,
                    ),
                }
                if hidden_states is not None and hidden_states.numel() > 0:
                    tensor_audit["hidden_states"] = _tensor_audit(
                        hidden_states,
                        persisted_dtype=torch.bfloat16,
                        path=hidden_path if save_raw_tensors else None,
                    )
                if weights is not None:
                    tensor_audit["expert_weights"] = _tensor_audit(
                        weights,
                        persisted_dtype=torch.float16,
                        path=weights_path,
                    )
                correlation_storage = {
                    "tokens_retained": int(router_logits.shape[1]),
                    "layer_indices": layer_indices,
                    "tensors": tensor_audit,
                    "projected_raw_payload_bytes": sum(
                        item["projected_serialized_payload_bytes"]
                        for name, item in tensor_audit.items()
                        if name in {"router_logits", "hidden_states"}
                    ),
                    "persisted_tensor_bytes": sum(
                        item["serialized_bytes"] for item in tensor_audit.values()
                    ),
                    "persisted_raw_tensor_bytes": sum(
                        item["serialized_bytes"]
                        for name, item in tensor_audit.items()
                        if name in {"router_logits", "hidden_states"}
                    ),
                }
                if feature_reducer is not None:
                    trace.metadata["correlation_storage"] = correlation_storage

                temporary_checkpoint = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
                checkpoint_payload: dict[str, Any] = expected_checkpoint
                if feature_reducer is not None:
                    checkpoint_payload = {
                        "config": expected_checkpoint,
                        "correlation_features": correlation_features,
                        "correlation_storage": correlation_storage,
                    }
                    if view_reducer is not None:
                        checkpoint_payload["correlation_views"] = correlation_views
                temporary_checkpoint.write_text(
                    json.dumps(checkpoint_payload, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_checkpoint, checkpoint_path)
                if not save_raw_tensors:
                    logits_path.unlink(missing_ok=True)
                    hidden_path.unlink(missing_ok=True)
                if not save_expert_weights:
                    weights_path.unlink(missing_ok=True)
                n_with_routing += 1
            else:
                logger.warning(
                    f"Empty router logits for {trace.dataset}/{trace.problem_id} — "
                    "trace written without routing data"
                )
            
            out_f.write(trace.model_dump_json() + "\n")

    if n_with_routing == 0:
        tmp_output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"No routing data were extracted from {len(traces)} input traces."
        )
    tmp_output_path.replace(output_path)

    logger.info(f"Finished extracting routing context. Output saved to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single forward pass to extract router logits")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to generated traces.jsonl",
    )
    parser.add_argument("--output", type=str, required=True, help="Path to output jsonl")
    parser.add_argument("--model_id", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N traces")
    parser.add_argument("--top-k", type=int, default=None, help="Number of top experts per token (default: read from model config, e.g. 8 for OLMoE)")
    parser.add_argument(
        "--extract-hidden-states",
        action="store_true",
        help="Also save per-layer generated-token hidden states for probing",
    )
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_CHOICES,
        default="none",
        help="Model weight quantization used during extraction (default: none)",
    )
    
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    input_file = Path(args.input)
    if input_file.exists():
        process_file(
            input_path=input_file,
            model_id=args.model_id,
            output_path=Path(args.output),
            limit=args.limit,
            top_k=args.top_k,
            extract_hidden_states=args.extract_hidden_states,
            quantization=args.quantization,
        )
    else:
        raise FileNotFoundError(f"Could not find input file: {input_file}")


if __name__ == "__main__":
    main()
