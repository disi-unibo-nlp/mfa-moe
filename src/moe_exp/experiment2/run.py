import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from moe_exp.schemas import TraceRecord
from moe_exp.models.loader import QUANTIZATION_CHOICES, load_model_and_tokenizer
from moe_exp.models.inference import extract_logs_single_pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
):
    """
    Run Experiment 2 offline-extraction loop over traces to compute routing dynamics.
    Saves per-trace tensors: router_logits, selected_experts, expert_weights,
    and optionally hidden_states for linear probing.

    top_k: number of experts selected per token. When None, it is read from the
    model config (num_experts_per_tok), so it is correct for any MoE model
    (OLMoE=8, Qwen1.5-MoE=4, …). An explicit value overrides the config.
    """
    logger.info(f"Loading model {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, quantization=quantization)

    config_top_k = getattr(model.config, "num_experts_per_tok", None)
    if top_k is None:
        if config_top_k is None:
            raise ValueError(
                f"Could not infer top_k from {model_id} config "
                "(no num_experts_per_tok); pass --top-k explicitly."
            )
        top_k = int(config_top_k)
        logger.info(f"Using top_k={top_k} from model config")
    elif config_top_k is not None and top_k != config_top_k:
        logger.warning(
            f"Requested top_k={top_k} differs from model config "
            f"num_experts_per_tok={config_top_k}"
        )

    traces: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            traces.append(json.loads(line))
            
    if limit is not None:
        traces = traces[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_dir = output_path.parent / "tensors"
    tensor_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Processing {len(traces)} traces for offline routing metrics computation...")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for trace_dict in tqdm(traces, desc="Extracting Routing"):
            trace = TraceRecord(**trace_dict)
            
            if not trace.cot_text.strip():
                out_f.write(trace.model_dump_json() + "\n")
                continue
                
            extracted = extract_logs_single_pass(
                model=model,
                tokenizer=tokenizer,
                problem=trace.prompt,
                cot_text=trace.cot_text,
                system_prompt=trace.system_prompt,
                extract_hidden_states=extract_hidden_states,
            )
            if extract_hidden_states:
                assert isinstance(extracted, tuple)
                router_logits, hidden_states = extracted
            else:
                assert isinstance(extracted, torch.Tensor)
                router_logits = extracted
                hidden_states = None
            
            # router_logits is (num_layers, seq_len, num_experts)
            if router_logits.numel() > 0:
                # Use safe filename
                safe_problem_id = trace.problem_id.replace("/", "_").replace("\\", "_")
                trace_id = f"{trace.dataset}_{safe_problem_id}"
                
                # Save router logits.
                # Store POSIX-style paths so the JSONL stays portable: tensors
                # are typically written on Windows during dev but re-read inside
                # the Linux Docker/SLURM pipeline, where backslash paths break.
                logits_path = tensor_dir / f"{trace_id}_logits.pt"
                torch.save(router_logits.to(torch.float32), logits_path)
                trace.model_logs.router_logits = logits_path.as_posix()

                if hidden_states is not None and hidden_states.numel() > 0:
                    hidden_path = tensor_dir / f"{trace_id}_hidden.pt"
                    torch.save(hidden_states.to(torch.float16), hidden_path)
                    trace.model_logs.hidden_states = hidden_path.as_posix()

                # Compute and save selected experts and weights
                selected, weights = compute_selected_experts(router_logits, top_k=top_k)

                experts_path = tensor_dir / f"{trace_id}_experts.pt"
                torch.save(selected.to(torch.int16), experts_path)
                trace.model_logs.selected_experts = experts_path.as_posix()

                weights_path = tensor_dir / f"{trace_id}_weights.pt"
                torch.save(weights.to(torch.float16), weights_path)
                trace.model_logs.expert_weights = weights_path.as_posix()
            else:
                logger.warning(
                    f"Empty router logits for {trace.dataset}/{trace.problem_id} — "
                    "trace written without routing data"
                )
            
            out_f.write(trace.model_dump_json() + "\n")

    logger.info(f"Finished extracting routing context. Output saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run single forward pass to extract router logits")
    parser.add_argument("--input", type=str, required=True, help="Path to input traces.jsonl from Exp 1")
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
    
    args = parser.parse_args()
    
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
        logger.error(f"Could not find input file: {input_file}")
