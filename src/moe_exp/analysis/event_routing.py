"""Event-centered routing dynamics analysis.

Computes routing metrics (entropy, expert-switch rate, top-k overlap, router
margin) in a window around reasoning events (backtracking, contradiction,
self-correction, first-error steps). This produces the central table from
the paper:

    Model | Event type | Entropy before | Entropy at | Expert-switch rate |
    Top-k overlap | Router margin

Usage
-----
    python -m moe_exp.analysis.event_routing \
        --input results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces_with_routing.jsonl \
        --output results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/event_routing.json \
        --window 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from moe_exp.schemas import TraceRecord
from moe_exp.models.inference import _format_prompt, _find_prompt_length

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-token routing metrics
# ---------------------------------------------------------------------------


def compute_token_entropy(router_logits: torch.Tensor) -> torch.Tensor:
    """Routing entropy per token, averaged across layers.

    Args:
        router_logits: (num_layers, seq_len, num_experts)
    Returns:
        (seq_len,) tensor of average entropy across layers.
    """
    log_probs = F.log_softmax(router_logits, dim=-1)
    probs = log_probs.exp()
    # Parenthesize the negation before .clamp(): sum(p*log p) is <= 0, so clamping
    # it before negating would zero out every value.
    entropy = (-torch.sum(probs * log_probs, dim=-1)).clamp(min=0.0)  # (L, T)
    return entropy.mean(dim=0)  # (T,)


def compute_expert_switch_rate(router_logits: torch.Tensor, top_k: int = 8) -> torch.Tensor:
    """Fraction of top-k experts that changed from previous token, averaged over layers.

    Args:
        router_logits: (num_layers, seq_len, num_experts)
    Returns:
        (seq_len,) tensor; first token is 0.0 by convention.
    """
    probs = F.softmax(router_logits, dim=-1)
    _, top_indices = torch.topk(probs, k=top_k, dim=-1)  # (L, T, k)

    num_layers, seq_len, _ = top_indices.shape
    switch_rates = torch.zeros(seq_len)

    for t in range(1, seq_len):
        layer_switches = []
        for layer in range(num_layers):
            prev_set = set(top_indices[layer, t - 1].tolist())
            curr_set = set(top_indices[layer, t].tolist())
            # Fraction of current experts not in previous set
            switched = len(curr_set - prev_set) / top_k
            layer_switches.append(switched)
        switch_rates[t] = sum(layer_switches) / num_layers

    return switch_rates


def compute_topk_overlap(router_logits: torch.Tensor, top_k: int = 8) -> torch.Tensor:
    """Jaccard overlap of top-k experts between adjacent tokens, averaged over layers.

    Args:
        router_logits: (num_layers, seq_len, num_experts)
    Returns:
        (seq_len,) tensor; first token is 1.0 by convention.
    """
    probs = F.softmax(router_logits, dim=-1)
    _, top_indices = torch.topk(probs, k=top_k, dim=-1)  # (L, T, k)

    num_layers, seq_len, _ = top_indices.shape
    overlaps = torch.ones(seq_len)

    for t in range(1, seq_len):
        layer_overlaps = []
        for layer in range(num_layers):
            prev_set = set(top_indices[layer, t - 1].tolist())
            curr_set = set(top_indices[layer, t].tolist())
            intersection = len(prev_set & curr_set)
            union = len(prev_set | curr_set)
            layer_overlaps.append(intersection / union if union > 0 else 1.0)
        overlaps[t] = sum(layer_overlaps) / num_layers

    return overlaps


def compute_router_margin(router_logits: torch.Tensor) -> torch.Tensor:
    """Gap between top-1 and top-2 expert probabilities, averaged across layers.

    A large margin = confident routing. A small margin = uncertain routing.

    Args:
        router_logits: (num_layers, seq_len, num_experts)
    Returns:
        (seq_len,) tensor of average margin.
    """
    probs = F.softmax(router_logits, dim=-1)  # (L, T, E)
    top2 = torch.topk(probs, k=2, dim=-1).values  # (L, T, 2)
    margin = (top2[:, :, 0] - top2[:, :, 1])  # (L, T)
    return margin.mean(dim=0)  # (T,)


# ---------------------------------------------------------------------------
# Step-to-token mapping
# ---------------------------------------------------------------------------


def _map_steps_to_token_ranges(
    steps: list[str],
    cot_text: str,
    tokenizer,
    formatted_prompt: str,
) -> list[tuple[int, int]]:
    """Map each step to (start_token_idx, end_token_idx) in the generation.

    Returns a list of (inclusive_start, exclusive_end) token index pairs,
    relative to the start of the generated tokens (0-based).
    """
    full_text = formatted_prompt + cot_text
    prompt_char_len = len(formatted_prompt)

    encoding = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding["offset_mapping"][0].tolist()

    prompt_len = _find_prompt_length(tokenizer, formatted_prompt, full_text)

    # Locate each step in cot_text
    step_char_ranges: list[tuple[int, int]] = []
    search_start = 0
    for step in steps:
        idx = cot_text.find(step, search_start)
        if idx == -1:
            stripped = step.strip()
            idx = cot_text.find(stripped, search_start)
            if idx == -1:
                step_char_ranges.append((-1, -1))
                continue
            step_char_ranges.append((idx, idx + len(stripped)))
            search_start = idx + len(stripped)
        else:
            step_char_ranges.append((idx, idx + len(step)))
            search_start = idx + len(step)

    # Convert character ranges to token ranges
    token_ranges: list[tuple[int, int]] = []
    for char_start, char_end in step_char_ranges:
        if char_start == -1:
            token_ranges.append((-1, -1))
            continue

        abs_start = prompt_char_len + char_start
        abs_end = prompt_char_len + char_end
        tok_start = None
        tok_end = None

        for tok_idx in range(prompt_len, len(offset_mapping)):
            tok_char_start, tok_char_end = offset_mapping[tok_idx]
            if tok_start is None and tok_char_end > abs_start:
                tok_start = tok_idx - prompt_len
            if tok_char_start < abs_end:
                tok_end = tok_idx - prompt_len + 1

        if tok_start is None or tok_end is None:
            token_ranges.append((-1, -1))
        else:
            token_ranges.append((tok_start, tok_end))

    return token_ranges


# ---------------------------------------------------------------------------
# Event-window aggregation
# ---------------------------------------------------------------------------


def _aggregate_window(
    metric: torch.Tensor | np.ndarray,
    center_tokens: list[int],
    window: int,
    seq_len: int,
) -> dict[str, float]:
    """Compute mean metric value before/at/after an event across multiple occurrences.

    center_tokens: list of token indices where the event occurs.
    window: number of tokens to look before and after the event center.
    """
    if not center_tokens:
        return {"before": float("nan"), "at": float("nan"), "after": float("nan")}

    if isinstance(metric, torch.Tensor):
        metric = metric.numpy()

    before_vals = []
    at_vals = []
    after_vals = []

    for ct in center_tokens:
        # "Before" window
        start = max(0, ct - window)
        if start < ct:
            before_vals.extend(metric[start:ct].tolist())
        # "At" the event
        at_vals.append(float(metric[ct]))
        # "After" window
        end = min(seq_len, ct + window + 1)
        if ct + 1 < end:
            after_vals.extend(metric[ct + 1:end].tolist())

    return {
        "before": float(np.mean(before_vals)) if before_vals else float("nan"),
        "at": float(np.mean(at_vals)) if at_vals else float("nan"),
        "after": float(np.mean(after_vals)) if after_vals else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_trace_events(
    trace: TraceRecord,
    router_logits: torch.Tensor,
    tokenizer,
    window: int = 5,
    top_k: int = 8,
) -> dict[str, Any] | None:
    """Compute event-centered routing metrics for a single trace.

    Returns None if the trace has no reasoning events to analyze.
    """
    if not trace.cot_text.strip() or router_logits.numel() == 0:
        return None

    events = trace.step_labels
    has_events = (
        events.backtracking_steps
        or events.contradiction_steps
        or events.self_correction_steps
        or events.first_error_step is not None
        or events.first_reasoning_event_step is not None
    )
    if not has_events:
        return None

    # Compute per-token metrics
    seq_len = router_logits.shape[1]
    entropy = compute_token_entropy(router_logits)
    switch_rate = compute_expert_switch_rate(router_logits, top_k=top_k)
    topk_overlap = compute_topk_overlap(router_logits, top_k=top_k)
    margin = compute_router_margin(router_logits)

    # Map steps to token ranges
    formatted_prompt = _format_prompt(tokenizer, trace.prompt)
    token_ranges = _map_steps_to_token_ranges(
        trace.steps, trace.cot_text, tokenizer, formatted_prompt
    )

    def _step_center_tokens(step_indices: list[int]) -> list[int]:
        """Get the center token for each step index."""
        centers = []
        for si in step_indices:
            if si < len(token_ranges):
                start, end = token_ranges[si]
                if start >= 0 and end > start:
                    center = (start + end) // 2
                    if center < seq_len:
                        centers.append(center)
        return centers

    result: dict[str, Any] = {
        "problem_id": trace.problem_id,
        "dataset": trace.dataset,
        "is_correct": trace.is_correct,
        "task_type": trace.task_type,
        "seq_len": seq_len,
        "events": {},
    }

    # Backtracking events
    if events.backtracking_steps:
        centers = _step_center_tokens(events.backtracking_steps)
        if centers:
            result["events"]["backtracking"] = {
                "n_events": len(centers),
                "entropy": _aggregate_window(entropy, centers, window, seq_len),
                "switch_rate": _aggregate_window(switch_rate, centers, window, seq_len),
                "topk_overlap": _aggregate_window(topk_overlap, centers, window, seq_len),
                "margin": _aggregate_window(margin, centers, window, seq_len),
            }

    # Contradiction events
    if events.contradiction_steps:
        centers = _step_center_tokens(events.contradiction_steps)
        if centers:
            result["events"]["contradiction"] = {
                "n_events": len(centers),
                "entropy": _aggregate_window(entropy, centers, window, seq_len),
                "switch_rate": _aggregate_window(switch_rate, centers, window, seq_len),
                "topk_overlap": _aggregate_window(topk_overlap, centers, window, seq_len),
                "margin": _aggregate_window(margin, centers, window, seq_len),
            }

    # Self-correction events
    if events.self_correction_steps:
        centers = _step_center_tokens(events.self_correction_steps)
        if centers:
            result["events"]["self_correction"] = {
                "n_events": len(centers),
                "entropy": _aggregate_window(entropy, centers, window, seq_len),
                "switch_rate": _aggregate_window(switch_rate, centers, window, seq_len),
                "topk_overlap": _aggregate_window(topk_overlap, centers, window, seq_len),
                "margin": _aggregate_window(margin, centers, window, seq_len),
            }

    # First error step (from gold labels)
    if events.first_error_step is not None:
        centers = _step_center_tokens([events.first_error_step])
        if centers:
            result["events"]["first_error"] = {
                "n_events": len(centers),
                "entropy": _aggregate_window(entropy, centers, window, seq_len),
                "switch_rate": _aggregate_window(switch_rate, centers, window, seq_len),
                "topk_overlap": _aggregate_window(topk_overlap, centers, window, seq_len),
                "margin": _aggregate_window(margin, centers, window, seq_len),
            }

    # First reasoning event: the earliest step where the model SIGNALS an issue
    # (earliest backtracking/contradiction). Distinct from first_error (gold) and
    # from the per-type rows, which aggregate all occurrences.
    if events.first_reasoning_event_step is not None:
        centers = _step_center_tokens([events.first_reasoning_event_step])
        if centers:
            result["events"]["first_reasoning_event"] = {
                "n_events": len(centers),
                "entropy": _aggregate_window(entropy, centers, window, seq_len),
                "switch_rate": _aggregate_window(switch_rate, centers, window, seq_len),
                "topk_overlap": _aggregate_window(topk_overlap, centers, window, seq_len),
                "margin": _aggregate_window(margin, centers, window, seq_len),
            }

    # Baseline: metrics for "normal" steps (no events)
    event_step_set = set(
        events.backtracking_steps
        + events.contradiction_steps
        + events.self_correction_steps
    )
    if events.first_error_step is not None:
        event_step_set.add(events.first_error_step)
    if events.first_reasoning_event_step is not None:
        event_step_set.add(events.first_reasoning_event_step)

    normal_steps = [i for i in range(len(trace.steps)) if i not in event_step_set]
    if normal_steps:
        centers = _step_center_tokens(normal_steps)
        if centers:
            result["events"]["normal"] = {
                "n_events": len(centers),
                "entropy": _aggregate_window(entropy, centers, window, seq_len),
                "switch_rate": _aggregate_window(switch_rate, centers, window, seq_len),
                "topk_overlap": _aggregate_window(topk_overlap, centers, window, seq_len),
                "margin": _aggregate_window(margin, centers, window, seq_len),
            }

    return result if result["events"] else None


# ---------------------------------------------------------------------------
# Aggregate summary across traces
# ---------------------------------------------------------------------------


def build_event_summary(per_trace_results: list[dict]) -> dict:
    """Aggregate event-centered metrics across all analyzed traces.

    Produces the Experiment 2 central table:
        Event type | Entropy before/at/after | Switch rate | Top-k overlap | Margin
    """
    event_types = [
        "normal", "backtracking", "contradiction", "self_correction",
        "first_error", "first_reasoning_event",
    ]
    metrics = ["entropy", "switch_rate", "topk_overlap", "margin"]
    phases = ["before", "at", "after"]

    summary: dict[str, dict] = {}

    for etype in event_types:
        values: dict[str, list[float]] = {f"{m}_{p}": [] for m in metrics for p in phases}
        n_traces = 0

        for result in per_trace_results:
            if etype in result.get("events", {}):
                n_traces += 1
                event_data = result["events"][etype]
                for m in metrics:
                    for p in phases:
                        v = event_data[m][p]
                        if not np.isnan(v):
                            values[f"{m}_{p}"].append(v)

        if n_traces == 0:
            continue

        row: dict[str, Any] = {"n_traces": n_traces}
        for key, vals in values.items():
            row[key] = float(np.mean(vals)) if vals else None
        summary[etype] = row

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute event-centered routing metrics around reasoning events"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to traces_with_routing.jsonl from Exp 2 (must have router_logits paths)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output JSON file with event-centered analysis"
    )
    parser.add_argument(
        "--window", type=int, default=5,
        help="Number of tokens before/after event center to aggregate (default: 5)"
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Number of top experts per token (default: read from model config, e.g. 8 for OLMoE)"
    )
    parser.add_argument(
        "--model_id", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="Model ID (for tokenizer loading)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N traces"
    )
    parser.add_argument(
        "--reasoning-only", action="store_true", default=False,
        help="Skip meta-reasoning traces (e.g. ProcessBench)"
    )
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    top_k = args.top_k
    if top_k is None:
        cfg = AutoConfig.from_pretrained(args.model_id)
        top_k = getattr(cfg, "num_experts_per_tok", None)
        if top_k is None:
            raise ValueError(
                f"Could not infer top_k from {args.model_id} config "
                "(no num_experts_per_tok); pass --top-k explicitly."
            )
        top_k = int(top_k)
        logger.info(f"Using top_k={top_k} from model config")

    logger.info(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_path = Path(args.input)
    traces_raw: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            traces_raw.append(json.loads(line))

    if args.limit:
        traces_raw = traces_raw[:args.limit]

    logger.info(f"Analyzing {len(traces_raw)} traces...")
    per_trace_results: list[dict] = []
    skipped = 0

    for trace_dict in traces_raw:
        trace = TraceRecord(**trace_dict)

        # Skip meta-reasoning if requested
        if args.reasoning_only and trace.task_type == "meta_reasoning":
            skipped += 1
            continue

        # Load router logits tensor
        if not trace.model_logs.router_logits:
            continue
        logits_path = Path(trace.model_logs.router_logits)
        if not logits_path.exists():
            continue

        router_logits = torch.load(logits_path, map_location="cpu", weights_only=True)

        result = analyze_trace_events(
            trace=trace,
            router_logits=router_logits,
            tokenizer=tokenizer,
            window=args.window,
            top_k=top_k,
        )
        if result is not None:
            per_trace_results.append(result)

    logger.info(
        f"Analyzed {len(per_trace_results)} traces with events "
        f"(skipped {skipped} meta-reasoning)"
    )

    # Build summary
    summary = build_event_summary(per_trace_results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "window": args.window,
            "top_k": top_k,
            "model_id": args.model_id,
            "n_traces_analyzed": len(per_trace_results),
            "n_traces_skipped": skipped,
        },
        "summary": summary,
        "per_trace": per_trace_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary table
    print("\n=== Event-Centered Routing Metrics ===")
    print(f"{'Event':<18} {'Entropy(before/at/after)':<32} {'Switch(before/at/after)':<32} {'Margin(before/at/after)':<32}")
    print("-" * 114)
    for etype, row in summary.items():
        def _fmt(metric: str) -> str:
            b = row.get(f"{metric}_before")
            a = row.get(f"{metric}_at")
            af = row.get(f"{metric}_after")
            return f"{b:.4f}/{a:.4f}/{af:.4f}" if all(x is not None for x in [b, a, af]) else "N/A"
        print(f"{etype:<18} {_fmt('entropy'):<32} {_fmt('switch_rate'):<32} {_fmt('margin'):<32}")


if __name__ == "__main__":
    main()
