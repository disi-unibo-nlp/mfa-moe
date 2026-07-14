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
    top_indices = torch.topk(router_logits, k=top_k, dim=-1).indices  # (L, T, k)
    seq_len = top_indices.shape[1]
    switch_rates = torch.zeros(seq_len, dtype=torch.float32)
    if seq_len < 2:
        return switch_rates

    # Each top-k list contains unique expert IDs.  Compare every current expert
    # with every previous expert, then count current IDs with no previous match.
    # This is exactly the set-difference definition above, but avoids Python
    # loops over every token and layer.
    previous = top_indices[:, :-1, :, None]       # (L, T-1, k, 1)
    current = top_indices[:, 1:, None, :]         # (L, T-1, 1, k)
    current_was_present = (previous == current).any(dim=2)  # (L, T-1, k)
    per_layer = 1.0 - current_was_present.float().mean(dim=-1)
    switch_rates[1:] = per_layer.mean(dim=0)
    return switch_rates


def compute_topk_overlap(router_logits: torch.Tensor, top_k: int = 8) -> torch.Tensor:
    """Jaccard overlap of top-k experts between adjacent tokens, averaged over layers.

    Args:
        router_logits: (num_layers, seq_len, num_experts)
    Returns:
        (seq_len,) tensor; first token is 1.0 by convention.
    """
    top_indices = torch.topk(router_logits, k=top_k, dim=-1).indices  # (L, T, k)
    seq_len = top_indices.shape[1]
    overlaps = torch.ones(seq_len, dtype=torch.float32)
    if seq_len < 2:
        return overlaps

    previous = top_indices[:, :-1, :, None]       # (L, T-1, k, 1)
    current = top_indices[:, 1:, None, :]         # (L, T-1, 1, k)
    intersection = (previous == current).any(dim=2).sum(dim=-1).float()
    union = 2 * top_k - intersection
    overlaps[1:] = (intersection / union).mean(dim=0)
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


def _matched_control_at(
    metric: torch.Tensor | np.ndarray,
    event_centers: list[int],
    normal_centers: list[int],
) -> dict[str, float] | None:
    """Compare each event token with the nearest normal step in the same trace.

    This produces one within-trace control per event, avoiding a comparison that
    is confounded by a trace's length or its overall routing profile.  Values are
    aggregated by trace later, rather than treating event tokens as independent.
    """
    if not event_centers or not normal_centers:
        return None
    if isinstance(metric, torch.Tensor):
        metric = metric.numpy()

    controls = [
        min(normal_centers, key=lambda center: (abs(center - event), center))
        for event in event_centers
    ]
    event_values = np.asarray([metric[center] for center in event_centers], dtype=float)
    control_values = np.asarray([metric[center] for center in controls], dtype=float)
    return {
        "n_events": len(event_centers),
        "event_at": float(event_values.mean()),
        "control_at": float(control_values.mean()),
        "delta_at": float((event_values - control_values).mean()),
    }


def _has_reasoning_events(trace: TraceRecord) -> bool:
    """Whether a trace contains an event that Experiment 2 can analyse."""
    events = trace.step_labels
    return bool(
        events.backtracking_steps
        or events.contradiction_steps
        or events.self_correction_steps
        or events.first_error_step is not None
        or events.first_reasoning_event_step is not None
    )


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
    if not _has_reasoning_events(trace):
        return None

    # Compute per-token metrics
    seq_len = router_logits.shape[1]
    entropy = compute_token_entropy(router_logits)
    switch_rate = compute_expert_switch_rate(router_logits, top_k=top_k)
    topk_overlap = compute_topk_overlap(router_logits, top_k=top_k)
    margin = compute_router_margin(router_logits)

    # Map steps to token ranges. Rebuild the prompt with the generation-time
    # system prompt so token indices line up with the Exp2 extraction.
    formatted_prompt = _format_prompt(tokenizer, trace.prompt, trace.system_prompt)
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
    normal_centers = _step_center_tokens(normal_steps)

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

    # Baseline: normal steps from the same event-containing trace.
    if normal_centers:
        centers = normal_centers
        if centers:
            result["events"]["normal"] = {
                "n_events": len(centers),
                "entropy": _aggregate_window(entropy, centers, window, seq_len),
                "switch_rate": _aggregate_window(switch_rate, centers, window, seq_len),
                "topk_overlap": _aggregate_window(topk_overlap, centers, window, seq_len),
                "margin": _aggregate_window(margin, centers, window, seq_len),
            }

    # Report a matched within-trace contrast as well as the descriptive normal
    # baseline.  Each event is paired with the nearest non-event reasoning step.
    event_steps_by_type = {
        "backtracking": events.backtracking_steps,
        "contradiction": events.contradiction_steps,
        "self_correction": events.self_correction_steps,
        "first_error": [events.first_error_step] if events.first_error_step is not None else [],
        "first_reasoning_event": (
            [events.first_reasoning_event_step]
            if events.first_reasoning_event_step is not None
            else []
        ),
    }
    metrics = {
        "entropy": entropy,
        "switch_rate": switch_rate,
        "topk_overlap": topk_overlap,
        "margin": margin,
    }
    for event_type, step_indices in event_steps_by_type.items():
        if event_type not in result["events"]:
            continue
        event_centers = _step_center_tokens(step_indices)
        matched_control = {
            name: comparison
            for name, metric in metrics.items()
            if (comparison := _matched_control_at(metric, event_centers, normal_centers)) is not None
        }
        if matched_control:
            result["events"][event_type]["matched_control"] = matched_control

    return result if result["events"] else None


# ---------------------------------------------------------------------------
# Aggregate summary across traces
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(
    values: list[float], rng: np.random.Generator, n_bootstrap: int
) -> list[float] | None:
    """Return a trace-level percentile bootstrap interval for a mean effect."""
    if len(values) < 5:
        return None
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(n_bootstrap, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return [float(low), float(high)]


def build_event_summary(
    per_trace_results: list[dict], n_bootstrap: int = 2000, seed: int = 42
) -> dict:
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
    rng = np.random.default_rng(seed)

    summary: dict[str, dict] = {}

    for etype in event_types:
        values: dict[str, list[float]] = {f"{m}_{p}": [] for m in metrics for p in phases}
        n_traces = 0
        n_event_occurrences = 0
        matched_deltas: dict[str, list[float]] = {metric: [] for metric in metrics}

        for result in per_trace_results:
            if etype in result.get("events", {}):
                n_traces += 1
                event_data = result["events"][etype]
                n_event_occurrences += int(event_data["n_events"])
                for m in metrics:
                    for p in phases:
                        v = event_data[m][p]
                        if not np.isnan(v):
                            values[f"{m}_{p}"].append(v)
                    matched = event_data.get("matched_control", {}).get(m)
                    if matched is not None:
                        matched_deltas[m].append(float(matched["delta_at"]))

        if n_traces == 0:
            continue

        row: dict[str, Any] = {
            "n_traces": n_traces,
            "n_event_occurrences": n_event_occurrences,
        }
        for key, vals in values.items():
            row[key] = float(np.mean(vals)) if vals else None
        paired_summary: dict[str, dict[str, Any]] = {}
        for metric, deltas in matched_deltas.items():
            if deltas:
                paired_summary[metric] = {
                    "n_traces": len(deltas),
                    "mean_delta_at": float(np.mean(deltas)),
                    "bootstrap_ci_95": _bootstrap_mean_ci(deltas, rng, n_bootstrap),
                }
        if paired_summary:
            row["matched_control"] = paired_summary
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
        help=(
            "Skip traces tagged task_type='meta_reasoning'. No current loader "
            "produces such traces; kept for forward compatibility."
        )
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Trace-level bootstrap replicates for matched-control intervals (default: 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for bootstrap intervals"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    traces_raw: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces_raw.append(json.loads(line))

    if args.limit:
        traces_raw = traces_raw[:args.limit]
    if not traces_raw:
        raise RuntimeError(
            f"Input {input_path} contains zero traces; event-routing analysis aborted."
        )

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

    logger.info(f"Scanning {len(traces_raw)} traces for labelled events...")
    per_trace_results: list[dict] = []
    meta_reasoning_skipped = 0
    no_event_traces = 0
    missing_router_logits = 0

    for trace_dict in traces_raw:
        trace = TraceRecord(**trace_dict)

        # Skip meta-reasoning if requested
        if args.reasoning_only and trace.task_type == "meta_reasoning":
            meta_reasoning_skipped += 1
            continue

        # Event-free traces cannot contribute to an event-centred statistic, so
        # avoid loading their potentially large tensors from disk.
        if not _has_reasoning_events(trace):
            no_event_traces += 1
            continue

        # Load router logits tensor
        if not trace.model_logs.router_logits:
            missing_router_logits += 1
            continue
        logits_path = Path(trace.model_logs.router_logits)
        if not logits_path.exists():
            missing_router_logits += 1
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
        f"from {len(traces_raw)} inputs; {no_event_traces} had no labelled event, "
        f"{missing_router_logits} lacked router logits, and "
        f"{meta_reasoning_skipped} were skipped as meta-reasoning."
    )

    # Build summary
    summary = build_event_summary(
        per_trace_results,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "window": args.window,
            "top_k": top_k,
            "model_id": args.model_id,
            "n_input_traces": len(traces_raw),
            "n_event_eligible_traces": len(per_trace_results),
            "n_traces_without_labelled_events": no_event_traces,
            "n_missing_router_logits": missing_router_logits,
            "n_meta_reasoning_skipped": meta_reasoning_skipped,
            "n_traces_analyzed": len(per_trace_results),
            # Retained for readers of earlier result files. It now has the
            # narrower meaning stated by n_meta_reasoning_skipped.
            "n_traces_skipped": meta_reasoning_skipped,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
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
