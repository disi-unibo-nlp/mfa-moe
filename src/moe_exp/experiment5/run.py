"""Experiment 5 — Expert behavior around reasoning events.

For each expert (per layer), measure over/under-use around reasoning events:

- activation frequency and weight mass per reasoning phase
  (normal / backtracking / contradiction / self-correction / first-error /
  final-answer). Phrasing note: we describe experts as associated with
  reasoning-state regions or transitions, NOT as "math/logic experts".
- expert usage in a ±window before vs after the first error and before vs
  after self-correction events (stratified by final trace outcome; this does
  not claim that the correction caused that outcome).
- global co-activation counts (joint top-k membership) and top-1 expert
  transition matrices, per layer, saved as .npz (too large for JSON).

"First correct derivation step" from the experiment plan has no label in the
current taxonomy and is not computed.

Runs offline over Exp2 outputs (router-logit tensors); no GPU needed.

Usage
-----
    python -m moe_exp.experiment5.run \
        --input results/exp2/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/traces_with_routing.jsonl \
        --output results/exp5/allenai--OLMoE-1B-7B-0924-Instruct/gsm8k/expert_events.json \
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
from moe_exp.models.inference import _format_prompt
from moe_exp.analysis.event_routing import _map_steps_to_token_ranges

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PHASES = [
    "normal", "backtracking", "contradiction", "self_correction",
    "first_error", "final_answer",
]

BEFORE_AFTER_GROUPS = [
    "first_error",
    "self_correction_finally_correct",
    "self_correction_finally_incorrect",
]


# ---------------------------------------------------------------------------
# Per-trace token bookkeeping
# ---------------------------------------------------------------------------


def _phase_token_sets(
    trace: TraceRecord,
    token_ranges: list[tuple[int, int]],
    seq_len: int,
) -> dict[str, set[int]]:
    """Assign generation tokens to reasoning phases.

    A token may belong to several event phases (e.g. a step that both
    backtracks and contradicts); "normal" is every token in no event phase.
    """

    def toks(step_indices: list[int]) -> set[int]:
        s: set[int] = set()
        for si in step_indices:
            if 0 <= si < len(token_ranges):
                start, end = token_ranges[si]
                if start >= 0 and end > start:
                    s.update(range(start, min(end, seq_len)))
        return s

    ev = trace.step_labels
    sets = {
        "backtracking": toks(ev.backtracking_steps),
        "contradiction": toks(ev.contradiction_steps),
        "self_correction": toks(ev.self_correction_steps),
        "first_error": toks([ev.first_error_step]) if ev.first_error_step is not None else set(),
        "final_answer": toks([len(trace.steps) - 1]) if trace.steps else set(),
    }
    event_union: set[int] = set().union(*sets.values())
    sets["normal"] = set(range(seq_len)) - event_union
    return sets


def _step_center_tokens(
    step_indices: list[int],
    token_ranges: list[tuple[int, int]],
    seq_len: int,
) -> list[int]:
    """Midpoint token of each step (same convention as event_routing)."""
    centers = []
    for si in step_indices:
        if 0 <= si < len(token_ranges):
            start, end = token_ranges[si]
            if start >= 0 and end > start:
                center = (start + end) // 2
                if center < seq_len:
                    centers.append(center)
    return centers


# ---------------------------------------------------------------------------
# Accumulator across traces
# ---------------------------------------------------------------------------


class _Accumulator:
    """Pools token-level expert statistics across all traces."""

    def __init__(self, num_layers: int, num_experts: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        shape = (num_layers, num_experts)
        self.usage = {p: np.zeros(shape) for p in PHASES}
        self.mass = {p: np.zeros(shape) for p in PHASES}
        self.n_tokens = {p: 0 for p in PHASES}
        # Global (all-token) matrices, per layer
        self.coactivation = np.zeros((num_layers, num_experts, num_experts), dtype=np.int64)
        self.transitions = np.zeros((num_layers, num_experts, num_experts), dtype=np.int64)
        # Before/after event windows
        self.ba_usage = {
            g: {"before": np.zeros(shape), "after": np.zeros(shape)}
            for g in BEFORE_AFTER_GROUPS
        }
        self.ba_tokens = {g: {"before": 0, "after": 0} for g in BEFORE_AFTER_GROUPS}
        self.ba_events = {g: 0 for g in BEFORE_AFTER_GROUPS}

    def add_trace(
        self,
        router_logits: torch.Tensor,
        trace: TraceRecord,
        token_ranges: list[tuple[int, int]],
        top_k: int,
        window: int,
    ) -> None:
        num_layers, seq_len, num_experts = router_logits.shape

        probs = F.softmax(router_logits.to(torch.float32), dim=-1)  # (L, T, E)
        weights, indices = torch.topk(probs, k=top_k, dim=-1)
        # Renormalized top-k weights = the mixture weights the model actually uses
        weights = weights / weights.sum(dim=-1, keepdim=True)

        onehot = torch.zeros_like(probs).scatter_(-1, indices, 1.0)  # (L, T, E)
        mass = torch.zeros_like(probs).scatter_(-1, indices, weights)  # (L, T, E)
        top1 = probs.argmax(dim=-1).numpy()  # (L, T)

        onehot_np = onehot.numpy()
        mass_np = mass.numpy()

        # --- Per-phase usage / weight mass ---
        phase_sets = _phase_token_sets(trace, token_ranges, seq_len)
        for phase, tok_set in phase_sets.items():
            if not tok_set:
                continue
            idx = sorted(tok_set)
            self.usage[phase] += onehot_np[:, idx, :].sum(axis=1)
            self.mass[phase] += mass_np[:, idx, :].sum(axis=1)
            self.n_tokens[phase] += len(idx)

        # --- Global co-activation and top-1 transition matrices ---
        for layer in range(num_layers):
            o = onehot_np[layer]  # (T, E)
            self.coactivation[layer] += (o.T @ o).astype(np.int64)
            if seq_len > 1:
                np.add.at(
                    self.transitions[layer],
                    (top1[layer, :-1], top1[layer, 1:]),
                    1,
                )

        # --- Before/after event windows ---
        ev = trace.step_labels

        def _add_window(group: str, centers: list[int]) -> None:
            for ct in centers:
                self.ba_events[group] += 1
                start = max(0, ct - window)
                if start < ct:
                    self.ba_usage[group]["before"] += onehot_np[:, start:ct, :].sum(axis=1)
                    self.ba_tokens[group]["before"] += ct - start
                end = min(seq_len, ct + window + 1)
                if ct + 1 < end:
                    self.ba_usage[group]["after"] += onehot_np[:, ct + 1:end, :].sum(axis=1)
                    self.ba_tokens[group]["after"] += end - (ct + 1)

        if ev.first_error_step is not None:
            _add_window(
                "first_error",
                _step_center_tokens([ev.first_error_step], token_ranges, seq_len),
            )
        if ev.self_correction_steps and trace.is_correct is not None:
            group = (
                "self_correction_finally_correct"
                if trace.is_correct
                else "self_correction_finally_incorrect"
            )
            _add_window(
                group,
                _step_center_tokens(ev.self_correction_steps, token_ranges, seq_len),
            )

    # -- Finalization ------------------------------------------------------

    def phase_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for phase in PHASES:
            n = self.n_tokens[phase]
            entry: dict[str, Any] = {"n_tokens": n}
            if n > 0:
                entry["activation_frequency"] = (self.usage[phase] / n).tolist()
                entry["weight_mass"] = (self.mass[phase] / n).tolist()
            out[phase] = entry
        return out

    def before_after_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for group in BEFORE_AFTER_GROUPS:
            entry: dict[str, Any] = {"n_events": self.ba_events[group]}
            for side in ("before", "after"):
                n = self.ba_tokens[group][side]
                entry[f"n_tokens_{side}"] = n
                if n > 0:
                    entry[f"activation_frequency_{side}"] = (
                        self.ba_usage[group][side] / n
                    ).tolist()
            out[group] = entry
        return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5 — per-expert usage patterns around reasoning events"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to traces_with_routing.jsonl from Exp 2 (must have router_logits paths)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output JSON; heavy matrices go to expert_arrays.npz next to it"
    )
    parser.add_argument(
        "--arrays-output", type=str, default=None,
        help=(
            "Optional path for co-activation/transition NPZ "
            "(default: expert_arrays.npz next to --output)"
        ),
    )
    parser.add_argument(
        "--window", type=int, default=5,
        help="Tokens before/after an event center for the before/after comparison (default: 5)"
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
    args = parser.parse_args()

    traces_raw: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces_raw.append(json.loads(line))
    if args.limit:
        traces_raw = traces_raw[:args.limit]
    if not traces_raw:
        raise RuntimeError(
            f"Input {args.input} contains zero traces; expert analysis aborted."
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

    logger.info(f"Analyzing {len(traces_raw)} traces...")
    accum: _Accumulator | None = None
    n_analyzed = 0

    for trace_dict in traces_raw:
        trace = TraceRecord(**trace_dict)

        if not trace.model_logs.router_logits or not trace.cot_text.strip():
            continue
        logits_path = Path(trace.model_logs.router_logits)
        if not logits_path.exists():
            continue

        router_logits = torch.load(logits_path, map_location="cpu", weights_only=True)
        if router_logits.numel() == 0:
            continue

        if accum is None:
            accum = _Accumulator(
                num_layers=router_logits.shape[0],
                num_experts=router_logits.shape[2],
            )

        # Rebuild the prompt with the generation-time system prompt so token
        # indices line up with the Exp2 extraction.
        formatted_prompt = _format_prompt(tokenizer, trace.prompt, trace.system_prompt)
        token_ranges = _map_steps_to_token_ranges(
            trace.steps, trace.cot_text, tokenizer, formatted_prompt
        )

        accum.add_trace(router_logits, trace, token_ranges, top_k, args.window)
        n_analyzed += 1

    if accum is None:
        raise RuntimeError("No traces with routing tensors were available to analyze.")

    logger.info(f"Analyzed {n_analyzed} traces")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Co-activation / transition matrices are (L, E, E) — store as npz, keep the
    # JSON portable with a POSIX-style relative path (same convention as Exp2).
    arrays_path = (
        Path(args.arrays_output)
        if args.arrays_output is not None
        else output_path.parent / "expert_arrays.npz"
    )
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        coactivation=accum.coactivation,
        transitions=accum.transitions,
    )

    output_data = {
        "config": {
            "window": args.window,
            "top_k": top_k,
            "model_id": args.model_id,
            "num_layers": accum.num_layers,
            "num_experts": accum.num_experts,
            "n_traces_analyzed": n_analyzed,
        },
        "phases": accum.phase_summary(),
        "before_after": accum.before_after_summary(),
        "arrays_path": arrays_path.as_posix(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_path} (matrices in {arrays_path})")

    # Console summary: per phase, the experts most over-used vs the normal
    # baseline (activation-frequency delta averaged over layers).
    normal_n = accum.n_tokens["normal"]
    if normal_n > 0:
        normal_freq = accum.usage["normal"] / normal_n
        print("\n=== Expert usage by reasoning phase (vs normal baseline) ===")
        print(f"{'Phase':<18} {'tokens':>8}   top over-used experts (mean delta freq across layers)")
        print("-" * 78)
        for phase in PHASES:
            n = accum.n_tokens[phase]
            if phase == "normal" or n == 0:
                continue
            delta = (accum.usage[phase] / n - normal_freq).mean(axis=0)  # (E,)
            top = np.argsort(delta)[::-1][:3]
            tops = ", ".join(f"e{int(e)} (+{delta[e]:.4f})" for e in top)
            print(f"{phase:<18} {n:>8}   {tops}")


if __name__ == "__main__":
    main()
