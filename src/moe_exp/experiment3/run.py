import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from moe_exp.schemas import TraceRecord
from moe_exp.models.loader import load_model_and_tokenizer
from moe_exp.models.inference import extract_logs_single_pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_routing_similarity(logits: torch.Tensor) -> np.ndarray:
    """
    Cosine similarity of router probability distributions.
    logits: (N, num_experts)
    Returns: (N, N) numpy array

    Uses a normalized matmul (cosine == dot product of L2-normalized vectors)
    instead of broadcasting cosine_similarity, which would materialize an
    (N, N, num_experts) intermediate and OOM for large N.
    """
    probs = F.softmax(logits, dim=-1)
    probs_norm = F.normalize(probs, p=2, dim=-1)
    sim = torch.matmul(probs_norm, probs_norm.transpose(0, 1))
    return sim.cpu().numpy()


def _compute_correlation_for_mask(
    h_sim_flat: np.ndarray,
    r_sim_flat: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute Pearson correlation for a subset of pairs defined by mask."""
    if np.sum(mask) < 10:
        return float("nan")
    result = pearsonr(h_sim_flat[mask], r_sim_flat[mask])
    return float(result[0])  # type: ignore[arg-type]


def _mantel_test(
    h_sim: np.ndarray,
    r_sim: np.ndarray,
    n_permutations: int = 1000,
    rng: np.random.RandomState | None = None,
) -> tuple[float, float]:
    """Mantel test: permutation-based significance for matrix correlation.
    
    Unlike Pearson p-values, this correctly handles non-independence of pairwise
    distances. Permutes rows/columns of one matrix and recomputes the correlation
    to build a null distribution.
    
    Returns (correlation, p_value).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    n = h_sim.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    
    h_flat = h_sim[upper_idx]
    r_flat = r_sim[upper_idx]
    
    observed_corr = float(np.corrcoef(h_flat, r_flat)[0, 1])
    
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        r_perm = r_sim[np.ix_(perm, perm)]
        r_perm_flat = r_perm[upper_idx]
        perm_corr = np.corrcoef(h_flat, r_perm_flat)[0, 1]
        if perm_corr >= observed_corr:
            count_ge += 1
    
    p_value = (count_ge + 1) / (n_permutations + 1)
    return observed_corr, p_value


def _get_backtracking_token_indices(
    traces: list[TraceRecord],
    tokenizer,
    formatted_prompts: list[str],
) -> list[set[int]]:
    """
    For each trace, identify token indices that fall within backtracking steps.
    Returns a list (one per trace) of sets of local token indices (0-based from gen start).
    
    Uses a single tokenization with offset_mapping to reliably map character
    positions to token positions, avoiding BPE boundary artifacts.
    """
    from moe_exp.models.inference import _find_prompt_length

    result = []
    for trace, fmt_prompt in zip(traces, formatted_prompts):
        bt_indices: set[int] = set()

        if not trace.step_labels.backtracking_steps or not trace.steps:
            result.append(bt_indices)
            continue

        full_text = fmt_prompt + trace.cot_text
        prompt_len = _find_prompt_length(tokenizer, fmt_prompt, full_text)
        prompt_char_len = len(fmt_prompt)

        # Tokenize the full text once with offset_mapping for char→token mapping
        encoding = tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encoding["offset_mapping"][0].tolist()  # list of (char_start, char_end)

        # Build a char→token index lookup for the generation portion
        # Token positions are relative to generation start (i.e., subtract prompt_len)
        def _char_to_gen_token(char_pos_in_cot: int) -> int | None:
            """Map a character position in cot_text to a generation token index."""
            abs_char = prompt_char_len + char_pos_in_cot
            for tok_idx in range(prompt_len, len(offset_mapping)):
                start, end = offset_mapping[tok_idx]
                if start <= abs_char < end:
                    return tok_idx - prompt_len
            return None

        # Locate each step in the original cot_text using progressive search
        # Steps may have been stripped, so we search for them as substrings
        cot = trace.cot_text
        step_char_ranges: list[tuple[int, int]] = []
        search_start = 0
        for step in trace.steps:
            # Search for the step content in the original text
            idx = cot.find(step, search_start)
            if idx == -1:
                # Fallback: try stripping both sides and searching more broadly
                stripped = step.strip()
                idx = cot.find(stripped, search_start)
                if idx == -1:
                    # Last resort: advance by 1 character from previous end
                    idx = search_start
                    step_char_ranges.append((idx, idx))
                    continue
                step_char_ranges.append((idx, idx + len(stripped)))
                search_start = idx + len(stripped)
            else:
                step_char_ranges.append((idx, idx + len(step)))
                search_start = idx + len(step)

        # For each backtracking step, map its character range to token indices
        for bt_step_idx in trace.step_labels.backtracking_steps:
            if bt_step_idx >= len(step_char_ranges):
                continue
            char_start, char_end = step_char_ranges[bt_step_idx]
            if char_start == char_end:
                continue  # Skip steps we couldn't locate

            # Find all tokens whose character span overlaps [char_start, char_end)
            abs_start = prompt_char_len + char_start
            abs_end = prompt_char_len + char_end
            for tok_idx in range(prompt_len, len(offset_mapping)):
                tok_char_start, tok_char_end = offset_mapping[tok_idx]
                # Token overlaps with step if intervals intersect
                if tok_char_start < abs_end and tok_char_end > abs_start:
                    bt_indices.add(tok_idx - prompt_len)

        result.append(bt_indices)
    return result


def process_file(
    input_path: Path,
    model_id: str,
    output_path: Path,
    limit: int | None = None,
    num_sampled_tokens: int = 1000,
    chunk_size: int = 20,
):
    """
    Run Experiment 3 offline-extraction loop over traces to compute geometric routing relations.
    Processes traces in chunks to limit memory usage.
    """
    logger.info(f"Loading model {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, quantization="none")

    traces_raw: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            traces_raw.append(json.loads(line))

    if limit is not None:
        traces_raw = traces_raw[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(traces_raw)} traces for geometry extraction...")

    # Accumulate per-layer token representations in streaming chunks
    # to avoid holding everything in GPU memory at once.
    # We store: layer -> list of (hidden_chunk, router_chunk, type_chunk, backtrack_chunk)
    num_layers: int | None = None
    layer_h_chunks: list[list[torch.Tensor]] = []
    layer_r_chunks: list[list[torch.Tensor]] = []
    token_type_chunks: list[np.ndarray] = []
    backtrack_flag_chunks: list[np.ndarray] = []

    from moe_exp.models.inference import _format_prompt

    for chunk_start in range(0, len(traces_raw), chunk_size):
        chunk_raw = traces_raw[chunk_start:chunk_start + chunk_size]
        chunk_traces: list[TraceRecord] = []
        chunk_router: list[torch.Tensor] = []
        chunk_hidden: list[torch.Tensor] = []
        # Tri-state correctness: 1 = correct, 0 = failed, -1 = unknown
        # (is_correct is None when answer comparison was ambiguous).
        chunk_correct: list[int] = []
        formatted_prompts: list[str] = []

        for trace_dict in tqdm(chunk_raw, desc=f"Chunk {chunk_start // chunk_size + 1}", leave=False):
            trace = TraceRecord(**trace_dict)
            if not trace.cot_text.strip():
                continue

            router_tensor, hidden_tensor = extract_logs_single_pass(
                model=model,
                tokenizer=tokenizer,
                problem=trace.prompt,
                cot_text=trace.cot_text,
                extract_hidden_states=True,
                system_prompt=trace.system_prompt,
            )

            if router_tensor.numel() > 0:
                chunk_router.append(router_tensor)
                chunk_hidden.append(hidden_tensor)
                if trace.is_correct is True:
                    chunk_correct.append(1)
                elif trace.is_correct is False:
                    chunk_correct.append(0)
                else:
                    chunk_correct.append(-1)
                chunk_traces.append(trace)
                formatted_prompts.append(
                    _format_prompt(tokenizer, trace.prompt, trace.system_prompt)
                )

        if not chunk_hidden:
            continue

        if num_layers is None:
            num_layers = chunk_router[0].shape[0]
            layer_h_chunks = [[] for _ in range(num_layers)]
            layer_r_chunks = [[] for _ in range(num_layers)]

        # Build backtracking token sets
        bt_token_sets = _get_backtracking_token_indices(chunk_traces, tokenizer, formatted_prompts)

        # Flatten tokens per layer and build type/backtrack arrays
        for layer in range(num_layers):
            h_cat = torch.cat([h[layer] for h in chunk_hidden], dim=0)
            r_cat = torch.cat([r[layer] for r in chunk_router], dim=0)
            layer_h_chunks[layer].append(h_cat)
            layer_r_chunks[layer].append(r_cat)

        # Build token-level metadata (same across layers since token count is per-trace)
        types = []
        bt_flags = []
        for i, h in enumerate(chunk_hidden):
            seq_len = h.shape[1]
            types.extend([chunk_correct[i]] * seq_len)
            bt_set = bt_token_sets[i]
            bt_flags.extend([t in bt_set for t in range(seq_len)])

        token_type_chunks.append(np.array(types, dtype=np.int8))
        backtrack_flag_chunks.append(np.array(bt_flags))

        # Free chunk memory
        del chunk_router, chunk_hidden

    if num_layers is None:
        logger.error("No valid traces extracted.")
        return

    # Concatenate metadata
    all_types = np.concatenate(token_type_chunks)
    all_bt_flags = np.concatenate(backtrack_flag_chunks)
    total_tokens = len(all_types)

    n_bt_tokens = int(all_bt_flags.sum())
    logger.info(f"Total tokens: {total_tokens} | Backtracking tokens: {n_bt_tokens}")
    logger.info("Computing similarities...")

    layer_results = []
    rnd = np.random.RandomState(42)

    # Two separate samples to avoid biasing population-level statistics:
    #
    #   base_indices: a uniform random sample over ALL tokens. Backtracking is
    #     rare (<1%), so this sample reflects the true population. It is used for
    #     the overall correlation / Mantel test and the correct/failed splits.
    #
    #   bt_indices: a stratified sample that guarantees backtracking tokens are
    #     present (all BT tokens + a random set of others). It is used ONLY for
    #     the backtracking-conditional correlation, where a uniform sample would
    #     contain ~0 BT-BT pairs.
    #
    # Mixing the oversampled BT tokens into the overall correlation (as a single
    # stratified sample would) makes "overall" non-representative of the corpus.
    bt_indices_all = np.where(all_bt_flags)[0]
    non_bt_indices_all = np.where(~all_bt_flags)[0]

    # --- Representative uniform sample ---
    all_idx = np.arange(total_tokens)
    n_base = min(num_sampled_tokens, total_tokens)
    base_indices = rnd.choice(all_idx, n_base, replace=False) if n_base < total_tokens else all_idx.copy()
    base_size = len(base_indices)
    if base_size < 2:
        logger.error("Not enough tokens to compute correlations.")
        return
    base_types = all_types[base_indices]

    # --- Stratified BT sample (for the backtracking-conditional metric only) ---
    n_bt_to_use = min(len(bt_indices_all), num_sampled_tokens // 4)  # cap BT at 25% of budget
    n_non_bt = min(num_sampled_tokens - n_bt_to_use, len(non_bt_indices_all))
    if n_bt_to_use > 0:
        bt_sample = rnd.choice(bt_indices_all, n_bt_to_use, replace=False) if n_bt_to_use < len(bt_indices_all) else bt_indices_all
    else:
        bt_sample = np.array([], dtype=np.intp)
    non_bt_sample = rnd.choice(non_bt_indices_all, n_non_bt, replace=False) if n_non_bt < len(non_bt_indices_all) else non_bt_indices_all
    bt_indices = np.concatenate([bt_sample, non_bt_sample]).astype(np.intp)
    rnd.shuffle(bt_indices)
    bt_flags_sampled = all_bt_flags[bt_indices]
    bt_size = len(bt_indices)

    logger.info(
        f"Base (uniform) sample: {base_size} tokens "
        f"({int(all_bt_flags[base_indices].sum())} backtracking). "
        f"BT-stratified sample: {bt_size} tokens "
        f"({int(bt_flags_sampled.sum())} backtracking)."
    )

    def _cosine_sim(h: torch.Tensor) -> np.ndarray:
        h_norm = F.normalize(h, p=2, dim=-1)
        return torch.matmul(h_norm, h_norm.transpose(0, 1)).cpu().numpy()

    for layer in range(num_layers):
        # Concatenate all chunks for this layer
        layer_h = torch.cat(layer_h_chunks[layer], dim=0)
        layer_r = torch.cat(layer_r_chunks[layer], dim=0)

        # --- Population-level metrics on the uniform sample ---
        base_h = layer_h[base_indices].to(torch.float32)
        base_r = layer_r[base_indices].to(torch.float32)
        h_sim = _cosine_sim(base_h)
        r_sim = get_routing_similarity(base_r)

        upper_idx = np.triu_indices(base_size, k=1)
        h_sim_flat = h_sim[upper_idx]
        r_sim_flat = r_sim[upper_idx]

        # Overall correlation using Mantel test (permutation-based p-value)
        # This correctly handles non-independence of pairwise similarities
        corr, p_value = _mantel_test(h_sim, r_sim, n_permutations=1000, rng=rnd)

        # Unknown correctness (-1) is excluded from both splits rather than
        # being lumped in with failed traces.
        types_i = base_types[upper_idx[0]]
        types_j = base_types[upper_idx[1]]
        correct_mask = (types_i == 1) & (types_j == 1)
        failed_mask = (types_i == 0) & (types_j == 0)
        corr_correct = _compute_correlation_for_mask(h_sim_flat, r_sim_flat, correct_mask)
        corr_failed = _compute_correlation_for_mask(h_sim_flat, r_sim_flat, failed_mask)

        # --- Backtracking-conditional metric on the stratified sample ---
        if bt_size >= 2:
            bt_h = layer_h[bt_indices].to(torch.float32)
            bt_r = layer_r[bt_indices].to(torch.float32)
            h_sim_bt = _cosine_sim(bt_h)
            r_sim_bt = get_routing_similarity(bt_r)
            bt_upper = np.triu_indices(bt_size, k=1)
            bt_i = bt_flags_sampled[bt_upper[0]]
            bt_j = bt_flags_sampled[bt_upper[1]]
            bt_mask = bt_i & bt_j
            corr_bt = _compute_correlation_for_mask(
                h_sim_bt[bt_upper], r_sim_bt[bt_upper], bt_mask
            )
            num_bt_pairs = int(np.sum(bt_mask))
        else:
            corr_bt = float("nan")
            num_bt_pairs = 0

        # Free full layer tensors
        del layer_h, layer_r

        layer_results.append({
            "layer": layer,
            "overall_correlation": corr,
            "correct_correlation": corr_correct,
            "failed_correlation": corr_failed,
            "backtracking_correlation": corr_bt,
            "p_value": p_value,
            "num_correct_pairs": int(np.sum(correct_mask)),
            "num_failed_pairs": int(np.sum(failed_mask)),
            "num_backtracking_pairs": num_bt_pairs,
        })

        logger.info(
            f"Layer {layer:02d} | Corr: {corr:.4f} | Correct: {corr_correct:.4f} "
            f"| Failed: {corr_failed:.4f} | Backtrack: {corr_bt:.4f}"
        )

    # Write overall results
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(layer_results, out_f, indent=2)

    logger.info(f"Finished Experiment 3 geometry analysis. Output saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 3: Geometric routing analysis")
    parser.add_argument("--input", type=str, required=True, help="Path to input traces.jsonl from Exp 1")
    parser.add_argument("--output", type=str, required=True, help="Path to output analysis JSON file")
    parser.add_argument("--model_id", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N traces")
    parser.add_argument("--samples", type=int, default=1000, help="Number of token samples for N^2 pairwise comparisons")
    parser.add_argument("--chunk-size", type=int, default=20, help="Number of traces to process per chunk (memory control)")

    args = parser.parse_args()

    input_file = Path(args.input)
    if input_file.exists():
        process_file(
            input_path=input_file,
            model_id=args.model_id,
            output_path=Path(args.output),
            limit=args.limit,
            num_sampled_tokens=args.samples,
            chunk_size=args.chunk_size,
        )
    else:
        logger.error(f"Could not find input file: {input_file}")
