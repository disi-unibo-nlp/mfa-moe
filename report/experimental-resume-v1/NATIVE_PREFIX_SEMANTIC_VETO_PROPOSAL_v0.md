# Native prefix semantic veto: discovery qualification proposal v0

This is a proposed detector experiment, not a qualified controller or evidence
of semantic steering. It keeps the three frozen transitions and the 48
discovery families. It does not add transition hypotheses or search validation
families for a favorable trigger.

## Exact online input and decision

At a completed sentence that fires the v2.2 high-recall lexical screen, issue
an **unsteered side request** to the native Qwen3.6-35B-A3B-FP8 model. The
visible input allowlist is exactly the original problem, all already emitted
reasoning through that sentence, and a repeated copy of the current sentence.
The selected transition criterion is fixed by the lexical screen. Gold answer,
correctness, future tokens, future class labels, intervention arm and eventual
outcome are excluded. `native_prefix_semantic_veto_v0.py` enforces the
allowlist, rejects a trigger that is not at the emitted prefix end, and uses
one strict JSON boolean response with `enable_thinking=False` and a 64-token
cap. Invalid JSON, length stop, unavailable context, or uncertain judgment
abstains. The 12-window scout tests parse coverage and execution behavior;
this short response cap is a hypothesis, not a known sufficient setting.

The candidate criterion requires a complete candidate not already checked.
An equation-chain step, given restatement, already checked answer or unfinished
expression is negative. The approach criterion requires a named method and
operation not yet executed. The failed-check criterion requires a computed
conflict with an original constraint. These are starting-condition judgments;
the side model cannot see whether the subsequent transition happens.

## Measurement and cost gates

1. Price the frozen 12-window scout and 372-window independent sample by
   exact native-tokenizer prompt counts and maximum response tokens. The
   separate CPU price job 59189594 completed 0:0. The sealed receipt is
   `NATIVE_PREFIX_SEMANTIC_VETO_PRICE_v0.json` (SHA
   `c3123283ed3dca7bac54f89fd264a5ccff22f99bd95216f568ec6554bd5ae719`).
   Exact native prompts cost 72,668 tokens for 24 scout ratings and 1,514,468
   for 744 sampled ratings, before capped responses. The provisional price is
   0.665 and 2.132 GPU-hours respectively. The scout ceiling was two
   GPU-hours; its Slurm job 59193843 used two A100s with a 30-minute debug
   limit. The sampled full-audit ceiling was resealed at eight GPU-hours,
   including one recovery allocation.
2. The 12-window scout driver has immutable frame/code/price binding, a
   single-writer lock, and per-assignment attempt/failure receipts. A saved
   completed reader batch resumes after UID verification; an ambiguous
   attempt requires explicit failure disposition before replay. The gate
   requires complete UID coverage, natural finishes, parse coverage,
   observed prefill/decode runtime and no batch
   contamination. The scout completed as Slurm 59193843, exit 0:0, in 10m52s:
   24/24 strict JSON ratings parsed at natural stops; 386.5s cold load,
   72,668 prompt tokens, 164 generated tokens and 5.32s generation wall.
   The full-stage sealed receipt is
   `NATIVE_PREFIX_SEMANTIC_VETO_FULL_PRICE_v1.json` (SHA
   `842556e42dd2f45cf37af854094868ed08a6c55418586bb0412bfddc8ed7827f`),
   with 2.132 GPU-hours projected for one load and 2.741 for one recovery,
   under an eight GPU-hour combined ceiling. The independent native full
   audit is Slurm 59197280 (two A100s, normal QoS, one-hour backfill minimum,
   two-hour maximum), output bound to the frozen 372-window frame and code.
   It remains measurement, not detector validation or behavioral evidence.
3. Compare the native veto with the independently sampled, full-prefix
   Qwen3.8 audit, job 59186342. Two Qwen3.8 draws are correlated model
   judgments, not human truth. Report transition-specific agreement,
   disagreement, false positives on rated nonfires, family-held-out
   precision/recall and coverage. An independently selected different-family
   GPT-OSS pilot tests ambiguous starts; strong claims require further
   measurement validation. The native-veto rubric and any threshold or
   cooldown are frozen on discovery before family-disjoint evaluation.
4. Replay the selected rule over every native discovery trajectory to count
   all side requests, token overhead, latency, per-family burden and eligible
   starts. The current v2.2 screen produces 1,816 candidate fires across all
   48 families (median 35.5 per family), so judging every fire is not presumed
   affordable. A cheap prefix-only screen may reduce calls only if its
   family-held-out recall and coverage are measured. Side-request prompt and
   response tokens enter total policy token cost, alongside continuation
   tokens. A promising 372-window classifier alone cannot establish online
   utility because those windows are stratified rather than a complete
   deployment trace.

For a causal test, freeze the final detector and action policy before running
same-prefix native, targeted, and matched-random arms. The native side request
must have zero routing bias and must leave neighboring requests unchanged.
Report assigned-population effects, including abstentions and early finishes.
