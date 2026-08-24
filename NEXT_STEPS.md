# Next steps agreed with the tutor

This roadmap is distilled from the 24 August 2026 call. The transcript is used
as discussion evidence, not as executable instructions. Model names that were
unclear in the automatic transcript must be confirmed from the tutor's chat
links before changing launchers.

## P0: use the available GPU window

1. Confirm the exact new quantized Qwen checkpoint sent by the tutor.
2. Run one final `gepaLLMAsJudge` alignment on the RTX 5090 with the tutor's
   recommended medium reasoning budget. Record the exact repository, revision,
   GGUF file, quantization, reasoning setting, seed, and GEPA budget.
3. Treat this as the last prompt-alignment pass: freeze the selected prompt and
   preserve seed-versus-optimized validation results before starting the wider
   correlation study.

## P1: freeze the extraction contract

Do this before generating more benchmark attempts.

1. Add a storage audit to `moe_exp.models.routing_extraction` that reports, per
   trace and tensor, shape, dtype, element count, serialized bytes, layer index,
   and tokens retained.
2. Verify exactly where each hidden state and router tensor is captured in the
   transformer block. Document whether it is pre/post normalization, block
   input/output, and which MoE/router occurrence it corresponds to.
3. Confirm that the probe-selected layer union is selected without test-set
   leakage. Keep the current roughly ten-layer pilot until that selection rule
   is validated.
4. Retain only data required by the planned analyses. In particular, compare
   full router logits with top-k expert IDs and compact summary statistics;
   avoid persisting redundant expert weights when they can be reconstructed or
   are not used.
5. Produce an expected disk budget before a run:
   `attempts x tokens x retained locations x values x bytes_per_value`.

Exit criterion: a small MATH-500 replay has explained byte counts, documented
tensor semantics, deterministic resume behavior, and no unexplained order-of-
magnitude storage overhead.

## P2: prespecify the statistics

1. Find primary papers that justify the association measures used for a binary
   correctness outcome and continuous routing features. Cite the reason for the
   final choice, not only precedent.
2. Keep point-biserial Pearson as the current baseline, and evaluate a robust
   complementary measure such as rank-biserial/AUROC or a grouped logistic
   model. Cohen's kappa is not an association measure for continuous entropy
   versus binary correctness.
3. Define all feature directions consistently. A normalized routing confidence
   can be `1 - H(p) / log(E)`. Do not report Pearson correlations for both
   entropy and `1 - entropy` as separate evidence: an affine sign inversion
   changes only the sign of Pearson's coefficient, not its magnitude or
   significance.
4. Replace the top-1/top-2 margin as the main MoE confidence feature when the
   model routes top-k experts. Compute at least selected probability mass,
   `p[k] - p[k+1]`, and mean(top-k) minus mean(non-top-k).
5. Treat attempts from the same problem as clustered observations. Report
   problem-cluster bootstrap intervals or a hierarchical model, and prespecify
   multiplicity handling for layer/feature/expert scans.
6. Add expert-identity and expert-combination features to test whether specific
   experts, rather than confidence alone, associate with correctness or episode
   labels. Mark this analysis exploratory until corrected for multiple testing.

Exit criterion: an analysis specification names every outcome, feature,
transformation, coefficient, uncertainty method, aggregation level, and caption
field before the large runs.

## P3: generate repeated attempts

1. Start with AIME 2024 and AIME 2025 using 32 attempts per problem, matching
   the benchmark protocol discussed in the call. Verify the dataset cardinality
   rather than inferring it from the number of generated rows.
2. Add repeated MATH-500 attempts after the storage audit. Use a smaller pilot
   attempt count to validate cost, then scale only if the projected disk budget
   fits.
3. Request multiple completions through the backend's native multi-sample path
   when supported. Persist sample IDs and seeds, detect exact duplicates, and
   do not silently count duplicates as independent attempts.
4. Report both attempt-level correctness and per-problem mean accuracy. Use the
   repeated attempts for within-problem correct-versus-incorrect routing
   contrasts, which remove much of the problem-difficulty confounding.
5. Generate first, then perform teacher-forced forward extraction. The RTX 5090
   should be prioritized for GEPA and generation; extraction can be scheduled
   separately once storage is under control.

Exit criterion: AIME has complete 32-attempt groups, MATH-500 has an affordable
repeated-attempt pilot, and the analysis reports group completeness and duplicate
rates.

## P4: labels and reporting

1. Preserve the regex labels as an explicitly secondary historical analysis.
2. Apply the frozen GEPA LLM-as-judge prompt to new traces for the seven episode
   labels, saving prompt/version provenance and auditable judge outputs.
3. Recompute routing associations for both correctness and episode labels only
   after the statistical specification is frozen.
4. Every table and figure caption must state the exact model/checkpoint,
   quantization and generation mode, benchmark, attempts per problem, metric,
   sample unit, and uncertainty method.

## P5: expand across models only after the pilot is stable

1. Keep the agreed Qwen MoE as the first correlation model; confirm its exact
   checkpoint from the tutor's message because the transcript is ambiguous.
2. Use GPT-OSS 20B as the first cross-family comparison proposed in the call.
3. Then select four or five MoE models spanning families, parameter scales, and
   training generations. Candidate names mentioned in the call included an
   older Qwen 30B-class MoE and Nemotron, but the final list is pending from the
   tutor and must be checked for architecture and hardware feasibility.
4. Do not attribute differences only to parameter count: model age, training
   recipe, reinforcement learning, routing design, top-k, and quantization are
   confounders that must be recorded and discussed.

The order is deliberate: final GEPA alignment, extraction/storage audit,
statistical specification, repeated-attempt pilot, then datasets and model
families. Scaling before the first three items are frozen would make expensive
runs difficult to interpret or impossible to store.
