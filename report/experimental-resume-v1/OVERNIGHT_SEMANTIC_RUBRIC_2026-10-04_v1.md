# Overnight routing comparison: blinded semantic measurement v1

The generation manifest freezes the horizon, 256 or 1,024 new tokens, before
submission. Each horizon/design is a separate exploratory discovery analysis.
The reader sees only the transition name, original problem, entire emitted
native reasoning prefix, triggering sentence and one assigned new reasoning
continuation. Arm, experts, seed, other continuations, gold answer, correctness,
future native text and execution metadata are excluded.

For candidate_to_verify, return {"target":true} only if the NEW continuation
substantively tests the same previously unchecked candidate or value against an
original constraint or by an independent calculation. Checking language,
repetition, unrelated calculation and a final answer alone are false.

For approach_to_commit, return {"target":true} only if the NEW continuation
commits to the identified approach through a concrete planning or implementation
step that advances the original problem. Restatement, generic planning and an
unrelated approach are false. If the approach is unclear from the prefix,
return false.

The prefix and triggering sentence supply context only and cannot satisfy the
outcome. Read only tokens before the first reasoning closure marker. Closure
without the target behavior and ambiguity are false. Return exactly
{"target":true} or {"target":false}. Two independently seeded Qwen3.8 draws
provide arm-blind LLM audits, not human ground truth or independent models.

The primary endpoint is both valid naturally stopped readers positive. All
assigned generation errors, missing route arrays, empty reasoning and invalid
reader responses stay in the denominator with no observed success; their rates
are reported by arm. The any-reader endpoint is a separate sensitivity. No new
response is sampled merely because a reader returned an invalid response.
Family bootstrap intervals simultaneously cover the contrasts, scopes and
primary endpoints frozen in each generation design. These discovery families
have already been analyzed and provide exploratory selection, not confirmation.
Neither a 256-token nor a 1,024-token continuation estimates original-prompt
16k accuracy or utility.
