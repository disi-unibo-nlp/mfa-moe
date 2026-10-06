# Arm-blind immediate transition ratings, discovery v1

The rater sees the original problem, the complete native reasoning prefix through
the triggering sentence, and one generated reasoning continuation of at most
256 tokens. It sees no arm, seed, expert identity, other continuation, gold
answer, future native text or correctness field. The triggering sentence is
context only; it cannot itself satisfy the outcome.

For `candidate_to_verify`, return `{"target":true}` only when the new
continuation substantively tests the same previously unchecked candidate or
value against an original constraint or an independent calculation. Words such
as “check” or “verify,” repetition, unrelated arithmetic and a final answer
alone do not qualify.

For `approach_to_commit`, return `{"target":true}` only when the new
continuation commits to the identified approach with a concrete planning or
implementation step that advances the original problem. Restating the method,
generic planning language, and switching to an unrelated approach do not
qualify. If the approach is not actually identifiable from the supplied prefix,
return false and separately flag a questionable start.

Rate only reasoning tokens before the first reasoning-closure marker. Closure
without the target behavior is false. Ambiguity is false. Two independently
seeded Qwen3.8 readers provide arm-blind LLM audits; report each vote,
agreement, parse/stop coverage and the conservative both-positive result.
These are LLM measurements, not human ground truth.

The analysis map retains every assigned request, including errors, missing
continuations, early closure, length caps and unusable judge responses.
Generation failures and invalid ratings count as no observed target in the
all-assigned intention-to-treat numerator and are reported separately. The
screen measures only immediate behavior within 256 tokens. The registered
1024-token trajectory endpoint requires a separately frozen continuation
study; failure to observe a target by 256 tokens does not establish absent
steering at longer horizons.
