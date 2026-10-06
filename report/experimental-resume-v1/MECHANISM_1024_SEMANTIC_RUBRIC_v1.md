# Mechanism validation: blinded 1,024-token semantic endpoint v1

Each Qwen3.8 reader sees one original problem, the complete native reasoning
prefix through the accepted triggering sentence, the frozen transition type,
and one generated reasoning continuation. It sees no arm, seed, expert set,
route, family, other continuation, later native text, or correctness field.
The triggering sentence is context and cannot itself satisfy the endpoint.

For `candidate_to_verify`, `{"target":true}` requires a substantive test of
the same previously unchecked candidate or value against an original
constraint or an independent calculation in the new continuation. Checking
words, repetition, unrelated arithmetic, and a final answer alone are false.

For `approach_to_commit`, `{"target":true}` requires commitment to the
identified approach through a concrete planning or implementation step that
advances the original problem. Restating the method, generic planning, or a
switch to an unrelated approach are false. If the approach is not identifiable
from the native prefix, return false.

Only tokens before the first reasoning-closure marker are decoded for rating.
The full available reasoning continuation up to 1,024 tokens is visible;
closure without the behavior and ambiguity are false. Two independently
seeded Qwen3.8 readers rate every gradeable continuation. A valid vote is an
exact JSON boolean with a natural `stop`; the conservative primary endpoint
requires both valid positive votes. Report each vote, parse/stop coverage,
agreement, closure, cap, and generation failures. These are LLM measurements,
not human ground truth.

Every assigned arm remains in the intention-to-treat denominator, including
generation errors, missing route arrays, zero reasoning, reader failure,
early closure, and length caps. Those with no valid positive pair contribute
zero observed target. Family-clustered contrasts compare target +1 with native
and matched random +1, and quantify native duplicate variation. A frozen
sensitivity analysis compares target +1 with the average of both native draws.
Only the single frozen action is evaluated; no second transition or action
order is inferred.
