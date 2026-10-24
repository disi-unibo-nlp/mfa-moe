# Candidate-start measurement diagnostic v1

This discovery-only review applies the already frozen trigger-1 definition in
`TRANSITION_RUBRIC_v0.1.md`. It does not revise an intervention, endpoint,
eligibility rule, or the causal primary analysis. The 52-row frame is blind to
prior Qwen and native votes. The separate key must not be sent to a reader.

For each record, the reader sees exactly `problem`, `emitted_prefix`, and
`triggering_sentence`. The emitted prefix ends immediately before the rated
sentence. Later text, gold answers, correctness, detector flags, expert arms,
and prior readers' ratings are forbidden. The reader should answer:

1. What specific value or answer is proposed by the triggering sentence?
   Quote only a short span from that sentence, or say `none`.
2. Is that value a **complete proposed value or answer** for a well-defined
   quantity at this point? A complete intermediate value may qualify if its
   quantity is identified. A rearranged equation, a partial expression, a
   setup fact copied from the problem, or an unfinished calculation does not
   qualify solely because it contains a number.
3. Has the same proposed value already been substantively checked earlier in
   the visible prefix against an original condition or independent
   computation? Identify the prior check if yes. Repetition and words such as
   “check” without evaluated evidence do not count as a substantive check.
4. Does the rated sentence itself evaluate the candidate against an original
   condition or independent calculation? If yes, it is a check rather than an
   unchecked start.

Record `yes`, `no`, or `uncertain` for the three judgments. The derived
unchecked-start judgment is `yes` only when item 2 is `yes` and items 3 and 4
are both `no`; `uncertain` on a required item yields `uncertain`. Report a
short rationale citing visible text. Do not infer intended future checks from
later native continuations. This diagnostic does **not** grade any later
transition or test whether a routing action helps.

The frame includes all 26 prior Qwen-both-true candidate rows and 26 distinct
Qwen-both-false v2.2-fire controls, matched by same family where available and
then nearest native-prefix length. These strata were selected using previous
ratings, so this diagnostic estimates neither population precision nor an
independent validation effect. Any model review is an LLM audit, and reader
disagreement must be preserved. Freeze a prospective controller rule and
evaluate it on independent families before treating it as qualified.
