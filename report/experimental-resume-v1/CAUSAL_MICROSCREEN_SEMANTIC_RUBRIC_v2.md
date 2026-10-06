# Arm-blind candidate-to-substantive-check rubric (v2)

Rate eligibility and outcome in separate, blinded tasks. The same Qwen3.8
model supplies two independent seeded reads of each task. These are LLM
audits of visible text, not human ground truth.

**Start eligibility:** For each of four unique frozen native prefixes, give
the reader only the original problem, the entire emitted prefix and its final
triggering sentence. No continuation or arm identity is visible. Return
exactly `{"start": true}` if that sentence presents a complete specific
candidate or computed value that the prefix has not yet substantively tested
against an original constraint or independent calculation. Unfinished
expressions, generic plans and previously checked candidates are false.

**Target outcome:** For each of 48 assigned continuations, give the reader
the same original problem and full emitted native prefix, its triggering
sentence, and exactly that continuation. Omit arm identity, seed, other
continuations and unemitted native future text. Return exactly
`{"target": true}` if the new continuation substantively tests that same
candidate against an original constraint or by an independent calculation.
Verification language, repetition, an unrelated calculation or a final answer
alone is false. Early reasoning closure without a substantive check is false.

Ambiguity is false in either task. Both readers must return valid stopped JSON
and agree positive for the conservative outcome. Preserve all assigned arms in
the later intention-to-treat join, even if a prefix fails start eligibility,
generation fails or the judge response is invalid. Report start eligibility
as a pre-treatment population descriptor rather than selecting arms on a
post-treatment rating.
