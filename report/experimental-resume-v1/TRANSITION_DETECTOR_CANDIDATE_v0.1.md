# Prefix transition candidates v0.1

`src/moe_exp/routing_control/transitions.py` is a prospective, prefix-only
proposal. It accepts the original problem and already emitted tokens/text
through the explicit adapter allowlist. Future completion, correctness, gold,
and class labels never enter a decision. Complete numeric candidates use the
qualified candidate parser's exact bounded grammar; an identified approach
requires a method, intent and operation in a completed sentence; a failed
check requires an explicit failure marker and visible equation or constraint
evidence. The latter two patterns are deliberately conservative and abstain on
vague language. All three are **unqualified semantic triggers**.

The discovery-only audit must report each transition's confusion matrix,
coverage, false-trigger rate, family count, and triggering token positions on
truncated prefixes, unfinished expressions, unsupported answer forms, and
reasoning closure. It must compare rule proposals with independent arm-blind
rubric ratings. A transition without sufficient supported positives and
acceptable errors is omitted. No action dictionary or validation enrollment
may treat this module's proposals as semantic truth before that sealed gate.

Six focused prefix tests and the existing routing-control suite passed on
2026-10-01. Tests establish code behavior, not semantic validity.
