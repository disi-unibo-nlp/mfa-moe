# Behavioral transition rubric v0.1

Frozen before any routing-action validation outputs exist. This is a content
rubric for independent arm-blind readers. Seven-class labels locate candidate
windows; they never by themselves satisfy a transition. Readers see the
original problem, the emitted reasoning through the rated sentence, and a
random UID. They do not see the arm, expert set, dose, correctness, gold
answer, other readers' ratings, or predicted class labels. Readers rate the
trigger and each later candidate event separately. Uncertain or inaccessible
text is recorded as uncertain and counts as no success in the assigned-arm
primary analysis; agreement and an uncertainty sensitivity are reported.

## Trigger 1: unchecked candidate

The prefix must contain a complete proposed value or answer. A boxed or
closed expression can be a candidate even if wrong; an unfinished expression,
symbolic form outside the qualified parser, or a mere intention to compute
is unsupported by the online detector. The trigger is not a verification.
A later sentence qualifies as **substantive verification** only if it tests
the candidate or an intermediate result against an original condition, derives
an independent computation and compares it, or substitutes it into a stated
equation. “Check,” “verified,” “looks right,” and a repeated answer without an
evaluated condition are insufficient.

## Trigger 2: identified approach

The prefix must tentatively name a specific method, construction, case split,
formula, or representation and the operation it would perform. General
uncertainty or a bare “try another way” is insufficient. A later sentence
qualifies as **committed planning or implementation** when it selects that
approach for a concrete next operation, or executes its first concrete step.
A disconnected plan or restatement of the tentative idea is insufficient.

## Trigger 3: visibly failed check

The prefix must display a check that produces a contradiction, an invalid
value, or a violated original condition, with enough computation or constraint
detail to see what failed. A statement that something “seems wrong” is
insufficient. A later sentence qualifies as **revised exploration or
planning** when it explicitly changes an assumption, method, branch, or next
operation in response to the failed check. Continuing the same computation
without repair is insufficient.

## Timing and record rules

The triggering sentence is excluded from every endpoint. The first later
qualifying sentence must begin before branch-relative token 1,024 and within
the same reasoning segment. A sentence that straddles the horizon is counted
only if it starts before the horizon. A class-only transition, answer closure,
or unobserved future text is no success. The frozen template may contain at
most two ordered actions; both corresponding events must occur in order for
trajectory success. A substantive event can fill only one slot. Nonfires,
early finishes, caps, crashes, and unscored outputs stay in intention-to-treat
denominators with an explicit receipt.

Report each reader's raw rating and their disagreement before adjudication.
Any LLM application of this rubric is reported as an LLM audit. Strong claims
about latent reasoning require separate measurement validation; observed
language is the operational endpoint.
