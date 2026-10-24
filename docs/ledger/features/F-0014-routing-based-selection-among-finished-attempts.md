# F-0014: Routing-based selection among finished attempts
- status: planned
- date: 2026-10-23
- request: Grilling answer Q1 (a and c): use routing also to choose among finished attempts; Q15 approved as a CPU analysis on the captures.
- purpose: Test whether routing read from finished attempts picks a correct attempt more often than majority vote, a random pick, or the shortest or longest attempt.
- acceptance: ATTEMPT_SELECTION.json per model with family-bootstrap intervals for routing consensus, outcome-profile and class-mix picks against majority vote, random (expected accuracy), shortest and longest, on all questions and on mixed-outcome questions.
- keywords: attempt selection, best-of-n, majority vote, routing consensus, RAD
- paths: src/steering/outcome/attempts.py, tests/steering/outcome
- decisions: D-0006

## Tasks

- [ ] F-0014.1 attempts.py with the three routing picks and four baselines, leave-question-out profiles.
- [ ] F-0014.2 Run on every model's captures after its class usage.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|
