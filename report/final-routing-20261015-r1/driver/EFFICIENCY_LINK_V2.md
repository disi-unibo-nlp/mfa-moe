# Efficiency link, eight models (v2)

Created 2026-10-23T09:21:17Z. `intervention_outcomes_used: false`. Machine-readable receipt:
`EFFICIENCY_LINK_PREREG.json` (sha256 of the script and of every input recorded there).

**What is predicted.** For each model and class, the sign of the native association between the
class of the current sentence and (i) the log of the reasoning tokens that remain after it
(`beta_tok`, negative = shorter path) and (ii) trace correctness (`beta_acc`), at the same
absolute position of the same question (question x position-bin fixed effects, no relative
position, no trace effects). Sign `+`/`-` only if the 95% family-cluster bootstrap interval
(2,000 draws, seed 20261017) excludes 0; otherwise `0`. Later test: Spearman correlation between
`beta_tok(m,c)` and the measured intervention token effect over (model, class) cells, plus the
sign-agreement rate; analogously for accuracy (positive correlation predicted).

### Signs of beta_tok (remaining reasoning tokens)

| model | Read | Analyze | Plan | Implement | Explore | Verify | Monitor |
|---|---|---|---|---|---|---|---|
| gpt | + | + | + | 0 | + | − | − |
| nemotron | + | + | + | + | 0 | − | − |
| qwen35 | 0 | + | + | 0 | + | − | − |
| qwen36 | 0 | + | + | + | 0 | − | − |
| gemma | + | + | + | − | + | − | − |
| glm | + | + | 0 | + | 0 | − | − |
| qwen330b | + | + | + | + | + | − | − |
| lfm25 | 0 | + | + | + | + | − | − |
| **pooled** | **+** | **+** | **+** | **+** | **+** | **−** | **−** |

### Signs of beta_acc (trace correct; truncated = incorrect)

| model | Read | Analyze | Plan | Implement | Explore | Verify | Monitor |
|---|---|---|---|---|---|---|---|
| gpt | − | 0 | 0 | 0 | − | 0 | 0 |
| nemotron | 0 | + | 0 | 0 | − | 0 | 0 |
| qwen35 | 0 | 0 | 0 | 0 | − | 0 | 0 |
| qwen36 | 0 | 0 | + | − | 0 | 0 | 0 |
| gemma | 0 | 0 | 0 | 0 | 0 | + | 0 |
| glm | 0 | + | 0 | 0 | 0 | 0 | 0 |
| qwen330b | 0 | 0 | 0 | + | 0 | 0 | 0 |
| lfm25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **pooled** | **−** | **0** | **0** | **0** | **−** | **0** | **0** |

### Pooled estimates (mean of the eight models, 95% interval)

| | Read | Analyze | Plan | Implement | Explore | Verify | Monitor |
|---|---|---|---|---|---|---|---|
| beta_tok | +0.077 [+0.065, +0.089] | +0.067 [+0.061, +0.073] | +0.055 [+0.047, +0.063] | +0.015 [+0.009, +0.022] | +0.024 [+0.018, +0.031] | -0.051 [-0.058, -0.044] | -0.197 [-0.212, -0.183] |
| beta_acc | -0.002 [-0.003, -0.000] | +0.000 [-0.000, +0.001] | +0.001 [-0.001, +0.002] | +0.001 [-0.000, +0.002] | -0.002 [-0.004, -0.001] | +0.001 [-0.000, +0.002] | -0.001 [-0.002, +0.000] |

### Consistency across models (at least 5 of 8 models share the sign, none opposite)

- tokens, sign by class: negative: Verify, Monitor; positive: Read, Analyze, Plan, Explore
- accuracy, sign by class: negative: none; positive: none

### Robustness (sensitivity variants; they never change the pre-registered signs)

- Models whose primary non-zero sign is reversed (opposite non-zero sign) by a variant, tokens:
  Read 5/8 (poscontrol).
- Same for accuracy: none.
- Between-question variant (dataset x position-bin effects), pooled signs. Tokens: Read 0 Analyze + Plan 0 Implement 0 Explore 0 Verify − Monitor −.
  Accuracy: Read 0 Analyze 0 Plan 0 Implement + Explore − Verify + Monitor 0.

### Caveats

- Association, not causation: class and remaining length are both outcomes of the same reasoning
  path; the later intervention test is what links them.
- Labelled sentences are a stratified SUBSET of each trace (median 8-33% of units); labels are
  single temperature-1.0 judge draws. Identical judge inputs inside a trace (repetition loops)
  are reduced to their first occurrence. Estimates are unweighted; 1/p design weights are a
  sensitivity analysis only.
- Truncation: for tokens, sentences of traces that hit the generation cap are excluded (their
  remaining length is censored); for accuracy they count as incorrect, with the variant excluding
  them reported as `acc_excl_trunc`. Truncated traces per model: gpt 0, nemotron 3, qwen35 3, qwen36 0, gemma 32, glm 8, qwen330b 27, lfm25 13.
- Accuracy is a trace-level outcome: with question fixed effects it is identified only from
  questions that have several attempts with different outcomes (0-37 questions per
  model), so `beta_acc` is far less precise than `beta_tok` and 46 of the
  56 model x class accuracy cells have sign `0`.
- `beta_tok` uses remaining REASONING tokens (the native tables do not store the completion
  token end position); the secondary `tok_total` adds the answer tokens.
- lfm25 is estimated after its interventions ran (no native table existed at pre-registration);
  the seven other models are identical to EFFICIENCY_LINK_PREREG.json.
