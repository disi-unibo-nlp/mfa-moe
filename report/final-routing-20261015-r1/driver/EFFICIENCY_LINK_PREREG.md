# Efficiency link pre-registration (outcome-blind)

Created 2026-10-17T00:09:58Z. `intervention_outcomes_used: false`. Machine-readable receipt:
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
| **pooled** | **−** | **0** | **0** | **0** | **−** | **0** | **0** |

### Pooled estimates (mean of the seven models, 95% interval)

| | Read | Analyze | Plan | Implement | Explore | Verify | Monitor |
|---|---|---|---|---|---|---|---|
| beta_tok | +0.085 [+0.073, +0.098] | +0.071 [+0.065, +0.076] | +0.055 [+0.047, +0.064] | +0.014 [+0.007, +0.021] | +0.024 [+0.018, +0.031] | -0.054 [-0.061, -0.047] | -0.196 [-0.212, -0.183] |
| beta_acc | -0.002 [-0.004, -0.001] | +0.000 [-0.001, +0.002] | +0.001 [-0.001, +0.002] | +0.001 [-0.000, +0.002] | -0.003 [-0.004, -0.002] | +0.001 [-0.000, +0.002] | -0.001 [-0.002, +0.000] |

### Consistency across models (at least 5 of 7 models share the sign, none opposite)

- tokens, sign by class: negative: Verify, Monitor; positive: Read, Analyze, Plan
- accuracy, sign by class: negative: none; positive: none

### Robustness (sensitivity variants; they never change the pre-registered signs)

- Models whose primary non-zero sign is reversed (opposite non-zero sign) by a variant, tokens:
  Read 5/7 (poscontrol).
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
  them reported as `acc_excl_trunc`. Truncated traces per model: gpt 0, nemotron 3, qwen35 3, qwen36 0, gemma 32, glm 8, qwen330b 27.
- Accuracy is a trace-level outcome: with question fixed effects it is identified only from
  questions that have several attempts with different outcomes (6-37 questions per
  model), so `beta_acc` is far less precise than `beta_tok` and 39 of the
  49 model x class accuracy cells have sign `0`.
- `beta_tok` uses remaining REASONING tokens (the native tables do not store the completion
  token end position); the secondary `tok_total` adds the answer tokens.
- lfm25 has no native lexicon table: its cells have no prediction.
