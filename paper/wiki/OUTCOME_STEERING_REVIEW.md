# Outcome steering r1: why the expert selection failed

Review of 2026-10-23 (Claude, at the author's request). It covers release
`report/outcome-steering-20261022-r1` (ledger F-0004, decision D-0003). Nothing in
that release was changed.

## Summary

We picked the right kind of experts (used more in wrong than in right attempts of the same
question), and on GPT-OSS this metric independently finds the experts the reference paper used.
But we nudged far too many of them, spread over every layer, chosen from too little data. An edit
that broad disturbs the model about as much as random experts do, so it never beat random experts.
The paper's own effect is small (a few accuracy points), so even a faithful copy may need more
problems than r1 had to detect it. r1 is therefore a test of a broad outcome-based edit, not of the
paper's narrow one, and its null result does not refute the paper.

## What the run did

- Models: GPT-OSS-20B (medium reasoning effort), Qwen3-30B-A3B, Nemotron.
- Selection data: campaign-v3 cohort B, 6-8 attempts per AIME/AMC question with routing captures.
- Metric, per (layer, expert):
  1. usage = how often the expert is selected per reasoning token, per attempt;
  2. z-score usage across the attempts of each question;
  3. score = mean over questions with both outcomes of (mean z of wrong attempts) minus
     (mean z of right attempts).
- Filters: keep experts whose score keeps its sign when any one question is left out; never touch
  the top 5% most-used experts per layer, or the top 10% Implement/Analyze experts.
- Size: best experts until they carry 1.2% of all routing slots over all layers (5-40 experts).
- Edit: subtract 0.4, 0.8 or 1.6 from their router logits at every prompt and reasoning token;
  the router still picks its own top-k.
- Controls: random experts matched by layer and usage quintile, unedited runs, and a
  stop-thinking-earlier baseline (native reasoning cut at 50/70/85%).

Selected sets (`<model>/OUTCOME_SELECTION.json`):

| Model | Attempts | Mixed questions | Experts nudged | Share of all routing | Layers used |
|---|---|---|---|---|---|
| GPT-OSS-20B | 276 | 37 | 20 | 1.21% | 12 of 24 |
| Qwen3-30B | 219 | 29 | 40 (cap) | 0.39% | 25 of 48, mostly 44-46 |
| Nemotron | 96 | 14 | 40 (cap) | 1.07% | 19 |

Results (`analysis/EFFECTS_OUTCOME.json`):
- Nudge vs matched random: no accuracy or length interval excludes zero for any model or strength.
- Nudge vs unedited accuracy: -5.0 to +1.7 points, all intervals include zero.
- The nudge is not gentle in effect: the nudged experts keep 24-34% of their top-k slots at 0.4,
  7-18% at 0.8 and 0-5% at 1.6.
- The automatic extension rule did not fire (`DECISION.json`, extend false).

## Reference papers

- **Beyond the Trace** (`chen2026beyond`, arXiv 2608.17638): the direct source. On GPT-OSS-20B at
  low effort it standardises expert usage within question over 5,834 runs and picks five experts
  used more when wrong (L12-E23, L12-E12, L12-E24, L16-E11, L20-E0), about 1.2% of routing in the
  three recorded layers. Subtracting 0.8 at every token shortened answers by 46 tokens (random
  sham: +48) and gave accuracy 0.406 vs 0.380 unedited vs 0.359 sham on 12 problems. Router edits
  are reported only for GPT-OSS-20B.
- **The Illusion of Specialization** (standing committee): motivated protecting the most-used
  experts.
- **SteerMoE**: inspired the earlier whole-reasoning runs (experts picked by reasoning-step risk
  difference); D-0003 rejected that for this run.
- **RICE**: background only.

The paper facts above come from the ELM deep-research extraction of 2026-10-22
(`mfa-moe/r/claude-review/why-steering-20261022/dr-bib/`). Per the author's rule they were not
re-read by Claude; the open points are listed under next steps.

## Checks run on 2026-10-23

Scripts: `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/claude-review/selection-review-20261023/`
(`null.py <model> 200`, `length.py <model>`, correlation-client env, CPU only, on the r1 usage
arrays in `campaign-runs/outcome-steering/`).

| Check | GPT-OSS | Nemotron | Qwen3-30B |
|---|---|---|---|
| Chosen set's mean score, real vs shuffled right/wrong labels (95th pct) | 0.40 vs 0.35 | 0.76 vs 0.61 | 0.77 vs 0.48 |
| Shuffles reaching the real score (of 200) | 0 | 0 | 0 |
| Experts passing the leave-one-out filter, real vs shuffled mean | 349 vs 290 | 955 vs 953 | 2148 vs 2020 |
| Top set picked on half the questions: score kept on the other half | ~40% | ~30% | ~55% |
| Correlation of the score with a within-question length score | 0.24 | 0.33 | 0.69 |

- The signal is real: the chosen sets beat shuffled labels in every model. (The shuffled top set is
  drawn without the protection rules, so this comparison is conservative.)
- The leave-one-out filter does nothing: shuffled labels pass as many experts.
- The scores are noisy: about half of each chosen set's score does not replicate on new questions.
  Nemotron, with 14 questions, is close to noise.
- For Qwen3-30B the score mostly means "used in longer attempts": wrong attempts there are 0.78
  within-question SDs longer (log length) than right ones, against 0.22 (GPT-OSS) and 0.27 (Nemotron).
- Replication on GPT-OSS: our score is positive and leave-one-out stable for all five of the
  paper's experts. L12-E12 and L12-E24 rank 5th and 6th of 768 and are in our set; L12-E23 ranks
  63rd, L20-E0 146th, L16-E11 164th.

## What we did wrong

1. Too many experts in too many places: 20-40 experts over all layers, against 5 in 3 middle layers.
2. Probably misread the paper's 1.2%: it appears to be a share of its three layers, but we applied
   it to all layers. On GPT-OSS that touches about 8 times more routing (about 1.16 against 0.14
   expert-slots per token, assuming the paper's "gated mass" counts the same thing).
3. Too little selection data: 96-276 attempts against 5,834.
4. A stability filter that does not filter (see checks).
5. A selection that tracks length rather than errors on Qwen3-30B.
6. Three models at once, although the paper showed the edit only on GPT-OSS, and at low effort
   where we used medium.

## What explains the bad results

These are hypotheses; none is tested directly.
- A broad edit acts like a random one. Once dozens of experts are nudged, target and random arms
  disturb the model about equally. This matches every earlier routing intervention in this repo:
  generic effects, never better than matched random (`report/ROUTING_STEERING.md`).
- Noisy picks dilute whatever real effect exists.
- The target effect is small: the paper's gain is about 2.6 points over unedited and 4.7 over
  sham on 12 problems, and 46 tokens of length. r1's accuracy intervals are about ±2.5 points, so
  a 2.6-point gain would likely not show as a clear result.

## Earlier wrong assumptions, and why this was not done before

- Reasoning steps map to experts. Every earlier run picked experts by step class (Plan, Implement,
  ...) and assumed switching them on or off would change how much of that step the model does.
  Inclusion moved; the step mix did not.
- "SteerMoE proved routing steering works." SteerMoE steered coarse behaviours (refusal, context
  use), compared mostly against no steering, and never tested fine-grained reasoning steps.
- Stronger is clearer. Forcing experts fully on or off was meant to make effects easy to see; it
  mostly broke models, with damage growing with the routing mass removed.
- Outcome-based selection arrived late. It was adopted on 2026-10-22, after the literature search
  found Beyond the Trace; before that the design followed the paper's step-label theme.
- Generalising before replicating. The recipe was stretched to three models and all layers at once,
  using existing captures, under a GPU deadline, instead of first copying it exactly on the one
  model it was shown on. This is a reading of the records: the ledger records the author's choices
  of models, strengths and protection, not why all layers were used.

## Recommended next steps

1. ELM check (no GPU): confirm the paper's 1.2% definition, its reasoning effort, and whether its 12
   evaluation problems overlap the selection questions.
2. Faithful GPT-OSS-only replication: the paper's 5 experts, and separately our top 5 in layers
   12/16/20; shift 0.8; low and medium effort; matched random and unedited arms. Size it from the
   measured rerun noise to detect about 3 accuracy points, which needs more than 128 families.
3. Other models only if step 2 is positive, after collecting more attempts per question.
4. If steering stays null, use routing the way the paper's main positive result does: to choose
   among finished attempts, or to decide when to stop and resample.
