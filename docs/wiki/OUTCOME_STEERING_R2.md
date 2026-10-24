# Outcome steering r2: design, runs, findings and open questions

Written 2026-10-24 (Claude, at the author's request). It follows `OUTCOME_STEERING_REVIEW.md`
(why r1 failed) and records the second campaign: what was decided, what ran, what the data shows so
far, and the questions the author raised about it. Ledger: D-0006, F-0013, F-0014. Code: branch
`outcome/r2` (worktree `.claude/worktrees/outcome-r2`), `src/moe_exp/outcome_steering/`.

## Status when this was written

- **Kept running, then stop.** GPT-OSS half-A steering release `report/outcome-steering-20261023-r6`
  (15 shards, its extraction 62132028 and analysis 62132037). Its receipt will be
  `report/outcome-steering-20261023-r6/analysis/EFFECTS_OUTCOME.json`.
- **Cancelled by the author's instruction.** The pick job that would have launched half B
  (`-r7`), and the extraction and analysis jobs of GLM, Qwen3-30B, Qwen3.5 and Qwen3.6. Their capture
  shards were already running and were left to end; their stores stay on FAST and can be extracted
  later with the release's own extraction stage.
- **No steering on the other seven models.** Nothing beyond the GPT-OSS half A is planned until the
  author decides.

## What was decided (grilling sessions of 2026-10-23)

| Item | Choice |
|---|---|
| Goal | A positive efficiency result from outcome-selected experts, plus routing-based choice among finished attempts; the expert pick must use the paper's class analysis |
| Data | The same 300 questions for all eight models, 6 native attempts each, medium effort, 32,768-token cap, every reasoning token's native expert ids and weights saved |
| Labels | Lexical (TF-IDF) readout only, no judge |
| Score | Per class: within-question z-scored expert usage on that class's tokens, wrong attempts minus right attempts |
| Sets | One per class (7) and a class-blind set; on GPT-OSS the class-blind set is Beyond the Trace's five experts |
| Where | The three routed layers at relative depth 1/2, 2/3, 5/6 (12, 16, 20 on GPT-OSS) |
| Size | Best experts until 1.2% of those layers' gate weight, minimum 5, maximum 10 |
| Edit | Lower the selected experts' router scores by 0.4, 0.8 or 1.2 at every prompt and reasoning token; the router still picks its own top-k and renormalises; the answer is never edited |
| Controls | Two disjoint random sets per expert set, same layers and usage band; unedited arm = the six capture attempts; stop-early baseline |
| Protocol | Half A of the questions runs every set and strength; a rule fixed in advance picks one cell; half B confirms it |
| Gate | GPT-OSS first; the other seven only after a joint review |

## What ran

| Run | Outcome |
|---|---|
| Capture r1 | Failed allocated validation: one test pinned an error message I had reworded. No shard ran. |
| Capture r2 | Validated and ran. GPT-OSS, LFM2.5, Gemma and Nemotron extracted and analysed. |
| GPT-OSS analysis, first try | Failed: the paper's five experts put three in one layer and usage band, leaving too few comparable experts for two random sets (GPT-OSS has 32 experts per layer). Fixed with a nearest-usage fallback for that fixed set only. |
| Half A r2 | Failed before sealing: `saldo` does not exist on compute nodes. |
| Half A r4 | Failed allocated validation: the sweep manifest used a field name the validation audit reads differently. Its analysis and pick were cancelled before they could act on empty data. |
| Half A r6 | Running. Fixes were checked with an end-to-end CPU dry run on real captures before launch. |

No failure spent GPU hours. Lessons: run the full `tests/study` suite locally before sealing, rehearse
every post-GPU step on real data before launch, and keep a failure watcher armed (memory
`wake-up-on-failure`).

The ELM grounding check (`mfa-moe/r/claude-review/outcome-r2-grounding-20261023/report.md`) settled:
Beyond the Trace's "1.2%" is a share of gate weight in layers 12/16/20; its selection and edit used
low effort; its 12 evaluation problems are not stated to be held out from selection; it shifted by
0.3 or 0.8 at every token and compared with no edit and one random equal-strength edit.

## Findings so far (before any r2 steering result)

**1. Little wrong-versus-right routing signal inside reasoning classes.** For each class, the score of
the best window experts against 200 label shuffles (share of shuffles reaching it) and how often the
chosen experts still score positive on held-out questions:

| Model | Mixed-outcome questions | Classes below 0.05 | Best class |
|---|---|---|---|
| GPT-OSS | 73 | none | Verify 0.15, held-out positive 90% |
| LFM2.5 | 23 | Explore 0.04 | Monitor 0.09, held-out positive 98% (partly length) |
| Gemma | 26 | none | Plan 0.12, held-out positive 80% |
| Nemotron | 26 | none | Plan and Monitor 0.19 |

With eight sets per model, one result near 0.05 is what chance gives. The best class differs by
model, so no reasoning class is a general lever. Most models get most questions all right or all
wrong, which leaves few questions to learn from.

**2. The paper's five experts do not replicate on new data.** On cohort B (hard AIME/AMC questions)
all five leaned toward wrong attempts. On the new 300 questions four of five lean toward right
attempts (L16-E11 t = -1.65, L20-E0 t = -2.08). Only 7 of 768 GPT-OSS expert positions pass |t| > 3,
against about 2 expected by chance.

**3. Routing does not pick better attempts than counting answers.** Accuracy of the attempt each
selector keeps, out of six finished attempts:

| Model | Majority vote | Random pick | Best routing selector |
|---|---|---|---|
| GPT-OSS | 0.807 | 0.748 | 0.770 (Verify experts) |
| LFM2.5 | 0.820 | 0.778 | 0.803 (routing consensus) |
| Gemma | 0.873 | 0.843 | 0.853 (Implement experts) |
| Nemotron | 0.827 | 0.762 | 0.773 (class mix) |

Majority vote is the standard self-consistency baseline: the most common final answer among the six
attempts. It is used only as the bar for this attempt-selection test, never in steering or selection.
On GPT-OSS, picking the attempt that uses the paper's five experts least does worse than random.

## Questions the author raised, and the answers

**What is steered, where and how?** Five experts per set in layers 12/16/20 (GPT-OSS), router scores
lowered by 0.4/0.8/1.2 on every prompt and reasoning token, against two matched random sets. This is
the same place and size as Beyond the Trace, much gentler than SteerMoE (forced in or out, up to 500
experts) and RICE (gate weight ×64), and static rather than state-dependent.

**Why exactly 5?** No metric of ours sets it. It is a minimum copied from Beyond the Trace. On every
model so far two or three experts already exceed 1.2% of the window's gate weight, so the minimum
decides, and the routing share of a set ranges from 1.2% to 5.4%. The signal table supports far fewer
reliable experts (close to none). Better rules for any further run: a fixed routing share per set
(equal strength across classes and models), and only experts whose score replicates on held-out
questions (classes without such experts are not steered).

**Why does earlier work seem to steer and we do not?** Earlier positives are weaker than they look:
SteerMoE moved coarse behaviours (refusal, context use) mostly against the unedited model; RICE gained
a few AIME problems with settings chosen on the test; Beyond the Trace reports +2.6 points on 12
problems with no accuracy interval. Audits that compare against matched random experts (The Illusion
of Specialization, the Command A+ modularity study, the causal audit of expert importance) mostly find
specialization dissolves, as ours does. Where we may be wrong: unstable outcome-linked experts,
medium instead of low effort, usage that marks failure rather than causes it (the router then hands
the work to similar experts), one static edit on every token, and few mixed-outcome questions.

**What role do the paper's metrics play in this experiment?** Too small a role, which weakens the
narrative. Only the class labels are used (to split tokens by class); the shared-core protection uses
the retention finding. Not used: the present-state readout, the class risk differences (only for
protection), the efficiency link (which classes precede shorter or longer reasoning) and the
next-state readout. A version driven by the paper's metrics would:
1. take experts that mark the class (risk differences) and lean toward wrong attempts within it
   (signal table);
2. choose classes and direction from the efficiency link: lower wrong-linked experts in classes tied
   to longer reasoning, raise right-linked experts in Verify and Monitor;
3. keep only experts that replicate on held-out questions, at a fixed routing share;
4. later, apply the edit only while the readout says the model is in the target class.

## Paper story this supports

Routing encodes the current reasoning activity (readout). Activities differ in how much reasoning
follows them (dynamics). Within an activity, routing carries little wrong-versus-right information
(signal table), and routing does not choose attempts better than majority vote. Static expert edits,
including a faithful copy of Beyond the Trace's set, are then tested against random edits (half A).
If half A is null, the claim is: routing reads the reasoning state, but fixed expert edits do not
steer it beyond a random edit, in line with the matched-control audits. The remaining distinct
levers are a low-effort GPT-OSS replication and a state-triggered edit.

## Files

- Capture release: `report/outcome-capture-20261023-r2/` (per model `SELECTION_R2.json`,
  `SIGNAL.json`, `ATTEMPT_SELECTION.json`); arrays on FAST under
  `campaign-runs/outcome-capture-20261023-r2/derived/`.
- Steering half A: `report/outcome-steering-20261023-r6/`.
- Failed attempts with receipts: capture r1 (`VALIDATION_FAILED.json`), steering r2
  (`LAUNCH_FAILED.json`), r4 (`VALIDATION_FAILED.json`).
- Grounding check: `mfa-moe/r/claude-review/outcome-r2-grounding-20261023/`.
