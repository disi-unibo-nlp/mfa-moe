# Outcome-selected gentle steering

Question: can we make a reasoning model think less without getting worse, by gently turning
down a few experts that the model tends to use when it ends up wrong?

The earlier whole-reasoning experiment (`steering.whole`) picked experts by reasoning-step class
(Plan, Implement, ...) and forced them on or off at every token. Almost nothing beat matched random
experts, and cutting the high-traffic "execution" experts broke the models. This package changes two
things, following "Beyond the Trace" (`chen2026beyond`):

1. Which experts: the ones a model uses more in its *wrong* attempts than in its *right* attempts
   of the *same* question (cohort B of the router captures: 6-8 attempts per AIME/AMC question).
2. How strongly: their router logit is lowered by a small `delta` (0.4, 0.8 or 1.6). The router
   still picks its own top-k, so no expert is forced in or out.

The base releases are the finished whole-reasoning releases in `report/`: `whole-reasoning-20261021-r7` for every model except GPT-OSS (`-r8`), with their top-ups and resumes, and `-r17/analysis-combined` for the final table of native and steered continuations. Ledger: feature F-0004, decision D-0003.

## Data flow

```
cohort-B routing captures ──usage()──> USAGE.npz / USAGE.json        (per attempt: expert use per token)
                                          │
base release WHOLE_NOMINATIONS.json ─> selection()                (score, stability, protection, controls)
                                          │
                                          ▼
                          OUTCOME_SELECTION.json  ──>  manifest.outcome_manifest()
                                                       │   target_nudge / random_nudge at each delta
 base releases' native runs (+ top-ups) ───────────>│   budget rows cut from the native runs
                                                       ▼
                          WHOLE.json manifest ──> steering.whole (prepare, run, extract)
                                                       │
 final whole-reasoning table of continuations ─────>│
                                                       ▼
                          analysis.analyze() ──> analysis/EFFECTS_OUTCOME.json
```

## Files

| File | What it does |
| --- | --- |
| `select.py` | Reads the cohort-B captures and scores every (layer, expert) by wrong-minus-right use, z-scored within each question. It keeps pairs whose sign survives leaving out any one question. It excludes the shared core (top 5% most used per layer) and the most execution-specific experts (top 10% Implement/Analyze). Then it takes the best pairs up to 1.2% of routing (5-40 pairs) and draws matched random controls. The r2 section selects per reasoning class inside three window layers (see below). |
| `manifest.py` | Builds the release rows: nudged and random arms at each delta, plus length-limit rows. A length-limit row is a native run cut at 50/70/85% of its reasoning, with the closing tokens appended so the model only writes its answer. A family whose question was used for selection gets a set selected without it. For r2 it also builds the halves A/B of the question roster and the sweep rows (`sweep_manifest`). |
| `analysis.py` | Bounded within-question contrasts: nudge vs random (primary), vs unedited, the rerun noise, the length-limit curve, and "nudge vs stopping at the same length". It also reports how much routing the nudged experts lost. `analyze_sweep` does the same for one half of the r2 sweep against the mean of the six capture attempts. |
| `decide.py` | The extension rule that was declared before any result: launch the next models only if a nudge is shorter than random without losing accuracy, or more accurate than random. |
| `__main__.py` | Command line: `usage`, `select` (preview, writes nothing), `derive`, `analyze`, `pick`, `decide`. |

Sealing the release is `steering.release outcome` (it also writes the code snapshot). The jobs are the
ordinary whole-reasoning ones (`scripts/steering/launch.sh`) plus `python -m steering.outcome analyze --release report/NAME`.
The files below belong to the second campaign and extend `select.py`, `manifest.py` and `analysis.py` (their `r2` sections).

The routing operator itself lives with the other operators. It is
`steering.operators.score_shift_masked`, applied by `steering.stream`. The
length-limit prefix lives in `steering.runtime.budget_tokens`.

## Second campaign (r2): experts chosen inside each reasoning class

Ledger F-0013 and F-0014, decision D-0006. r1 nudged 20-40 experts over every layer, chosen from 37 or
fewer questions, and behaved like a random edit. r2 keeps the outcome idea and changes the rest:

1. **Fresh data** (`capture.py`). Every model answers the same 300 questions six times with unedited
   routing. Every reasoning token keeps its native expert ids and weights at every routed layer
   (`capture_mode: native_ids` in `steering.stream`; the arrays are saved as `AUDIT.npz` per unit).
2. **Classes** (`classes.py`). Each reasoning sentence gets a class from the lexical readout (the
   TF-IDF classifier used for every continuation in this study); each token inherits it. Expert use is
   counted per class per attempt.
3. **Selection** (`select.py`, r2 section). For each class, an expert scores high when the model uses it
   more on that class's tokens in wrong attempts than in right attempts of the same question. Only the
   three layers at depth 1/2, 2/3 and 5/6 are candidates (layers 12/16/20 of GPT-OSS, where "Beyond the
   Trace" found its five experts). The best pairs are taken until they carry 1.2% of those layers' gate
   weight, the paper's own definition, at most 10. Each set gets two disjoint random sets
   matched by layer and usage (nearest usage in the layer when a fixed set leaves too few experts in
   its quintile). On GPT-OSS the class-blind set is the paper's five.
4. **Evidence** (`evidence.py`). Per class: the real score against 200 label shuffles, replication on
   held-out questions, and how much the score just tracks attempt length.
5. **Choosing among finished attempts** (`attempts.py`). Routing picks one of six attempts; it is compared
   with majority vote, a random pick and the shortest and longest attempt.
6. **Sweep** (`manifest.py`, `analysis.py`, `pick.py`). The questions are split into halves A and B. Half A
   runs every set at shifts 0.4/0.8/1.2 against both random sets; the rule in `pick.py` (fixed before any
   result) chooses the cell that half B confirms. The unedited arm is the six capture attempts.

```
capture release ──extract──> derive.py: classes -> select (SELECTION_R2) -> evidence (SIGNAL)
                                                                         -> attempts (ATTEMPT_SELECTION)
SELECTION_R2 ──> sweep half A (OUTCOME_SWEEP.json) ──analysis──> pick.py (PICK.json) ──> sweep half B
```

| File | What it does |
| --- | --- |
| `capture.py` | The 300-question roster and the six native rows per question with routing saved (`CAPTURE_NATIVE.json`). |
| `classes.py` | Lexical labels, per-token classes and per-class expert counts per attempt (`USAGE_CLASS`). |
| `evidence.py` | The per-class signal table (`SIGNAL.json`). |
| `attempts.py` | Routing-based choice among finished attempts (`ATTEMPT_SELECTION.json`). |
| `derive.py` | Runs classes, selection, evidence and attempts for one model of a capture release. |
| `pick.py` | The frozen pick rule; writes `PICK.json` and tells the launcher whether half B has cells to run. |

The sealed receipts of the campaign are `report/outcome-capture-20261023-r2` (captures and derived
selections) and `report/outcome-steering-20261023-r6` (half A of GPT-OSS; its analysis never ran, the campaign
was stopped). Commands:

```bash
scripts/steering/whole.sh capture NAME TESTS --shards gpt:3,glm:15 --cutoff 2026-10-30T23:00     # seal + jobs (oc-*)
scripts/steering/outcome_derive.sh NAME gpt glm      # CPU chain per model, each after its extraction
scripts/steering/whole.sh sweep NAME_A TESTS --half A --capture NAME --models gpt --shards 15 --cutoff 2026-10-30T23:00   # jobs os-*
sbatch --dependency=afterok:ANALYSIS_JOB scripts/steering/outcome_pick.sh NAME_A gpt NAME_B TESTS --capture NAME --shards 2 --cutoff ...   # half B
```

## Running r1

```bash
scripts/steering/outcome_usage.sh                # 1. CPU job, ~5 min: expert use per cohort-B attempt
PYTHONPATH=src python -m steering.outcome select    # 2. look at the expert sets (no files)
scripts/steering/whole.sh outcome outcome-steering-20261022-r1 <passed tests> --shards 3 --cutoff 2026-10-23T23:00   # 3. seal + jobs
sbatch scripts/steering/outcome_next.sh report/<release>/analysis/EFFECTS_OUTCOME.json <next name> <tests> --cutoff ... --models ...   # 4. the extension rule
```

Outputs:
- `report/<release>/`: receipts, `<model>/OUTCOME_SELECTION.json`, manifests, job records, and
  `analysis/EFFECTS_OUTCOME.json`.
- `campaign-runs/outcome-steering/` on FAST: the usage arrays. A capture or sweep release keeps its unit stores
  on FAST too (`report/<release>/stores` is a link) and `<capture>/derived/<model>/USAGE_CLASS.npz` beside them.

Reading the results: token contrasts are `log(1+T_control) - log(1+T_treated)`, so a positive value
means shorter reasoning. A run that hits its limit or never finishes keeps an interval, never a
guessed value.

<!-- ledger:begin -->
## Ledger

Decisions and features that govern this folder (generated by `ledger sync`):

- [D-0003](../../../docs/ledger/decisions/D-0003-outcome-selected-gentle-routing-nudge.md) [active] Outcome-selected gentle routing nudge: chosen Select experts by outcome on campaign-v3 cohort B (per-attempt usage per reasoning token, z-scored within question, mean wrong-minus-right over mixed-outcome questions, sign stable under leave-one-question-out). Exclude the shared core (top 5% native frequency per layer) and the most execution-specific experts (union over folds of the top 10% positive Implement/Analyze risk differences). Take stable positive pairs until 1.2% of native routing mass (5 to 40 pairs; fill to 5 by score with an unstable_fill flag if needed). Families whose question is in cohort B use the set selected without that question. Subtract delta in {0.4, 0.8, 1.6} from the selected experts' router logits at every prompt and reasoning token, keeping native top-k and renormalisation. Matched random controls with the same layers, frequency quintiles and exclusions. Reuse the base releases' native and native_repeat rows. Length-limit baseline: cut each family's native reasoning at 50/70/85% of its length, append the closing suffix and generate only the answer (allowance min(2048, remaining)).; scope: The outcome-steering experiment (F-0004) on gpt, qwen330b and nemotron; extension to the other models with cohort-B captures only if the three-model result is positive.; not: Changes to the finished whole-reasoning releases r7-r17 or their analysis; manuscript claims before a sealed receipt.
- [D-0006](../../../docs/ledger/decisions/D-0006-class-conditioned-outcome-steering-design-r2.md) [active] Class-conditioned outcome steering design (r2): chosen Captures: the same 300 questions for every model (aime24 30, aime25 30, amc23 40, olympiad 100, minerva 50, math500 50; digest-ranked within benchmark, one question per family, no outcome read), 6 native attempts each, medium effort, original prompt tokens, 32,768-token completion cap on every row of r2, native top-k ids and weights of every reasoning token at every routed layer. Labels: lexical readout v1 (TF-IDF), out-of-fold by FOLDS.json; no judge. Score per class c: within-question z-scored usage on tokens labelled c, mean of wrong minus mean of right, over questions with both outcomes; capped or unknown attempts excluded. Sets: one per class (7) plus a class-blind set (GPT-OSS: Beyond the Trace's five pairs L12-E23, L12-E12, L12-E24, L16-E11, L20-E0; other models: the plain outcome set). Set rule from related work: candidates only in the three routed layers at relative depth 1/2, 2/3, 5/6 (n//2, 2n//3, 5n//6; 12/16/20 on GPT-OSS, Beyond the Trace); best positive pairs until they carry 1.2% of those layers' native gate weight per reasoning token, at most 10 pairs (the paper's own definition, confirmed by the ELM grounding check of 2026-10-23: "1.2% of the gated mass across the three recorded layers", gate weight accumulated over routed assignments; the slot share is reported beside it); committee experts (top 5% native use per layer, Illusion of Specialization) and the top-10% Implement/Analyze experts never nudged; logit shift with native top-k and renormalisation (SteerMoE-style operator). Each evaluation family uses sets selected without its own question. Arms: target nudge and two disjoint matched random draws (same layer, native-frequency quintile) per set and shift; shifts 0.4, 0.8, 1.2 at every prompt and reasoning token; unedited arm = the 6 capture attempts; stop-early baseline at 50/70/85% of a native attempt. Halves: the 300 questions split by a benchmark-stratified digest into A and B. Pick rule (frozen now): among half-A cells with accuracy against native no worse than -5 points (point estimate) take the most negative tokens-against-mean-random point estimate; if none qualifies, the best accuracy-against-random estimate. Half B confirms that cell and the class-blind set at its best shift. Primary reported contrast: tokens against the mean of the two random draws with accuracy against native noninferior within 5 points, on half B; every half-A cell is exploration; no success threshold gates anything. Attempt selection on the captures: routing consensus, outcome-profile and class-mix scores against majority vote, random, shortest and longest.; scope: The second outcome-steering campaign (F-0013, F-0014): native captures on all eight models, class-conditioned expert selection, the GPT-OSS nudge sweep on question half A, the automatic pick and its confirmation on half B, and the attempt-selection analysis. The other seven models' sweeps are decided jointly after the GPT-OSS half-B receipt.; not: Release r1 and its analysis (kept unchanged, reported as the broad-edit test); the whole-reasoning releases r7-r20; GPT-OSS low/high reasoning effort (a later GPT-only extension); manuscript claims before a sealed receipt.
- [F-0004](../../../docs/ledger/features/F-0004-outcome-selected-gentle-steering-experiment.md) [active 6/6] Outcome-selected gentle steering experiment
- [F-0013](../../../docs/ledger/features/F-0013-class-conditioned-outcome-steering-on-eight-models.md) [active 3/7] Class-conditioned outcome steering on eight models

<!-- ledger:end -->
