# Outcome-selected gentle steering

Question: can we make a reasoning model think less without getting worse, by gently turning
down a few experts that the model tends to use when it ends up wrong?

The earlier whole-reasoning experiment (`steering.whole_reasoning`) picked experts by reasoning-step class
(Plan, Implement, ...) and forced them on or off at every token. Almost nothing beat matched random
experts, and cutting the high-traffic "execution" experts broke the models. This package changes two
things, following "Beyond the Trace" (`chen2026beyond`):

1. Which experts: the ones a model uses more in its *wrong* attempts than in its *right* attempts
   of the *same* question (cohort B of the router captures: 6-8 attempts per AIME/AMC question).
2. How strongly: their router logit is lowered by a small `delta` (0.4, 0.8 or 1.6). The router
   still picks its own top-k, so no expert is forced in or out.

The base releases are the finished whole-reasoning releases in `report/`: `whole-reasoning-20261021-r7` for every model except GPT-OSS (`-r8`), with their top-ups and resumes, and `-r17/analysis-combined` for the final table of native and steered continuations. Ledger: task `outcome-steering-r1`, decision `gentle-routing-nudge`.

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
                          WHOLE.json manifest ──> steering.whole_reasoning (prepare, run, extract)
                                                       │
 final whole-reasoning table of continuations ─────>│
                                                       ▼
                          analysis.analyze() ──> analysis/EFFECTS_OUTCOME.json
```

## Files

| File | What it does |
| --- | --- |
| `select.py` | Reads the cohort-B captures and scores every (layer, expert) by wrong-minus-right use, z-scored within each question. It keeps pairs whose sign survives leaving out any one question. It excludes the shared core (top 5% most used per layer) and the most execution-specific experts (top 10% Implement/Analyze). Then it takes the best pairs up to 1.2% of routing (5-40 pairs) and draws matched random controls. The r2 section selects per reasoning class inside three window layers (see below). `usage` refuses a model whose `USAGE.json` is already in the folder it writes to, so measured tables are never overwritten. |
| `manifest.py` | Builds the release rows: nudged and random arms at each delta, plus length-limit rows. A length-limit row is a native run cut at 50/70/85% of its reasoning, with the closing tokens appended so the model only writes its answer. A family whose question was used for selection gets a set selected without it. For r2 it also builds the halves A/B of the question roster and the sweep rows (`sweep_manifest`). |
| `analysis.py` | Bounded within-question contrasts: nudge vs random (primary), vs unedited, the rerun noise, the length-limit curve, and "nudge vs stopping at the same length". It also reports how much routing the nudged experts lost. `analyze_sweep` does the same for one half of the r2 sweep against the mean of the six capture attempts. |
| `decide.py` | The extension rule that was declared before any result: launch the next models only if a nudge is shorter than random without losing accuracy, or more accurate than random. |
| `__main__.py` | Command line: `usage` (`--reports DIR --out DIR`), `select` (preview, writes nothing), `derive`, `analyze`, `pick`, `decide`. A model is a Hugging Face id or a short key. |

Sealing the release is `steering.release outcome` (it also writes the code snapshot). The jobs are the
ordinary whole-reasoning ones (`scripts/steering/submit_release_jobs.sh`) plus `python -m steering.outcome analyze --release report/NAME`.
The files below belong to the second campaign and extend `select.py`, `manifest.py` and `analysis.py` (their `r2` sections).

The routing operator itself lives with the other operators. It is
`steering.operators.score_shift_masked`, applied by `steering.stream`. The
length-limit prefix lives in `steering.runtime.budget_tokens`.

## Second campaign (r2): experts chosen inside each reasoning class

Ledger: tasks `outcome-steering-r2` and `attempt-selection-by-routing`, decision `class-conditioned-outcome-design`. r1 nudged 20-40 experts over every layer, chosen from 37 or
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
scripts/steering/seal_release.sh capture NAME TESTS --shards gpt:3,glm:15 --cutoff 2026-10-30T23:00     # seal
scripts/steering/submit_release_jobs.sh report/NAME                                                      # jobs oc-*
scripts/steering/derive_outcome_sets.sh report/NAME openai/gpt-oss-20b zai-org/GLM-4.7-Flash            # CPU chain per model, each after its extraction
scripts/steering/seal_release.sh sweep NAME_A TESTS --half A --capture NAME --models gpt --shards 15 --cutoff 2026-10-30T23:00
scripts/steering/submit_release_jobs.sh report/NAME_A                                                    # jobs os-*
sbatch --dependency=afterok:ANALYSIS_JOB scripts/steering/pick_and_launch_half_b.sbatch report/NAME_A openai/gpt-oss-20b NAME_B TESTS --capture NAME --shards 2 --cutoff ...   # half B
```

## Running r1

```bash
scripts/steering/measure_expert_usage.sh report OUT_DIR openai/gpt-oss-20b     # 1. CPU job, ~5 min: expert use per cohort-B attempt (the release reads select.BULK)
PYTHONPATH=src python -m steering.outcome select    # 2. look at the expert sets (no files)
scripts/steering/seal_release.sh outcome outcome-steering-20261022-r1 <passed tests> --shards 3 --cutoff 2026-10-23T23:00   # 3. seal
scripts/steering/submit_release_jobs.sh report/outcome-steering-20261022-r1                                                  #    and submit
sbatch scripts/steering/decide_and_launch_next.sbatch report/<release>/analysis/EFFECTS_OUTCOME.json <next name> <tests> --cutoff ... --models ...   # 4. the extension rule
```

Outputs:
- `report/<release>/`: receipts, `<model>/OUTCOME_SELECTION.json`, manifests, job records, and
  `analysis/EFFECTS_OUTCOME.json`.
- `campaign-runs/outcome-steering/` on FAST: the usage arrays. A capture or sweep release keeps its unit stores
  on FAST too (`report/<release>/stores` is a link) and `<capture>/derived/<model>/USAGE_CLASS.npz` beside them.

Reading the results: token contrasts are `log(1+T_control) - log(1+T_treated)`, so a positive value
means shorter reasoning. A run that hits its limit or never finishes keeps an interval, never a
guessed value.
