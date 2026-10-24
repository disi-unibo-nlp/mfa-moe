# Outcome-selected gentle steering

**Question.** Can we make a reasoning model think less without getting worse, by gently turning
down a few experts that the model tends to use when it ends up wrong?

The earlier whole-reasoning experiment (`steering.whole`) picked experts by reasoning-step class
(Plan, Implement, ...) and forced them on or off at every token. Almost nothing beat matched random
experts, and cutting the high-traffic "execution" experts broke the models. This package changes two
things, following "Beyond the Trace" (`chen2026beyond`):

1. **Which experts.** The ones a model uses more in its *wrong* attempts than in its *right* attempts
   of the *same* question (campaign-v3 cohort B, 6-8 attempts per AIME/AMC question).
2. **How strongly.** Their router logit is lowered by a small `delta` (0.4, 0.8 or 1.6). The router
   still picks its own top-k, so no expert is forced in or out.

Ledger: feature F-0004, decision D-0003.

## Data flow

```
cohort-B routing captures ──usage()──> USAGE.npz / USAGE.json        (per attempt: expert use per token)
                                          │
r7 WHOLE_NOMINATIONS.json ──────────> selection()                  (score, stability, protection, controls)
                                          │
                                          ▼
                          OUTCOME_SELECTION.json  ──>  manifest.outcome_manifest()
                                                       │   target_nudge / random_nudge at each delta
 base releases' native runs (r7/r8 + top-ups) ────────>│   budget rows cut from the native runs
                                                       ▼
                          WHOLE.json manifest ──> steering.whole (prepare, run, extract)
                                                       │
 final whole-reasoning table (r17 continuations) ─────>│
                                                       ▼
                          analysis.analyze() ──> analysis/EFFECTS_OUTCOME.json
```

## Files

| File | What it does |
| --- | --- |
| `select.py` | Reads the cohort-B captures and scores every (layer, expert) by wrong-minus-right use, z-scored within each question. It keeps pairs whose sign survives leaving out any one question. It excludes the shared core (top 5% most used per layer) and the most execution-specific experts (top 10% Implement/Analyze). Then it takes the best pairs up to 1.2% of routing (5-40 pairs) and draws matched random controls. |
| `manifest.py` | Builds the release rows: nudged and random arms at each delta, plus length-limit rows. A length-limit row is a native run cut at 50/70/85% of its reasoning, with the closing tokens appended so the model only writes its answer. A family whose question was used for selection gets a set selected without it. |
| `analysis.py` | Bounded within-question contrasts: nudge vs random (primary), vs unedited, the rerun noise, the length-limit curve, and "nudge vs stopping at the same length". It also reports how much routing the nudged experts lost. |
| `decide.py` | The extension rule that was declared before any result: launch the next models only if a nudge is shorter than random without losing accuracy, or more accurate than random. |
| `__main__.py` | Command line: `usage`, `select` (preview, writes nothing), `analyze`, `decide`. |

Sealing the release is `steering.release outcome` (it also writes the code snapshot). The jobs are the
ordinary whole-reasoning ones (`scripts/steering/launch.sh`) plus `python -m steering.outcome analyze`.

The routing operator itself lives with the other operators. It is
`steering.operators.score_shift_masked`, applied by `steering.stream`. The
length-limit prefix lives in `steering.runtime.budget_tokens`.

## Running it

```bash
scripts/steering/outcome_usage.sh                # 1. CPU job, ~5 min: expert use per cohort-B attempt
PYTHONPATH=src python -m steering.outcome select    # 2. look at the expert sets (no files)
scripts/steering/whole.sh outcome outcome-steering-20261022-r1 <passed tests> --shards 3 --cutoff 2026-10-23T23:00   # 3. seal + jobs
sbatch scripts/steering/outcome_next.sh report/<release>/analysis/EFFECTS_OUTCOME.json <next name> <tests> --cutoff ... --models ...   # 4. the extension rule
```

Outputs:
- `report/<release>/`: receipts, `<model>/OUTCOME_SELECTION.json`, manifests, job records, and
  `analysis/EFFECTS_OUTCOME.json`.
- `campaign-runs/outcome-steering/` on FAST: the usage arrays.

Reading the results: token contrasts are `log(1+T_control) - log(1+T_treated)`, so a positive value
means shorter reasoning. A run that hits its limit or never finishes keeps an interval, never a
guessed value.
