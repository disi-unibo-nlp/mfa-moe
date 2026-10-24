# routing: walkthroughs

## expert-selection-trace: Routing, one expert-selection coverage row
- commit: e63f4d8dc28cc751351c09cdb329d323de567c6b
- source: report/research-campaign-v3-20261014-r6 (CAMPAIGN.json, FOLDS.json, gpt/SOURCES.json, the class-count archive it names for GPT), report/expert-memory-study-v1/gpt/SELECTION.json (sha256 f8627ebd) and the Overleaf data file figures/journal/data/selection_coverage.csv; every value below comes from a run on 2026-10-25 of the code at this commit on those files (`export` ran as Slurm job 62189479)
- code: src/routing/expert_selection.py, scripts/routing/export_selection_coverage.sh, tests/routing/test_expert_selection.py

**Input:** One row of the table behind the paper's selection figure: GPT-OSS-20B, held-out fold 0, keep
24 of its 32 experts in every layer, selector `state_balanced`. The row reads `minimum_class_coverage`
0.9675551447589305 on 298 held-out question families. The question the table answers: if only three
quarters of the experts are kept, how much of the routing mass of the least-covered state survives on
questions the selection never saw?
`scripts/routing/export_selection_coverage.sh CAMPAIGN_DIR OUT_DIR` writes the table with
`python -m routing.expert_selection export CAMPAIGN_DIR OUT_DIR` (CAMPAIGN_DIR is `report/expert-memory-study-v1`), but
`export` only copies numbers from the stored `SELECTION.json` receipts. The numbers are made by
`expert_selection.analyze`, which this walkthrough reruns on the saved counts and compares with that receipt.

### 1. Counts (`expert_selection.counts_for`, `expert_selection.validate_counts`)

The class-count archive of the GPT cohort A is joined to the 1,547 attempts of `gpt/SOURCES.json`.

| Array | Shape | Meaning |
|---|---|---|
| `counts` | (1547, 9, 24, 32), uint32 | per attempt, class, layer and expert: tokens of that class that selected the expert |
| `tokens` | (1547, 9) | tokens of the class (7 states, all reasoning, all labelled) |

`validate_counts` requires that the selected slots of every cell equal its tokens times the native top-k (4).
The analysis uses the 7 state classes, `counts[:, :7]`.

### 2. Folds (`expert_selection.family_folds`)

Folds hold whole question families, because attempts of one question are never independent. The 1,486
families (of 1,547 attempts) are sorted by `digest([SEED, 'fold', family])` and dealt out as `i % 5`.

| | Value |
|---|---|
| `folds_sha256` rebuilt / in the saved receipt | 86e9f66dc7ee538471bca1df576fea62937e2274a4ec8bde55305997ec72ed90 / the same |
| fold 0: training families / test families | 1,188 / 298 |
| training families that have Read tokens, layers 0 to 2 | 949 each |

### 3. Scores (`expert_selection.scores`)

On the training families only, each family's counts are averaged, turned into shares per class and layer,
and combined two ways into a (24 layers x 32 experts) score: `balanced` gives every state the same weight,
`frequency` pools the labelled tokens so that frequent states dominate. A class or layer without training
support is an error, not a zero.

### 4. Keep sets (`expert_selection.keep_set`)

Each layer keeps its 24 best experts by score (ties by expert index); the `random` arm draws 24 per layer from
a hash seeded by model, layer and expert. State-balanced fold 0 at layer 0 keeps
[1, 2, 3, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23, 24, 25, 26, 28, 29, 31] and drops
[0, 4, 6, 11, 18, 20, 27, 30].

### 5. Held-out coverage (`expert_selection.coverage`, inside `expert_selection.analyze`)

On the 298 test families, the share of each state's routing slots that fall on kept experts is averaged
within families, then over layers, and the weakest state decides `minimum_class_coverage`.

| Mean layer coverage | Read | Analyze | Plan | Implement | Explore | Verify | Monitor | minimum |
|---|---|---|---|---|---|---|---|---|
| state_balanced | 0.9697 | 0.9736 | 0.9750 | 0.9707 | 0.9768 | 0.9778 | 0.9676 | **0.9675551447589305** |
| labelled_frequency | 0.9663 | 0.9744 | 0.9749 | 0.9730 | 0.9753 | 0.9768 | 0.9505 | 0.9505093153831535 |
| random | 0.7833 | 0.7806 | 0.7881 | 0.7703 | 0.7860 | 0.7842 | 0.7853 | 0.7703260232265086 |

The fold 0 results, the stability of the fold masks, the final masks and the training support equal the saved
`SELECTION.json` field by field (masks and held-out values for all three arms are identical).

### 6. The row (`expert_selection.export`)

`export` writes one row per model, fold and arm for the seven models of `CAMPAIGN.json` (105 rows, each model
needing its `SELECTION.json`). The three rows for `gpt`, fold 0, are `labelled_frequency` 0.9505093153831535,
`random` 0.7703260232265086 and `state_balanced` 0.9675551447589305, each with `test_families` 298 and
`minimum_complete` True. All 105 rows equal the rows of `figures/journal/data/selection_coverage.csv` in the Overleaf
checkout, which holds 15 more rows for LFM2.5 from its own receipt. The CPU job took 5 s (1.3 s of CPU).
`tests/routing/test_expert_selection.py` and `test_store.py` (11 tests, 11 subtests) pass on the login node in 12.3 s.

### Worth noticing

- The weakest state is Monitor in both selectors, the rarest one. Balancing the states buys 1.7 points there
  (0.9676 against 0.9505). On the other six states the two selectors differ by at most 0.34 points: the balanced
  set is the higher on four of them, the frequency set on Analyze and Implement.
- Keeping 24 of 32 experts at random covers only 77 percent of the weakest state's slots, so the 95 to 97
  percent of the learned sets comes from the selection, not from the keep ratio.
- The unit is the question family, not the attempt: 1,547 attempts form 1,486 families, split 1,188 to 298.
- `export` and `analyze` are separate on purpose. The exported table is a copy of receipts; the rerun of
  `analyze` (loading the counts, joining them to the attempts, five folds and the final fit) took 92 s on the
  login node and reproduced every field it computes (folds, masks, held-out coverage, stability, support). The
  saved receipt also carries five fields added after the analysis (`all_reasoning_sensitivity`, `campaign_sha256`,
  `elapsed_seconds`, `peak_rss_kib`, `source_files`), so the two seals differ.
- `inclusion_statistics`, in the same module, is the function that the whole-reasoning nominations of
  whole-nomination-trace use for their risk differences.
