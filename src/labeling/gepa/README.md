# gepaLLMAsJudge

`gepaLLMAsJudge` optimizes the prompt used by Qwen 3.6 27B to assign one of seven
adapted Schoenfeld Episode Theory labels:

`Read`, `Analyze`, `Plan`, `Implement`, `Explore`, `Verify`, `Monitor`.

Run it with `PYTHONPATH=src`; it needs `dspy` and `scipy`. The frozen judge (the program every other
folder loads) came out of `scripts/labeling/optimize_judge.sh`.

## Context and data isolation

Each target unit is classified with:

- the SAT problem statement, choices, table, and figure description from
  `SAT.json`;
- the previous response unit;
- the current target unit;
- the next response unit.

Correct answers and rationales are never copied into model-visible context.
Documents are split before units are flattened, so no response crosses train,
validation, outer-CV, or locked-test boundaries.

The prompt defines a whole-unit precedence rule for compound annotations such
as a `**Final Answer**` marker combined with a boxed answer. Gold labels are
never silently modified. Every run writes `annotation_audit_*.json`.

## Prompt and optimization

The default `few-shot` prompt uses 21 hand-audited synthetic contrastive
examples: three per class. They target the recurring boundaries
`Read`/`Analyze`, `Analyze`/`Verify`, `Analyze`/`Implement`, `Plan`/`Monitor`,
and `Explore`/`Analyze`. Because the examples are synthetic, validation and
test annotations cannot leak into the prompt.

GEPA now uses an LLM-as-a-judge reward by default. The judge is the same Qwen
model being optimized, but it runs with reasoning disabled for this stage. For
each candidate label it returns a short free-text explanation and four scores
from 1 to 5:

- `GOLD_AGREEMENT`: agreement with the authoritative annotation;
- `FUNCTIONAL_FIT`: fit to the target unit's reasoning function;
- `CONTEXTUAL_COHERENCE`: use of problem and neighboring-unit context;
- `BOUNDARY_PRECISION`: distinction from the closest competing label.

The GEPA reward is a normalized weighted mean: gold agreement has weight 0.40,
functional fit 0.25, contextual coherence 0.15, and boundary precision 0.20.
These are optimization-feedback dimensions, not final experiment metrics.
Final prompt selection still uses gold-label balanced accuracy or macro-F1
plus the per-class recall safety gate. The former static rewards remain
available with `--gepa-reward balanced` and `--gepa-reward exact` for ablations.
Every judge call is also appended to `judge_feedback_*.jsonl`, including the
candidate and gold labels, four component scores, normalized composite score,
and free-text reasoning. This makes judge quality auditable independently of
GEPA's binary checkpoint.

After optimization, a safety gate compares the seed and optimized prompts on
validation balanced accuracy. The optimized prompt is rejected if it fails to
improve the selection metric or lowers any class recall by more than 0.10.
Configure this with:

```bash
--selection-metric balanced_accuracy
--max-class-recall-drop 0.10
```

Reports include strict accuracy, balanced accuracy, macro-F1, per-class
precision/recall/F1, Cohen's kappa, Kendall's tau-b, and valid-output coverage.
Kendall's tau-b is retained for comparability but should not be treated as the
primary metric unless the configured label ordering is substantively justified.

## Recommended workflow

First compare configurations using nested response-grouped cross-validation:

```bash
python -m labeling.gepa.run \
  --dataset-dir data/Schoenfeld_Reasoning \
  --prompt-variant few-shot \
  --few-shot-examples 21 \
  --gepa-reward llm-judge \
  --selection-metric balanced_accuracy \
  --cv-folds 5 \
  --locked-test-documents 6 \
  --gepa-auto heavy \
  --output-dir results/gepaLLMAsJudge/context-llmjudge-cv
```

For each outer fold, GEPA uses only an inner training and validation split. The
selected fold prompt is then evaluated on an untouched outer fold. The six
locked-test responses are excluded from the entire process. Cross-validation
does not produce one deployable prompt; use it to choose the configuration.

Fit the chosen configuration without evaluating the locked test:

```bash
python -m labeling.gepa.run \
  --dataset-dir data/Schoenfeld_Reasoning \
  --prompt-variant few-shot \
  --few-shot-examples 21 \
  --gepa-reward llm-judge \
  --selection-metric balanced_accuracy \
  --gepa-auto heavy \
  --output-dir results/gepaLLMAsJudge/context-llmjudge-final
```

Only after configuration and prompt-selection rules are frozen, explicitly
authorize the one-time locked-test evaluation:

```bash
python -m labeling.gepa.run \
  --dataset-dir data/Schoenfeld_Reasoning \
  --prompt-variant few-shot \
  --few-shot-examples 21 \
  --gepa-reward llm-judge \
  --selection-metric balanced_accuracy \
  --gepa-auto heavy \
  --evaluate-locked-test \
  --output-dir results/gepaLLMAsJudge/context-llmjudge-locked-test
```

The current historical test split has already been inspected in earlier
experiments. A genuinely final result requires newly held-out responses or an
external evaluation corpus.

## Reproducing the frozen judge

`scripts/labeling/optimize_judge.sh API_BASE OUTPUT_DIR` runs the Qwen3.8-27B alignment pass
that produced `results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3`: the heavy GEPA budget,
medium reasoning effort, 4096 reflection tokens, and the earlier 40-example optimized prompt
(`results/exp0a/context-llmjudge-20k-s42/gepa_optimized_prompt_20260811_135905.txt`) as the
external seed. Everything else is the default (seed 42, LLM-judge reward, balanced-accuracy
selection, 26/6/6 split). Add `--evaluate-locked-test` only to the frozen final evaluation job.
Always use a fresh output directory when changing the reward configuration, because GEPA resumes
checkpoints found under that directory's `gepa_logs/`.

The splits are fixed in `run.py` (`TRAIN_DOCUMENTS`, `VAL_DOCUMENTS`): 26 / 6 / 6 whole
responses for the seed given with `--seed`.

Outputs include the seed, GEPA-optimized, and safety-selected prompts/programs;
validation or outer-fold predictions; split response IDs; annotation audit;
aggregate and per-class metrics; per-call `judge_feedback_*.jsonl`; and an
append-only `stats.csv`.
