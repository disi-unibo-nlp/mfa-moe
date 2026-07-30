# Experiment 0a — context-aware GEPA episode judge

Experiment 0a optimizes the prompt used by Qwen 3.6 27B to assign one of seven
adapted Schoenfeld Episode Theory labels:

`Read`, `Analyze`, `Plan`, `Implement`, `Explore`, `Verify`, `Monitor`.

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
never silently modified. Every run writes `annotation_audit_*.json`, and the
same audit can be run without a model:

```bash
python -m moe_exp.experiment0a.audit \
  --dataset-dir data/Schoenfeld_Reasoning \
  --output results/exp0a/annotation_audit.json
```

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
Final prompt selection still uses gold-label accuracy, balanced accuracy, or
macro-F1 plus the per-class recall safety gate. The former static rewards remain
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
python -m moe_exp.experiment0a.run \
  --dataset-dir data/Schoenfeld_Reasoning \
  --prompt-variant few-shot \
  --few-shot-examples 21 \
  --gepa-reward llm-judge \
  --selection-metric balanced_accuracy \
  --cv-folds 5 \
  --locked-test-documents 6 \
  --gepa-auto heavy \
  --output-dir results/exp0a/context-llmjudge-cv
```

For each outer fold, GEPA uses only an inner training and validation split. The
selected fold prompt is then evaluated on an untouched outer fold. The six
locked-test responses are excluded from the entire process. Cross-validation
does not produce one deployable prompt; use it to choose the configuration.

Fit the chosen configuration without evaluating the locked test:

```bash
python -m moe_exp.experiment0a.run \
  --dataset-dir data/Schoenfeld_Reasoning \
  --prompt-variant few-shot \
  --few-shot-examples 21 \
  --gepa-reward llm-judge \
  --selection-metric balanced_accuracy \
  --gepa-auto heavy \
  --output-dir results/exp0a/context-llmjudge-final
```

Only after configuration and prompt-selection rules are frozen, explicitly
authorize the one-time locked-test evaluation:

```bash
python -m moe_exp.experiment0a.run \
  --dataset-dir data/Schoenfeld_Reasoning \
  --prompt-variant few-shot \
  --few-shot-examples 21 \
  --gepa-reward llm-judge \
  --selection-metric balanced_accuracy \
  --gepa-auto heavy \
  --evaluate-locked-test \
  --output-dir results/exp0a/context-llmjudge-locked-test
```

The current historical test split has already been inspected in earlier
experiments. A genuinely final result requires newly held-out responses or an
external evaluation corpus.

## SLURM with Docker

Build the two images once:

```bash
docker build -t llama.cpp:localcuda src/common/llamacpp
docker build -t moe-mfa-experiments:latest .
mkdir -p slurm_logs
```

Run five-fold nested cross-validation:

```bash
sbatch run_experiment0a.sh \
  --prompt-variant few-shot \
  --few-shot-examples 21 \
  --gepa-reward llm-judge \
  --cv-folds 5 \
  --gepa-auto heavy \
  --output-dir results/exp0a/context-llmjudge-cv-s42
```

Run the final fit, still without test evaluation:

```bash
sbatch run_experiment0a.sh \
  --prompt-variant few-shot \
  --few-shot-examples 21 \
  --gepa-reward llm-judge \
  --gepa-auto heavy \
  --output-dir results/exp0a/context-llmjudge-final-s42
```

Add `--evaluate-locked-test` only to the frozen final evaluation job. Each
candidate evaluation now requires both a classifier call and a judge call, so
budgeted runs are substantially slower. Keep judge reasoning disabled on the
3090; reasoning-enabled ablations should be reserved for the 5090.
Always use a fresh output directory when changing reward configuration because
GEPA resumes checkpoints found under that directory's `gepa_logs/`.

The SLURM launcher now defaults to GEPA's `heavy` budget and allows up to 4096
tokens for each reflection response. The allocation itself is unchanged: one
GPU, 64 GB host memory, and a 48-hour wall-clock limit. Override the budget
explicitly with `--max-metric-calls N` when a fixed, auditable compute budget is
preferred.

Outputs include the seed, GEPA-optimized, and safety-selected prompts/programs;
validation or outer-fold predictions; split response IDs; annotation audit;
aggregate and per-class metrics; per-call `judge_feedback_*.jsonl`; and an
append-only `stats.csv`.
