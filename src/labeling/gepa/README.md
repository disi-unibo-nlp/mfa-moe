# src/labeling/gepa

Optimises the prompt of the judge that labels reasoning sentences with the seven states (Read, Analyze, Plan, Implement, Explore, Verify, Monitor). This was done once. The result is the frozen judge that every label set in the project uses.

| File | What it does |
| --- | --- |
| `optimize_judge.py` | The command: loads the data, runs GEPA, applies the safety check, evaluates, writes everything. |
| `data.py` | The labelled Schoenfeld responses, response-grouped splits, and an audit of units that merit a human look. |
| `prompts.py` | The seed instructions for the seven labels and a bank of 21 hand-written contrastive examples (three per label). |
| `metrics.py` | Strict and balanced accuracy, macro-F1, per-class precision and recall, Cohen's kappa, Kendall's tau-b. |

## Run

Needs `dspy`, `gepa` and `scipy`, and a running OpenAI-compatible server. The launcher is `scripts/labeling/optimize_judge_prompt.sh SERVER_URL DATASET_DIR SEED_PROMPT OUT_DIR`; it runs

```bash
PYTHONPATH=src:... python -m labeling.gepa.optimize_judge --dataset-dir DATASET_DIR --api-base SERVER_URL \
  --output-dir OUT_DIR --gepa-auto heavy --enable-thinking --reflection-max-tokens 4096 \
  --seed-prompt-file SEED_PROMPT
```

The frozen judge came from `DATASET_DIR=data/Schoenfeld_Reasoning` and `SEED_PROMPT=results/exp0a/context-llmjudge-20k-s42/gepa_optimized_prompt_20260811_135905.txt`. Everything else is the default: seed 42, the 21-example few-shot prompt, the LLM-judge reward, balanced-accuracy selection, medium reasoning effort, and a split of 26 training, 6 validation and 6 locked-test responses. The frozen judge is `results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json`. Use a fresh output folder when you change the reward, because GEPA resumes checkpoints it finds under the folder's `gepa_logs/`.

Output in `--output-dir`: `selected_program_<time>.json` (the program `labeling.batch_predictor` loads), the seed and optimised prompts and programs, validation predictions, split response ids, `annotation_audit_<time>.json`, `results_<time>.json`, `judge_feedback_<time>.jsonl` and `stats.csv`.

## How it is set up

- Data: the released Schoenfeld Reasoning annotations (`data/Schoenfeld_Reasoning`). Each unit is judged with the problem statement, the previous unit, the unit itself and the next unit. Correct answers and rationales are never shown to the model. Responses are split before units are flattened, so no response sits in two of training, validation, cross-validation fold or locked test.
- Prompt: the default `few-shot` prompt appends 21 synthetic contrastive examples, so no validation or test annotation can leak into it.
- Reward: the judge is the same model that is being optimised. For each candidate label it gives a short explanation and four scores from 1 to 5 (gold agreement 0.40, functional fit 0.25, contextual coherence 0.15, boundary precision 0.20). These guide GEPA; they are not result metrics. Every judge call is logged in `judge_feedback_*.jsonl`. `--gepa-reward balanced` and `exact` give the older static rewards.
- Safety check: the optimised prompt is rejected if it does not improve the selection metric on validation, or if any class recall drops by more than 0.10 (`--max-class-recall-drop`).
- Locked test: touched only with `--evaluate-locked-test`, once, after the configuration is frozen, and never in cross-validation (`--cv-folds`). The current test split has already been inspected in earlier experiments, so a final claim needs new held-out responses.

To compare configurations first, add `--cv-folds 5 --locked-test-documents 6`. Cross-validation chooses a configuration; it does not produce one deployable prompt.

Paper use: the appendix on external annotation development.

Tests: [`tests/labeling/test_gepaLLMAsJudge.py`](../../../tests/labeling/README.md).
