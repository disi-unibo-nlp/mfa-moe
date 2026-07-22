# Experiment 0a — GEPA-optimized sentence judge

Experiment 0a optimizes the instruction prompt used by Qwen 3.6 27B to assign
one of the seven sentence-level categories from the adapted Schoenfeld Episode
Theory:

`Read`, `Analyze`, `Plan`, `Implement`, `Explore`, `Verify`, `Monitor`.

Each classification request contains exactly one sentence. It does not contain
the full response or neighboring reasoning. The official 38-response corpus is
split by complete response (26 train, 6 validation, 6 held-out test by default)
before its sentences are flattened, so sentences from one response cannot leak
between splits.

Official corpus: <https://github.com/MingLiiii/Schoenfeld_Reasoning>

## Optimization and evaluation

GEPA requires a score for each individual example. Cohen's kappa and Kendall's
tau-b are not defined for one sentence, so GEPA receives exact-match reward:
`1` for the correct class and `0` otherwise, plus explicit gold-versus-predicted
feedback for reflection.

After optimization, reviewer agreement is computed globally across all
validation or test sentences:

- Cohen's kappa;
- Kendall's tau-b, using the guidebook label order shown above;
- exact accuracy and valid-output coverage.

Raw agreement coefficients are preserved. The optional composite reporting
score rescales each coefficient from `[-1, 1]` to `[0, 1]` before averaging;
negative agreement is not clipped.

## Prompt variants

`base` contains the seven definitions and distinctions. `few-shot` appends
individual gold training sentences, selecting one example for every class when
seven examples are requested. Validation and test sentences are never eligible.

```bash
python -m moe_exp.experiment0a.run \
  --dataset-dir data/Schoenfeld_Reasoning \
  --prompt-variant few-shot \
  --few-shot-examples 7 \
  --gepa-auto light
```

For a plumbing check only:

```bash
python -m moe_exp.experiment0a.run \
  --dataset-dir data/Schoenfeld_Reasoning \
  --train-documents 1 \
  --val-documents 1 \
  --test-documents 1 \
  --max-units-per-document 2 \
  --max-full-evals 1
```

## SLURM with Docker

Build the two images once:

```bash
docker build -t llama.cpp:localcuda src/common/llamacpp
docker build -t moe-mfa-experiments:latest .
mkdir -p slurm_logs
```

Run the base and few-shot conditions separately with the same seed and budget:

```bash
sbatch run_experiment0a.sh \
  --prompt-variant base \
  --output-dir results/exp0a/qwen3.6-27b-base \
  --gepa-auto light \
  --seed 42

sbatch run_experiment0a.sh \
  --prompt-variant few-shot \
  --few-shot-examples 7 \
  --output-dir results/exp0a/qwen3.6-27b-few-shot \
  --gepa-auto light \
  --seed 42
```

The launcher defaults to:

- repository: `/home/tassinari/moe-mfaExperiments`;
- dataset: `/home/tassinari/moe-mfaExperiments/data/Schoenfeld_Reasoning`;
- model: `/llms/Qwen3.6-27B-UD-Q4_K_XL.gguf`;
- one llama.cpp slot and one evaluator thread;
- an 8192-token server context, sufficient for one sentence plus few-shot examples.

Outputs contain the constructed seed prompt, optimized prompt, split response
IDs, sentence predictions, corpus-level validation/test agreement, and an
append-only CSV summary.
