# Experiment 0a — GEPA-optimized episode judge

This experiment optimizes the instruction prompt used by a local LLM to assign
the two annotation levels from Li et al., *Understanding the Thinking Process of
Reasoning Models: A Perspective from Schoenfeld's Episode Theory*:

- paragraph level: `General`, `Explore`, `Verify`
- sentence level: `Read`, `Analyze`, `Plan`, `Implement`, `Explore`, `Verify`, `Monitor`

The official 38-response gold corpus is split by complete response (26 train,
6 validation, 6 held-out test by default), preventing sentences from the same
response from leaking across splits. GEPA sees train and validation only. The
test set is evaluated once after optimization.

Official corpus: <https://github.com/MingLiiii/Schoenfeld_Reasoning>

## Setup

Install the project and Experiment 0a dependencies:

```bash
pip install -e '.[exp0a]'
git clone https://github.com/MingLiiii/Schoenfeld_Reasoning.git data/Schoenfeld_Reasoning
```

Build and launch the llama.cpp server already provided in `src/common`:

```bash
cd src/common/llamacpp
docker build -t llama.cpp:localcuda .
./serve_llamacpp.sh
```

`serve_llamacpp.sh` currently defaults to the locally stored
`Qwen3.6-27B-UD-Q4_K_XL.gguf`, served as `local-llamacpp` on port 8080.

Then run from the repository root:

```bash
bash ./src/moe_exp/experiment0a/run_gepa.sh
```

## SLURM with Docker

On the cluster, build both images once from the repository root:

```bash
docker build -t llama.cpp:localcuda src/common/llamacpp
docker build -t moe-mfa-experiments:latest .
mkdir -p slurm_logs
```

The launcher defaults to the same cluster paths as `run_pipeline.sh`:

- repository: `/home/tassinari/moe-mfaExperiments`
- dataset: `/home/tassinari/moe-mfaExperiments/data/Schoenfeld_Reasoning`
- model: `/llms/Qwen3.6-27B-UD-Q4_K_XL.gguf`

Submit a full run with a light GEPA budget:

```bash
sbatch run_experiment0a.sh --gepa-auto light
```

Or specify paths and a fixed optimization budget:

```bash
sbatch run_experiment0a.sh \
  --dataset-dir /path/to/Schoenfeld_Reasoning \
  --model-dir /llms \
  --model-name Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --max-full-evals 10 \
  --seed 42
```

The job starts llama.cpp and GEPA in separate containers on a private Docker
network, waits for the model server to become healthy, and removes the server
container when the job exits. Only llama.cpp receives the allocated GPU. With
the conservative defaults there is one server slot and one evaluator thread;
raise `--parallel` and `--num-threads` together only if GPU memory allows it.

For a fast end-to-end SLURM plumbing check:

```bash
sbatch run_experiment0a.sh \
  --train-documents 1 \
  --val-documents 1 \
  --test-documents 1 \
  --max-units-per-document 2 \
  --max-full-evals 1
```

Monitor it with `tail -f slurm_logs/<job-id>.out` and inspect results under
`results/exp0a/qwen3.6-27b`.

For a small plumbing check:

```bash
python -m moe_exp.experiment0a.run \
  --dataset-dir data/Schoenfeld_Reasoning \
  --train-documents 1 \
  --val-documents 1 \
  --test-documents 1 \
  --max-units-per-document 2 \
  --max-full-evals 1 \
  --num-threads 1
```

The two limiting flags are plumbing checks only. Do not use them for reported
experimental results.

Exactly one GEPA budget is required: `--gepa-auto`, `--max-full-evals`, or
`--max-metric-calls`.

## Metric

For each complete response, the scorer computes Cohen's kappa and Kendall's
tau-b independently for both annotation levels. The default GEPA reward is the
equal-weight mean of the four coefficients after clipping negative values to
zero. Raw coefficients and exact accuracies are preserved in result files.

The label orders used for tau-b are the guidebook presentation orders shown
above. These labels are nominal rather than ordinal, so this tau interpretation
must be treated as an explicitly requested experimental convention; Cohen's
kappa is the statistically natural reviewer-agreement measure. Undefined
constant-sequence agreement is defined as 1 for identical sequences and 0
otherwise.

Outputs under `results/exp0a` include the optimized prompt, serialized DSPy
program when supported, split IDs, validation scores, held-out predictions,
corpus-level test agreement, and an append-only CSV summary.
