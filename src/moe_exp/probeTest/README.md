# probeTest — gold Schoenfeld episode probes

This experiment asks whether the seven sentence-level categories released by
Li et al. are linearly decodable from
Qwen, GPT-OSS, and Gemma hidden states. The original run uses
`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`. This is the official Qwen
GPTQ 4-bit checkpoint; unlike on-load bitsandbytes conversion, its fused MoE
expert weights are already quantized and fit on a 32 GiB RTX 5090.
It combines the [Schoenfeld gold corpus](https://arxiv.org/abs/2509.14662) with
the layer-wise probe protocol from
[LLM Reasoning as Trajectories](https://aclanthology.org/2026.acl-long.1237/)
and its [official implementation](https://github.com/slhleosun/reasoning-trajectory).

## What is forwarded

No response is regenerated for probe training. The input to each model is:

1. the released SAT `Instruction`, formatted as a user message;
2. the corresponding released DeepSeek-R1 response, supplied in teacher
   forcing as the assistant continuation.

Every labeled sentence is matched verbatim and monotonically against the
original response before tokenization. The feature for a sentence is the
hidden state at the token immediately preceding its first token. It is
therefore a causal, prospective boundary probe: it cannot read the target
sentence or any later text.

The current dataset snapshot contains 38 responses and 3,125 annotated units.
Exactly 38 of them—one per response—contain the synthetic `</think>` closing
boundary and are all labeled `Monitor`. They are excluded by default, yielding
the 3,087 sentences reported in the paper and avoiding a trivial marker
shortcut. `--include-think-boundary-units` restores all 3,125 for an explicit
ablation; both raw and retained counts are recorded in the manifest.

## Probe protocol

For each of the seven labels and each hidden-state index, the experiment trains
one binary target-vs-rest logistic regression. This matches the ACL reference:

- sentence-level stratified 80/20 split with seed 42;
- `LogisticRegression(max_iter=2000, class_weight="balanced")`;
- library defaults made explicit: `solver="lbfgs"`, L2 penalty, `C=1.0`;
- raw activations, with no standardization and no PCA.

The Qwen configuration has 40 language-model layers. Transformers returns 41
hidden-state tensors: index 0 is the embedding output and indices 1–40 are the
successive layer outputs.

The paper's sentence-level split is reproduced even though units from one
response can occur in both train and test. Each saved split contains response
IDs so this limitation is auditable. A response-grouped generalization study
would be a separate protocol, not a silent change to this replication.

## Benchmark inspection stage

After training, the `all` command applies the saved probes to 20 examples from
each of GSM8K, MATH, PRM800K, and ProcessBench. This stage is also available by
itself through the `label` command or the launcher's `--label-only` flag.

This inspection stage does not generate new solutions:

- GSM8K uses the benchmark's reference rationale;
- MATH uses the benchmark's reference solution;
- ProcessBench uses its supplied step-by-step solution;
- PRM800K uses the reconstructed rated path already defined by the project
  loader (good prefix followed by the first negatively rated completion).

Reference reasoning is split conservatively at sentence punctuation and line
boundaries. Each resulting unit is teacher-forced through the same probe
checkpoint, using the same causal pre-unit boundary definition as probe
training. For every unit, the stage saves all seven best-layer one-vs-rest
scores, all probes above their binary 0.5 threshold, and an inspection label
chosen by the largest score. Because these classifiers were trained
independently with balanced class weights, the scores are not calibrated
multiclass probabilities. The argmax labels are review candidates, not new
ground truth.

## Cluster run

The launcher defaults to:

```text
model       Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
HF_HOME     /llms
results     results/probeTest/qwen3.5-35b-a3b-gptq-int4
precision   checkpoint-native GPTQ 4-bit weights
```

Submit from the repository root. The launcher defaults to `faretra`, matching
the existing project launcher for machine 40; an explicit Slurm option can
override it:

```bash
docker build -t moe-mfa-experiments:latest .
sbatch src/moe_exp/probeTest/run_slurm.sh
# Override only if needed:
sbatch --nodelist=<other-hostname> src/moe_exp/probeTest/run_slurm.sh
```

If the model is absent, `from_pretrained` downloads it into `HF_HOME`; otherwise
the cached snapshot is reused. The extraction is sharded by response and is
resumable. An existing shard is reused only when dataset digest, model,
revision, quantization, prompt, and boundary definition all match.

For an environment with CUDA PyTorch and the project already installed:

```bash
sbatch src/moe_exp/probeTest/run_slurm.sh --local
```

The local environment also needs the `probe` extra (GPTQModel) and a
torchvision build matching its CUDA-enabled PyTorch installation. The
project Docker image installs the CUDA 12.8 build of torchvision automatically.
The launcher persists GPTQModel's compiled CUDA kernels below `/llms/.cache`,
so their one-time compilation is reused by later runs.

To run only the benchmark stage from the already completed probes:

```bash
CUDA_VISIBLE_DEVICES=0 bash src/moe_exp/probeTest/run_slurm.sh --label-only
```

The direct equivalent is:

```bash
python -m moe_exp.probeTest.run label \
  --probe-results results/probeTest/qwen3.5-35b-a3b-gptq-int4/probes/results.json \
  --output-dir results/probeTest/qwen3.5-35b-a3b-gptq-int4/benchmark_labels \
  --examples-per-benchmark 20
```

To test only the alignment and forward path on one response, do not use the
`all` command—the subset may not contain enough instances of every class:

```bash
export HF_HOME=/llms
python -m moe_exp.probeTest.run extract \
  --dataset-dir data/Schoenfeld_Reasoning \
  --output-dir results/probeTest/smoke/activations \
  --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --quantization gptq-4bit \
  --max-documents 1
```

## GPT-OSS and Gemma layer selection

Run the full gold extraction and layer-wise probe stages for each model:

```bash
IMAGE_NAME=moe-mfa-experiments:quantized-forward \
bash src/moe_exp/probeTest/run_slurm.sh \
  --model openai/gpt-oss-20b --skip-benchmark-labeling

IMAGE_NAME=moe-mfa-experiments:quantized-forward \
bash src/moe_exp/probeTest/run_slurm.sh \
  --model google/gemma-4-26B-A4B-it --skip-benchmark-labeling
```

The launcher selects `mxfp4-bf16` for OSS and the existing text-only
`bnb-4bit` expert adapter for Gemma. OSS expands the checkpoint's quantized
MXFP4 values to BF16 and offloads excess weights to CPU. This avoids native
Triton compiler crashes on this RTX 5090; it does not recover the original
pre-quantization weights. Both probes and correlation replay use this mode.
The explicit `mxfp4` mode remains available for compatible native runtimes. Outputs go to separate `gpt-oss-20b` and
`gemma-4-26b-a4b-it-nf4` directories below `results/probeTest`. Use
`--model-revision` to pin the checkpoint revision. The corpus, causal boundary,
sentence split, classifier settings, and accuracy selection match Qwen.
Benchmark labeling is optional and is not needed to select correlation layers.

OSS supplies 25 hidden states (24 decoder layers); Gemma supplies 31 (30 layers).
The correlation pipeline automatically reads each model's results and keeps
only selected indices with a corresponding router. See
[model support](../correlation_pipeline/MODEL_SUPPORT.md#probe-layers).

### Completed runs (2026-09-15)

Both runs use all 3,087 units and the unchanged seed-42 protocol. Layer indices
are zero-based decoder inputs for correlation replay.

| Target | OSS layer | Accuracy | Gemma layer | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Read | 19 | 0.9304 | 21 | 0.9207 |
| Analyze | 21 | 0.7654 | 27 | 0.7896 |
| Plan | 15 | 0.9239 | 25 | 0.9159 |
| Implement | 21 | 0.8916 | 29 | 0.8819 |
| Explore | 23 | 0.9239 | 16 | 0.9239 |
| Verify | 21 | 0.8770 | 19 | 0.8447 |
| Monitor | 15 | 0.9320 | 22 | 0.9320 |

**Correlation layer unions:** OSS `15, 19, 21, 23`; Gemma
`16, 19, 21, 22, 25, 27, 29`. Every selected index has a router.
Both selections passed a full-checkpoint GPU check through the correlation
loader and exact-token replay, with finite router and hidden-state tensors.
Each run saves `correlation_replay_validation.json`; the combined audit is
`results/probeTest/model_layer_selection.json`.

Gemma: 217 fits, no convergence warnings. OSS: 175 fits, four hit the fixed
2,000-iteration limit (Analyze at 1 and 20; Plan and Verify at 1). None of
the seven selected fits hit the limit. The warnings are retained in the
full results; the iteration budget was not changed to improve selection.

Pinned revisions:
- OSS: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Gemma: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`

Full per-layer accuracy, F1, AUC, convergence diagnostics, saved classifiers,
and split indices are in each model's `probes/` directory.

## Outputs

```text
probeTest/qwen3.5-35b-a3b-gptq-int4/
├── activations/
│   ├── manifest.json
│   └── shards/
│       ├── 01-<response-id>.pt
│       └── 01-<response-id>.json
├── probes/
    ├── classifiers/<label>_layer_<index>.pkl
    ├── splits/<label>.npz
    ├── unit_index.jsonl
    ├── results.json
    ├── layerwise_metrics.csv
    └── layerwise_accuracy.png
└── benchmark_labels/
    ├── manifest.json
    ├── gsm8k/
    │   ├── inspection.md
    │   ├── predictions.jsonl
    │   └── records/*.json
    ├── math/...
    ├── prm800k/...
    └── processbench/...
```

Only boundary vectors are persisted, not token-by-token hidden states. At
float32 this is roughly 1 GiB for 3,087 units × 41 hidden-state indices × 2,048
dimensions, plus small classifier and report files.

## Response-grouped evaluation of saved activations

`probe --protocol grouped` evaluates generalization to unseen responses in the
same corpus. The default `sentence` protocol and `all` pipeline are unchanged.
No model loading, extraction, GPU, or generation is required by grouped fitting.

The grouped protocol uses five `StratifiedGroupKFold` outer folds, with complete
responses as groups and seven-class labels for approximate stratification.
Inside each outer development set, one `GroupShuffleSplit` reserves 20% of its
responses for validation. Each target's layer maximizes validation AUROC
(ties select the lowest index). Its classifier is then refitted on all outer
development responses and evaluated on the untouched outer test responses.
Classifier settings match the replication. There is no global best-layer
selection from these outer test scores and no change to existing routing layers.

A second logistic regression uses only `log1p` of the number of preceding
annotated units and preceding response tokens. This prefix-position baseline
uses no total response length or future content; token counts depend on the
encoder's tokenizer. It controls a simple positional shortcut, not all possible
position or formatting effects.

From an environment with the project's CPU dependencies installed:

```bash
for model in qwen3.5-35b-a3b-gptq-int4 gpt-oss-20b gemma-4-26b-a4b-it-nf4; do
  CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python -m moe_exp.probeTest.run probe --protocol grouped \
    --manifest "results/probeTest/$model/activations/manifest.json" \
    --output-dir "results/probeTest/$model/probes_grouped_s42" \
    --split-plan results/probeTest/grouped_s42/split_plan.json \
    --outer-folds 5 --validation-size 0.2 --seed 42 --workers 4 \
    --bootstrap-replicates 5000 --bootstrap-seed 42
done
```

For the existing Docker image, wrap that loop in `bash -c` with this prefix:

```bash
docker run --rm --network none --memory 16g \
  -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e PYTHONPATH=/workspace/src -e MPLCONFIGDIR=/tmp/mpl \
  -v "$PWD:/workspace" -w /workspace \
  --entrypoint bash moe-mfa-experiments:latest -c '<loop above>'
```

The shared split plan verifies response IDs, unit indices, labels, and exact
text across encoders. A nonempty legacy output directory is rejected. Completed
fold/target checkpoints resume only when input hashes, code, library versions,
and settings match; use a new output directory for a changed experiment.

Outputs include `run_config.json`, `split_plan.json`, `unit_index.jsonl`,
`folds/<fold>/<target>.json` and `.pkl`, `predictions.jsonl`, `results.json`, and
`status.json`. Each unit has exactly one outer-test prediction per target.
The summaries report the unweighted mean of the five fold metrics, and macro
means across the seven targets. Each fold metric weights its sentences equally.
The 95% percentile intervals resample whole responses independently within each
outer test fold, preserving all units and pairing probe/baseline scores. They
condition on the fitted models, chosen layers, and split plan; they do not
include refitting or split-seed variation. Draws missing a binary class in any
fold are omitted for AUROC/balanced accuracy and the valid count is retained.
The legacy/grouped comparison changes splitting and the selection criterion,
so its difference is not an isolated estimate of response leakage.

Regenerate the shared report/thesis tables and figure from completed runs with:

```bash
python3 report/generate_grouped_probe_results.py
```
