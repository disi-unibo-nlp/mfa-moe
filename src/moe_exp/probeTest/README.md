# probeTest — gold Schoenfeld episode probes

This experiment asks whether the seven sentence-level categories released by
Li et al. are linearly decodable from
`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` hidden states. This is the official Qwen
GPTQ 4-bit checkpoint; unlike on-load bitsandbytes conversion, its fused MoE
expert weights are already quantized and fit on a 32 GiB RTX 5090.
It combines the [Schoenfeld gold corpus](https://arxiv.org/abs/2509.14662) with
the layer-wise probe protocol from
[LLM Reasoning as Trajectories](https://aclanthology.org/2026.acl-long.1237/)
and its [official implementation](https://github.com/slhleosun/reasoning-trajectory).

## What is forwarded

No response is regenerated for probe training. The input to Qwen is:

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
boundaries. Each resulting unit is teacher-forced through the same Qwen
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
