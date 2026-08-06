# probeTest — gold Schoenfeld episode probes

This experiment asks whether the seven sentence-level categories released by
Li et al. are linearly decodable from `Qwen/Qwen3.6-35B-A3B` hidden states.
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

## Cluster run

The launcher defaults to:

```text
model       Qwen/Qwen3.6-35B-A3B
HF_HOME     /gringotts/hf_home
results     /gringotts/home/tassinari/results/probeTest/qwen3.6-35b-a3b
precision   bitsandbytes NF4 weights, bfloat16 forward
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

To test only the alignment and forward path on one response, do not use the
`all` command—the subset may not contain enough instances of every class:

```bash
export HF_HOME=/gringotts/hf_home
python -m moe_exp.probeTest.run extract \
  --dataset-dir data/Schoenfeld_Reasoning \
  --output-dir /gringotts/home/tassinari/results/probeTest/smoke/activations \
  --model Qwen/Qwen3.6-35B-A3B \
  --quantization bnb-4bit \
  --max-documents 1
```

## Outputs

```text
probeTest/qwen3.6-35b-a3b/
├── activations/
│   ├── manifest.json
│   └── shards/
│       ├── 01-<response-id>.pt
│       └── 01-<response-id>.json
└── probes/
    ├── classifiers/<label>_layer_<index>.pkl
    ├── splits/<label>.npz
    ├── unit_index.jsonl
    ├── results.json
    ├── layerwise_metrics.csv
    └── layerwise_accuracy.png
```

Only boundary vectors are persisted, not token-by-token hidden states. At
float32 this is roughly 1 GiB for 3,087 units × 41 hidden-state indices × 2,048
dimensions, plus small classifier and report files.
