# Ordered-routing cache experiment

Tests whether mild identity steering reduces expert loads at a fixed resident
expert budget, alongside its effect on answer accuracy. This folder is isolated
from existing generation experiments and never overwrites their artifacts.

## Run

From the repository root, with the existing guiding Docker image and local model:

```bash
bash src/moe_exp/moe_cache_streaming/run_pair.sh
```

Defaults: GPT-OSS +1, the existing `oss_strength_1` policy and held-out sampling
prompts, 24 distinct problems selected by deterministic dataset round-robin,
one sequence at a time, 32,768 output-token limit, frozen 2026-09-22 template date.
All six datasets are represented when sufficient prompts exist. Selection never
uses correctness. Both conditions use identical prompts, seeds and limits.
A 24-problem run is exploratory, not sufficient to establish a small accuracy loss.

Results go into a timestamped `results/moe_cache_streaming/oss_*` directory.
Each condition launches a fresh model process. Outputs must not already exist;
interrupted runs are retained as incomplete, not silently resumed or reused.

Short plumbing test (accuracy is meaningless with eight output tokens):

```bash
LIMIT=1 MAX_TOKENS=8 MAX_MODEL_LEN=4096 \
  bash src/moe_exp/moe_cache_streaming/run_pair.sh
```

Custom budgets and sample size:

```bash
LIMIT=60 BUDGETS="8 12 16 24 32" \
OUTPUT_ROOT=results/moe_cache_streaming/oss_cache_60 \
  bash src/moe_exp/moe_cache_streaming/run_pair.sh
```

Other overrides: `MODEL`, `POLICY`, `PROMPTS`, `STRENGTH`, `TEMPLATE_DATE`,
`IMAGE_NAME`, `HF_CACHE_DIR`, `CUDA_VISIBLE_DEVICES`. For Qwen, supply its model,
policy and sampling prompts, and use budgets at least 16 (top-8 plus eight pins).
Only architectures with one supported routed gate per layer are accepted.
`DRY_RUN=true` prints Docker commands without running them.

## Capture and controls

Captures every routed layer, including unsteered ones, after the identity hook.
These are hook-computed top-k IDs, not separately observed native dispatch;
native tied-score selection may differ. Each response saves a compressed ordered
sequence of top-k sets per layer. Full prefill is excluded. The first generated
token is predicted by prefill, so N generated tokens must correspond to exactly
N-1 decode records at every layer. Shape/length mismatches fail the experiment.

Execution requires synchronous scheduling, eager mode, TP=1, one sequence, bounded chunked prefill, no prefix
cache or speculative decoding. All prefill chunks are excluded by counting actual
prompt tokens before recording decode. `PREFILL_CHUNK_SIZE` defaults to 2048,
independently of the context and output limits. Launcher logs are saved as
`baseline.log` and `guided.log` in the output root. Asynchronous scheduling is explicitly disabled: it can execute extra decode steps
before observing a natural stop, breaking the one-step-per-returned-token contract.
Length mismatches remain errors rather than being silently trimmed. If capture
validation fails, `failed_completion.json` retains the generated answer and stop
metadata for diagnosis. GPU-to-CPU recording adds overhead. The saved
`instrumented_seconds` is diagnostic resident-weight runtime, not an SSD speed
measurement. It must not be used to claim production throughput improvement.

The analysis evaluates four conditions at every cache budget:

1. Original routing with per-layer LRU.
2. Original routing with calibration-frequency pins and remaining capacity LRU.
3. Guided routing with boosted experts pinned and remaining capacity LRU.
4. Guided routing with ordinary LRU, to separate routing and pinning effects.

Pins are allocated only on policy layers; other layers use LRU in every
condition. Frequency pins use the same number of slots as target pins, selected
from saved calibration statistics, never evaluation traces. Budgets are identical
per layer and condition. Slots represent equal-sized experts within a layer;
this does not specify total GPU bytes or account for KV-cache capacity.

Each response starts cold. Initial pin loads count, even if those experts are
unused. All top-k experts for a token are requested simultaneously; current
experts cannot evict one another. Sorted expert IDs break simultaneous recency
ties. Capacity must fit the union of pins and current top-k; the launcher requires
at least `top_k + max_pin_count`. We do not assume a separate uncounted scratch
cache. Prefill warming, cross-response reuse, batching and prefetch are excluded.

## Results and limits

`analysis/summary.md` and `summary.json` report accuracy, hit fraction, total
expert loads per answer and loads per decode token, including all routed layers.
Loads per answer matter because guidance can change completion length.
Input, sampling, policy, version and artifact hashes are checked before pairing.
Scoring uses the existing `correlation_pipeline.score_completion` contract;
its results should not be mixed with the thesis's later offline rescoring without
applying the same rescoring to these new outputs.

A positive result is fewer simulated loads for guided generation at an acceptable
observed accuracy cost across the intended tasks. Sample size, seed variation,
and an independently chosen accuracy tolerance still matter. These data do not
measure bytes read from an SSD or end-to-end speed under offloading.

Actual SSD inference needs a backend that selectively loads expert weights.
The current vLLM backend keeps weights resident. This experiment provides the
ordered traces and controlled comparisons needed to justify implementing that
backend; it does not pretend to implement offloading by adding disk-read delays.

CPU verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_moe_cache_streaming.py -v
```

## Qwen and Gemma smoke commands

These use each model's own policy and original held-out sampling prompts.
The Gemma policy and split were prepared from the existing Gemma routing and
NVFP4 generation artifacts; policy metadata preserves the replay-checkpoint
provenance. They are also suitable inputs for a later full guiding evaluation.

```bash
MODEL=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
POLICY=results/moe_identity_guiding/qwen_global/policy.json \
PROMPTS=results/moe_identity_guiding/qwen_global/split/prompts.full.sampling.jsonl \
OUTPUT_ROOT="results/moe_cache_streaming/qwen_smoke_$(date -u +%Y%m%dT%H%M%SZ)" \
LIMIT=1 MAX_TOKENS=16 BUDGETS="16 32 64" \
  bash src/moe_exp/moe_cache_streaming/run_pair.sh

MODEL=nvidia/Gemma-4-26B-A4B-NVFP4 \
POLICY=results/moe_identity_guiding/gemma_global/policy.json \
PROMPTS=results/moe_identity_guiding/gemma_global/split/prompts.full.sampling.jsonl \
OUTPUT_ROOT="results/moe_cache_streaming/gemma_smoke_$(date -u +%Y%m%dT%H%M%SZ)" \
LIMIT=1 MAX_TOKENS=16 BUDGETS="16 32 64" \
  bash src/moe_exp/moe_cache_streaming/run_pair.sh
```

Run sequentially on a single GPU. Both profiles use text-only loading and
synchronous scheduling. These short capped runs check loading, hooks, trace
lengths and analysis only; they do not validate answer accuracy, long-context
stability or natural termination on these model families.
