# Correlation pipeline

This pipeline generates reasoning attempts through vLLM, tags their sentences
with the frozen GEPA-selected classifier, teacher-forces the saved attempts
through the matching Hugging Face MoE checkpoint, and measures associations
among correctness, reasoning events, hidden-state trajectories and routing.

`run_all.sh` uses eight concurrent requests for both generation and tagging.
Tagging logs overall selected-sentence totals, percentage complete, remaining
sentences, reused checkpoints, processing speed, and estimated time remaining
every 10 seconds during concurrent requests and at dataset boundaries. Routine
LiteLLM/HTTP request logs are suppressed; warnings and errors remain visible.
On resume, only matching validated labels count as already complete; ETA uses
newly completed sentences from the current run.

To tag already saved generations in a Slurm job, submit from the repository root:

```bash
mkdir -p slurm_logs
sbatch src/moe_exp/correlation_pipeline/tag_slurm.sh \
  --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --generation-dir results/correlation_pipeline/reasoning-vllm-v1/sampling/generation
```

`--model` identifies the model that generated the traces. The directory must
contain `<model slug>/<dataset>/traces.jsonl`; for example, the slug above is
`Qwen--Qwen3.5-35B-A3B-GPTQ-Int4`. Dataset folders with saved traces are discovered
automatically, or select them with `--datasets math500 aime24`. This job performs
tagging only. Using sampled inputs preserves their sentence selection; using
`results/correlation_pipeline/generation` tags all sentences in the raw traces.
Matching annotation checkpoints resume automatically.

The job uses the same `moe-mfa-experiments:latest` client image and
`vllm/vllm-openai:v0.29.0` judge image as the local workflow, with high judge
reasoning effort by default for this labeling-only wrapper and judge MTP disabled.
The underlying `run_all.sh` default remains low for existing local workflows.
Pass `--judge-reasoning-effort low`, `--judge-reasoning-effort medium`, or
`--judge-reasoning-effort high` to override the wrapper default. Docker, both
images, the repository, and `/llms` must be available on the compute node.
Override image/cache paths through `IMAGE_NAME`, `VLLM_IMAGE`, and `HF_CACHE_DIR`.
The default allocation is one GPU, eight CPUs, 64 GB RAM, and four days; override
cluster-specific resources before the script path, for example `sbatch --partition=gpu --time=2-00:00:00 ...`.
The judge tensor-parallel size defaults to one; pass `--judge-tensor-parallel-size 2`
when allocating two GPUs. `--judge-workers 16` sets sixteen concurrent sentence
requests and the judge server's `--max-num-seqs 16`.

For the GPT-OSS calibration, use only the first 16 real `math500` traces; this
avoids the per-dataset meaning of `--limit 16` becoming 16 traces for every
benchmark:

```bash
sbatch --job-name=gptoss-qwen-cal --account=IscrC_MIOSR \
  --partition=boost_usr_prod --qos=normal --time=04:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:2 --mem=120G \
  --mail-type=BEGIN,END,FAIL --mail-user=lorenzo.molfetta@unibo.it \
  src/moe_exp/correlation_pipeline/tag_slurm.sh \
  --model openai/gpt-oss-20b \
  --generation-dir results/correlation_pipeline/gpt-oss-20b/generation \
  --datasets math500 --limit 16 --judge-workers 16 \
  --judge-tensor-parallel-size 2 --judge-reasoning-effort high \
  --judge-model unsloth/Qwen3.8-27B-NVFP4 \
  --judge-program results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json \
  --results-dir results/correlation_pipeline/gpt-oss-20b/calibration-qwen3.8-high
```

Outputs default to
`results/correlation_pipeline/reasoning-vllm-v1/annotations/<model slug>/`.
Use `--results-dir results/tagging-other-run` to isolate another run. Logs go to
`slurm_logs/tag-<job ID>.out` and `.err`. Set `JUDGE_PORT` to a free port for
concurrent jobs on the same node. The Slurm submit directory locates the repo;
set `PHYS_DIR` explicitly if submitting elsewhere. Preview locally with
`bash src/moe_exp/correlation_pipeline/tag_slurm.sh --model ... --generation-dir ... --dry-run`.

Each vLLM server enables prefix caching, allows eight sequences, and batches up to
8,192 tokens. Generation uses MTP with three speculative tokens; judge MTP is
disabled by default after a CUDA launch timeout in v0.29.0 speculative GDN
attention on the RTX 5090. Use `--judge-speculation mtp` only to explicitly
opt in. The servers run one
after another on the same GPU and stop before forward extraction begins.

Generation uses `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`, including its saved MTP head.
The judge uses `unsloth/Qwen3.8-27B-NVFP4` and the existing frozen GEPA program.
These retain the original base models with vLLM-compatible quantizations for
the 32 GB RTX 5090. The judge uses `--enforce-eager`, FP8 KV cache and text-only
loading for memory headroom. Forward extraction still loads
`unsloth/Qwen3.5-35B-A3B` through Unsloth with runtime 4-bit quantization.

Generation retains its 32,768-token completion budget and a 49,152-token context
to leave room for prompts. Judge context is 32,768, thinking is enabled with low
reasoning effort when `run_all.sh` is invoked directly, temperature is 0, and the
completion budget is 4,096 tokens. `tag_slurm.sh` changes only the
labeling-workflow default to high; its `--judge-reasoning-effort` option accepts
`low`, `medium`, or `high`.
Changing reasoning effort invalidates existing annotation checkpoints; labels
are recomputed under the new setting.

The quantized checkpoints differ from the old GGUF artifacts. New generations
use their actual checkpoint name, and vLLM annotations/analyses live under
`reasoning-vllm-v1`. Existing `reasoning-v1` labels are retained as a separate
run; they are not treated as outputs from the new quantized judge.

## Benchmark protocol

The first eight benchmarks follow the SPIRAL evaluation suite. Generation is
zero-shot with temperature 0.6 and top-p 0.95. The current completion budget is
32,768 tokens (increased from the original 8,192-token pilot).

| Name | Source / configuration | Sampling |
|---|---|---:|
| `math500` | `HuggingFaceH4/MATH-500`, test | pass@1 |
| `aime24` | `math-ai/aime24`, test | avg@32 |
| `aime25` | `math-ai/aime25`, test | avg@32 |
| `olympiad` | `Hothan/OlympiadBench`, `OE_TO_maths_en_COMP` | pass@1 |
| `amc23` | `math-ai/amc23`, test | avg@32 |
| `minerva` | `math-ai/minervamath`, test | pass@1 |
| `gpqa_diamond` | public Simple Evals Diamond CSV | avg@10 with seed-0 permuted choices |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro`, test | pass@1 |

GPQA-Diamond follows SPIRAL's released evaluator: each question is attempted ten
times with a fresh deterministic permutation of its four answer choices. The
attempts remain grouped under the same source question during analysis.

The defaults are `math500`, `aime24`, `aime25`, `olympiad`, `amc23`, and `minerva`.
GPQA-Diamond and MMLU-Pro remain available through explicit `--datasets` selection.
The four existing
repository loaders remain available by explicit selection: `gsm8k`, `math`,
`prm800k`, and `processbench`. The last two provide reference process labels in
their original use, but newly generated answers have no aligned gold error step.
ProcessBench generation is therefore retained for representation/routing and
reasoning-event analysis and is explicitly unscored for final-answer accuracy.

`math-verify` performs symbolic math scoring. Multiple-choice datasets require
the requested `\boxed{LETTER}` answer. A normalized exact/numeric fallback is
recorded explicitly if symbolic parsing cannot score an answer.

## Stratified 100k sampling and labelling (all benchmark sources)

`sample_stratified.py` selects the 100,000 sentence labels of every source
instead of the error-rate-weighted sentence sample above. It keeps one
completion per `source_problem_id` (diversity first), splits each dataset's
scored traces into equal-count `reasoning_tokens` quartiles, and allocates
sentence quotas proportional to the chosen universe's
`dataset x correct|incorrect x Q1..Q4` supply. Inside a cell the quota is spread
max-min fairly across problems, and each cell quota is split evenly across four
25,000-identity parts.

The same recipe (seed 42) is applied to every source; the balanced sample roots
live next to each source's generation:

| source key | generation model | sampling root (`--output-dir`) |
| --- | --- | --- |
| `gpt` | `openai/gpt-oss-20b` | `results/correlation_pipeline/gpt-oss-20b/reasoning-vllm-v1/sampling-100k-stratified-v2` |
| `gemma` | `nvidia/Gemma-4-26B-A4B-NVFP4` | `results/correlation_pipeline/gemma-nvfp4-nf4/reasoning-vllm-v1/sampling-100k-stratified` |
| `qwen36` | `Qwen/Qwen3.6-35B-A3B-FP8` | `results/correlation_pipeline/qwen36-35b-a3b-fp8/reasoning-vllm-v1/sampling-100k-stratified` |
| `nemotron` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | `results/correlation_pipeline/nemotron-nvfp4-dspark/reasoning-vllm-v1/sampling-100k-stratified` |
| `glm` | `zai-org/GLM-4.7-Flash` | `results/correlation_pipeline/glm-4.7-flash/reasoning-vllm-v1/sampling-100k-stratified` |
| `qwen330b` | `Qwen/Qwen3-30B-A3B` | `results/correlation_pipeline/qwen3-30b-a3b/reasoning-vllm-v1/sampling-100k-stratified` |

The earlier GPT-OSS labels (a 100k prefix of the full corpus: `math500` plus
three `aime24` problems) and the earlier Gemma labels (the error-weighted
`sample_tagging.py` population under `reasoning-vllm-v1/sampling`) are
superseded by this recipe. The Nemotron sampling was produced with it on
2026-09-18 and is reused unchanged (the current code reproduces the same plan
hash), but its labels are re-run with the uniform launcher below.

Run it from the repository root (CPU only; any Python with pydantic works, a
2-5 GB corpus takes one to three minutes):

```bash
PYTHONPATH=src python -m moe_exp.correlation_pipeline.sample_stratified \
  --generation-dir results/correlation_pipeline/gpt-oss-20b/generation \
  --generation-model openai/gpt-oss-20b \
  --output-dir results/correlation_pipeline/gpt-oss-20b/reasoning-vllm-v1/sampling-100k-stratified \
  --dry-run
```

Drop `--dry-run` to write `<output-dir>/generation/<model-slug>/...` (combined
100k), four self-contained `<output-dir>/parts/part-XX/...` roots and
`sampling_manifest.json`. Truncated/unscored traces are excluded and counted in
the manifest, which also records per-dataset cell supplies, the chosen
`sample_id` per problem, share deviations and plan hashes. Re-running with the
same seed is idempotent; a different plan into the same directory is refused.

### Reasoning-length metric

The length stratum comes from `sample_stratified.reasoning_tokens`, which
resolves in three tiers and records which one won per dataset in the manifest's
`eligible.<dataset>.length_metric_source`:

1. the server's `usage.completion_tokens_details.reasoning_tokens` when it is
   positive;
2. otherwise a count derived from the saved `metadata.token_replay`: the
   completion offsets are character spans into `cot_text`, so locating
   `reasoning_content` inside it identifies exactly which completion tokens are
   reasoning. This needs no tokenizer and no chat-format knowledge;
3. otherwise the trace is dropped and counted in
   `dropped_missing_reasoning_tokens`.

Tier 2 exists because vLLM does not populate the usage field for every model
family. The gpt-oss harmony path reports a constant `reasoning_tokens: 0` while
still parsing `reasoning_content` correctly, so the first gpt-oss sample stratified
100k units into a single length bucket whose quartiles were decided by the
`trace_sha256` tiebreak alone. The sampler now also **refuses** a dataset whose
eligible traces all report the same length, which is what makes that failure
loud instead of silent. Gemma, Qwen3.6 and Nemotron resolve entirely through
tier 1, and their selections are byte-identical before and after the change.

### Labelling the parts on LEONARDO

`sbatch/native_stratified_annotate.sbatch` labels one part of one source with
the usual Qwen3.8-27B judge (TP=2, MTP=3, batch 64, reasoning effort low, 12 h
limit, checkpoint resume). It reads only
`<sampling root>/parts/part-XX/generation/<slug>` and writes only
`/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-stratified/<source>/part-XX`.
Ports are unique per source and part (`gpt` 43000+part, `gemma` 43100+part,
`qwen36` 43200+part, `nemotron` 43300+part, `glm` 43400+part, `qwen330b`
43500+part) because two 2-GPU jobs can share a node:

```bash
STRAT=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-stratified
for src in gpt gemma qwen36 nemotron glm qwen330b; do
  case $src in gpt) base=43000 ;; gemma) base=43100 ;; qwen36) base=43200 ;;
                nemotron) base=43300 ;; glm) base=43400 ;; qwen330b) base=43500 ;; esac
  for p in 0 1 2 3; do
    sbatch --parsable sbatch/native_stratified_annotate.sbatch \
      --source $src --part $p --port $((base + p)) \
      --output-dir $STRAT/$src/part-0$p
  done
done
```

The 2026-09-18 Nemotron labels from `sbatch/native_nemotron_annotate.sbatch`
(same contract, ports 42000+part, output under `qwen38-mtp3-production/nemotron`)
are superseded by the uniform rerun; that launcher is kept for provenance.
Each part is internally stratified, so a single part is representative of the
global mix. The plan-only form of the client is still available for inspection:

```bash
<judge-env>/bin/python -m moe_exp.correlation_pipeline.annotation_batch \
  --trace-root <sampling root>/parts/part-00/generation/<model-slug> \
  --output-dir <scratch-part-00> \
  --part 0 --parts 1 --part-size 25000 --total 25000 \
  --judge-program results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json \
  --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:41800/v1 \
  --plan-only
```

### Generating a new source model

`sbatch/native_source_generate.sbatch` benchmarks a BF16 dense-MoE checkpoint on
two A100s and writes a trace root the sampler can consume. The vLLM argv comes
from `sbatch/native_source_server.sh`, which is a pure-stdout helper so the argv
is testable without Slurm. Per-source settings are not interchangeable:

| `--source` | model | attention | parser | context | temp / top-k |
| --- | --- | --- | --- | --- | --- |
| `glm` | `zai-org/GLM-4.7-Flash` | `TRITON_MLA` | `glm45` | 49152 | 1.0 / 0 |
| `qwen330b` | `Qwen/Qwen3-30B-A3B` | `FLASH_ATTN` | `qwen3` | 40960 | 0.6 / 20 |

GLM-4.7-Flash is routed through vLLM's MLA kernels (`glm4_moe_lite` is listed in
`is_deepseek_mla()`), where `FLASH_ATTN` is not a valid backend and `TRITON_MLA`
is the only MLA backend that supports SM80. Qwen3-30B-A3B is plain GQA, caps at
its native 40,960-token window, and has no MTP head, so `--speculation mtp` is
refused for it. The existing Nemotron and Qwen3.6 generation launchers are left
alone: their trace roots are published provenance.

Probe a new checkpoint before spending a full run. `sbatch/native_source_probe.sbatch`
is a 30-minute `boost_qos_dbg` job over a four-problem slice whose `SMOKE_OK`
verdict requires, besides the trace and token-replay counts, that
`sample_stratified.reasoning_tokens` resolves to a positive and **non-constant**
value on every trace. That is the sampler's own predicate, so a passing probe
guarantees the corpus is samplable. Note `boost_qos_dbg` allows at most two
running or pending jobs per user, which is exactly two probes.

```bash
sbatch sbatch/native_source_probe.sbatch --source glm
sbatch sbatch/native_source_probe.sbatch --source qwen330b

# only once both verdicts are SMOKE_OK
for ds in math500 aime24 aime25 olympiad amc23 minerva; do
  sbatch sbatch/native_source_generate.sbatch --source glm      --job-label "$ds" --datasets "$ds"
  sbatch sbatch/native_source_generate.sbatch --source qwen330b --job-label "$ds" --datasets "$ds"
done
```

### Merged deliverable

`labels_export.py` verifies the four parts of every source (complete summaries,
matching `annotations_sha256`, record schema, identities unique across parts,
per-dataset trace counts equal to the part manifests), concatenates them into
`<merged-root>/<source>/annotations.json` + `summary.json`, and writes
`manifest.json` (judge settings, code revision, Slurm launch record),
`SHA256SUMS.txt`, `README.txt` and a zip under the gitignored `data/labels/`.
`sbatch/mfa_export_labels.sbatch` runs `--verify-only` and then the export on
the serial partition; submit it with `--dependency=afterok:<labeling job ids>`
and `--launch-record <json>`.

```bash
PYTHONPATH=src python -m moe_exp.correlation_pipeline.labels_export --verify-only
PYTHONPATH=src python -m moe_exp.correlation_pipeline.labels_export \
  --launch-record /leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-stratified/launch_record.json
```

## 1. Generate with vLLM

The host Python on the current machine is 3.10, while the project requires
Python 3.11. Use the Docker launcher below, following the same pattern as
`probeTest/run_slurm.sh`; do not run `pip install -e .` on the host.

Build or refresh the project image once from the repository root:

```bash
docker build -t moe-mfa-experiments:latest .
```

### One-command full experiment

Install the separate, pinned inference-server image:

```bash
docker pull vllm/vllm-openai:v0.29.0
bash src/moe_exp/correlation_pipeline/run_all.sh
```

The project image runs the Python stages; vLLM runs in its own image. The shared
`/llms` Hugging Face cache is mounted into both. Models download there on first
use. `VLLM_IMAGE` can override the server image.

The stages are generation, sentence sampling, tagging, forward replay, and analysis.
Before tagging, the launcher selects one attempt per problem (`sample_id=0` for
repeated benchmarks) and at most 100,000 sentences across the selected benchmarks.
Benchmark quotas are proportional to error rate, computed from all scored source
attempts; sentences are sampled uniformly within each benchmark with seed 42.
The existing proportional allocation can select fewer sentences when supply is
limited. Sampling fails if accuracy is unavailable or all error rates are zero.
Use `--max-sentences N` to change the cap. The manifest and sampled generations
are saved under `reasoning-vllm-v1/sampling`; tagging and all forward views use
those generations, while the original generation files remain intact.
`--skip-annotate` reuses that sampled input and its annotations. Use
`--skip-sampling` only when intentionally supplying inputs directly (as the
separate `run_sampled_tagging.sh` wrapper does). `--skip-tagging` bypasses
sampling and keeps the full-data workflow. A changed sampling plan requires a
new results directory.

Analysis
includes correctness/events, metric-pair correlations, expert identities/pairs,
and full-reasoning, class and position views. Both vLLM servers listen inside
their container on `0.0.0.0:41800`; the local API is
`http://127.0.0.1:41800/v1`. Only one server runs at a time.

Use `--model` to select the checkpoint for generation, forward replay, and
analysis while keeping the tagging model and frozen GEPA program unchanged:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --model ORG/MODEL \
  --results-dir results/correlation_pipeline/my-model
```

If generation needs a separate quantized checkpoint of the same model, also
pass `--generation-model ORG/MODEL-QUANTIZED`. This option (or the
`GENERATION_MODEL` environment variable) overrides generation only, regardless
of option order. The generation manifest records `--model` as the matching
forward target. Use the same options when resuming saved runs.

Omitting `--model` retains generation's `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` and
forward/analysis's `unsloth/Qwen3.5-35B-A3B` defaults. Model profiles now select
the server parser, MTP behavior, context default, and forward loading for
Qwen3.6, Qwen3.5, Qwen3, Nemotron 3.5 Lightning, Gemma4, GLM-4.7-Flash, and
GPT-OSS-20B. New runs save and replay the actual server token IDs, including
each model's native reasoning-channel format.

The probe indices remain unchanged. GPT-OSS has no layers matching those
indices and requires an explicit `--all-router-layers` override for forward
replay. See [MODEL_SUPPORT.md](MODEL_SUPPORT.md) for the model table, commands,
memory settings, layer mappings, and verification limits.

To tag the already saved GGUF generations without regenerating answers:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh --skip-generate \
  --generation-model qwen3.5-35b-a3b-mtp-ud-q4-k-xl
```

Generation is stored under `results/correlation_pipeline/generation/<model>`.
Annotations, forward checkpoints and analyses are under
`results/correlation_pipeline/reasoning-vllm-v1`. `--results-dir` changes that
common root; `--generation-dir` overrides the generation directory. Use distinct
result roots when comparing generation checkpoints. Server logs are saved as
`slurm_logs/correlation-vllm-<generation|judge>-<timestamp>.log`.

A small isolated run including all four stages:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --datasets math500 --results-dir results/correlation_pipeline/smoke \
  --max-items 1 --samples-per-problem 1 --bootstrap-samples 20
```

`--workers` and `--judge-workers` control concurrent generation and tagging
requests; both default to eight. Sample counts per problem remain unchanged.
vLLM may temporarily queue some requests when GPU cache space is full; eight
workers do not guarantee eight active decodes for long contexts. Keep the tested
`GPU_MEMORY_UTILIZATION=0.90` default on this 32 GB GPU: higher limits caused
out-of-memory failures in the judge pilot.
Each sentence keeps the same question/previous/current/next context. Completed
labels are saved atomically even when requests finish out of order; final output
is restored to source order.

To run without any sentence tagging:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh --skip-tagging
```

This keeps generation, forward replay, whole-reasoning and position views,
correctness/event correlations, metric-pair correlations and expert analyses.
It omits class views (Read, Analyze, Plan, Implement, Explore, Verify, Monitor),
and requires neither a judge server/program nor annotation files. The output
root is `results/correlation_pipeline/reasoning-vllm-untagged-v1`; generations
keep their usual directory. This separates the reduced outputs from tagged
analyses. Adding labels later reuses generations but requires another forward
replay to compute the class features.

`--skip-annotate` instead reuses existing completed annotations and retains
class views. It is useful after tagging has finished. `--skip-generate`,
`--skip-forward`, and `--skip-analyze` skip their respective stages; skipping
forward requires checkpoints containing the selected views. For analysis only,
pass `--skip-generate --skip-annotate --skip-forward` for tagged outputs, or
`--skip-generate --skip-tagging --skip-forward` for untagged outputs.

An already-running tmux command keeps its original options. To switch it to
untagged mode, interrupt that command and relaunch with `--skip-tagging` and the
same generation settings; completed generation shards are reused. Omit
`--skip-generate` while some datasets still need generation.
`--limit N` limits saved traces in tagging/extraction/analysis; `--max-items`
and `--samples-per-problem` affect generation only. `--dry-run` prints the
complete plan without Docker calls or output changes.

For manual server management, the equivalent generation command is:

```bash
docker run --rm --gpus device=0 --ipc=host \
  -v /llms:/llms -e HF_HOME=/llms -p 127.0.0.1:41800:41800 \
  --entrypoint vllm vllm/vllm-openai:v0.29.0 \
  serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --served-model-name Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --host 0.0.0.0 --port 41800 --api-key local-vllm-key \
  --max-model-len 49152 --enable-prefix-caching --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 --generation-config vllm --language-model-only
```

Then, in another shell:

```bash
bash src/moe_exp/correlation_pipeline/run_docker.sh generate
```

Each trace records its checkpoint, exact chat messages, sampling settings and
seed. Its manifest retains the matching forward target
`unsloth/Qwen3.5-35B-A3B`; extraction rejects a different target. The new model
directory is `Qwen--Qwen3.5-35B-A3B-GPTQ-Int4`.

See [REASONING_VIEWS.md](REASONING_VIEWS.md) for tests and analysis contracts.
The older llama.cpp launcher remains available to reproduce the historical
performance comparison below.

### Measure generation speed

`--workers` controls concurrent HTTP requests independently of benchmark repeat
counts. vLLM's `--max-num-seqs 8` enables actual concurrent decoding. The previous
llama.cpp pilot used one MTP slot, so its one-versus-two-worker result does not
predict vLLM throughput. See the historical measurements in
[the performance report](../../../report/generation_performance.md).

Compare workers on identical saved prompts and seeds without writing to the
generation dataset. This standard-library diagnostic also runs on host Python:

```bash
PYTHONPATH=src python3 -m moe_exp.correlation_pipeline.profile_generation \
  --generation-dir results/correlation_pipeline/generation/qwen3.5-35b-a3b-mtp-ud-q4-k-xl \
  --output-dir results/generation_performance/new-comparison \
  --base-url http://127.0.0.1:8080/v1 \
  --workers 1 2 --requests 32 --repeats 2 --max-tokens 1024
```

The output directory must be new. Each setting executes all 32 requests, with
the setting order reversed on the second repeat. A short warmup precedes each
trial. `summary.json` records actual server slots, wall throughput and token
limit hits; `requests.jsonl` retains request latencies and response hashes.
These capped outputs measure speed and are not accuracy evaluations. Request
latencies include queueing and must not be summed for wall throughput.

To attribute a completed serial run's actual time to benchmarks, match its
server timing log to the saved generation shards:

```bash
PYTHONPATH=src python3 -m moe_exp.correlation_pipeline.audit_generation_performance \
  --generation-dir results/correlation_pipeline/generation/qwen3.5-35b-a3b-mtp-ud-q4-k-xl \
  --server-log slurm_logs/correlation-mtp-20260903_152848.log \
  --output results/generation_performance/historical.json
```

This audit requires one complete serial log whose completed requests match all
the shards under the supplied directory. It verifies every completion token
count and checks timestamp alignment; it rejects mismatched or overwritten
runs. Partial benchmark shards are reported separately from completed datasets.

### Sentence classes and reasoning positions

The pipeline also supports a frozen GEPA sentence-annotation stage and full,
class-specific, and position-specific reasoning views. See
[REASONING_VIEWS.md](REASONING_VIEWS.md) for the complete rerun command,
unit tests, smoke runs, checkpoint rules, and output interpretation.

## 2. Teacher-forced forward pass

Stop the vLLM server to release VRAM. Forward replay uses the corresponding
Qwen3.5 base model through the existing Unsloth backend:

```bash
src/moe_exp/correlation_pipeline/run_docker.sh forward \
  --quantization unsloth-4bit
```

Hidden states are extracted by default. `--router-only` removes hidden
trajectory and hidden/router geometry features. By default, extraction reads
`results/probeTest/qwen3.5-35b-a3b-gptq-int4/probes/results.json` and retains
only the union of its `best_by_target` layers. Pass `--probe-results` to use
another completed probe run. Probe hidden-state indices without a corresponding
router layer (currently index 40) are reported and excluded. The original layer
numbers are stored in each trace and preserved in analysis column names.

The forward stage now reduces router and hidden tensors on the fly. Normalized
confidence (derived from entropy), top-k confidence measures, switching, overlap, hidden
norm, trajectory distance, and hidden/router geometry are written into each
trace as compact scalar features. Full router and hidden tensors are released
without serialization. Top-k expert IDs remain on disk because expert-identity
and expert-combination analyses need them and they are small. The default output
is:

```text
results/correlation_pipeline/forward/<hf-model>/<dataset>/
  traces_with_routing.jsonl
  tensors/*_experts.pt
  tensors/*_extraction.json
```

Each trace contains a storage audit with tensor shapes, dtypes, element counts,
projected raw bytes, and actual persisted bytes. The model-level `summary.json`
aggregates these values and reports the raw payload avoided. The geometry metric
uses at most 128 uniformly sampled tokens by default; change this contract with
`--max-geometry-tokens`.

Use `--save-raw-tensors` only for a small audit or ablation run. It additionally
writes `*_logits.pt` and `*_hidden.pt` while retaining the same compact
features. The correlation stage never saves normalized expert-weight tensors.
Analysis excludes generations with `finish_reason=length` or completion tokens
at or above `max_tokens` from all trace, expert, and reasoning-view statistics.
Unknown token-limit status is retained. Problem-level avg@n statistics still
require all n attempts, so groups made incomplete by this filter are ineligible.
The output records excluded counts per dataset; the generation-budget audit
continues to cover all input traces. Existing analysis artifacts must be rerun
to apply this policy.

Entropy is retained in the compact audit values, but the analyzer reports its
normalized confidence transform instead of treating both affine-equivalent
quantities as separate correlation evidence.

The GGUF and Hugging Face checkpoints represent the same post-trained model but
use different 4-bit formats, so their numerical activations are not identical.
Quantizing either copy does not change the tokenization/chat-template
requirement; the exact generation messages are replayed with the Hugging Face
tokenizer. Qwen3.5 exposes `Qwen3_5MoeForConditionalGeneration`; Unsloth's
`FastModel` keeps the standard forward API used to request router logits and
hidden states. Extraction calls the decoder backbone directly, so it does not
materialize vocabulary logits or the CausalLM wrapper's auxiliary
load-balancing loss. KV caching is also disabled. Qwen3.5's router API returns
probabilities despite the `router_logits` name; extraction stores equivalent
log-probabilities so downstream softmax-based metrics preserve the native router
distribution exactly.

The Docker launcher sets `UNSLOTH_DISABLE_STATISTICS=1` to avoid an unrelated
Hub telemetry probe and places Unsloth/Torch compile caches under the writable
shared cache. The loader also defaults `UNSLOTH_DISABLE_VENDORED_FLA=1` and
`UNSLOTH_COMPILE_DISABLE=1`: with Torch 2.11, the vendored gated-delta Triton
kernel and the compiled router/hidden-state graph terminate natively, while the
official pure-Torch/eager fallbacks complete within the RTX 5090 memory budget.
These settings can be overridden for a future compatible stack; they do not
disable model downloads or alter weights.

## 3. Analyze correlations

```bash
src/moe_exp/correlation_pipeline/run_docker.sh analyze \
  --bootstrap-samples 500
```

This reads the compact features directly (and remains backward-compatible with
older raw-tensor runs) and writes:

```text
results/correlation_pipeline/analysis/<hf-model>/
  trace_features.csv
  correlations.json
```

Features include per-layer and across-layer summaries of router entropy,
margin, top-1 switching, top-k overlap, hidden-state norm, consecutive hidden
trajectory distance, and hidden/router geometry correlation. Binary outcomes
are final correctness, backtracking, contradiction, and self-correction.

Repeated attempts are not treated as independent. The output contains:

1. exploratory point-biserial trace-level correlations;
2. problem-cluster bootstrap intervals for prespecified across-layer and
  structure features;
3. problem-level Spearman associations between pass rate and feature mean or
   variability; and
4. within-problem correct-minus-incorrect feature contrasts for prompts that
   produce both outcomes.

GPQA-Diamond is excluded from the last contrast because its ten attempts also
permute answer positions; mixing correct and incorrect attempts would confound
reasoning differences with option-position bias. Its trace-level and
problem-level results remain available.

Naive independent-sample p-values and layer-specific feature rankings are
retained only as exploratory diagnostics and are labelled as such. Any later episode classifier can add numeric values under
`TraceRecord.metadata["episode_features"]`; the analyzer automatically includes
them as `episode_*` columns.
