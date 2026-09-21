# MoE margin guiding

Second guiding experiment from the tutor call, especially **44:53–45:19**:
identify margin values associated with higher accuracy, then encourage those
values during generation. The target-range implementation and top-k boundary
default were confirmed by the user. No reasoning-class labels are used.

This follows `moe_identity_guiding`'s structure: `prepare.py` splits problems,
`calibration.py` freezes a policy, `routing.py` hooks native routers, and `run.py`
generates and compares matched conditions. `run_global.sh` supplies the same
Qwen/OSS/Gemma presets, and `run_docker.sh` uses the existing guiding image.
Problem keys, model presets, native router specifications, prompt loading and
paired scoring/comparison are reused from the existing experiments.

The older `moe_guiding` experiment targets Mixtral and replaces an expert using
a different margin definition. Its top-2 replacement rule does not apply here.

## Metric and calibration

Probabilities are the full softmax of router logits, before selected-expert
renormalization. With probabilities sorted descending:

```text
router_boundary_margin = p[k] - p[k+1]     # default, one-based ranks
router_margin          = p[1] - p[2]       # optional legacy analysis metric
```

Calibration consumes `metadata.correlation_features` from the correlation
pipeline, including its expert counts, layer IDs, and per-layer metric values.
Raw `[layers, tokens, experts]` `model_logs.router_logits` tensors are a fallback;
`--tensor-base-dir` controls relative tensor paths. Selected-expert tensors alone
cannot recover a margin. Cached boundary margins must be nonnegative, as expected
for the supported native softmax top-k routers.

Each observation is one trace's mean margin at one layer. Each problem has total
weight one, divided equally among its attempts. For each layer:

1. Partition the observed margins using problem-weighted quantile cuts
   (`--bins 5`). Ties stay together, so fewer bins may result.
2. Record every bin's accuracy, lift over calibration accuracy, distinct-problem
   support and observed minimum/maximum margin.
3. Among bins supported by at least `--min-support 4` problems and positive lift,
   choose the highest accuracy; break ties by support, then lower margin.
4. Use that bin's observed minimum/maximum as the frozen target range. Skip
   layers without eligible bins; fail if no layer is eligible.

The policy records all bins, targets, source hashes, calibration problem IDs,
and feature provenance. Calibration needs scored correct **and** incorrect
answers. No held-out correctness is used to select a target. If tuning bin count,
support or strength, use a separate validation split before final evaluation.

## Intervention

Let `r = k` for boundary margin, or `r = 1` for top-1/top-2 margin. For each live
token at a selected layer, let `m` be its margin and `[lo, hi]` the learned range:

```text
t = clamp(m, lo, hi)
d = m + strength * (t - m)                 # strength in [0, 1]
```

To reduce the margin, mix the full probability vector with the uniform
distribution: `q = (1-a)*p + a/E`, where `a = (m-d)/m`.
To increase it, mix with uniform mass on the current top-r experts:
`q = (1-a)*p + a*u_top_r`, where `a = (d-m)/(1/r-m)`.
The hook returns `log(q)` in the gate's original dtype, leaving native dispatch
and expert-weight normalization in place. Within-range rows and strength-zero
controls retain their original logits exactly.

In exact arithmetic, this reaches `d` and preserves expert ordering except at
singular endpoints. Mixture coefficients are capped just below one to avoid
zero probabilities and fully uniform ties. FP16/BF16 rounding can cause ties or
small target errors. Diagnostics report actual post-cast margins, distance to
the target range, changed logit rows, and changed top-k sets. Expert identities
are not preferentially selected, but their relative mixture weights can change.

Learning ranges from **trace means**, then steering **individual token margins**
is an experimental assumption, not a causal conclusion or a guarantee of higher
accuracy. The paired generation experiment measures its effect. The hook applies
to both prefill and decode, independently of reasoning classes. Counters include
engine padding/recomputation; tensor-parallel worker counters must not be summed
as independent tokens.

## Run

From the repository root, preview the full workflow without Docker execution:

```bash
DRY_RUN=true bash src/moe_exp/moe_margin_guiding/run_global.sh all
```

Run preparation, fitting, baseline generation, guided generation and comparison:

```bash
MODEL_PROFILE=qwen bash src/moe_exp/moe_margin_guiding/run_global.sh all
# Other presets: MODEL_PROFILE=oss or MODEL_PROFILE=gemma
```

The launcher defaults to `full`: all original attempts (32 for AIME24, AIME25
and AMC23; one for MATH500, Minerva and Olympiad). Use `single` as the second
argument for one attempt per held-out problem. Stages can be run separately:

```bash
bash src/moe_exp/moe_margin_guiding/run_global.sh prepare
bash src/moe_exp/moe_margin_guiding/run_global.sh fit
STRENGTH=0.5 bash src/moe_exp/moe_margin_guiding/run_global.sh generate full
bash src/moe_exp/moe_margin_guiding/run_global.sh compare full
```

Use a new `OUTPUT_ROOT` for another experiment; existing splits, policies and
run manifests are protected from overwrites. Other environment overrides include
`MODEL`, `GENERATION_ROOT`, `ROUTING_ROOT`, `CALIBRATION_FRACTION`, `METRIC`, `BINS`,
`MIN_SUPPORT`, `MAX_NUM_SEQS`, `TEMPERATURE` and `TOP_P`. The default output root is
`results/moe_margin_guiding/<profile>_global`. As in identity guiding, generation
defaults to 32,768 output tokens, 49,152 context tokens and concurrency 16.

The full-population split is stratified by dataset and disjoint by source
problem, keeping repeated attempts together. Evaluation includes held-out
generation problems regardless of saved correctness or routing availability.
Calibration preserves cached margin features. You can also prepare a smaller
experiment directly from scored routing traces:

```bash
bash src/moe_exp/moe_margin_guiding/run_docker.sh prepare \
  --traces /path/to/traces_with_routing.jsonl --output-dir /workspace/results/margin/split
bash src/moe_exp/moe_margin_guiding/run_docker.sh fit \
  --traces results/margin/split/calibration.jsonl --output results/margin/policy.json
```

For this smaller preparation, prompts are `split/prompts.jsonl` and the eligible
population is restricted to scored traces with margin data. Use the `generate`
subcommand with `--policy`, `--prompts`, `--condition baseline|guided`, and a fresh
`--output-dir`. Paths inside Docker must refer to mounted locations; relative
repository paths are recommended.

Each condition saves `manifest.json` and `generations.jsonl`. Comparison reuses
the identity experiment's checks: same policy, prompts, model settings, sampling,
versions and tokenized inputs, with complete paired IDs. It reports accuracy,
mean generated tokens, truncations, and wrong-to-right/right-to-wrong counts.

## Runtime support and validation

The worker supports the existing Qwen3.5, GPT-OSS and Gemma4 native router
locations. Expert dimensions are checked against the loaded checkpoint. It
requires eager execution and tensor parallelism only; prefix caching is disabled.
Expert parallelism/load balancing, pipeline/data parallelism, speculative
decoding and DBO are rejected, as is the OSS ROCm path that bypasses its router.
Every selected hook must process tokens or the run fails.

The Qwen and Gemma presets inherit the existing workflow's transfer from replay
checkpoints to quantized generation checkpoints. Numeric margin ranges may shift
between those checkpoints, so inspect achieved margins in the diagnostics.

CPU tests cover both metrics and intervention directions, native precision,
mixture-weight changes, cached/raw calibration, problem-disjoint preparation,
overlap rejection, router hook locations and launcher previews:

```bash
python -m pytest tests/test_moe_margin_guiding.py -q -p no:cacheprovider
```

No end-to-end GPU generation result is claimed by these tests.

### Sampling matched to the original generations

Both baseline and guided runs use temperature 0.6, top-p 0.95, disabled top-k,
and 32,768 maximum output tokens. The launcher recovers each attempt's original
seed and chat-template options from the generation traces (including Gemma's
`enable_thinking`). Baseline and guided conditions use identical per-attempt seeds.
Sampling or model mismatches fail before model loading.

New runs are saved under `<output-root>/sampling/full/{baseline,guided}`; old
greedy results remain intact. Stochastic runs must generate fresh completions.
`all` reuses an existing split and policy, then generates both conditions and
compares them; it resumes compatible incremental runs and skips validated completed runs.


## Faster execution, baseline reuse, and recovery

Global launchers now use `--diagnostics minimal` and `--resume`. Minimal diagnostics
retain per-layer hook call/token counts and the mandatory hook-activity checks,
without extra top-k comparisons, expert histograms, or post-intervention margin
statistics. The intervention arithmetic is unchanged. Use `DIAGNOSTICS=full` on
a fresh run (or `generate --diagnostics full`) for detailed router validation.
Unavailable statistics are omitted, rather than reported as zero.

Before loading a baseline model, generation searches completed runs beneath
`results/moe_identity_guiding` and `results/moe_margin_guiding`. Reuse requires
matching model/revision, engine settings, software versions, prompt-file hash,
sampling configuration, and every saved attempt's input and seed. The source
must have complete coverage and diagnostics proving no guiding hooks were
installed. Different guiding policies and the two known worker-extension names
are allowed, since baseline execution applies neither policy. Unknown worker
extensions are rejected. This search applies to all model profiles, not just Qwen.
`--baseline-search-root PATH` overrides the search roots and can be repeated;
`--no-reuse-baseline` requests a fresh baseline in a fresh output directory.

A reused baseline is copied into the destination, preserving its source manifest
and content hashes in `reused_baseline`. The destination records the current
comparison policy; the nested source manifest records the actual originating
execution. Comparison verifies the reused completion hash and source compatibility.
The source run remains intact. Existing completed destinations are validated and
skipped, not replaced by a different baseline.

Generation retains vLLM continuous batching and writes each completed, scored
attempt to `generations.jsonl`, flushing and syncing it to disk. The manifest
records `completed_count` and `expected_count`. Completions are saved in completion
order; comparison joins by ID. No partial token sequence is checkpointed.

Rerun the same global command after interruption, or pass `--resume` to `generate`.
Saved attempts are validated and skipped; unfinished attempts restart with their
original seeds. Resume refuses changed run settings, duplicate IDs, incompatible
records, or another writer holding the run lock. A torn final JSONL line is
removed on resume; corruption in complete lines is rejected. Scheduling after a
restart can differ, so seeds alone do not guarantee bit-identical regenerated
answers. Hook reports from earlier sessions are retained separately.

Incomplete legacy runs without the incremental execution marker cannot be
resumed automatically: they may still be executing in an older process. Let them
finish, or use a fresh output directory after stopping them. These code changes
do not alter processes that are already running.
