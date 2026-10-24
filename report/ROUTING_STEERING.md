# Routing steering implementation and evidence status

This implements the campaign interfaces and experiment machinery for the supplied
plan. **The investigation is not complete:** full feature extraction, candidate
selection, prospective fitting, annotation collection, GPU qualification,
calibration and the causal pilot have not run. No steering benefit is established.

## Observed results

- The prior accepted compact audit reproduced **exactly**, including all source
  hashes: [reproduction](routing-steering-audit-20260928/evidence-reproduced.json).
  Its minimum common-range BH-adjusted value is 0.2190262971.
- The frozen inventory contains **8,603 deduplicated model-attempts**, 12 accepted
  cohorts and seven models. It includes the newly accepted Gemma A and Qwen3-30B A
  captures. Five models have accepted A captures; GLM and Qwen3.6 have B only.
- B contains **895 model-attempts, 51 distinct questions across models**. The
  inventory preserves all A/B memberships, original arm probabilities and every
  separate capture receipt. It does not turn B into a representative benchmark.
- One Qwen3.5 A attempt is capped and adjudicated correct. Correctness, termination
  and unresolved outcomes remain separate; extended continuations are excluded.
- Shared question folds reserve **300 A-only questions** and leave **1,247** for
  development. B's entire question union is excluded from the reserve. These are
  exploratory reserved questions, because earlier aggregates inspected the corpus.
- [Question-balanced outcome/cost tables](routing-steering-audit-20260928/frozen-baselines.md)
  now include the two new A cohorts. Their
  [JSON artifact](routing-steering-audit-20260928/frozen-baselines.json) contains
  joint question bootstrap intervals, natural-completion sensitivity, denominators,
  unresolved counts, input hashes and code provenance. These are descriptive
  baseline tables, not corrected routing associations or a steering frontier.

Frozen inventory and folds are under
`results/correlation_pipeline/routing-steering-v1/{inventory,folds}.json`; those
larger input manifests remain ignored as experiment artifacts. `inventory.md`
contains the cohort coverage table. A zero **verified** window/tensor count there
means packet extraction is pending, not that the metric is absent.

## Implemented interfaces

| Component | Implementation | Scope |
|---|---|---|
| Versioned artifacts | `dynamics/contracts.py` | Content bindings, immutable/idempotent writes, populations, global question folds, timing checks |
| Campaign inputs | `dynamics/campaign.py` | Acceptance/receipt/outcome/sampling joins; exact source-token checks; stored windows first; optional retained-tensor occupancy; same-layer whole/reasoning scope summaries; measured sizing estimates |
| Expert contrasts | `dynamics/contrasts.py` | Matched question/position/state contrasts; per-token selection RD and normalized dispatched-weight RD; secondary slot-share/log-odds/nPMI; joint question bootstrap; all-expert BH families; fold-local selection; stable singleton group cards; sibling sequence controls |
| Prediction | `dynamics/prospective.py` | Four outer/three inner frozen folds; cumulative position/text/prefix-state/static/ordered blocks; training-only transforms and state readouts; estimated history; classification/cost errors, calibration and paired improvements; A calibration and eligible-B weighting |
| State events | `dynamics/state_features.py` | Features before destination-overlapping tokens; gaps break histories; caps censor events; natural termination competes with transitions |
| Annotation plan | `dynamics/annotation_plan.py` | At most 16 B question pairs, three first/centred/last blocks of 40 sentences, 3,840 labels before exact reuse; 200 paired prefix-only labels |
| Native actions | `moe_guiding/native.py` | Backend-delegated native selection; signed pre-selection bias; selected-mixture reweighting preserving routed scale; timing, realized displacement and no-op evidence contracts |
| Runtime adapter | `moe_guiding/native_integration.py` | GPT/Qwen3 native router-instance hooks; one request, eager execution, exact positions, no prefix caching/speculation/EPLB; phase termination and skipped-action telemetry; reject paths bypassing hooks |
| Mechanism controls | `moe_guiding/mechanisms.py` | Same-layer/group-size prevalence matching and equal-compute verified-continuation likelihood panels, separate from free-generation benefit |
| Branches | `continuation.py`, `dynamics/experiment.py` | Arbitrary exact prefixes; original effective cap; 16 questions × 2 seeds × 5 arms; optional 64 timing branches reusing baseline blocks; immutable locked resume; paired question analysis and point-estimate frontier |
| Native execution | `moe_guiding/native_run.py` | Guarded Slurm-only runner for a previously qualified, frozen manifest and engine configuration |

The legacy Mixtral top-2 commands and historical extended-continuation mode remain
available. The legacy dynamics evaluator now puts order-invariant entropy
summaries in its static block and calls retrospective next-class tasks
`oracle_next_class`.

New reductions retain expert weight occupancies directly. Old packets can recover
them from selected IDs and executed weights, after hash verification. They cannot
recover unselected candidate scores or finer hidden trajectories.

Card-v3 stores the sampler under `generation_config.sampler`, and often records
`max_tokens: null` with the actual cap in `metadata.effective_max_tokens`. The new
branch path resolves that exact effective cap. It does not substitute the
historical continuation module's separate 131,072-token extension budget.

## Analysis and claim boundaries

The contrast suite accepts state/destination/recovery rows only with the necessary
contiguous labels and timing. Landmark extraction does not manufacture those
labels. `state_features.event_rows` and `annotation_plan.prepare` provide the
interfaces for the label stage; they do not invoke a paid annotation service.

Entropy decomposition is order invariant and belongs in the static baseline.
Entropy difference is dependent on local and marginal entropy, not another
independent mechanistic finding. Turnover and ordered/shuffled drift belong in
the ordered increment. Boundary thresholds and expert support are fitted inside
training splits. Candidate selection uses training questions, with sign checked
on held-out questions in at least three of four folds. At most three singleton
layer-specific groups per model are retained across the implemented accuracy and
efficient-success nominations.

Prefix-only state readouts require prefix-only training labels. Existing look-ahead
labels remain discovery strata or targets. The current predictive output does
**not automatically qualify a trigger**: useful lead time and a frozen trigger
threshold still require validation. A stable candidate can instead enter the
prespecified fixed-landmark mechanism test.

The pilot builder requires native no-op/output/attribution evidence, historical
backend revalidation, development-prefix dose records, and matched sham evidence.
The implemented initial approximation tolerance is **5% ± 2 percentage points**
changed expert sets for each selection-bias direction and sham; this is recorded
in the frozen specification. Reweighting records its achieved weight displacement.
Neither small gaps nor likelihood improvement qualify action benefit.

The native adapter has been inspected against the installed vLLM source and tested
with synthetic modules. **It has not passed GPU qualification on GPT or Qwen3.**
Monolithic MoE kernels and unobserved routing paths are rejected. Native selected
layer executions are reported; total executions across all layers/shared experts
are left unavailable until qualified instrumentation measures them. Router timing
includes instrumentation and native selection; paired latency and no-op overhead
must be measured before interpreting an efficiency result.

The paired analyzer excludes unknown correctness from that metric and returns
`awaiting_adjudication` while any outcome is unresolved. It keeps total/reasoning
cost, caps, routing manipulation, state metrics and latency distinct. Causal
conclusions, once a randomized pilot runs, are conditional on these exact prefixes,
not whole-prompt policy value or deployment safety.

## Reproduction commands

Use the existing project numerical environment with one BLAS thread and the
repository `src` on `PYTHONPATH`. All new stages are available through:

```text
python -m moe_exp.correlation_pipeline.dynamics steering inventory --manifests DIR --acceptance DIR --outcomes FILE --output inventory.json
python -m moe_exp.correlation_pipeline.dynamics steering folds --inventory inventory.json --output folds.json
python -m moe_exp.correlation_pipeline.dynamics steering baselines --inventory inventory.json --output baselines.json
python -m moe_exp.correlation_pipeline.dynamics steering extract --inventory inventory.json --output feature-dir --include-weights
python -m moe_exp.correlation_pipeline.dynamics steering analyze --features feature-dir/manifest.json --folds folds.json --output analysis.json --evaluate
python -m moe_exp.correlation_pipeline.dynamics steering pilot-plan --prefixes prefixes.json --folds folds.json --candidate candidate.json --calibration calibration.json --qualification qualification.json --output branches.json
python -m moe_exp.moe_guiding.native_run --branches branches.json --engine engine.json --output pilot-dir
python -m moe_exp.correlation_pipeline.dynamics steering pilot-analysis --branches branches.json --results RESULT_FILES --output paired.json
```

Full extraction, fitting and inference belong in authorized Slurm jobs. Sizing
outputs carry population `sizing_only` and are rejected as scientific analysis
inputs. Failed/partial extraction resumes per immutable attempt artifact; changed
inputs/configuration/code require a fresh output directory. Branches use a frozen
code binding and per-branch locks, so incompatible runs cannot mix or count a
branch twice. A crash before result persistence can repeat compute, never a saved
result.

## Resource gate and outstanding work

`sbatch/routing_steering_size.sbatch` completed as **job 58911879**:

- 24 existing captures: shortest and longest in each of 12 accepted cohorts.
- 1,300,951 completion tokens represented; 1,707,169,534 compressed input bytes.
- One requested CPU, 24 GiB RAM, at most 30 minutes, no GPU or model load.
- Private repository outputs; staging remains in the user's private WORK subtree.
- Slurm reports `COMPLETED`, exit `0:0`, elapsed **00:04:57**, on `login13`.
- One CPU was requested; Slurm allocated two. The Python step used one CPU and
  reported peak RSS **6,813,248 KiB (6.50 GiB)**. No GPU was allocated.
- All 24 captures passed receipt, packet, exact-token and retained-tensor checks.
  Twenty-two had usable stored windows. All 24 had native identity semantics;
  the same 22 had window-level occupancy and gaps. The ten sampled A captures
  had annotations; the fourteen sampled B captures did not. This is coverage
  of the sizing sample, not an estimate of full-population missingness.
- Execution used frozen source `84c9396d5358551e527aa206481d4abeef94dc318aca990282479e945301e1ac`.
  The sizing manifest binding is
  `9991b7ca96a00bd30377880e05a7d4fee27cd3f98cfd1fffa29718ae0e8742c8`.

Live checks identified `login01.leonardo.local`, user `lmolfett`, account
`iscrc_miosr`, and serial QoS limits of two running jobs, ten submitted jobs,
eight CPUs and 30,800 MiB per user. Other jobs already use this QoS.

The concrete next proposal is four extraction shards, one CPU requested and
12 GiB RAM each, four hours maximum, concurrency two; followed by a one-CPU,
12-GiB, 30-minute merge/check job. Maximum requested processing time is 16.5
CPU-hours; the observed two-CPU allocation granularity could make the allocation
ceiling 33 CPU-hours. The user authorized this envelope on 2026-09-28. At
submission, other serial jobs occupied eight of the ten allowed submission
slots, so the four logical shards were packed into two workers within one job
using `sbatch/routing_steering_extract_packed.sbatch`. This preserves the data,
12-GiB worker memory and storage limits, while reducing the maximum requested
CPU time to **8.5 CPU-hours including verification**. Each worker runs two
shards sequentially; the packed job has a four-hour wall limit.

**Extraction job 58915268** was submitted and verified running on `login13`,
with two concurrent Python steps. It requests one physical core per worker and
24 GiB total memory; Slurm allocates four logical CPUs across those two cores.
**Verification job 58915273**
requests one CPU, 12 GiB and 30 minutes, with confirmed dependency
`afterok:58915268`. It remains pending on that dependency. These are actual
job IDs, not scheduler dry-run IDs. Neither job uses a GPU, performs inference
or creates labels. Completion and scientific results remain pending. The
submission commands, full returned output and verification are recorded in
`results/correlation_pipeline/routing-steering-v1/full-submission-58915268.json`.

Two short launch attempts were stopped during concurrency verification:
58914637 (131 seconds) inherited the complete default tmpfs GRES per step;
58915046 (81 seconds) placed the two requested CPUs on sibling hardware threads
of one physical core. Their unused dependent merge jobs were automatically
cancelled by Slurm. The corrected launcher uses `--gres=none` for steps,
`--hint=nomultithread` and physical-core binding. It still stages only in private
WORK. The two stopped launches used 0.118 allocated CPU-hours in total. Including
them and the merge's observed allocation granularity, the final allocation
ceiling is 17.12 CPU-hours, below the approved 33-hour possible allocation ceiling.
Frozen extraction code and inputs are unchanged, so any completed artifacts can
resume without being mixed with incompatible results.

Interim check on **2026-09-28 at approximately 16:09 CEST**: shards 0 and 1
completed successfully (2,151 attempts each; step exit codes `0:0`) in about
81–82 minutes. Shards 2 and 3 are running. Around 54% of the 8,603 attempt
artifacts have been saved; the completed shards contain 6.42 GiB of output.
The extraction error log is empty and the merge remains pending on its success
dependency. These are progress observations, not final acceptance.

Across the two completed shards, 4,295 of 4,302 unique attempts have usable
windows, executed-weight occupancy and gaps. All have verified native identity
metadata. Only 16 of 443 inspected B attempts have attached sentence annotations;
all inspected A attempts have attached annotations (this does not imply dense
sentence coverage). Cohort coverage counts retain A/B memberships and must not
be summed as unique attempts. The partial coverage artifact is
`results/correlation_pipeline/routing-steering-v1/status-20260928-1609.json`.
No new contrast, predictive or causal result has been computed from this partial
extraction.

Measured projections are recorded in
`results/correlation_pipeline/routing-steering-v1/resources-startup-separated.json`:
7.51 total processing hours including 25% margin and four startup allowances,
1.85–1.90 projected hours per shard, and 17.33 GiB conservative output projection.
The full read covers 141,932,838,440 compressed input bytes. The output allowance
is 25 GiB; each shard aborts before writing attempt artifacts beyond 6 GiB.
Existing WORK filesystem headroom was 156 GiB at preflight.

The first 9,337-token extraction took 79.04 seconds, including lazy numerical
imports; other Gemma measurements were 0.129 seconds for 524 tokens and about
19 seconds for 131,072 tokens. Initialization was not timed separately. The
startup-separated projection charges the entire first duration once per worker
and estimates steady work from the remaining observations. The unadjusted,
deliberately conservative estimate remains in `resources.json`; it charges the
startup-inclusive excess per attempt. Neither projection is a throughput guarantee.

Shards use immutable source snapshots, compatible resume manifests and a verified
merge requiring all 8,603 attempts exactly once. Sizing-only artifacts remain
ineligible for scientific analysis. The new `resources` and `merge-extractions`
CLI stages expose these checks without submitting any jobs.

The accepted dense Qwen3.6 pilot receipts report 64 traces, 30,187 labeled
sentences and 640,800 windows. Its bnb-4bit replay has separate provenance; its
existing analysis manifest still says `running`, with no finished analysis
tables present. The capture/label receipts do not establish method-validation
completion. GPU calibration/inference and labeling still need measured proposals.

Still outstanding: complete packet/tensor availability verification; full routing
reconciliation and retrospective difficulty adjustments; accepted dense-pilot
method replay; candidate cards from real held-out features; prospective results
and useful-lead-time checks; prefix-only annotation comparisons; native GPU parity
and historical-backend comparison; calibrated real policy/prefix manifests;
likelihood-panel execution; 160 pilot continuations and optional timing arms;
adjudication and paired causal analysis. A larger trial remains a later decision
requiring fresh questions, an accuracy margin, minimum useful saving, and measured
variance/cost. No such trial is authorized or automatically scheduled.

## Verification

The initial regression run passed **104 tests with one optional integration skip**
across `test_routing_steering.py`, `test_routing_dynamics.py`, `test_moe_guiding.py`
and `test_moe_guiding_launcher.py`. Checks include native k and scale,
request timing/isolation, original-budget branches, paired timing controls,
annotation reuse/budgets, matched scopes, equal-compute likelihood, fold leakage
and resume integrity. `git diff --check` and sizing-script shell syntax passed.
The latest original-sampling-probability, question-identity and termination checks
passed with an unchanged frozen inventory payload.

After the measured sizing run, the updated routing/steering suites passed
**60 tests in 199.72 seconds**. New checks cover model-specific efficiency
matching, vector/scalar bootstrap agreement with missing questions, complete
shard unions, startup accounting and storage-limit aborts. Frozen source
`ed9332b65d989d1fd09b9285d66e0a629faa292d119f2ac5e820b2ad52acdfbb`
was read back and its digest verified. Both full-extraction launchers passed
final Slurm dry runs; their hypothetical job IDs are not submissions.

A bounded real-packet check on a GPT capture of 256 completion tokens (compressed
packet below 1 MiB) verified its source/packet hashes, exact token IDs, retained
windows, native IDs, gaps and attached labels. Its effective original budget is
130,973 tokens. It supplies no prospective row at token 256 because the attempt
has already ended there. Tensor occupancy was not read in this bounded check.

No checkpoint was loaded, no inference/training ran, no package/environment was
installed, and no commit or push was made.
