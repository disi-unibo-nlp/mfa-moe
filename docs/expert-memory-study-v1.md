# Seven-model expert memory study v1

This is a private research harness for the October 15 accuracy–memory result.
It does not change, dispatch, cancel or replace the verification v3 campaign.
The implementation request adopted the numerical defaults in
`report/expert-memory-study-v1/PROTOCOL.json`. The subsequent instruction,
"Go on and submit everything we need", authorizes necessary finite allocations
under the measured-price, qualification and complete final-feasibility gates.
`WORKLOAD_AUTHORIZATION-r1.json` preserves that instruction and its scope.

## Entry and source boundaries

Run `scripts/expert_memory_study/run.py --help` with the existing inference
environment and `PYTHONPATH=src`. Preparation, card creation, source checking,
population reservation, mask freezing, sizing and export are bounded operations.
Real saved-data analysis, qualification, generation and grading run only in a
Slurm Python step through an exact card and a separate authorization receipt.
CPU/GPU scripts reject direct login-shell execution. All writes stay in private
HOME/WORK. There are no new environments or checkpoint copies.

`prepare --output report/expert-memory-study-v1` binds seven native model
contracts, the existing count archives, model/backend sources and all 35
scientific/29 readiness files of the preserved campaign. Once sealed, repair
this harness in a new version and reconcile existing intents. Never edit a
sealed source while its jobs are pending.

## Saved observations and selection

The historical source has one outcome-blind attempt per question, 1,547 shared
questions and 1,486 duplicate families. The archive identity joins and selected
slot counts are checked before use. Five common folds group duplicate families.
Only training-fold counts rank experts. Each family has total weight one and
its duplicate questions share that weight.

For state balancing, average within-family class selected-slot shares across
supported families, then average equally across all seven classes. For the
frequency control, pool the same family-weighted labelled counts by observed
token mass. Neither uses correctness. Per-layer/per-native-group quotas and
original-ID tie breaking match. Random priorities are SHA256 over seed 20261008,
model, original layer, group and original expert ID; those fixed sets are shared
across folds. Inference about random controls is conditional on these masks.
All-reasoning frequency is an offline sensitivity, not the primary quality claim.

`measure-cpu` loads the largest archive and fits one training fold. It records
load/fit walltime and peak RSS. Selection cards require that receipt and fit five
folds plus a full-training mask. `freeze-masks` requires all seven results.
No pilot generation can precede that freeze. All seven classes need positive
training support; held-out unsupported cells are missing, not zero coverage.

Transitions reuse original sentence tables on the common 522-question/509-family
population. Original consecutive indices, known labels and identical reasoning
segments are necessary for an edge. The export includes family support,
label-coverage denominators and dwell censoring. Endpoint dwell lengths are
observed lower lengths, not estimates of complete uncensored durations.

## Compact loading and qualification

The explicit `ExpertMemoryWorker` installs construction hooks before model load
in every rank. The registered private `expert_memory_v1` loader retains original
router width and native selection, masks removed logits and selection correction
biases, then remaps original IDs to compact executor IDs. Expert tensors and all
numbered packed metadata are filtered while streaming existing safetensors.
GPT fused weights/scales/biases are sliced along the verified expert dimension;
its config is copied only for native loader reshaping and restored immediately.
Gemma's custom routing and learned scales stay native. Shared/dense paths stay
native. Declared draft layers are skipped consistently with the native model.

Qualification uses the final frozen masks for all four arms and aliases equal
policies. For each compact policy it loads one complete original resident expert
layer at a time through the original model loader and compares same-state
prefill/decode output, route weights and remapped IDs for every routed layer.
The tolerances are routing 1e-6/1e-5 and BF16 output 1e-2/1e-2 (absolute/relative).
It restores module/static-context state after each reference and closes each
engine. It also exercises a 40,958-token prefill followed by two decode tokens,
without sampled cross-run replay or a repeat-only full-cap draw.

Monolithic kernels that bypass the router, missing packed-scale coverage,
unclassified fused metadata, unsupported expert parallelism or backend changes
fail closed. These are implementation boundaries, not observed model failures.
The code is unqualified until actual two-rank GPU receipts pass. A source review
or synthetic CPU test cannot establish GPU fidelity.

Memory records unique expert backing storage, unique whole-model tensor storage,
allocator allocation/reservation, process NVML residence and load/decode peaks.
All arms use the same frozen 8 GiB per-rank cache reservation and must resolve
exactly equal KV/hybrid-state cache specifications; automatic extra cache use
cannot hide or substitute for expert savings. Qualification reports actual
storage reduction; the nominal 25% expert-count reduction is not a measured
whole-model or allocator reduction.

## Population, pilot and final analysis

`freeze-population` requires a complete source-bound later-exposure ledger,
preserves historical duplicate graph edges and excludes the 128 verification
families. The pre-later-exposure count must equal 4,351; discrepancies require
reconciliation. The ledger must list reviewed source files, question IDs,
family IDs and its review cutoff. An incomplete template is not accepted.
One deterministic representative per family is ordered proportionally by
subject/level; the first 32 families are pilot-only. Gold solutions live in the
separate full-MATH question table used by offline grading, not generation inputs.

All seven masks precede pilot assignments. Seeds depend only on family and are
shared across arms/models. Pricing retains one original pilot assignment per
role, then uses its cold load, prefill, decode and closeout times with a 25%
reservation margin. Equal policies alias completed receipts. Existing native
benchmark runs do not match this runtime/output cap by default.

The frozen output cap is 32,768 and context 40,960. Inputs cannot be truncated
or reduce that cap. A production assignment that does not fit records a failure.
The grader is the existing deterministic benchmark scorer. Strict accepts,
capped/missing answers, uncapped rejects and execution failures are retained;
semantic accuracy uses explicit unresolved bounds without a replacement judge.
An accepted extracted answer can remain correct even if generation was capped;
otherwise capped/missing-final answers are known protocol failures.

`size-final` releases discordance counts and exact one-sided simultaneous upper
bounds, without signs or arm means. Unknown semantic pairs count as discordant
conservatively. The prescribed rule fixes N once at a multiple of 32 from
640–3,712. It cannot silently shrink to fit quota/deadline. Final dispatch needs
a complete allocation feasibility/approval receipt and measured prices, not
only a sample-size number. Analysis resamples the same families across all
models, reports 21 simultaneous contrasts and unknown-outcome bounds, and
marks degenerate resampling uncertainty unavailable. There is no retention margin.

`plan-final` binds all 28 role-price receipts, frozen masks/roster and a recent
source-bound account/storage/deadline observation. It prices all remaining
canonical draws into finite shards of at most eight hours, excluding policy
aliases and completed receipts. It reports infeasibility without changing N.
`approve-final-plan` accepts only an exact complete-allocation user approval
with scheduler and deadline confirmation. Individual final cards must then
match a shard of that approved plan and receive their own exact authorization.
Neither command submits a job. Alias prices use `price --source-result` and
retain the original measured draw/closeout. Original generation commits the
corresponding identical-policy alias receipts without further inference.

## Durable finite execution

Cards name an exhibit, missing cell, sources, independent units, controls,
unfavorable-result interpretation, finite resources and price. User approval
is a separate sealed receipt binding the exact card and resources. `submit`
records an immutable intent before `sbatch`; ambiguous outcomes cannot cause a
duplicate submission. Accepted IDs are read back with `scontrol`. An unresolved
intent requires live accounting/comment reconciliation. The original scientific
harness does not dispatch successors. A separately versioned finite Slurm handoff,
`scripts/experimental_resume/expert_memory_study_v1_handoff_r1.py`, advances only
this study through qualifications, retained prices, measured pilot shards and
offline grading/sizing. Its controller is held until two bounded all-ancestor
failure guards are accepted, and submission children alone have inherited
scheduler variables removed. Invalid guard dependencies terminate automatically.
It uses no login daemon and never changes verification controller ownership.

`reconcile-submission` joins the exact private Slurm comment to queue and
accounting observations. A missing match remains unknown and never authorizes
resubmission. A terminal scheduler record permits reconciliation of the handoff,
without inferring a scientific result. Overlapping unresolved assignment/output
intents prevent another submission, including when the card hash differs.

Assignment starts and results are immutable. A started but uncommitted item
requires terminal-state reconciliation; no automatic infrastructure retry is
enabled. The journal supports one explicit authorized retry only when a future
card permits it and no draw was committed. Failures are included in analysis.
Closeout timing has its own receipt, bound to the original generation receipt.
Scheduling completion is not scientific success.

CPU measurement: serial/normal, 4 CPUs, 16 GiB, 20 minutes. Proposed qualification:
production/normal, 2 GPUs, 16 CPUs, 120 GiB, up to two hours per model; all-seven
ceiling 224 billing core-hours/28 GPU-hours. These are reservations, not prices.
Seven qualification cards are now authorized and accepted in the engineering
order, connected by `afterany` so every model can supply an engineering cell.
Acceptance does not establish qualification. The finite continuation releases
pricing only if all seven frozen loaders pass. A failed capability retains an
explicit limitation and leaves accuracy unmeasured. Its final allocation screen
preserves N and distinguishes hard entitlement/storage failure from a conservative
single-job deadline screen; a complete scheduler assessment remains necessary
before final submission. The user's workload authorization remains valid.
Generation stops by October 13
23:59:59 UTC; October 14–15 are for closeout and manuscript work.

`export` uses a fresh destination, binds observed inputs and retains explicit
UNMEASURED cells. Transition PDFs and coverage/accuracy CSVs appear only when
their source receipts exist. Canonical manuscript integration is a separate
source-bound editorial snapshot; compilation establishes no scientific result.
Observed exports include class/layer held-out coverage, fold-mask Jaccard and
minimum-class fold plots, engineering memory, and joint all-assignment scores
with actual generation memory and its observed denominator. Minimum class
coverage averages layers within class; the stricter minimum class/layer cell
is a separate statistic. Summed per-rank peaks are labelled separately from
end-of-generation allocator and NVML residence.
