# Active analysis and preserved experiment contracts

## Active v3 continuation — 2026-10-11

Use the separately sealed [r3 continuation](../../scripts/research_campaign_v3_r3/README.md),
root `report/research-campaign-v3-20261011-r3/`.
[ADR20](DECISIONS.md#adr-20261011-20-storage-recovery-and-separately-bound-v3-continuation)
records verified storage recovery, completed-analysis reuse and measured storage
admission. All 77 regression tests passed. Accepted initial owners are import
59975081, snapshot 59975083, controller 59975084 and guards 59975086/59975088;
all are released with exact resource/dependency readbacks. The prior r2 owners
retired with zero runtime after protected transfer. Follow this campaign's newest
handoff and immutable journals; the r2 launch below is historical.

GPT's completed 81 pairs remain evidence; six repaired native analyses and
retained prices are still required before subsequent GPU qualification. There
are no new natural draws, altered nominations or additional model-selection
decisions. Q31–Q38, the original total budget and October 15 reporting persist.
Acceptance is orchestration ownership, not scientific completion.

## Campaign v3 authorized repair — 2026-10-10

The user has authorized implementation and protected launch of the separately
sealed [v3 launcher](../../scripts/research_campaign_v3/README.md), using v2 and
its v1 ancestry. [ADR19](DECISIONS.md#adr-20261010-19-separately-sealed-campaign-v3-repair-and-authorized-launch)
records the fixed scope and automatic staged admission. The audit below remains
the source of failure provenance. V1/v2 source seals and failed receipts stay intact.
Read [observed v3 status](CAMPAIGN_V3_STATUS.md), the new campaign's `READINESS.json`,
`LAUNCH.json`, `OWNERSHIP_TRANSFER.json` and newest protected handoff before
assessing ownership or results. Authorization and implementation do not establish
scientific qualification, measured savings or complete science.

Observed launch uses `report/research-campaign-v3-20261009-r2/`, with 69 passing
contracts and accepted source/snapshot/controller/failure jobs
59900682/59900687/59900703/59900706/59900709. R1 preserves a zero-runtime held-job
readback rejection; r2 repairs the exact one-node range parser under a fresh seal.
The four v2 reporting owners are now verified zero-runtime cancellations.
Use the r2 binding and newest handoff; keep the inherited audit below as history.

## Campaign v2 audit and final decision requirements — 2026-10-09

The user requested an audit of everything completed, failed and still needed
to decide the final runs. [ADR18](DECISIONS.md#adr-20261009-18-campaign-audit-and-final-decision-requirements)
records this analysis. The [reproducible audit](../../report/model-analysis-review-20261009/DECISION_AUDIT-r1.json)
binds compact receipts, live resources, source checks and small diagnostic
fixtures. It is not a rerun of every raw scientific fit, a new experiment
design, or an observed relaunch. Q31–Q38 remain fixed.

**Observed state:** v2 exported an incomplete terminal report. Seven derivation
jobs failed with the same non-finite JSON serialization exception. Seven
full-A ownership caches and seven exact-attempt native-accuracy summaries were
written; six retained GPU pilot measurements completed. Qwen3-30B's pilot did
not run. There are no running or pending scientific jobs. Snapshot 59783334
and controller/failure owners 59794746/59794749/59794753 wait for October 15;
they will report the terminal state, not repair or resume science.
`REPORT_COMPLETE=true` and `SCIENCE_COMPLETE=false` must remain distinct.

### Existing evidence to retain

| Evidence / catalogue | Completed result and scope | Consequence for final runs |
|---|---|---|
| D01–D04 / X01–X02: benchmark, labels, native accuracy | Both seven-model benchmark corpora and original scores exist. V2 exact-attempt native metadata/accuracy outputs also exist despite the later derivation failure. | Reuse original attempts, strict scores, annotation provenance and unknown bounds. Do not regenerate native baselines or restart J1. |
| X05: CLEAN routing readout | Seven models, 522 questions/509 families each; saved class-readout and lexical-null gains. Clean next-class results cover GPT/Qwen3.6 only. | Supports routing information about visible annotated classes. No new probes or five-model anticipation fits are required by the adopted design. |
| X11: held-out expert selection | All seven, 1,486 families/five folds; state balancing wins minimum held-out coverage in all five folds for five models, frequency in Qwen3.5/Qwen3.6. | Supports a model-dependent selection comparison, not accuracy after removal. Retain both directions and the original weighting of the published result. |
| X06 / X13: trajectories and coordination | Common-population transitions, all-seven top-1 path studies and clean GPT B1 already exist. V1 additionally completed one full-top-k layer-pair extraction per model. | Reuse completed results within their original populations. New full-top-k inference still needs repaired support/intervals; the seven v1 pairs lack family sufficient-statistic files. |
| X08–X10: historical intervention and qualification | Routing engagement exists; FRESH semantic gain remains imprecise. Some global steering policies harmed accuracy. Five v3 recognizers and the shared J1 audit failed their original gates. | Do not relaunch failed recognizers, relax gates, or repeat completed dense recovery. Old incomplete GLM/Qwen3.5 work and 560 historical FRESH reader pairs do not block the revised seven-class experiment. |
| X12 / v2 pilots: physical execution | Six saved-sequence instrumentation pilots pass both signs, multiple experts, prefix penalties and TP checks. No new full-context/compact-shape qualification or target-role prices were reached. | Retain successful instrumentation and native router banks. Engineering PASS does not establish causal benefit, resident savings or pruning accuracy. |

The stronger observational findings and mixed selection results are already in
the published paper snapshot recorded by ADR17. This audit adds no new effect
estimate. Historical uncertain/harmful results remain part of the evidence.

### Defects and limitations to resolve before relaunch

1. **Missing-value round trip.** `derive.py` reloads inherited
   `native_dynamics_estimands.parquet` with Pandas; missing values become NaN.
   `statistics.py::bootstrap_panel` hashes the original records using strict
   JSON, which rejects them. This occurs in all seven real artifacts and is
   reproduced by the audit fixture. Normalize missing inputs consistently
   before aggregation, hashing and publication; retain undefined bootstrap
   draws and unestimable cells. Test the actual inherited Parquet path, not
   only dictionaries containing `None`. Reuse the completed exact-attempt
   accuracy outputs where their bindings match.
2. **Shared-failure propagation and first-model gate.** `dispatch.py` creates
   the global failure marker only for measure/qualify/freeze/execute. All seven
   derivations were recorded as shared failures without that marker.
   `controller.py` then classified Qwen3-30B as missing model inputs and launched
   the other six measurements. Its source/mapping files exist. A shared analysis
   exception must stop dependent GPU work; a genuine isolated capability/data
   limitation must preserve independent cells. Add a fake-scheduler case for
   this exact inherited-state sequence. Slurm exit0 alone is not stage success.
3. **REAP is only partially qualified.** Qwen3.5/Qwen3.6 pass the retained
   reconstruction check. GPT, Nemotron, Gemma and GLM do not. Their banks have
   verified hashes and 64 unique positions at every required layer, so router
   calibration data can be considered separately from invalid output-norm
   scores. Diagnose native weighting/quantized-kernel reconstruction before
   expanding those norm measurements. Repair an established implementation
   defect or retain an explicit method/model limitation; do not relax the
   fidelity threshold or call unsupported REAP scores a qualified baseline.
4. **A later memory export would crash.** Execution rows store masks as
   `{reference, sha256, counts}`. `export.py::_resident` instead converts every
   mask key to an integer native-layer ID. The audit reproduces its
   `invalid literal for int(): 'reference'` failure. Use the common mask resolver
   or stored count vector and test a real referenced mask through final export.
5. **Decision contrasts and plots are incomplete.** The memory panel currently
   bootstraps arm levels without forming paired pruned-minus-native accuracy
   differences. Add those declared contrasts, support and unknown bounds using
   the same family draws. The overview causal heatmap pools signs/depths and
   shows target-arm probabilities; final causal exhibits also need separate
   signs/sites and target-minus-native/random effects. These changes complete
   the adopted reporting contract; they must not select promising outcomes.
6. **Multi-generation reuse is not implemented by simply copying v2.**
   `resume.py` imports executions/scores/annotations only from the immediate
   parent, omits measurement banks and ownership caches, and reads inherited
   units only one level deep. V2 itself references v1 units.
   `_inherited_actual` also sums only the immediate parent's jobs. A replacement
   needs transitive, deduplicated evidence and accounting across v1/v2, preserved
   instrumentation identities for compatible banks, and typed component reuse.
   Preserve all failed receipts and link corrected replacements to their defects.
7. **CPU execution layout dominates elapsed time.** There are 899 layer-pair
   cells at gaps 1/2/4/8; only seven first-pair extractions exist. V1 rereads raw
   captures twice for each pair, and v2 retains that layout. Reuse compact
   extraction and bounded family sufficient-statistic shards across pairs;
   retain nominated IDs, benchmark/family weighting, the original 10,000 paired
   draws and split linear/log-odds allowance. Measure the revised CPU role before
   admitting its remaining work. Check the finite controller/handoff forecast
   against the actual number of shards, including scoring and annotation.

All corrections require a separately sealed implementation/root. Do not edit
v1/v2 or their terminal receipts in place. Test importing v2 and its v1 ancestry,
the actual Parquet fixture, shared and isolated failures, supported and
unsupported method cells, referenced masks, interrupted exports and terminal
accounting before another `submit`. Reconcile the four accepted reporting
owners under the old lock and prearm replacement failure owners before any
ownership transfer. No ownership transfer occurred during this audit.

### Minimum remaining work, preserving the adopted study

| Order / claim artifact | Reuse first | Genuinely missing work and acceptance boundary |
|---|---|---|
| Repair the saved-data atlas and inference (D04/D05/X06/X13) | Seven ownership caches, exact native-accuracy outputs, original counts/routes, frozen folds/nominations; compatible v1 supplementary components | Recompute seven affected dynamics summaries; finish Qwen3.6 supplementary extraction and rebind the others where compatible; correct seven existing pair analyses and extract the 892 absent pairs. Preserve every nominated/unestimable outcome and explicit per-panel populations. No natural generation. |
| Complete saved-native measurement (D05/X12) | Six verified pilot banks and their successful hook/selector checks | Run the missing Qwen3-30B retained pilot on its existing two-GPU shape. Resolve four REAP fidelity limits, then collect only absent saved-sequence router/norm inputs. There are 1,547 assigned sequences/model: six of 10,829 have measurement receipts. No new natural calibration solutions. |
| Freeze and qualify the memory/causal operators (X11/X12) | Existing discovery scores, family map, all original rosters/configuration identities | Complete five discovery-only calibration folds/model; freeze missing weighted masks and numeric doses/pulses; test full native context and every distinct compact count vector, including unequal GPT MXFP4. Verify physical storage release, fixed KV capacity and within-execution masked equivalence. No sampled cross-run replay requirement. |
| Reach the pre-production decision point (Q33–Q38) | Completed retained measurements and compatible original annotations | Retain the first required original target-role outcome for each runtime/shape and first genuinely missing original-annotator batch. Include all remaining generation, annotations, scoring, analysis, storage and protected owners in the 25% admission margin. Export feasibility if the full residual workload does not fit; do not silently shrink samples/methods/models. |
| Reach the scientific decision point | Reuse native controls and all already committed original assignments | Execute only admitted, untouched frozen rows; score and annotate with the original contracts; export paired accuracy–resident-memory curves and class/depth/sign causal responses with uncertainty, unknowns and typed failed/unavailable cells. A failed qualification is interpretable; a missing execution is not a null effect. |

The fixed pruning roster contains 420 question rows in 400 distinct families;
its benchmark family counts are 30/30/40/100/100/100. Causal selection has 131
question rows in 129 distinct families and one assigned prefix per
benchmark/family/model; its family counts are 12/12/15/30/30/30. Nemotron and
Qwen3-30B each have one unavailable assigned prefix, with no replacement.
These counts replace rough maximum 180/600-family intuition for this campaign.
At a 50% binary rate, simple independent-family marginal 95% half-widths are
about 18–28 percentage points for 12–30 families and about 10–18 points for 30–100,
before pairing, equal-benchmark weighting, missingness and simultaneous panels.
The central configuration has broader support than the sparse sensitivity
indices. Tokens, sentences, folds, arms and seven model copies are not additional
independent families. Report realized support; neither an interval crossing zero
nor an unsupported REAP cell establishes equivalence.

### Cost, storage and elapsed-time assessment

The live balance reports 41,918 local hours before outstanding reservations.
The original 42,050-hour authorization baseline remains unchanged. The audit
records current campaign/inherited accounting, four pending reporting jobs,
and exact quota; approximately 82.04 GiB remains after the mandatory 10 GiB reserve
at this observation. This is not a complete residual admission.

Using the existing conservative measurement formula and the six retained pilot
times, their 9,276 remaining saved-sequence measurements require approximately
2,347.72 local hours, or 2,934.65 with the 25% margin, and 7.77 GiB of bank output.
This excludes Qwen3-30B and all later stages. It is a component forecast from
the existing implementation, not a submitted workload or full-study price.

The 892 unextracted coordination pairs extrapolate to 980.58 worker-hours from
the seven v1 first-pair timings. Current analysis jobs request 24 GiB, while the
live serial aggregate limit is 30,800 MiB: only one such job fits. The resulting
40.86 worker-days, before queue delays and other CPU stages, is a diagnostic of
the current execution layout, not a repaired-pipeline ETA. New weighting,
uncertainty code and extraction layout must be timed before promising a date.
October 15 remains the reporting milestone. Fixing serialization alone does not
make the full result set achievable by then.

The **pre-production decision** is whether the fixed all-seven workload is
qualified and feasible at measured prices. Its inputs are repaired saved-data
summaries, valid operators/masks and a complete residual forecast. The **final
scientific decision** needs the paired memory/accuracy and causal/control
outcomes, or explicit qualified reasons those cells cannot be measured. It
cannot be made from pilot success or retrospective coverage alone. Choose the
eventual research/package direction from those results while preserving mixed,
negative and imprecise outcomes; do not adapt frozen configurations to the first
favorable held-out observations.

## Preserved campaign authorizations and launch history

**Authorized instrumentation repair — 2026-10-09 ([ADR16](DECISIONS.md#adr-20261009-16-authorized-campaign-v2-instrumentation-repair-and-handoff)):**
the user explicitly authorized the versioned [v2 launcher](../../scripts/research_campaign_v2/run.py),
CPU repair gates, staged seven-model GPU qualification and protected transfer
from v1. The repaired root is `report/research-campaign-v2-20261009-r1/`.
Compatible completed artifacts are referenced; scientific enrollment and
Q31–Q38 remain fixed. The original budget includes inherited expenditure and
reservations. See that root's readiness, ownership, handoff and admission receipts
for observed state; authorization and submission do not establish scientific success.

**Authorized revised campaign — 2026-10-09 ([ADR15](DECISIONS.md#adr-20261009-15-authorized-unattended-research-campaign-v1)):**
the user explicitly authorized implementation and launch under the supplied final
contract. [Launcher](../../scripts/research_campaign_v1/run.py) owns prepare, dry-run,
submit, worker, status and closeout. All seven models, six benchmarks, original folds,
signed class/depth interventions and five methods × two allocations × three removal
levels remain in scope. Authority is at most 42,050 local accounting hours, with
automatic measured complete-residual admission and terminal export on infeasibility.
October 15 is a snapshot milestone. Q31–Q38 are settled in ADR15. Pending-design text
below is historical; the source-to-experiment matrix remains the reuse catalogue.
Older campaigns and their controllers remain stopped.

**Observed launch:**
[research-campaign-v1-20261009-r1](../../report/research-campaign-v1-20261009-r1/LAUNCH.json)
has verified initial jobs59756885/59756888/59756890/59756891/59756892.
Source resolution is running; the protected controller chain owns progression.
[Readiness](../../report/research-campaign-v1-20261009-r1/READINESS.json) records
30 passing tests, source preservation and live scheduler checks. No final
scientific outcome or production-admission claim follows from submission.

**USER STOP — 2026-10-08 (ADR20261008 12):** all 15 owned Slurm jobs were cancelled at the user's explicit request; the live queue is empty. Both campaigns' scientific work and automatic continuation are stopped. This supersedes prior dispatch/continuation authorization, including ADR06 and ADR11. Do not submit, retry, requeue, release or restart controllers until the user explicitly authorizes resumed/revised work after reviewing claims and reuse. Preserve completed outputs, sealed sources and immutable receipts. Earlier statuses below are historical. [Verified stop receipt](../../report/experiment-stop-20261008-r1/STOP_RECEIPT.json).

## Current analysis and memory priorities

**Joint design review in progress ([ADR14](DECISIONS.md#adr-20261008-14-joint-research-design-from-existing-evidence)):**
explanatory analysis and measured accuracy–memory curves are required.
The primary states are the seven observable reasoning classes. Generalization
uses held-out problem families within all six math benchmarks, with transfer
as separate robustness work. The latest Q7 correction requests analysis as the
removal percentage increases, without a loss threshold or required improvement;
this supersedes earlier preservation-as-success language. Compare native and
strong same-memory pruning throughout. Individual expert
profiles versus cross-layer coordination are a core analytic comparison; method
complexity is conditional on discovery evidence. All seven are reported without
requiring seven positive outcomes. Accuracy uses equal benchmark weights and
equal family weights within benchmark, alongside all per-benchmark results.
A separate new all-seven causal reasoning intervention is also required (Q9).
Removal levels are 0%, 12.5%, 25% and 50%. Steering promotes and suppresses
all seven classes on common target-independent saved prefixes, with matched
random controls and a shared native branch. Analyse every routed layer and
nominate early/middle/late sites per model/class using discovery-only statistics.
Immediate class response and subsequent dynamics are equally important;
continuation accuracy and token cost are also measured. An online policy is
gated secondary work. Exact nomination, dose, horizons, support/uncertainty and
cost remain to be settled; no allocation follows from these design choices.
**Latest corrections, Q16/Q17:** no new training, auxiliary fitted probes or
learned controllers. Nomination uses explicit statistics on discovery families.
Pulse/window/selection thresholds are configurable indexed settings, not a
single prematurely fixed choice. The common runner must accept a threshold
index and support Slurm arrays while reusing compatible outputs across analysis
settings. Different counterfactual executions remain separately costed work.
**Q18 is accepted:** empirical expert sets receive signed native score offsets,
followed by the architecture's usual selector and k. Exact nomination rules and
grids remain open.
**Q20 is accepted:** the main new coordination analysis retains full native
top-k selections; existing dominant-expert path results are reused alongside it.
**Q19 is accepted:** measure current-class and source-conditioned next-class
associations; nominate intervention layers/sets from current-class association
on discovery families. Exact empirical ranking/support rules remain open.
**Q21 is accepted:** compare empirical selection with REAP, all-reasoning-token
frequency, matched-label frequency and matched random, sharing the native
baseline. Inspect retained REAP inputs before proposing any absent measurements.
**Q22/Q23 are accepted:** equal and unequal layer allocations are both main-study
comparisons, at the same total routed-expert-byte budgets. Unequal allocation
uses each method's own expert score per stored byte, retaining native layer
constraints. The early/middle/late intervention sites do not restrict which
layers may be pruned.
**Q24/Q25 are accepted:** inclusion-rate difference nominates class-associated
sets/layers, with weight/enrichment companions; reuse the existing five family
folds for discovery/evaluation separation. No new trained predictor is involved.
**Q26–Q28 are accepted:** retain each model's own saved native decoding profile;
balance benchmarks then families in discovery as well as evaluation; use the
exact original seven-class annotator for missing labels. No repeat-only GPU
reproducibility checks. Later declared statistical runs may estimate means and
standard deviations; no run count or new allocation is authorized by that intent.
**Q29 is accepted:** cross-benchmark robustness uses benchmark-excluded
nomination/mask statistics and native routing coverage from retained data.
No transfer-specific memory-accuracy or causal generation branch is included.
**Q30 is accepted:** report complete held-out sensitivity curves. Define the
initial grid and nomination rules from discovery data; retain and identify
later settings motivated by held-out results as exploratory extensions.
Numerical intervention settings remain open pending saved-distribution review.
Earlier contracts below preserve work for reuse; they are not a newly agreed
experiment list. Each accepted claim must acquire exact existing inputs,
missing measurements, model/dataset scope, outputs, controls, metrics,
uncertainty and finite cost. Completed fits and path studies must be reviewed
before proposing extensions. All computation remains stopped.

## Current source-to-experiment matrix

**Review draft, not a submission list.** This is the single recap requested in
ADR14. The research question is how native expert allocation and cross-layer
coordination relate to the seven observed reasoning operations and their
transitions, how direct routing changes affect those trajectories at different
depths, and how accuracy changes as resident expert memory is reduced. Immediate
response and dynamics have equal importance. New work uses direct statistics
and routing interventions, without training a model, probe or controller.

Every row covers all seven checkpoints and all six existing benchmarks unless
the retained evidence has narrower coverage, explicitly stated below. `W` is
the exact private root in [the catalogue](EXPERIMENT_INVENTORY.md#locations-and-identity-conventions);
`<model>` expands to its seven model keys. **Available** means a completed output
exists; **derive** means an agreed new statistic can use retained inputs;
**inspect** means compatibility is not established; **measure** means the
requested outcome has not been observed. Several statuses can apply to one row.

| Paper question / planned exhibit | Exact reusable inputs and completed outputs | Residual work and output, subject to the open method choices |
|---|---|---|
| **Corpus, native accuracy and annotation coverage** — population table | D01–D04/X01–X02. Both complete generation corpora; `W/campaign-v3/outcomes/outcomes_v3-99bf8d59506e2893.parquet`; `report/model-analysis-review-20261008/{benchmark_cells.csv,cohort_populations.csv}`; original annotation qualification. **Available:** 84 model/benchmark/corpus score cells and all-seven label inventories. | **Reuse** scores/coverage. **Derive only if needed:** the agreed equal-benchmark/equal-family aggregate and uncertainty from retained rows. No native generation or regrading to reproduce these scores. Preserve unequal native profiles, unknown grades and annotation validity limits. |
| **Individual expert association across depth** — class × layer atlas | D04/D05/X04/X07/X11. `W/v3_analysis/results-r2/<model>/A/{class_counts.npz,attempts.parquet,provenance.json,E_expert_enrichment.csv.gz,E_class_experts.csv}`; `report/model-analysis-review-20261008/EXPERT_ASSOCIATION_METHOD-r1.json`; `report/expert-memory-study-v1/exhibits-r2/class_layer_coverage.csv`. **Available:** all-layer profiles and held-out coverage/stability for all seven. | **Reuse** original profiles with original weighting. **Derive only the missing contrast:** discovery/evaluation-separated empirical inclusion differences, enrichment and stability, balancing benchmarks then families under the agreed controls; a nomination table with native layer IDs, expert sets and support. Saved slot shares must be converted before being called per-token inclusion. No new layer classifier. |
| **Native state trajectories and boundary behavior** — transition/dynamics atlas | D04/D05/D07/X06. `report/expert-memory-study-v1/TRANSITIONS.json`; `W/dynamics-routing/results/<model>/A/{sentences.parquet,hist.npz}`; `report/routing-study-v1/NATIVE_BOUNDARY_PROFILES.json`; `report/routing-study-v1/dense-recovery-v3/TRAJECTORY_RESULT.json`. **Available:** all-seven common-population transitions/dwell and narrower Qwen3.6 dense/boundary trajectories. | **Reuse** matrices/support/dwell. **Derive** only absent lag/window/source/progress summaries from valid original adjacency: occupancy, transition counts/probabilities, observed dwell/re-entry and routing turnover. The sentence cache also retains first16/last16 marginals for source-before-destination association. Sparse labels cannot supply complete trajectories; do not bridge gaps or repeat completed label recovery. |
| **Individual expertise versus cross-layer coordination** — depth-pair/path comparison | D05/X05/X13. `W/depth-paths/results/part1/<model>.json`, `results/s3/nmi/<model>.json`, `results/part2/results.json`; `W/forum/tests/r3_context/readout-results.v3.json` and saved losses; `W/steering-v1/runs/resume-v1/r3e-b1-clean-v2/result.json`. **Available:** all-seven dominant-expert path/dependence and combined readout results; GPT/Qwen3.6 correctness comparisons; controlled class-specific coordination for GPT. | **Reuse** these outputs with top-1/exposure/weighting limits. The compact `W/depth-paths/results/s3/_tok/<model>/<row>.npz` files support their original top-1 representation. **Derive/inspect:** new full-top-k co-selection needs the compatible GPT compact cache or original token-aligned selected-ID captures, joined to retained labels. Marginal histograms cannot recover same-token co-selection. A product-of-marginals reference does not control generic pair coupling as B1 does. No new B1 or probe fitting follows from missing cells. |
| **Routing/activation behavior** — depth and trajectory descriptors | D05/X04. `W/v3_analysis/results-r2/<model>/A/features.parquet`; source `src/moe_exp/correlation_pipeline/features.py`. **Available:** all-seven aggregate hidden norm, hidden step distance and hidden/router geometry, including native layer columns; route counts/weights/margins under their own schemas. | **Reuse** available summaries. **Inspect** granularity before any new activation claim: aggregate hidden metrics, token-level hidden vectors, selected gate weights and per-expert output norms are different inputs. Output norms were not established in inspected producers/schemas; exhaustive archive availability remains uninspected. No hidden-state recapture is justified merely by naming an activation analysis. |
| **Causal class response and equally important dynamics across depth** — response matrix and trajectory panels | D02/D04/D05/D07/D08/X09/X10. Saved labelled prefixes/routes; historical Qwen3.6 assigned continuations/dense labels; `src/moe_exp/verification_study/` native adapters and `report/verification-study-v3/` qualification receipts. **Available:** historical engagement/behavioral evidence and all-seven native execution patterns. | **Inspect** compatible prefix/branch/label outputs first. **Measure only absent assignments:** all seven targets, both signs, nominated early/middle/late sites, common target-independent prefixes, native/matched-random branches. Export first-sentence class probabilities and assignment-level dynamic trajectories, closure/unknowns, actual routing displacement, continuation accuracy and tokens. Historical narrow interventions do not fill the new model/class/site/configuration grid. |
| **Accuracy as expert memory is removed** — accuracy–resident-memory curves | D05/D09/X11/X12. `report/expert-memory-study-v1/{SAVED_DATA_CLOSEOUT-r1.json,FROZEN_MASKS.json}`, `<model>/SELECTION.json`, `exhibits-r2/{selection_coverage,mask_overlap_stability,class_layer_coverage}.csv`; `report/model-analysis-review-20261008/{CHECKPOINT_HEADER_INVENTORY.json,labelled_expert_coverage.csv}`. **Available:** 25% held-out masks/coverage/stability, static bytes, and descriptive coverage at **every integer keep count from k to E**, all seven. | **Reuse** the full descriptive curve and compatible masks. **Derive only absent** masks/coverage for 12.5%/25%/50%, including changed benchmark weighting, method or layer allocation where required. **Inspect** native-control compatibility and strong-baseline inputs. **Measure:** actual resident removal and assigned answer accuracy for selection/baselines at 0%, 12.5%, 25%, 50%. Counts/bytes cannot supply post-removal answers or resident savings. No loss threshold or required positive result. |
| **Cross-benchmark robustness** — separate transfer exhibit | The same retained identities/counts/routes/labels and original per-benchmark outputs; `report/experimental-resume-v1/family-freeze.json`. | **Q29: saved-data derivation only.** Exclude one benchmark and overlapping families from discovery; measure nomination/association stability and native routing coverage on that benchmark. Keep the accepted fold identities and balance the remaining discovery benchmarks. Reuse any necessary primary-study native measurement cache. This does not establish transfer accuracy or causal effects; no transfer-specific generation is included. |

Full-A association covers 1,547 questions / 1,486 duplicate families. The common
readout/transition population covers 522 questions / 509 families with sparse
labels. They are neither interchangeable nor independent. Attempts, layers,
tokens and model copies do not increase independent family support. Existing
fitted results keep their original interpretation; reuse does not relabel them
as training-free. Five absent clean anticipation fits are not proposed work.

Q20 settles the coordination representation: use aligned native selected sets
at each token/layer for the main extension. All-seven top-1 caches remain useful
for reproducing their own path summaries, but cannot provide full top-k joints.
Inspect the available GPT full-top-k compact cache and seven-model raw capture
manifests/schemas before scheduling only the necessary offline extraction.
No new natural generation is implied by this representation choice.

**Preserve the exact family split:** the saved selection folds partition the
full original 1,486-family universe into 298/297/297/297/297 families and are
shared by all seven models. They are assigned by sorting the seed-bound family
hashes, then taking index modulo five. Rebuilding on the 509-family sentence
subset would change the assignment. Join any reduced prefix/sentence roster to
the existing full-universe map. A historical family uses its fold's mask;
`final_masks` used all historical families and are not held-out masks for them.
The old fresh `PILOT_ASSIGNMENTS` roster has no historical fold dispatch and
uses final masks, so its execution interface needs a versioned fold-aware
extension for this design.

**More precise memory reuse:** all seven `<model>/SELECTION.json` files also
contain five `all_reasoning_sensitivity` fold masks at 25% removal, with coverage
and overlap. Reuse these. They do not contain a final all-population
all-reasoning mask or complete score/rank arrays. The full integer-keep curves
save coverage, not expert identities. Existing masks are sorted by ID, not
importance; truncating a stored mask cannot recover another quota. Missing
masks are deterministic aggregations/rankings from the existing class counts
(`ALL_REASONING` slot 7; `ALL_LABELLED` slot 8), without generation or training.
Reuse also requires matching the scoring rule and weights: old masks do not
implement the now-accepted equal-benchmark discovery weighting, unequal layer
allocation, REAP or new retention levels. Derive only missing weighted scores
and masks from the retained inputs; compare actual mask identities before
deduplicating downstream work. Reusing the fold map does not automatically
make every old mask the requested new mask.

Q21 adopts [REAP](https://arxiv.org/abs/2510.13999) as the published output-aware
baseline. Its [official implementation](https://github.com/CerebrasResearch/reap)
uses router weights and expert activation norms and documents a selected-weight
normalization correction. Pin the scoring definition and source version;
architecture-native execution scaling remains a separate contract. Saved
mixture weights and aggregate hidden-state norms are not substitutes for
per-expert output norms. Inspect existing captures before proposing a forward
measurement on saved discovery sequences. Benchmark/model support, calibration
population/token budget and physical-removal fidelity must be checked; official
code availability is not qualification on all seven local checkpoints.

**Equal versus unequal allocation:** preserve each method's scoring definition
and discovery population, then compare its layerwise ranking at equal quotas
with global score-per-byte selection at the same total budget. Keep at least
the experts required for native k/group constraints at every routed layer;
shared experts and other required tensors remain accounted separately. Current
contracts have one routing group per model. Do not silently normalize a score
differently across layers in the unequal arm; its cross-layer scale is part of
the declared method. The random comparator needs a recorded random priority
rule under the same feasible budgets, not an unspecified favorable allocation.

The equal-allocation keep counts are already determined by Q10 and native
geometry; they are not new arbitrary thresholds:

| Model | Native experts / k per routed layer | Keep at 12.5% removal | Keep at 25% | Keep at 50% |
|---|---|---:|---:|---:|
| GPT-OSS | 32 / 4 | 28 | 24 | 16 |
| Nemotron | 128 / 6 | 112 | 96 | 64 |
| Qwen3.5 / Qwen3.6 | 256 / 8 | 224 | 192 | 128 |
| Gemma / Qwen3-30B | 128 / 8 | 112 | 96 | 64 |
| GLM | 64 / 4 | 56 | 48 | 32 |

These values are integral and exceed native k; they do not prove successful
compact loading. Unequal allocation gives a method-specific vector of native
layer keep counts and the same total byte budget. The existing GPT MXFP4 loader
requires one retained count across layers, so unequal allocation needs a new
versioned implementation and measurement. Padding experts to a common size or
leaving removed weights resident cannot count as actual savings. Preserve native
correction biases, scaling, shared paths and quantization metadata/ID mapping.

**Intervention-dose reuse has its own input requirement:** standard retained
A/B captures and v3 discovery files store selected IDs/weights, not all native
router scores. The v3 score bank was transient worker state. A retained legacy
Qwen full-score fixture uses another quantization/revision and an outcome-selected
engineering population; it is not a current-native calibration bank. See
[D05](EXPERIMENT_INVENTORY.md#d05-routing-captures-count-archives-and-common-population).
Arbitrary offset sweeps cannot be reconstructed from top-k weights alone. First
resolve any compatible bank; if needed, propose one retained forward measurement
on saved sequences, collecting router vectors and REAP output norms together
where feasible. This still does not supply autoregressive counterfactual outcomes.

### Measurement and reporting contract under review

| Block | Quantities to report | Controls and interpretation |
|---|---|---|
| Class/layer association and nomination | Native inclusion-rate difference, empirical enrichment, weight occupancy where available, supporting families, rank/set stability | Discovery-only rules; benchmark/progress and source-class conditioning as appropriate; held-out descriptive confirmation. No one-expert/one-operation or causal claim from an association. |
| Native transitions and routing dynamics | Valid adjacent transition counts/probabilities; class occupancy; observed dwell/re-entry with censoring; set overlap/turnover and ID-aligned weight displacement at indexed lags | Never connect annotation gaps. Preserve temporal order versus marginal composition. Use model-native weight scales; normalized comparisons need an explicit separate definition. |
| Cross-layer coordination | Empirical joint co-selection versus class-conditioned marginal references; MI/JSD-type summaries; proposed supported pair log-odds-ratio contrasts between individual classes, beside marginal inclusion differences; original saved NMI/B1/readout results | Top-1 projection versus full top-k explicit. Product marginals also remove generic coupling. The proposed odds-ratio contrast asks whether pair coupling changes with class, with common support and family-level uncertainty. No predictive sufficiency or causal mediation claim. See the exact proposal in [METRICS](METRICS.md#empirical-full-top-k-coordination-specification-for-review). |
| Assigned causal response and dynamics | Full target-by-observed-class response, both signs/sites; first complete sentence; time-resolved class occupancy/events/counts, dwell/re-entry, routing engagement and recovery; continuation accuracy/tokens | Native and same-layer matched-random controls; common prefixes; closure/unknowns explicit. Assignment-level dynamic contrasts are causal estimands; post-treatment state-conditioned transition matrices are descriptions of different arm populations. |
| Memory curves | Actual expert/whole-model resident bytes, answer accuracy and paired difference at each removal level, unknown bounds, tokens and execution cost separately | Native, empirical selection, frequency/random and agreed strong same-memory pruning; valid native k/groups/shared paths, common evaluation profiles and controlled KV/workspace accounting. No preservation claim from nonsignificance. |

Family-level uncertainty must respect benchmark weighting and paired arms/models;
repeated prefixes/seeds stay clustered. Exact comparison families, simultaneous
intervals, support/precision and missingness rules still need agreement before
a final batch. Continuous estimates/support come before arbitrary nomination
cutoffs. An unfavorable result or model difference remains in the paper.

### Reuse checks required in each executable card

1. Name the exact retained result/input and producer contract, with model,
   benchmark, family population, checkpoint and annotation coverage.
2. Compare the requested measurement with its fields, estimand, routing
   representation, decoding/operator and completion status. New names, schemas
   or source seals alone do not establish a missing scientific measurement.
3. Mark each cell **reuse output**, **derive from saved input**, **compatibility
   unresolved**, or **new outcome required**, with evidence. Preserve failures,
   partial/cancelled work and original scope.
4. Resolve compatibility before scheduling generation/capture and record any
   actual incompatibility. Reuse compatible native/counterfactual branches;
   never substitute an untreated suffix for an unobserved intervention.
5. List only residual tasks, their independent support/expected uncertainty,
   controls, interpretation and measured resource bounds.

The audit inspected manifests, compact results, schemas, code and bounded
record/header samples. It did **not** exhaustively validate all archive payloads
or cross-runtime native-baseline joins. Keep unresolved cells explicit instead
of claiming an unsupported 100% compatibility guarantee. No ambiguous cell
silently becomes a fresh-generation task. Check saved-prefix enrollment for
future-outcome selection; compare tokenizer/checkpoint, prompt/prefix tokens,
sampling and penalty history, cap/termination, population and scoring. A v3
receipt-schema rejection alone is not scientific proof that new native data
are necessary. Exact independent sampled-run replay is not required.

### Indexed configuration and execution contract

The implementation sequence follows the missing measurements, rather than
restarting old campaigns:

1. Resolve the result/input/assignment reuse manifest; reuse finished outputs
   and derive only absent balanced summaries, nominations and mask variants from
   retained counts/routes. Keep new statistical work separate from file checks.
2. If needed after compatibility inspection, collect missing native router
   vectors and REAP expert-output norms from saved discovery sequences, combining
   the required measurements where feasible. No new natural solutions are needed
   merely to produce calibration inputs.
3. Execute only absent controlled intervention or physical-pruning assignments
   under the declared finite configurations, after their sampling/cost contracts
   and workload authorization exist. Reuse compatible native branches, identical
   actions/masks and completed labels; preserve the original class annotator.
4. Export all configured curves/effects and uncertainty. Add later declared
   statistical replicates for mean/SD when selected and costed; never repeat a
   completed assignment solely to check exact GPU reproducibility.

Q17 requires one common entry point accepting `--threshold-index`, explicit
model/analysis selection and an immutable configuration table. A Slurm array
index maps deterministically to a recorded task/configuration row. This
interface is **specified, not implemented or qualified** by this review.

An **execution configuration** includes source prompt/prefix, checkpoint/runtime,
seed and explicit statistical replicate ID, pruning mask or native layer/expert set/signed offset/pulse, decoding and
cap. An **analysis configuration** includes observation windows, lags and
summaries applied to compatible outputs. A cutoff that changes the nominated
set/layer/mask changes the execution dependency. Many analysis indices can reuse
one execution; identical actions can be aliased after identity checks. The
shared native branch is not repeated for each target, sign, site or window.

The dry-run manifest must enumerate exact inputs, cache hits, partial results,
dependencies, residual CPU/GPU/annotation tasks and costs before submission.
Grids are versioned: an index never changes meaning. Config-incompatible caches
must not be silently overwritten or treated as absent. Preserve all settings
and results; resume only agreed missing assignments. No pulse, strength,
selection cutoff or dynamics horizon is fixed here. Longer windows require
retained support; changed intervention strengths/pulses generally require new
counterfactual continuations.

Q30 requires the complete held-out curves for the declared configurations,
including unfavorable, unsupported and unfinished cells with their correct
status. Initial grid construction and nomination use discovery data. A setting
added after inspecting held-out outcomes remains reportable, but is explicitly
an exploratory extension with its own immutable index and rationale. It must
not replace the original sweep or be presented as a prospectively selected
winner. This reporting choice does not authorize an unlimited sweep.

Reproducibility checks must not create repeat-only GPU assignments. A changed
sample from the same seed is not a failed qualification or an automatic retry
trigger. Later planned repetitions have distinct replicate IDs and a stated
variance-estimation purpose; preserve and average all runs rather than selecting
a favorable repeat. Report run-level mean/SD separately from uncertainty across
problem families. Seeds/repetitions do not multiply independent family support,
and variation across the five different evaluation folds is not run-to-run SD.

**Existing implementation to reuse:**

- `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/steering/moe_steer/{spec.py,ops.py}`
  already supports expert sets and signed biases; `runner.py` has manifest/shard
  execution and committed-UID reuse, and `manifests.py` indexes saved token
  prefixes. Its engine is Qwen3.6-specific, so this is not all-seven qualification.
- `src/moe_exp/verification_study/adapters.py` provides seven native selector
  implementations and within-execution checks, but its sealed action is one
  expert, nonnegative bias and a fixed pulse.
- `src/moe_exp/expert_memory_study/{journal.py,stages.py}` provides immutable
  assignment reuse and identical-mask aliasing. The sealed protocol fixes 75%
  retention/four arms. `scripts/expert_memory_study/run.py run-card` is an
  existing finite-card entry point, not the revised threshold runner.
- `scripts/paper/export_labelled_expert_coverage_r1.py` already exported every
  integer keep-count curve; changing a plotted threshold does not require
  another scientific analysis or generation.

Integrate these patterns under new versioned bindings, preserving sealed sources
and stopped controllers. Before deduplication, one nomination/dose/pulse setting
has **85 continuations per prefix/model/seed**: one native plus 7 classes × 3
sites × 2 signs × target/random. Saved prefixes do not make absent branches
free. Independent sample size, measured prices and workload authorization remain
required before submission.

The memory design has **31 method/allocation/removal conditions per evaluated
question/model/replicate before reuse and mask aliases**: one native plus five
selection methods × two layer allocations × three nonzero removal levels.
This counts conditions, not jobs or qualified configurations. Each historical
family uses only its own evaluation fold's masks; five folds do not multiply
its evaluation count. Actual task counts follow the reuse manifest and selected
statistical replication, not this upper-level condition count.

### Decisions still needed before a final executable recap

The direct full-top-k comparison/null now has a concrete statistical proposal
in [METRICS](METRICS.md#empirical-full-top-k-coordination-specification-for-review),
pending shared-design confirmation. REAP input compatibility remains an evidence
check. Bind family support, consistent scoring,
uncertainty, model-native profile identities, configurable intervention grids
and finite measured allocations to the cards. Later replication count/budget
remain separate. These are explicit gaps, not silently adopted defaults.

### Measurement bindings to preserve

The current review also identified these measurement bindings, which must be
resolved in the revised cards rather than inherited from the historical text:

- **Seven-class annotations:** all 28 original D04 batches bind the same saved
  program, Qwen3.8 reader revision and decoding configuration. Their original
  DSPy pipeline can label new identified windows without training or the failed
  candidate-verification recognizer. The later direct dense adapter has different
  formatting, temperature and output cap. Q28 retains the original D04 contract;
  do not silently mix them or repeat completed annotation recovery. Reuse checks
  include the problem and neighboring-sentence context, not sentence text alone.
- **Accuracy:** `W/v3_analysis/results-r2/<model>/A/attempts.parquet.correct`
  comes from `adjudicated_correct`. The D03 outcome sidecar separately retains
  `strict_correct`, scorer contract 3, termination and historical adjudication
  status. Comparing that adjudicated `correct` column to newly strict-scored
  pruned outputs would mix graders. Reuse the matching saved strict fields for
  a common strict comparison. Unknown-outcome bounds need termination and, for
  missing-final distinctions, original trace/final-content metadata. Historical
  J1 acceptance and the later failed shared J1 audit retain separate scopes.

The revised design stops at unresolved input/measurement contracts; the older
execution authority and defaults below are preserved history.

## Preserved pre-stop memory contracts

**Reuse prerequisite:** every future computation proposal must cite the relevant [D/X/R catalogue entries](EXPERIMENT_INVENTORY.md#r04-requirement-for-any-future-computation-proposal) and exact existing artifacts, state what necessary information they cannot supply, and distinguish missing files from uninspected compatibility and genuinely unmeasured outcomes. Existing solutions, Verify labels and routing captures must be inspected before fresh generation is proposed. All computation remains stopped pending explicit resumed/revised authorization.

**Current completion:** [X11](EXPERIMENT_INVENTORY.md#x11-family-held-out-expert-selection-and-stability) and [X06](EXPERIMENT_INVENTORY.md#x06-state-transitions-and-native-boundaries) catalogue completed all-seven selection/coverage/stability and common transitions. [D08/X10](EXPERIMENT_INVENTORY.md#d08-v3-discovery-and-candidate-specific-judgments) reconcile six48-receipt discovery rosters,42 retained Qwen3.5 receipts and five recognition failures. Historical schedules/prices/accepted cards below are preserved for review, not current dispatch authority.

**ADR20261008 09 supersedes the previous deadline ordering:** paper results from all-seven analysis, with reduced resident expert memory and jointly measured accuracy as the first deployment objective. Computational Linguistics is the target and October 15 is the result cutoff. The package is an eventual paper output; public API and general deployment work are deferred. No external validators or failure-centered fallback. Preserve unfavorable/unknown evidence, existing accepted jobs and their original contracts.

**Completed on October 8 without new inference:** [all-seven result inventory](../../report/model-analysis-review-20261008/INVENTORY.json),84 original model/benchmark score cells across two distinct corpora, clean readout/anticipation exports, [checkpoint-header byte inventory](../../report/model-analysis-review-20261008/CHECKPOINT_HEADER_INVENTORY.json), and [labelled-slot concentration curves](../../report/model-analysis-review-20261008/LABELLED_EXPERT_COVERAGE-r1.json). The new curve took 12.341 seconds from compact saved summaries. These are source/export measurements, not a price for subsequent NPZ fits or GPU pruning.

| Next work | Exact paper claim and reusable sources | Missing measurement, controls and gate |
|---|---|---|
| **1. Saved-data expert-selection results — completed** | C2: profile-informed keep-sets can be specified and compared on held-out training-family folds. Reuse all-seven per-attempt `class_counts.npz`, question/family joins, current expert/transition outputs and actual tensor-header bytes. | Completed seven-model five-fold/full-training fits, coverage/stability and common transitions are in [X11/X06](EXPERIMENT_INVENTORY.md#x11-family-held-out-expert-selection-and-stability) and the saved-data closeout. Reuse them; do not repeat the same fits. Coverage does not measure resident savings or accuracy under removal. |
| **2. Resident-loader qualification and retained pricing** (initial path by October 10–11) | C3 engineering prerequisite: expert removal actually changes allocated storage while preserving the declared router. Reuse native adapters, header accounting and same-state fixtures; preserve frozen v3 sources. | New implementation must omit allocations, map IDs, preserve groups/k/shared paths and handle packed scales. Compare all-keep same-state behavior, then valid keep-set allocation, fixed-KV memory and a retained outcome-blind role price. Exact GPU shape/time is not yet measured or authorized. Complete all seven capability cells even if some remain unsupported. |
| **3. Frozen held-out accuracy/memory evaluation** (October 11–13 only if ready) | C3 empirical table: paired original-prompt accuracy and realized expert/whole-model memory at one meaningful byte target. Reuse native outputs only where checkpoint, decoding, questions and runtime are actually compatible. | Full-model/profile/frequency/matched-random arms, and faithful output-aware baseline if claimed. Equal bytes and comparable layer budgets; assigned failures/caps/unknown grades remain included. Independent families, shared across-model resampling, justified precision, frozen selection, measured prices and explicit workload approval are required. |
| **4. Evidence and manuscript freeze** (October 14–15) | Existing C1 results plus whichever C2/C3 cells are observed and source-bound, in the CL manuscript. | Unmeasured resident/accuracy cells stay explicit. The deadline does not authorize weaker controls, selected favorable results or one-model substitution for seven-model scope. |

The two immediate cards are specified in [the source review](debate/package-analysis-review-20261008.md#the-two-concrete-next-cards). They are scientific implementation/design cards, **not submission-ready allocations**. The later user correction defers their public-package integration suggestions; retain only code needed for the named paper result. No new `sbatch` was issued for this review. Existing benchmark/readout, repaired probes, dense labels, historical deactivation and GPT B4 should not be rerun merely for fresh provenance. Five additional anticipation caches are not a prerequisite for the static-memory question.

**Budget and precision:** static checkpoint bytes are not runtime resident bytes, and prior positive-pulse/native prices do not price a new loader. The 128-family verification reservation is excluded from the new test population. At 96 independent pairs and hypothetical 25% discordance, a nominal normal 95% half-width would be about 10 percentage points, before multiplicity or missingness; two seeds do not double independent support. ADR10 adopts a 25% routed-byte removal planning target and a blinded 32-family pilot sizing rule for 21 contrasts at a five-percentage-point planning half-width. No accuracy-retention margin is assumed. Equal percentages need not mean equal absolute bytes or equal whole-model savings. Later exposure reconciliation, measured prices and complete allocation feasibility remain required.

**Baseline input gate:** selected IDs/weights do not reconstruct REAP-style expert output norms. Inventory retained inputs first. If a faithful output-aware comparison is necessary and norms are absent, a separately priced teacher-forced pass on existing training sequences may be justified; fresh generation is not the default. No such pass is currently authorized.

**Adopted implementation defaults (ADR10):** the separate
[expert-memory v1 harness](../../docs/expert-memory-study-v1.md) implements
25% routed-byte removal, four arms (native/state-balanced/pooled labelled-token
frequency/fixed random), five common historical-family folds, output cap32768,
context40960 and the frozen32-family pilot/21-contrast sample-size rule.
This supersedes ADR09's unselected-percentage status. The primary baseline
uses exactly the same labelled observations; no quality claim against conventional
all-token pruning or output-aware pruning is adopted. Static header bytes per
expert are equal within each checkpoint, so the specified layer quotas match
75% serialized expert-byte retention; actual GPU storage remains unmeasured.

The first reviewable card is
`report/expert-memory-study-v1/cards/CPU_MEASUREMENT.json`:4CPUs,16GiB,20minutes,
serial/normal, largest-archive load/one-fold RSS/time and one-model transition
timing. **Later workload authorization (ADR11)** supersedes the awaiting-approval
status: the user instructed "Go on and submit everything we need". Measurement
59715886 completed0:0 in40seconds; the eight measured saved-data cards completed
with all-seven selections and common transitions. Their allocation ceiling was
1.6core-hours and observed cost0.218core-hours, in addition to the measurement's
0.044core-hours. Final masks, the reconciled4351-family fresh population and
896pilot assignments are frozen before prospective outcomes.

Seven qualification jobs59717241/59717247/59717256/59717261/59717266/59717268/
59717269 are accepted in Qwen3-30B/GLM/Qwen3.6/Qwen3.5/GPT/Gemma/Nemotron order,
joined by `afterany`. Ceiling224core-hours/28GPU-hours is an engineering
reservation, not a production price or observed capability. A separately sealed
finite Slurm handoff advances this study without changing verification ownership.
It checks all-seven capability, retains original pilot pricing assignments, and
releases only measured finite pilot/CPU closeout cards. Retained pilots
price later work; frozen final N cannot be reduced to fit the deadline or quota.
The retrospective fits and transitions are now observed in
[exhibits-r2](../../report/expert-memory-study-v1/exhibits-r2/EXPORT.json).
GPU capability and joint accuracy/memory remain unmeasured. Final submission still
requires the complete measured entitlement/storage/scheduler/deadline assessment;
ADR11 authorization was subsequently superseded by ADR12. Every remaining allocation is stopped; resumed/revised computation requires explicit user authorization.

**Publication gate applies throughout (ADR20261008 08):** name the exact exhibit, prove what existing artifacts cannot supply, justify independent support/uncertainty, define controls and interpretation, and bound minimum necessary measured cost before submission. The analysis-first direction changes priority, not existing source seals or accepted jobs.

## Preserved minimal-verification v3 campaign

The following fixed contract governs the already accepted verification campaign. Its accuracy/output-token endpoint is distinct from the newly prioritized resident-memory endpoint. This record creates no additional dispatch or cancellation authority.

The user adopted minimal verification v3 and instructed **“Implement the plan.”** All seven checkpoints remain required: GPT-OSS, Nemotron, Qwen3.5, Qwen3.6, Gemma, GLM and Qwen3-30B. There is no arbitrary campaign ceiling. Each allocation still has finite resources, a claim/model cell, a price basis, an artifact and an observed job ID. Account entitlement and storage constrain submissions.

The implementation is [verification_study](../../src/moe_exp/verification_study/) with [CLI](../../scripts/experimental_resume/verification_study_v3.py), GPU/CPU scripts, sealed manifests and receipts. Executable contract: `report/verification-study-v3/PROTOCOL.json`. [Observed status](EXPERIMENT_STATUS.md) comes from receipts. Original v1/v2 code and artifacts remain unchanged.

## Fixed design

| Item | Adopted contract |
|---|---|
| Aim | Causal verification control or well-supported limits; accuracy/output-token utility jointly measured |
| Eligibility | Specific checkable intermediate/complete numeric/symbolic candidate against an original-problem constraint, not already checked |
| Nomination | One singleton/model from last256 already-emitted rows in checked-versus-unchecked destination profiles. Family/source-native-trace matched; no exact position/candidate matching or anticipation claim. Require ten matched families and positive associations in all three discovery family folds. No held-out/final outcomes. |
| Operator | Positive raw-logit bias followed by the native selector; preserve native k, groups, sigmoid/softmax, shared paths and learned scales |
| Pulse | One256-prediction pulse; stop at operational native closure |
| Dose/control | Same-state native discovery inclusion increment0.25±0.025. Same-layer random pool excludes target; exposure difference≤0.025, own calibrated dose and ID-aligned relative weight displacement mismatch≤20%. Empty pool fails qualification. |
| Recognition | Eligible and already-checked decisions each need≥10 apparent families and≥70% apparent family precision. Disjoint held-out uncertainty/coverage and symbolic audit; no invented held-out precision cutoff. Up to128 prefixes;36/12 discovery-family split, family-balanced strata. |
| Validation |32 independent families/two native seeds/full16k pipeline coverage; no favorable-effect selection |
| Arms/seeds | Native, target, matched random; two seeds; no new incumbent arm |
| Inference | Shared family resampling across checkpoints; seeds averaged first;42 contrasts,50000 draws; unknown binary outcomes retain worst-case bounds; degenerate uncertainty is unavailable |
| Utility | Generator+online-controller output tokens. Inputs, walltime, annotation/grading, capture and research recovery separately itemized. No accuracy-loss margin adopted. |

The current cheap recognizer emits zero controller tokens. A semantic-veto fallback remains blocked until an isolated executor and exact prices qualify. Smaller-budget, SSD-preloading, approach/commitment and context extensions remain secondary.

## Separate experiments

**Conditional assay:** Native original-prompt sources on96 final families/seeds0/1/cap16384. Freeze the first qualified seed0 prefix per family and every no-eligible/terminal-unknown path before treatment, without seed1 fallback. Fresh native/target/random continuations use seeds0/1/cap1024. Primary event: verification before native closure within1024 tokens. Closure is a competing event; retain every assigned branch, without conditioning on observed closure. Maximum4032 short continuations across seven models.

**Original-prompt policy utility:** All96 families×two seeds×three arms/cap16384, including no-fire/failures. Reuse only exactly compatible original native sources; conditional fresh native continuations are a different population. Maximum4032 long final runs. Discovery/validation separately accounted. Nonsignificant accuracy loss does not establish retention.

## Order and paper artifacts

| Stage | Gate/reuse | Output and failure interpretation |
|---|---|---|
|0 Evidence | Saved seven-model CLEAN readout, two anticipation results, completed dense ledger | Generated paper table and missing-cell matrix; no duplicated fits/recovery |
|1 Backend | Pinned native selectors and engineering fixtures |14 observed cases/model: geometry/parity, k/dose,256 boundary, closure, supplied-prefix alignment, online trigger, isolation/cleanup, candidate cases and16k feasibility. Cross-run comparisons were diagnostic only. Failure of a valid execution invariant is engineering non-identification. |
|2 Discovery/prices | Check source compatibility; retain one-assignment price pilot |48 discovery families/model, compressed capture,128-prefix audit, role prices. Production needs measured costs; narrowly bounded first measurements are outcome-blind. |
|3 Recognizer/nominee/dose | Blinded draws, apparent-family gates, destination profiles, same-state native bank | Detector, singleton, bias, matched pool; failure never relaxes rules or starts a replacement sweep |
|4 Validation | Exact backend/runtime/detector/policy and prices |32-family native pipeline coverage, without an effect-sign gate |
|5 Sources/freeze |192 exact native final sources/model | Immutable seed0 ledger and both treated rosters; pending paths cannot become ineligible |
|6 Experiments | Frozen enrollment/policy and priced finite shards | All-assignment capture, dose/timing, closure/no-fire and failures; one automatic assignment retry maximum |
|7 Closeout | Blinded closure-bounded semantic views; gold offline; preserve scorer-v3/J1 | Assigned outcome bounds, reader agreement, strict/adjudicated accuracy, tokens and recovery. Exact J1 deployment uses existing100-item frozen audit/gate. |
|8 Paper | Bound closeout and shared-family inference | Generated tables/provenance and updated wiki/manuscript; backend PASS is not a behavioral effect |

GPT/Qwen3.6 anticipation is already measured; the other five lack corresponding prepared K2 caches. All seven have native A routing histograms. A sentence-level extension must bind its available population/features and price; do not duplicate completed fits or start a tensor producer to fill a table. Seven-model CLEAN readout is already usable.

**Prospective engineering correction (ADR 20261008 07):** exact cross-execution token or route equality is not a qualification requirement for sampled GPU generation. Do not run new repeat-only fixtures or retry because independent sampled outputs differ. Verify actual within-execution capture, same-state inactive-path fidelity, architecture semantics, token alignment and lifecycle instead. All seven active backends are already qualified; preserve their sealed sources and receipts.

## Execution and continuity

**Publication gate before any newly defined experiment (ADR20261008 08):** cite applicable catalogue entries and exact source files, identify precisely what those artifacts cannot supply, then establish the exact claim/table/figure, demonstrate the missing input after checking existing artifacts for reuse, justify independent support and expected uncertainty, verify interpretable controls/measurement for positive and negative outcomes, and bound the minimum necessary measured cost. New code, source seals or fresh generations are not contributions by themselves. An unanswered gate calls for evidence inspection or design correction before quota is spent. This requirement supplements the finite job-card contract and authorizes no new jobs or alteration of a sealed campaign.

The CLI supports preparation, qualification, finite generation, audit framing/rating, recognition, nomination, calibration, validation, final freeze, semantic windows, strict/J1 grading, assigned-population assembly, analysis, prices and paper export. `submit-task` compiles a sealed stage request. The task wrapper accepts only study commands, binds code/campaign/resources and injects the allocation deadline.

First-wave `afterany` closes out all models even if qualification fails. Later gates inspect scientific PASS receipts, not only Slurm state. Completed receipts are immutable; attempt journals allow at most two invocations/assignment. Submission intents/comments prevent blind duplicate submission; unresolved intents require reconciliation. Qualification repairs need new versioned source bindings and reconciliation of all affected jobs. Preserve the source closure while jobs are pending or running. The active revision and result root are recorded in [RUNBOOK.md](RUNBOOK.md); previous partial receipts retain their original bindings.

Every job must change a named claim/model cell and produce a paper artifact with exact bindings, population, controls, endpoints, negative/imprecise interpretation, finite resources/prices, retries and closeout. Failure leaves a model explicitly unqualified and does not reduce seven-model scope.
