# Seven model scope and evidence coverage

**Observed v2 coverage, October9 ([ADR18](DECISIONS.md#adr-20261009-18-campaign-audit-and-final-decision-requirements)):**
all seven full-A ownership caches and exact-attempt accuracy summaries exist.
GPT, Nemotron, Gemma, GLM, Qwen3.5 and Qwen3.6 passed a retained single-sequence
instrumentation pilot. Only Qwen3.5/Qwen3.6 passed its REAP reconstruction check;
the other four have a method fidelity limitation. Qwen3-30B's pilot was skipped
after the common analysis error, not a measured model failure. All seven repaired
dynamics summaries failed on serialization. Full measurement, calibration,
compact qualification and assigned effects remain outstanding.
[Detailed reuse, defects, support and decision requirements](NEXT_EXPERIMENTS.md#campaign-v2-audit-and-final-decision-requirements--2026-10-09).
Historical continuing/waiting statements below are superseded by this observation
for v2 and by the October8 stop for older campaigns.

**2026-10-09 revised campaign authorization:** [ADR15](DECISIONS.md#adr-20261009-15-authorized-unattended-research-campaign-v1)
and the common research-campaign-v1 launcher supersede the stop only for the newly
authorized finite study. All seven models remain in scope; capability, missing
measurements and admission are checked independently. Existing observations and
old recognition/qualification limits retain their original scientific scope.
The old controllers remain stopped. Historical statuses below must not dispatch them.

**USER STOP — 2026-10-08 (ADR20261008 12):** all 15 owned Slurm jobs were cancelled at the user's explicit request; the live queue is empty. Both campaigns' scientific work and automatic continuation are stopped. This supersedes prior dispatch/continuation authorization, including ADR06 and ADR11. Do not submit, retry, requeue, release or restart controllers until the user explicitly authorizes resumed/revised work after reviewing claims and reuse. Preserve completed outputs, sealed sources and immutable receipts. Earlier statuses below are historical. [Verified stop receipt](../../report/experiment-stop-20261008-r1/STOP_RECEIPT.json).

**User requirement, 2026-10-07:** work on all previously analyzed and benchmarked models, including causal verification. The current roster has seven checkpoints. Model readiness controls execution order; it does not remove a model from the research scope.

**October 8 scope amendment (ADR09):** all seven remain required for analysis and eventual usefulness evaluation; the new primary deployment endpoint is resident expert memory with joint accuracy. The following historical verification status retains its original protocol. Equivalent next-class caches for five missing models are not prerequisites for static expert selection.

## Current analysis and memory coverage

Direct collection/result/code locators are in the [experiment inventory](EXPERIMENT_INVENTORY.md), especially [D04/D05](EXPERIMENT_INVENTORY.md#d04-ab-cohorts-and-seven-class-sentence-annotations) and [X11](EXPERIMENT_INVENTORY.md#x11-family-held-out-expert-selection-and-stability).

| Evidence or capability | Seven-model availability | Remaining boundary |
|---|---|---|
| Math benchmarks | All42 model/benchmark cells in each of two 32,529-attempt corpora; 1,547 shared questions/model | Native profiles differ; 33 unknown card-profile scores, saved per-cell uncertainty and strict/adjudicated denominators preserved |
| Present-class routing readout | All seven, 522 questions/509 families each | Predicts automated visible episodes, not latent states or deployment utility |
| Descriptive expert profiles and per-attempt counts | All seven A cohorts;1,547 common questions each | Full-cohort associations are not training-only keep-set evaluation |
| Clean next-class anticipation | GPT and Qwen3.6 | Five equivalent clean results absent; do not silently expand two-model claims |
| New labelled-slot concentration curves | All seven; same-cohort frequency ranking and state coverage | No held-out stability, accuracy, executed-weight or memory-saving claim |
| Family-held-out expert selection | All seven; five common folds on1547questions/1486families, plus full frozen masks | Saved slot coverage and mask stability; no physical-removal accuracy inference |
| Consecutive observed sentence transitions | All seven; common522questions/509families, original indices and segments | Sparse label coverage7.0–28.1%; missing labels break adjacency, dwell endpoints are censored |
| Actual serialized expert bytes | All seven; tensor-header inventory and model-specific quantization metadata | Static payload, not GPU resident bytes; GLM index discrepancy retained |
| Physical resident removal and joint accuracy | No completed qualified all-seven result found | New capability-aware loader, frozen selection and held-out original-prompt evaluation required |

[Inventory and detailed audit](../../report/model-analysis-review-20261008/AUDIT_NOTES.md), [new findings](FINDINGS.md#e12-analysis-first-seven-model-review--2026-10-08), [current priorities](NEXT_EXPERIMENTS.md#current-analysis-and-memory-priorities). B contains 51 questions in the union but only one in the all-seven intersection; it remains a selected case-control cohort, separate from A and full benchmark runs.

**Expert-memory v1 implementation (ADR10):** a separate private harness now
covers all seven models with the same four-arm/family-fold design. Its quota is
75% of routed experts per layer: GPT24, Nemotron96, Qwen3.5/3.6 each192, Gemma96,
GLM48 and Qwen3-30B96. Synthetic CPU contracts exercise mapping, packed metadata,
selection isolation and inference boundaries. These checks are not GPU qualification.
**ADR11 supersedes the awaiting-approval/fit status:** all-seven train-fold fits
and transitions are observed, and all final masks are frozen before outcomes.
State balancing has higher minimum held-out class coverage in all five folds
for GPT/Nemotron/Gemma/GLM/Qwen3-30B; labelled frequency is higher for Qwen3.5/3.6.
Coverage is not accuracy. The reconciled fresh population leaves4351eligible
families, with32pilot-only and4319in the final order. Seven bounded GPU
qualification jobs are accepted under explicit user authorization; physical
loader qualification, realized memory and prospective accuracy remain
**UNMEASURED**. [Observed export](../../report/expert-memory-study-v1/exhibits-r2/EXPORT.json),
[qualification submissions](../../report/expert-memory-study-v1/QUALIFICATION_SUBMISSIONS-r1.json).
The preserved v3 status below is a separate scientific protocol and its accepted
ownership remains unchanged.

## Reconciled verification coverage at the stop

All seven v3 backends retain14/14 PASS. [D08](EXPERIMENT_INVENTORY.md#d08-v3-discovery-and-candidate-specific-judgments) distinguishes canonical-file presence from independently validated scientific closeout: six models have48 discovery receipts and128-record frames; Qwen3.5 has42 retained receipts and no full frame. GLM recognition is unmeasured; its retained reader was cancelled before runtime. Qwen3.6's saved256-draw recognition FAIL (28.57% eligible/28 apparent families;8.33% already-checked/12) supersedes its earlier audit wait. Five models now have recorded recognition limits, with causal/utility effects unmeasured. The separate shared J1 FAIL remains unchanged. Every experiment is stopped; incomplete work is not a scientific FAIL.

## Preserved minimal-verification status

**Earlier progress snapshots below are historical.** They preserve measurements/reservations at their original cutoff; the reconciliation above supersedes continuing/waiting language. Older independently validated closeouts remain valid for their scope.

**All seven v3 backends passed all 14 qualification checks. All seven retained discovery pilots completed with validated captures and `STEP_COMPLETE` signals.** Actual measured role prices now size the remaining 47 traces for each checkpoint. Exact continuation cards, submissions and serial callback limits are recorded in [IMPLEMENTATION.md](IMPLEMENTATION.md). Current scientific evidence readiness is [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md). Backend PASS qualifies the new executor; semantic support, calibrated controls and causal/utility outcomes still need evidence. The model-specific v2 results below retain their historical scope.

The observed 2026-10-08 execution checkpoint has complete48-trace discovery rosters for **GPT-OSS, Nemotron, Gemma, Qwen3-30B and Qwen3.6**, each with a source-bound128-prefix audit and no missing discovery assignment. Qwen3.5 and GLM retain their original ongoing discovery cards. GPT's, Nemotron's, Gemma's and Qwen3-30B's complete recognizer FAIL results are recorded below; Qwen3.6 completed its retained reader measurement and is completing its priced full audit. This supersedes pilot-only/partial discovery progress for those five models, without measuring causal or utility effects. `AUTOMATION_BOOTSTRAP_CLOSEOUT-r5.json`, `QWEN330B_DISCOVERY_ADVANCEMENT-r1.json`, `GEMMA_DISCOVERY_ADVANCEMENT-r1.json`, `QWEN36_DISCOVERY_ADVANCEMENT-r1.json`, `QWEN36_READER_ADVANCEMENT-r1.json` and [IMPLEMENTATION.md](IMPLEMENTATION.md) bind the observed advancement.

The shared equivalence grader is now **FAIL** on the full100-item audit:97 agent-consensus matches, zero false equivalences on clear-wrong controls, and three errors in the Minerva stratum against a maximum of two. This grading limit applies uniformly to all seven models. Their independent semantic/control qualification continues; unresolved uncapped strict grades remain unknown accuracy in the existing bounds. It does not remove a model or establish a causal null. [Observed closeout](../../report/verification-study-v3/J1_QUALIFICATION_CLOSEOUT-r1.json)

GPT's full256-draw audit now gives **recognition FAIL**: eligibility confirmation precision22.41% across29 apparent families; already-checked18.75% across8 families. The original gate requires70% and10 families for each. All draws executed, but only84/128 prefixes have two usable judgments. This supersedes GPT's unmeasured recognizer status above, without qualifying a target or measuring a causal null. GPT stays in the roster as a controller qualification limit; subsequent full-audit outcomes are recorded below. [Full audit closeout](../../report/verification-study-v3/GPT_ELIGIBILITY_CLOSEOUT-r1.json), [Sol review](debate/sol-gpt-recognition-review.md)

Qwen3-30B's retained reader measurement completed successfully and supplied its own observed role/context price. Two committed bulk cards plus the retained prefix cover the original128-prefix audit without overlap or extra draws. Their combined196.0 billing core-hours are an upper reservation, not measured expenditure or semantic evidence. First bulk shard59674100 has now completed: all56 original prefixes/112 draws validate, with97 stopped and15 truncated outputs, allocating25.271 billing core-hours. That partial-audit wait is superseded by the completed full-audit result below; both fixed draws and original identities are preserved. [Reader advancement](../../report/verification-study-v3/QWEN330B_READER_ADVANCEMENT-r1.json), [completed shard](../../report/verification-study-v3/QWEN330B_READER_SLICE_CLOSEOUT-r1.json)

Nemotron's complete256-draw audit also has **recognition FAIL**: eligibility confirmation25% across34 apparent families; already-checked10% across10 families. Both support minima pass, but both precision gates fail. There are191 usable stopped draws,37 truncated and28 content-invalid draws;81/128 prefixes have two usable judgments. Parent and Sol reproduced the full ledger and qualification exactly. Nemotron stops before nomination/treatment, with causal and utility outcomes unmeasured. GPT's separate limit remains visible; Qwen3-30B's later result is recorded below. [Full audit closeout](../../report/verification-study-v3/NEMOTRON_ELIGIBILITY_CLOSEOUT-r1.json), [Sol review](debate/sol-nemotron-recognition-review.md)

Gemma's retained reader also completed and supplied its own role/context price. Its original index47 and bulk cards59675163/59675165 cover all128 prefixes exactly once, with both fixed draws. The combined143.467 billing core-hours are an upper reservation; the retained measurement actually allocated3.396 core-hours. One retained draw was truncated and remains unknown. First bulk shard59675163 now validates all47 original prefixes/94 draws:82 normally stopped and12 truncated outputs, allocating22.018 billing core-hours. That partial-audit wait is superseded by the completed full-audit result below; original draws and identities remain preserved. [Reader advancement](../../report/verification-study-v3/GEMMA_READER_ADVANCEMENT-r1.json), [completed shard](../../report/verification-study-v3/GEMMA_READER_SLICE_CLOSEOUT-r1.json)

Qwen3-30B's complete256-draw audit now also has **recognition FAIL**: eligibility confirmation10.10% across33 apparent families; already-checked33.33% across12. Both support minima pass; both precision gates fail. There are193 usable stopped draws,39 truncated and24 content-invalid draws, with80 usable pairs. Parent and Sol reproduce the complete ledger and original qualification exactly. Its causal/utility cells remain unmeasured before nomination/treatment; Gemma's subsequent outcome is recorded below. The three successful reader allocations consumed61.173 billing core-hours, separate from the earlier upper reservations. [Full audit closeout](../../report/verification-study-v3/QWEN330B_ELIGIBILITY_CLOSEOUT-r1.json), [Sol review](debate/sol-qwen330b-recognition-review.md)

Gemma's complete256-draw audit now also has **recognition FAIL**: eligibility confirmation25.52% across32 apparent families; already-checked41.67% across12. Both support minima pass and both precision gates fail. There are209 usable stopped draws,30 truncated and17 content-invalid draws, with91 usable pairs. Parent and Sol exactly reproduce the original ledger and qualification. Its causal/utility outcomes remain unmeasured before nomination/treatment. Qwen3.5, Qwen3.6 and GLM continue under the same gates. The three successful Gemma reader allocations consumed63.364 billing core-hours, separate from the earlier upper reservations. [Full audit closeout](../../report/verification-study-v3/GEMMA_ELIGIBILITY_CLOSEOUT-r1.json), [Sol review](debate/sol-gemma-recognition-review.md)

Qwen3.6's complete48-trace discovery and128-prefix audit are independently validated, including archive hashes and exact accepted job slices. Its four successful native allocations consumed250.911 billing core-hours. The retained largest-input reader at original index51 has completed with16,679 input tokens and both fixed draws, allocating4.391 billing core-hours. Its measured role/context price supports two running bulk cards covering the other127 prefixes exactly once; their280.533 core-hour upper reservation is not spent cost. The full256-draw semantic audit remains unmeasured. This is discovery advancement, with recognition/causal/utility outcomes unmeasured. [Discovery closeout](../../report/verification-study-v3/QWEN36_DISCOVERY_ADVANCEMENT-r1.json), [reader advancement](../../report/verification-study-v3/QWEN36_READER_ADVANCEMENT-r1.json), [independent stage review](debate/sol-qwen36-discovery-review.md)

## Existing native data and routing contracts

Discovery count disclosure is available in [the source-bound count table](../../report/verification-study-v3/DISCOVERY_AUDIT_COUNTS-r1.csv) and [its receipt](../../report/verification-study-v3/DISCOVERY_AUDIT_COUNTS-r1.json). The design uses48 shared problem families:48 native solutions/model, up to128 audit records/model and256 fixed reader outputs/model. This is336 planned solutions,896 audit records and1792 reader outputs, not that many independent problems. The audit splits36/12 families; the four completed reader audits represent46–48 families and127–128 unique prefix positions each, with only80–91 records having both draws usable. Already-checked held-out support is only2–4 families. These counts support an exploratory/qualification table, not precise causal or accuracy-retention claims. That preserved count table had five validated native rosters and four closed reader audits at its cutoff. The [catalogue](EXPERIMENT_INVENTORY.md#d08-v3-discovery-and-candidate-specific-judgments) now records six48-receipt rosters,42 Qwen3.5 receipts and five recognition FAIL artifacts, separating later file presence from full validation. Incomplete work is not counted as zero.

The latest cross-model observational summary is September 29 `cross-4`, containing all seven A and B roots. It supersedes the September 28 audit's three-model A coverage. Each A root has 1,547 attempts/questions. Label percentages below are median observed sentence coverage, not annotation accuracy. [Latest source](SOURCES.md#s17-latest-seven-model-observational-analysis)

| Key and checkpoint | Captured routed layers | Routed pool / native k | A scored / 1,547 | Median labelled sentence share | B attempts / questions |
|---|---:|---:|---:|---:|---:|
| `gpt` GPT-OSS-20B | 24 | 32 / 4 | 1,546 | 36.36% | 276 / 37 |
| `nemotron` Nemotron-3.5-Lightning-30B-A3B-NVFP4 | 23 | 128 / 6 | 1,547 | 24.29% | 96 / 14 |
| `qwen35` Qwen3.5-35B-A3B-GPTQInt4 | 40 | 256 / 8 | 1,546 | 8.61% | 49 / 7 |
| `qwen36` Qwen3.6-35B-A3B-FP8 | 40 | 256 / 8 | 1,545 | 7.93% | 47 / 6 |
| `gemma` Gemma-4-26B-A4B-NVFP4 | 30 | 128 / 8 | 1,545 | 18.12% | 107 / 15 |
| `glm` GLM-4.7-Flash | 46 | 64 / 4 | 1,545 | 16.44% | 101 / 14 |
| `qwen330b` Qwen3-30B-A3B | 48 | 128 / 8 | 1,545 | 17.81% | 219 / 31 |

Nemotron has 52 network layers but 23 captured routed layers; GLM has 47 network layers, with the first dense. Preserve shared-expert paths wherever present, including GLM, Nemotron and the Qwen3.5/3.6 implementation. Score spaces differ: some adapters expose probabilities/log-probabilities, others logits or sigmoid affinities with selection-correction biases. Bind exact local checkpoint configuration and adapter before action.

B totals 895 attempts but only **51 distinct questions in the union**. Do not sum the per-model question counts as independent problems. Sparse labels cannot be joined across missing sentences to create trajectories.

## A common routing result already supported on all seven

Every clean R3-D present-class readout has 522 questions/509 families. The table reports routing-augmented minus retrospective judge-context baseline loss; negative favors the augmentation. These are model-specific simultaneous CI7 results under the saved procedure, not a fresh joint across-model causal test. All seven corresponding lexical-null excess intervals also exclude zero. [Readout evidence](SOURCES.md#s07-clean-readout-and-anticipation)

| Model | Observed sentences | Loss difference | Simultaneous CI7 |
|---|---:|---:|---|
| GPT-OSS | 37,549 | −0.07858 | [−0.09567, −0.06179] |
| Nemotron | 35,334 | −0.13721 | [−0.15628, −0.11819] |
| Qwen3.5 | 34,740 | −0.14090 | [−0.15359, −0.12781] |
| Qwen3.6 | 35,484 | −0.13899 | [−0.15276, −0.12601] |
| Gemma | 31,902 | −0.14866 | [−0.16136, −0.13692] |
| GLM | 38,931 | −0.13679 | [−0.15302, −0.12035] |
| Qwen3-30B | 36,445 | −0.09427 | [−0.10929, −0.08100] |

This is direct routing information, unlike the three-model external hidden-state probe. Neither result alone proves a deployable recognizer or useful control.

## What each model still needs for the common causal question

| Model | Native observational analysis | Clean next-class routing result | New substantive verification study status |
|---|---|---|---|
| GPT-OSS | Available, including clean readout | Available: 509 questions / 10,209 pairs; contained sensitivity retained | v3 backend PASS; full recognizer FAIL. Stops before nomination/treatment; causal and utility outcomes unmeasured. Historical steering remains separate. |
| Nemotron | Available, including clean readout | No completed clean result found in the preserved anticipation source | v3 backend PASS; full recognizer FAIL. Stops before nomination/treatment; causal and utility outcomes unmeasured. |
| Qwen3.5 | Available, including clean readout | No completed clean result found | v3 backend PASS;42/48 discovery receipts retained at user stop, no full frame/recognition. Later outcomes unmeasured. |
| Qwen3.6 | Available, including clean readout | Available: 494 questions / 2,492 pairs; contained sensitivity retained | v3 backend PASS;48 discovery receipts/128-prefix frame and saved full recognizer FAIL. Causal/utility outcomes unmeasured. Historical A/B/C/FRESH remain separate. |
| Gemma | Available, including clean readout | No completed clean result found | v3 backend PASS; full recognizer FAIL. Stops before nomination/treatment; causal and utility outcomes unmeasured. |
| GLM | Available, including clean readout | No completed clean result found | v3 backend PASS;48 discovery receipts/128-prefix frame present; full archive/accounting closeout not independently reproduced here. Recognition unmeasured; retained reader cancelled before runtime. |
| Qwen3-30B | Available, including clean readout | No completed clean result found | v3 backend PASS; full recognizer FAIL. Stops before nomination/treatment; causal and utility outcomes unmeasured. |

The three-model grouped hidden-probe result is archived evidence. The other four models do not have an equivalent completed grouped-probe result in the inspected sources. If a seven-model hidden-probe claim is retained, those are missing cells; routing readout is already the stronger common coverage to build on. Do not imply that every analysis already exists for every model.

## Commonalities and differences to put in the paper

1. **Common readable episode information:** supported by all seven clean routing readouts, with model-specific effect magnitudes and source scope.
2. **No universal confidence/entropy prescription:** confidence/correctness directions and adjusted routing associations vary with model, position and population.
3. **Potential common behavioral association:** latest cross-4 reports an adjusted pooled Explore-fraction association r=−0.06241 [−0.08800, −0.03674], same direction 7/7. Its random-effects pooling across models does not by itself handle shared benchmark questions. Treat it as descriptive; it cannot justify suppressing Explore or define a causal policy.
4. **Prospective information is not equally documented:** present-class readout is seven-model; clean next-class prediction is currently two-model; fine token timing is a further question.
5. **Causal generalization is unfinished:** no Qwen result substitutes for the remaining models. Within the preserved v3 campaign, the shared test is verification using each model's conditional targets and native-k-preserving operator. ADR09 makes resident-memory reduction with joint accuracy the current deployment objective.

Use identical scientific definitions and equivalent controls, not identical expert IDs or logit doses. Report every model's effect, support, dose, missingness and interval, then heterogeneity on shared eligible families. Jointly resample shared families for a combined claim. Include sensitivity treating the related Qwen checkpoints as an architecture group; seven checkpoints are not seven unrelated architectural replications.

The user adopted the bounded singleton design in ADR-06 and instructed implementation. Qualification and execution are staged through explicit gates and measured prices. Preserve historical Qwen comparisons under their original contracts; all seven participate in that preserved verification question; the current memory/accuracy extension is separate. The exact active operator and finite job cards are in [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md).
