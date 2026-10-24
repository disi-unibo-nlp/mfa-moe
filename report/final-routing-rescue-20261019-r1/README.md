# Steering experiment readiness

Observed 2026-10-20 on `login05.leonardo.local` as `lmolfett`. The native queue
is empty. **The rescue is incomplete and is not ready for a paper claim or a
blind resubmission.** This audit submitted no jobs and changed no manuscript.

## What actually ran

| Stage | Observed result | Evidence |
|---|---|---|
| Earlier reduced eight-model study | 29,377 executable continuations represented in the saved analysis; 7,091 model/family/class pairs | `../final-routing-20261015-r1/driver/analysis-repaired/EFFECTS.json` and per-model `ASSIGNMENTS.json` |
| Earlier strong activation, GPT/Nemotron/Qwen3-30B/Gemma | Saved analysis has 7,710 continuations including native comparisons and 3,598 paired cells; activation only | `../final-routing-20261015-r1/driver/analysis-strong/EFFECTS.json` |
| Rescue GPT, job 61497098 | Slurm COMPLETED 0:0; 3,712/3,712 stored results complete | `OBSERVED_STATUS.json`, `GENERATION_PROVENANCE_AUDIT.json` |
| Rescue Qwen3.6, job 61497099 | Slurm COMPLETED 0:0; 1,705/3,712 complete, 2,007 untouched | Same receipts |
| Rescue outcome extraction, 61497678/79 | Both COMPLETED 0:0; successful extraction does not qualify generation provenance | Same accounting; original logs under the main report root |
| Reader planning, 61497693/94 | Both FAILED 1:0: no first-complete sentences found because the planner read the wrong store | `../final-routing-20261015-r1/rescue-reader-plan-*-*.err` |
| Rescue reader GPU and final rescue analysis | No reader GPU submission; no complete reader-qualified rescue result | Queue/accounting and missing required task/annotation evidence |

The earlier eight-model executable counts are GPT 3,741; Nemotron 3,712;
Qwen3.5 3,741; Qwen3.6 3,741; Gemma 3,741; GLM 3,741; Qwen3-30B 3,712;
LFM2.5 3,248. The 551-row difference from a fully available 129-family panel
is explained by unavailable prefixes, not demonstrated extraction loss.
The existing pooled lexical target-versus-random occupancy estimates are
-0.0010597743500415124 for activation and +0.0009469033241026483 for signed
deactivation. Their simultaneous intervals contain zero. These are findings
under the short-pulse protocol and its lexical measurement, not evidence that
expert steering is impossible.
The earlier strong-activation summary likewise has intervals crossing zero:
target-minus-random occupancy 0.0003733546053595565 and first-sentence effect
0.002711640211640212. It contains no strong-deactivation experiment. These
historical summaries were read without recomputing or overwriting them.

The lexical reader already has some validation on earlier steered outputs:
`driver/lexical-validation/L3/task1_variants_ctx_pool.csv` reports pooled-model
accuracy 0.7415664760126065 and macro-F1 0.6994867891568074 on 17,134 contexts
from 277 families. `task2_differential_accuracy_ctx_pool.csv` reports
target-minus-random accuracy -0.0018619308311826899 with interval
[-0.011757524609189246, 0.009055398997514296]. These files are under the earlier
main report root. They argue against assuming gross differential reader error
without testing it; they do not validate the new rescue distribution or every
trajectory metric.

## Provenance failure in the rescue

The full audit checked all 7,424 manifest assignments and every present
OUTPUT/RESULT self-hash and assignment join. All those hashes pass. Hash
validity does not establish that a continuation used one experimental version.

The driver retained the same assignment IDs and store across source changes.
It skipped already completed results and resumed earlier partial outputs.
Consequently, successful r8 scheduler jobs reused earlier-version science:

- 29 completed GPT outputs have earlier, uncapped allowances and exceed 512 tokens.
- Earlier failed-operator jobs contributed positive token sequences to 16 GPT
  and eight Qwen3.6 continuations. This flags unqualified source fidelity; it
  does not prove that every earlier native token was numerically incorrect.
- Eight Qwen continuations resumed tokens from r4 with a changed segment seed.
- 5,380 assignments meet the audit's r8 positive-source and 512-token allowance
  condition. This is a necessary provenance condition, not full qualification.
- The planned all-routed-layer telemetry for 64 pre-specified engineering cells
  is absent. Existing action-layer captures cannot establish downstream compensation.

Preserve every earlier output and failed allocation. Do not pool uncapped
accuracy/token-cost outcomes with capped outputs, silently drop invalid units,
or treat untouched units as negative effects. Any salvage analysis needs the
original assignment denominator, explicit unknowns and source-specific joins.

## Corrections prepared in this audit

The exact 15 files in the r8 source receipt were archived and read back before
editing: `SOURCE-r8.tar.gz`, SHA-256
`6f9e0a34b8bddaded7d14ef9302e1b89bb7fd04b7ebfa5fb7adbea90f16e4307`.
This preserves that receipt's contents; its original dependency list was not a
complete execution seal.

1. `study/rescue_reader.py` reads `<bulk>/driver/<model>/rescue/units`, uses
   `outcomes.record` for the existing prefix-straddling rules, and retains
   every assignment in an availability inventory. Exact raw context inputs
   produce the frozen reader IDs. It validates result/output/assignment joins,
   source jobs, allowance, seed, full roster and task links before loading a
   reader. Recorded profiles must match assignments and study profiles. The
   frozen parent campaign, actual annotator program bytes, source tables,
   extraction/reader code and inherited reader runtime are checked. Zero-token
   earlier admissions are distinguished from earlier tokens.
   A mixed or incomplete cohort fails this gate. New tasks use the separate
   `RESCUE_READER_TASKS-r1.json`; no original task or output is overwritten.
2. `study/effects.py` swaps interval endpoints when negating deactivation
   contrasts. Previously some missingness intervals could have lower > upper.
   The saved main pooled intervals had equal identification endpoints, so this
   bug does not explain their small central estimates.
3. The unused `final_routing.operator.score_override` helper used raw-logit
   extrema in a log-softmax score matrix. Its scale is corrected, and tests now
   also exercise the actual executed selector against an independent reference.
   The generation selector's semantics were not changed by this audit.
4. `rescue_source.py` binds code, installed runtime, full manifests, source
   tables and the selected study binding. A new release must explicitly select
   its source receipt. A separate HOME/WORK binding may change only output/control
   locations; the original binding stays unchanged. The driver rejects old
   unbound stores and changed execution identities before recovery or model
   loading. No new live binding, identity or generation store was created.
5. Rescue completion now inventories every assigned row. Both the post-engine
   and already-terminal paths publish state counts; an unknown, partial or
   untouched assignment keeps the run incomplete. The rescue CLI exits 2 after
   preserving its incomplete receipt, preventing an `afterok` reader from
   proceeding. The generation wrapper forwards its source-receipt argument;
   generation/reader staging uses the permitted WORK tools directory.
6. Analysis restores planned missing assignments, keeps token costs unknown,
   uses shared family bootstrap draws within benchmarks and exports accuracy
   bounds. Wide identification bounds explicitly mark point aliases as
   compatibility-only. Gapped/incomplete dynamics, empty post-pulse windows,
   missing unbounded metrics and unqualified calibration are not reported as
   observed zero effects. New analyses require a fresh output path; existing
   effect artifacts were not recomputed or overwritten.

These corrections are development changes, not a released source binding.
The old r8 verification appropriately fails on changed source files. Do not
regenerate that old receipt or overwrite historical results to clear a gate.

Three independent GPT-6.1-sol reviewers at xhigh effort examined implementation
and provenance, measurement and paper methods, and conversation continuity.
They exchanged objections and cross-reviewed the fixes. Their review caught
an introduced file-hash versus receipt-hash mismatch before release; the
correction now also passes on the actual saved parent campaign, program and
both source tables. The more expensive saved-token replay did not finish the
bounded login-node check and was stopped. No full reader plan, GPU qualification
or corrected scientific analysis has been completed by this audit.

`AUDIT_VALIDATION-r1.json` records **52 passing focused CPU tests**: 11 reader,
17 analysis, 12 execution/completion and 12 operator/science tests. The edited
Slurm wrappers pass `bash -n`, and `git diff --check` passes. Actual saved
campaign/program/source metadata checks pass. The repository-wide conduct
check still reports 29 inherited findings in manuscript tooling and long or
undocumented functions; the active ledger also lacks committed walkthroughs.
Those broader checks are not reported as clean. The final validation receipt
hash is `9f137bf5ad6eccc1e703559ddd2c8bb8bd4d6bb74ddbb6020d55a7fdaf76a3d2`.

## Recent discussion reconciliation

The last substantive Claude conversation is `d436564b-c6b3-4db0-ad89-02887d57f862`.
Its October 19 claim that no GPU work remained preceded the rescue decision.
Its unfinished manuscript work included the final LFM readout/export, accepted
text-judge fixes, 49 page-review items and the final paper verification.

The substantive latest DeepSeek-local-Codex conversation is
`01a15561-134c-7280-99fe-c302e4bc8479` (provider DeepSeek). At October 19 20:53 UTC
the user selected the bounded rescue and at 20:55 said “Implement the plan.”
Its tool-only continuation ended after accepting the failed reader-planning
jobs. It did not reach final rescue interpretation or a source commit.
A newer short DeepSeek thread contained no superseding scientific decision.

The latest historical OpenAI Codex experiment root is
`01a140b3-8015-7473-a5cb-a0d5afea1ddb`, verified as provider OpenAI and source
VS Code. Its last substantive conclusion was October 15 at 22:09:45 UTC
(October 16 at 00:09 Rome): the accounting mount failure occurred before model
loading; closeout completed; local correction `3c8fbfd` passed 53 tests; the
rejected accounting-only SSH action still required an explicit exception.
No later exception was found. The subsequent reduced study and rescue supersede
that conversation's experiment-status snapshot, without authorizing the
rejected SSH action.

The approved rescue includes all first-complete sentences, a stratified
512-context trajectory validation subset, and 64 pre-specified engineering
cells with all routed layers captured. The 512 contexts are annotation
contexts, distinct from the 512-token intervention window. These obligations
remain separate. The first-complete sentence after the prefix is during the
rescue pulse. With pulse length equal to the continuation cap, this rescue has
no observed post-pulse window; a true post-pulse endpoint needs a separately
specified longer continuation, not relabeling the existing endpoint.
The approved text did not specify subset strata, quotas, selection seeds or
engineering-cell membership. Choices frozen now must retain their actual
chronology; they cannot be described as choices frozen before the r8 outputs.

## What the papers support

[SteerMoE / (De)Activation](https://arxiv.org/html/2509.09660v2) changes router
scores and studies safety/faithfulness. Its useful procedures are contrastive
expert discovery, overrides on a common log-softmax scale, sustained exposure,
expert-count selection using task performance and fluency, and random/bottom-ranked
expert controls. Successful control of those behaviors is evidence for routing
as an actuator; it does not fix the effect size or direction for Schoenfeld
reasoning episodes.

The original selected expert sets are associated with their classes: all
35 inspected GPT and all 35 Qwen3.6 selected fold/class cells had positive
held-out count excess, with median stability 1. That establishes an association,
not the causal direction of an intervention. Main-study exposure was only
13 tokens for GPT and 14 for Qwen3.6 versus roughly 320–352-token measurement
horizons. The hard-selection implementation can include a selected expert
while retaining a small mixture weight. Weight displacement and downstream
response are therefore necessary checks before interpreting a near-zero effect.

The local `../iclr2027_conference_anonym_private.pdf`, *Rational Manifolds*, uses
hidden-state reconstruction differences. Pages 4–6 describe conditional
sentence-wise interventions, interventions restricted to thinking, annotations
of both native and intervened outputs, validation-site sweeps and an independent
test split. Figure 4/page 7 shows that steering away from episode-associated
regions sometimes increases that episode. Appendix B/page 17 and the final
result tables provide the site-screening/composition procedure.

We can reuse the separation of discovery, validation and test; measured
intervention exposure; native and generic-perturbation controls; semantic
annotation on intervened text; the full response matrix; and accuracy/token
tradeoffs. A validation search for sites that improve accuracy would be a new
optimization study. It must not retrospectively replace this study's primary
endpoint or select only favorable held-out results.

The next measurements should distinguish these explanations:

| Observation to establish | Interpretation supported |
|---|---|
| Expert selection changes but mixture weight barely moves | Weak delivered intervention; selection alone is insufficient evidence of dose |
| Semantic reader changes while lexical trajectory labels do not | Reader attenuation or distribution shift |
| Both readers agree on changes in other classes | Cross-class response or a mismatch between the nominated axis and intended behavior |
| Action-layer changes vanish at later layers | Possible compensation; requires the missing all-layer captures |
| Target and random overrides similarly damage fluency/accuracy | Generic perturbation, without demonstrated class-specific control |

Failure to establish any of these leaves limited causal control as a possible
result. The earlier successful papers do not logically imply that our null
must come from measurement error.

## Remaining work before another submission

1. **Prepare and qualify an explicit corrected cohort and release.** The
   execution-identity guard is implemented; select and bind the actual fresh
   storage location and reviewed source. Do not automatically regenerate the
   entire cohort or adopt old units into the new identity.
   Adjudicate the 37 incompatible completions and retain all unknown assignments;
   2,007 Qwen assignments still need an authorized completion disposition.
2. **Complete the engineering evidence.** Qualify actual inclusion/exclusion,
   mixture-weight displacement, inactive-path fidelity and native-repeat
   variability. Preserve the 64-cell all-layer requirement; those missing
   captures require new allocated execution if that requirement is retained.
3. **Finish the reader pipeline on the adjudicated cohort.** Run the corrected
   CPU task planner, measure unique contexts and reusable annotations, freeze
   the label-blind trajectory subset, and price all reader work. The current
   two-hour reader cannot be assumed to finish all contexts. First-overlap and
   first-complete endpoints require distinct joins where prefixes straddle.
4. **Integrate validated reader evidence and re-run analysis under a new output
   path.** Planned denominators, sparse-bootstrap handling, accuracy bounds and
   gap rejection are corrected, but no new scientific results were computed.
   Reader-calibration uncertainty, both directions of the full response matrix
   with simultaneous intervals, censored dynamics, reader/lexical comparison
   and the approved heterogeneity/pulse/full-window checks remain unfinished.
   The old Rogan–Gladen `bound_ci` used bound point estimates rather than
   sampling intervals; the correction marks calibration unqualified instead
   of preserving that confidence claim. First-overlap and first-complete
   readouts still need explicit endpoint joins.
5. **Review, commit and seal the exact candidate, then admit its resource
   envelope.** Complete real worker-path validation before release. Final paper
   integration and manuscript checks follow the resulting scientific evidence.
   Existing source receipts, failures, manuscript edits and historical effects
   remain preserved.

These are prerequisites and pending obligations, not a declaration that a
final executable launch graph has been completed.

## Charged resources and authorization

`OBSERVED_STATUS.json` records all rescue attempts, including failures and
cancellations: **16.122222222222224 GPU-hours** used. Against the approved
20-GPU-hour ceiling, **3.8777777777777764 GPU-hours** remain. The originally
planned 2-GPU, 120-minute reader alone would bring usage to
**20.122222222222224 GPU-hours**, before completing missing generations or
engineering evidence. CPU-only analysis/planning usage in this audit's accounting
scope is **1.6544444444444444 CPU-hours**. The original phrase “8 CPU-hours”
needs consistent treatment as a CPU-only allowance; GPU jobs also consume CPUs.
No allowance is reset by calling the next version a repair.

Native `saldo -b` reports 68,000 project local hours and 32,254 consumed, with
end date 2027-02-04. Project credit is separate from permission to exceed the
bounded rescue. The live association remains `iscrc_miosr` with the previously
listed Booster QoS grants. No pending October 16 SSH-accounting exception was
used or retried.

No further Slurm submission was made. A revised, measured completion budget
and explicit authorization are required before expanding this bounded rescue.
