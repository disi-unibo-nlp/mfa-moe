# Parallel routing execution — October 5, 2026

This runbook continues the October 4 execution record. The user authorized the
required CPU/GPU work, long jobs, and additional hours. Qualification, immutable
inputs, complete stage pricing, and exact job/artifact verification remain
technical gates. Submission does not establish successful execution.

## Experimental comparisons

| Comparison | Frozen purpose | Assigned generation |
|---|---|---:|
| A | Separate versus paired correlation-selected verification experts | 128 continuations, 1,024-token cap |
| B | One versus two pulses, with matched random and native controls | 156 continuations, 1,024-token cap |
| C | Logit bias, forced inclusion, and selected-weight reweighting | 208 continuations, 256-token cap |
| Fresh | All frozen methods on the independent extension, with shared controls | 3,480 continuations, 1,024-token cap |
| Utility | Native versus the frozen selected policy from original prompts | 96 families × two arms × two seeds, 16,384-token cap; first two families provide runtime pricing |

Fresh enrollment is complete: 123 families and 134 starts, comprising 100
verification and 34 commitment starts. The 220-family parent pool and all
exclusions remain recorded. No winner from the 13-family pilot selects fresh
methods or families. The utility policy selector uses the complete fresh
comparison with its original deterministic tie rules.

The sparse actions use the previously selected experts and preserve top-k=8
and the shared expert. Repeated pulses test exposure, not the order of different
actions. Dense seven-class labels preserve gaps and report transitions, dwell,
re-entry, loops, expert turnover, and selection-frequency velocity/acceleration.
The route arrays contain expert IDs; these are not full gate-distribution
measurements. Actual intervention telemetry is reported separately.

## Active submission graph

The following graph was established and verified through 00:31 CEST. Later state
snapshots are saved as `OVERNIGHT_SLURM_<UTC>.json`.

| Work | Job IDs / dependencies |
|---|---|
| A generation | `59344807_0–3`; all COMPLETED, 0:0, 128 requests |
| B generation | `59344808_0–6`; all COMPLETED, 0:0, 156 requests |
| A sealing and semantic readers | `59347821 → 59344824` completed; readers `59350067_0–1 → 59350069` analysis |
| B sealing and semantic readers | `59347824 → 59344828` completed; readers `59349207_0–1 → 59349213` analysis |
| A dense labels / trajectories | `59350613_0–20 → 59350617`, following corrected CPU launch |
| B dense labels / trajectories | `59350396_0–26 → 59350400`, following corrected CPU launch |
| A / B target engagement | `59348108` / `59348109`; both COMPLETED, 0:0 |
| A / B paper builds | `59350111` / `59349261`, after the respective semantic analyses |
| Supplemental pulse qualification | `59348871`, three fixed requests; COMPLETED, 0:0, PASS |
| Combined operator qualification | `59349546`; COMPLETED, 0:0, PASS, 31 engineering requests bound |
| C prepare/dispatch | `59349549 → 59349552`, after combined qualification |
| C generation / semantic price / dispatch | `59351430_0–3 → 59351433 → 59351438`; 208 assigned requests |
| C dense / engagement | `59351477 → 59351483` / `59351472` |
| Fresh prepare/dispatch | `59349583 → 59349585`, after combined qualification and already verified enrollment |
| Fresh generation / semantic price / dispatch | `59351429_0–133 → 59351434 → 59351437`; 3,480 assigned requests |
| Fresh dense / engagement | `59351476 → 59351482` / `59351474` |
| Fresh strict-format sensitivity | `59349974 → 59351479`, then the exact fresh analysis |
| Utility pilot attachment | `59349601 → 59351478`; resolves fresh analysis, selected policy and independent utility qualification |
| Original-prompt engine qualification | `59346566`: COMPLETED, 0:0; `PASS_ENGINEERING`, all 13 cases |
| Historical capped-reader sensitivity | `59347889`: qualification COMPLETED, 0:0; replay `59347891_0–4 → 59347894` analysis |
| Native veto-format sensitivity | `59348469`: COMPLETED, 0:0; fixed 134 starts, CPU only |

Fresh strict-format outcome sensitivity has its own frozen plan and root
submission receipt under `overnight-fresh-veto-submissions-v2`. It joins the
unchanged semantic outcomes after primary analysis and never modifies the
main enrollment or replaces the original strict-veto flags.

## Observed findings

B's 15 routing contrasts and A's eight routing contrasts are separate secondary
multiplicity families. The saved intervals use 50,000 paired family bootstrap
replicates with Bonferroni percentile correction; these are approximations
with small family counts. They are not exact randomization intervals.

A's paired verification experts increase normalized target inclusion by
0.1376 versus native, simultaneous 95% interval [0.0763, 0.1882], across eight
families. Both individual expert arms also have positive measured inclusion
effects. These findings verify that the proposed interventions reach their
routing targets; the semantic comparison determines whether that changes the
intended behavior.

B's routing measurement is complete. Across its 13 families, one pulse increases
normalized target-expert inclusion by 0.2940 versus native routing, with a
simultaneous 95% interval [0.1277, 0.4761]. Relative to matched random routing,
the increase is 0.3080 [0.1394, 0.4985]. This endpoint counts target inclusion
over intended pulse opportunities, including zeros after early closure. The
corresponding blinded semantic outcomes are pending. A routing first-stage
effect does not establish a useful reasoning change.

The fixed native-prefix veto sensitivity recovered 237 naturally stopped
ratings whose valid final JSON was wrapped in Markdown fences. Coverage is now
262/268 ratings, with 68 strict starts across 63 families. Six ratings remain
length-capped and unknown. The original primary enrollment and original two
strict starts remain unchanged; the new 68-start analysis is explicitly a
format-adjudicated sensitivity, with the same 28 semantic contrasts and a
separate token multiplicity family.

The historical 1,024-token semantic stage remains exploratory because its
128-family feasibility gate failed. Its 760 assignments cover 95 starts and 78
families. Target minus native is −2.105 percentage points, simultaneous 95%
interval [−10.417, +5.978]; target minus matched random is +0.526 points
[−8.211, +8.560]. Neither comparison establishes semantic steering.

The separate 4,096-token reader-cap qualification recovered valid stopped JSON
for all four fixed capped ratings. Three used more than 1,024 tokens; only two
preserved the old emitted text as an exact prefix. Changed batching therefore
is disclosed, and the replay remains a sensitivity rather than replacement
primary evidence. Its full 180-rating replay includes every originally capped
or malformed rating, without arm/outcome-based selection.

## Preserved execution failures and corrections

1. The initial A/B startup imported the wrong package before selecting the
   qualified worker. The new entry point fixes import order. Original failed
   arrays, logs and allocation costs are preserved.
2. The old routing checker bounded changed expert *slots* by token *rows*.
   The corrected CPU audit uses the top-8 slot bound, preserves target-hit,
   force inclusion, reweight membership, TP parity, closure and pulse checks,
   and records each superseded checker reason. No generation is repeated for
   that dimensional error.
3. Original operator qualification `59344841` and CPU adjudication `59347826`
   remain FAIL. After the checker correction, 27/28 cases pass; the repeated-
   pulse fixture closed after 24 tokens and never reached the second pulse.
   Supplement `59348871` used the fixed first existing fixture whose five
   original runs reached 1,024 tokens. Both TP ranks executed the full windows
   [0,256) and [512,768); the three requests reached 1,024 tokens and inactive
   parity passed. Combined v4 qualification binds both original failures and
   new raw evidence. Cancelled descendants had zero runtime and were replaced
   by the graph above. Stochastic native duplicates still diverged; this does
   not establish global engine equivalence.
4. A nested CPU dispatcher exported its CPU-affinity mask to a GPU job.
   `59346750` failed before model launch; its children had zero runtime.
   The corrected shared helper removes parent scheduler/step/device-placement
   variables only from the environment supplied to `sbatch`. The original
   helper is archived. Eighteen scheduling/checker regression tests passed.
   Two already queued dense CPU children also inherited the old mask:
   `59345851` and `59345854` failed before processing. Corrected launches
   `59350372/59350375/59350378` reused the verified B preparation and created
   the dense arrays above.
5. Slurm may age completed jobs out of its dependency cache. Completed
   prerequisites now require successful `sacct` accounting and matching sealed
   artifacts; active prerequisites retain explicit dependencies.

## Utility and paper outputs

The four-GPU original-prompt engine passed engineering qualification, including
native inactive behavior, the three operators, preemption/recompute and
reasoning closure. It does not establish semantic control or utility. The
runtime pilot uses the first two frozen utility families, eight canonical
assignments, and a four-GPU allocation of at most eight hours. Its exact IDs
can be reused by the full 96-family stage only with unchanged policy, sampler,
source and execution bindings.

The full production continuation is implemented and sealed around the frozen
engine, with exact pilot import, disjoint shards, bounded recovery, all-cost
pricing and a 384-assignment reconciliation index. All 208 qualified scientific
source hashes are unchanged. Every load, side query,
nonfire, failed attempt and shutdown counts. The previous 4.75-GPU-hour utility
line is not an adequate production estimate. An empirical runtime projection
is distinguished from an allocation ceiling and a mathematical work bound.

`UTILITY_OUTCOME_MEASUREMENT_PLAN_v3.json` freezes offline strict/J1 grading and
utility inference before original-prompt outcomes. Its tested adapter retains
the original convention: naturally stopped answers receive strict/J1 grading;
caps and committed errors are operationally wrong. Missing executions and
unknown complete-token costs remain visible through identification bounds.
Complete results receive paired family-bootstrap simultaneous accuracy/token
intervals and a joint confidence-region figure. Similar accuracy is not a
noninferiority or equivalence result.

`UTILITY_PRODUCTION_CONFIG_v1.json` binds the v4 pilot proposal, v3 outcome
measurement and v2 J1 resource envelope. It allows 32 concurrent four-GPU
shards, retains complete families, and prices from the measured pilot with
an explicit 2× stress factor. The initial allocation and one infrastructure
recovery wave share a 16,000 billing-core-hour generation allowance; the
reserve covers at most ceil(0.25 × initial shard count) allocations. A
separate 216 GPU-hour maximum envelope covers all 384 possible J1 items,
three votes and all allowed transport attempts. Exact prompts reduce or
confirm that grading price before submission. These reservations are not
observed spending or a completion-time guarantee. Live account commitments
are reconciled before production.

The production adapter passed eight focused tests. Outcome v3 passed eight
tests; the J1 chain passed 19 tests and real repeated scorer imports. Frozen
v2 outcome files are preserved: v3 repairs an import collision and compares
the actual tokenizer vocabulary fingerprint instead of its file-byte hash.
J1 entry v2 pins the qualified scoring package before shared helper imports.
Neither repair changes the scientific model, controller, judge or sampler.

The first production follower `59351999` failed in five seconds because
`saldo` was unavailable on the CPU-node command path; its grading child
`59352001` was cancelled with zero runtime. Separately, C/Fresh paper
attachments `59349561/59349594` rejected reversed CLI arguments before
submitting children. These operational recoveries are being verified;
the generation and semantic pipelines above are unaffected.

At 00:28:51 CEST, the saved allocation snapshot reports 59.442 GPU-hours for
the tracked parallel chain, including preserved failures and provisional
running allocations. This is not the total historical project expenditure.
Use `audit_parallel_jobs_v3.py` for subsequent snapshots, including nested
utility J1 jobs.

The legacy paper addendum is already generated at
`routing-paper-v3-legacy-104ee9fbd9387320/`, with 12 claim records, a contrast
TSV and PDF/SVG figures. Each new semantic stage has an automatic paper job.
Target-engagement analyses produce their own CSV, PDF and PNG artifacts.

Historical confirm exposure, K1's matched-population scope, K2's uncertainty,
G2's non-rejection interpretation and G3's dose-selection role remain explicit.
Completed clean observational analyses, depth summaries and X2 are reused.
The failed serial/eager NLL qualification is preserved. X3 retains its legacy
diagnostic role and remains behind the new steering study in priority.

## Verified follow-up on October 5

The live queue is empty at this inspection. All 134 fresh generation array
tasks (`59351429_0–133`) and all four C tasks (`59351430_0–3`) completed
`0:0`. The sealed fresh stage receipt reports 3,480 assigned continuations,
zero errors, 3,480 routed arrays, 3,426,894 emitted tokens, 3,111 length stops
and 369 natural stops. Its seal is
`b037fad3097751c79f4de04ef360fb324181f007cea0bea8b490ed73258e0ecd`.

Fresh grading preparation `59351434` failed before reader submission at
`build_overnight_blind_frame_v2.reader_context`, with `KeyError: 'family'`.
The fresh reader source stores analysis metadata separately from visible
reader inputs; a versioned exact-provenance adapter is being prepared.
Generation results are complete and will be reused without GPU regeneration.
Dependent fresh semantic, dense, engagement and utility attachments need
explicit recovery after their new prerequisites are verified.

The C reader array `59356203_0–2`, semantic analysis `59356205`, dense label
array `59356361_0–9` and dense analysis `59356363` all completed `0:0`.
A/B semantic and dense outputs are also available for a source audit.
No original-prompt utility result exists yet. The production follower's
`saldo` command-path failure is being repaired separately from the fresh
measurement schema error.

Verified identity remains `login05.leonardo.local` / `lmolfett`, and the live
Slurm association is `iscrc_miosr` with `normal` among allowed QoS. Current
`saldo -b` reports 21,350 of 68,000 local billing hours consumed (31.4%),
with 1,251 of 7,391 monthly hours consumed. These figures are project
accounting, not a GPU-hour total for the steering study.

### Completed-source audit and independent recovery

The earlier pending statements above are superseded by verified saved results.
A/B/C semantic and dense analyses are complete. No small-study semantic interval
strictly excludes zero. A pair minus native is +6.25 percentage points
[-25.00,+50.00]; B repeat minus once is -3.85 [-26.92,+15.38]; C reweight minus
matched random is +7.69 [0.00,+26.92]. These are separate small-family,
Bonferroni bootstrap approximations, not a validated steering success. Valid
reader pairs are 110/128, 141/156 and 200/208 respectively. All C generations
reached the 256-token cap. Dense valid stopped labels are 5,936/5,965,
7,764/7,787 and 2,867/2,876; gaps and incomplete tails still break adjacency.

The historical full 180-rating recovery is also complete, sealed in
`MECHANISM_SEMANTIC_CAP_RECOVERY_SENSITIVITY_v2.json`. Valid reader pairs rise
from 632/760 to 749/760. Target minus native is 0.00 percentage points
[-7.895,+7.865]; target minus random is +2.632 [-5.056,+10.111]. Only 73/179
formerly capped replays retain the exact old text as a prefix, so this remains
a changed-batching sensitivity and does not replace the registered analysis.

Independent fresh routing analysis `59378652` completed `0:0` in 1m31s.
Source, plan, assigned-record, telemetry and figure bindings verified. Its
`RESULT.json` seal is
`ff8710f2f9d45b3aa1ab365c36e403cf0225f66470d1ab2feff1777838731cf4`.
Across the 100 verification starts/families, bias increases normalized target
inclusion by 0.1376 [0.1186,0.1576] versus native; force increases it by 0.8871
[0.8303,0.9263]. Across 34 commitment starts/families, bias increases it by
0.3507 [0.2528,0.4571]. These simultaneous intervals belong to the frozen
routing multiplicity family. They demonstrate routing engagement, while the
fresh semantic comparison remains ungraded. Reweighting changes selected
weights and requires its same-state dose telemetry; membership alone is
insensitive to that operator. All 3,480 assignments remain in the analysis.

C paper recovery `59378922` was submitted and verified by exact job ID using
the unchanged v3 scientific builder through the prepared named-argument v4
entry. It directly builds from completed C analysis `59356205`, avoiding an
aged dependency. It completed `0:0` in seven seconds; generated figures and
48 new claim records passed artifact-hash verification. The addendum is
`routing-paper-v3-c-9037a894f431788f/`, with summary seal
`ae4f974d99e7ed1d7d41caff33bf69907d46a8d3d3d75ddc3699cf4c64738ebd`.
The six prepared named-argument recovery tests passed in the existing
`correlation-client-3.11` test environment.

### Restored measurement graph (10:38 CEST)

The additive fresh schema amendment is sealed at
`ae2dc509e2951056e315322a6c9f887bfae07b74c82a9aac4dd7f63b40e3bada`.
Twenty focused tests and all four wrapper syntax checks passed. All 134 real
start contexts, 208 qualified scientific source hashes and 32 frozen semantic
measurement source hashes remain verified. Original failed jobs and receipts
are preserved.

| Recovery / downstream work | Verified jobs and status |
|---|---|
| Fresh frame and exact reader price | `59379611`: COMPLETED `0:0`, 4m47s |
| Fresh reader dispatch | `59379614`: COMPLETED `0:0`, 12s |
| Fresh GPU readers | `59380006_0–39`; first two RUNNING, remaining 38 pending Priority |
| Fresh primary semantic analysis | `59380008`, after all 40 reader tasks |
| Format-veto sensitivity | `59379758` completed attachment; `59380077`, after primary analysis |
| Fresh paper addendum | `59379763` completed attachment; `59380073`, after primary analysis |
| Utility policy selection | `59379677 → 59380075` completed; guarded selector `59380092`, after primary analysis |
| Fresh dense preparation | `59380018`: RUNNING independently from semantic readers |
| Fresh dense GPU dispatcher | `59380029`, after exact dense preparation and complete price validation |

All 3,480 saved continuations are gradeable. The fresh blinded frame seal is
`c8d56005fb77d164d17a558a2c53424b863112534cdf5f40850a19077201a993`;
the arm-map seal is
`23d956c8f5714b32ed62152bfdb34cb2b741dea6aff5464edca5446c1eaa732e`.
The exact reader price is 6,960 ratings in 40 two-GPU shards with two-hour
walls, projected complete cost 137.8637 GPU-hours, below the 191.06-hour
frozen measurement envelope. This is a projection, not observed spending.
Independent post-preparation measurement/provenance validation passed.

Dense recovery has its own sealed amendment
`872e5dcd10020e7530db176617aca4c97a8105a4e9b89fb96f97c379da91ea6e`,
19 passing regression tests and four checked wrappers. Its separate derived
source context has seal
`0a38092042aee6a67e4782664aa4eed6a055223a8e37d950a5bbbb182c55a1d3`;
the original source seal is never assigned to normalized bytes. Outputs use
the dedicated `generated-dense-fresh-metadata-recovery-v1/` parent. Neither
recovery repeats generation or selects rows from observed intervention effects.

The utility pilot v5 attachment and guarded selector are restored successfully.
Production follower `59379679` failed `2:0` in two seconds before Python or
GPU launch: the absolute `/cineca/bin/saldo` executable is not mounted on its
compute node. Grading follower `59379681` was cancelled with zero runtime.
Thus the absolute-PATH correction is insufficient; live account transport
needs qualification before a separate production restoration. The valid
pilot/selector chain must not be duplicated. Native live balance and queued
commitment checks remain required; a cached balance is not substituted.

A final utility source audit caught three late environment-export blocks
written after root preparation. They were restored to the exact already sealed
operational bytes before the pending jobs executed. Root independently verified
both complete eight-file operational closures. All frozen science and existing
receipts remained unchanged. No further edits to bound operational files are
permitted without a separate version and amendment.

`audit_parallel_jobs_v4.py` adds restored dense and format-sensitivity receipts
to the bounded allocation ledger. Snapshot `OVERNIGHT_SLURM_20261005T083825Z.json`
reports 308.2361 tracked GPU-hours, including preserved failures and provisional
running allocations. It is not the total historical project expenditure.
Direct root submissions are separately sealed under
`overnight-root-recovery-submissions-2026-10-05/DIRECT_JOBS_v1.json`.

No original-prompt utility results or fresh semantic-effect estimates exist at
this checkpoint. GPU start times and a paper-completion ETA remain unverified:
38 reader tasks lack scheduler start estimates. Production QoS has a documented
65-full-node minimum and must not be used to accelerate two-GPU reader shards.

The two-minute, one-CPU native-account transport probe `59381728` completed
its qualification attempt with `FAILED`, `1:0`, four seconds. The sealed failure
receipt (`utility-account-transport-probe-v1/PROBE.json`,
`7f24b349122ccac7e1d05d56537fbd8168cd076b73781f269a5a9d59e36f61b3`)
records SSH authentication rejection on the actual compute node, before any
balance query. Existing credentials and host trust were unchanged. A finite
true-login-node orchestration adapter is being prepared: allocated CPU jobs
prepare/account and publish readiness; bounded native login calls read live
balance/commitments and submit exact Slurm children. It must not forge Slurm
environment variables, use cached account balances or introduce a service.
The valid v5 pilot and selector receipts remain the predecessors.

At the last live inspection, fresh reader tasks `59380006_0` and `_1` are
running on `lrdn0648`, with both two-GPU allocations in model startup; the
other 38 tasks still wait on Priority. Dense CPU preparation remains running
on `viz16`. The separate unresolved native NLL qualification is now undergoing
a source/failure audit in parallel, with additive fixes/proposals only and no
GPU resubmission without an exact qualified price.

### Native continuation attached and closeout recovery

The fresh semantic chain remains canonical: `59380006 → 59380008`, with
format sensitivity `59380077`, paper build `59380073`, and guarded utility
selector `59380092`. At the pre-audit inspection, reader tasks 0–2 had
completed `0:0`, tasks 3–6 were running, and tasks 7–39 were pending. The
independent dense array `59385734_0–651` and analysis `59385736` remain queued.
No generation replay or additional GPU comparison was submitted in this recovery.

Utility native orchestration v3 is now attached once. Its durable
`utility-production-root-submissions-v3-a5a6d77abd5ec32e/ATTACH_CHAIN.json`
has seal `337d8667f0b4bbdb6c048de4882b01223a15f168ba67f2d5bd647fd6ce9cb62b`.
Exact job `59386634` was read back as the authorized two-CPU, five-minute
`lrd_all_viz/normal` successor of the existing selector `59380092`.
The existing v5 selector and pilot attachment were reused.

All nine focused native-orchestration tests and ten exact bound wrapper syntax
checks passed. The v3 operations seal remains
`8a3000cc60dfef0bb5872cc7615ceabc919a5bf67f2ffc43f02daf4c1732a946`;
both operational files and the frozen utility scientific/grading closures
validated. The attachment's native live snapshot observed 46,650 billing hours
remaining, 22,048.6911 hours of current project queue commitments, and
24,601.3089 uncommitted hours against a 17,984-hour requirement including the
generation/recovery allowance, grading reserve, and pilot. These are a dated
reservation check, not observed experiment spending or a future budget guarantee.

The additive `routing_study_closeout_v2.py` audits outcomes on actual CPU Slurm
steps and consumes at most one utility readiness receipt per native call.
Its `advance --submit` mode waits for successful terminal producer accounting,
then delegates to the unchanged sealed v3 dispatcher for fresh balance/queue
checks. No readiness exists yet; the attachment is pending its selector.
The tool creates an immutable study index, semantic contrast table, synthesis,
NLL audit, and reconciliation of the existing advisory reviews. Missing assigned
ratings and labels remain explicit unknowns. A completed study is reported only
after terminal accounting and required artifacts verify.

The first closeout CPU attempt `59388426` failed `1:0` after 27 seconds. Its
utility bootstrap pinned the immutable scoring package, while dense validation
requires its separately frozen repository module paths. The source bytes were
unchanged; the path-aware closure correctly rejected the mixed namespace.
The v1 plan, script, output and failure are preserved. V2 uses a clean dense
subprocess and binds that failed attempt in its separate plan. Eight closeout
tests, the clean dense source-closure check, and the v2 wrapper syntax check
passed. Initial v2 CPU audit `59389128` completed `0:0` in 31 seconds. Its
`STUDY_INDEX.json` seal is
`60a490bf8b7922087b517be9b20201e49468db8af4e5df5c162ab2fbcf39b761`.
It verifies A/B/C semantic outcomes, all three completed dense results, four
paper addenda, their saved figures, and fresh routing-engagement artifacts.
At that snapshot 528/6,960 fresh ratings were accounted in complete shards:
466 valid stopped ratings and 62 capped/malformed measurements. The other
6,432 ratings and all 187,524 dense labels remain explicit unknowns. The
snapshot reports 321.9661 GPU-hours for its broader closeout job set including
legacy X2 and preserved score-calibration attempts; it is not the same job set
as the parallel-only snapshots or a total project charge.

The following exact CPU checkpoints are attached and verified:
`59389450` follows terminal accounting of fresh analysis, sensitivity, paper
and selector jobs; `59389577` follows dense trajectory analysis `59385736`.
Each is a two-CPU, zero-GPU, one-hour maximum allocation. Their receipts and
immutable reports live under `routing-study-closeout-v2/`.

`continue_routing_study_v1.py` passed four additional no-submission tests and
is sealed by `routing-study-closeout-v2/CONTINUATION.json`
(`94e1d358184dfeb3a59e89c7641ed29820cb89f28d672d99cb0c2bb743c351dc`).
The finite native call is:

```bash
/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python -B scripts/experimental_resume/continue_routing_study_v1.py step --submit
```

It consumes at most one verified utility readiness receipt, preserving the
existing selector, eight-assignment pilot import and single reserved recovery
wave. After canonical J1 dispatch, it attaches final CPU closeout once. Already
terminal predecessors are verified through accounting and omitted from new
Slurm dependencies to avoid aged controller-cache entries. The call must run
on the actual native login shell after readiness is produced; no automatic
login service or background polling is installed. Existing user authorization
covers these finite dispatches within the frozen envelopes.

The native NLL distinction is persisted in the first audit's `NLL_AUDIT.json`:
legacy X2 measured all 3,354 assigned sequences and passed the frozen Q3 repeat
rule at batch sizes one and eight. This is not bitwise engine equivalence.
Counterfactual qualification `59189255` and score-API calibration jobs
`59196354/59198795` remain failed, with their measurements, numerical checks,
hashes and allocation costs preserved. They do not require an X2 rerun.

Existing October 2 debate files A, D and D2 and their adjudication are present.
Their early pending-state statements are historical. Closeout maps their
findings to verified repairs, current limitations, and superseded concerns;
it does not claim a new independent moderator review. Distinct substantive-action
ordering and X3 remain deferred. Fresh semantic effects and original-prompt
utility are still pending at this checkpoint.

The dated recovery ledger is `RECOVERY_LEDGER_2026-10-05_v1.json`, sealed at
`e8a9e357a0f8900b958435cbb4ac224372a8b1883c82492f4fee81bcb0da5ebe`.
Its job snapshot supersedes the earlier queue counts. Ledger references and
generated report hashes passed independent readback verification. Slurm's
start-estimate query returns `N/A` for the remaining fresh reader and dense
array ranges; no completion date has been established.
