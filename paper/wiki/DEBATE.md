# Model debate and adjudication

Three historical DeepSeek reviews, including the implementation follow-up, were run through the existing local Codex provider, configured as `deepseek-flash` / **DeepSeek V4.1 Flash**, reasoning effort **max**, ephemeral session and read-only sandbox. All exited zero and produced saved responses. The user explicitly approved disclosure of the research packet and follow-up to `api.deepseek.com` after automatic review initially rejected the sensitive transfer. Credentials and the full private PDF were excluded. Configuration identifies the requested model; it is not independent attestation of the provider's backend weights.

A separate **GPT-6.1 Sol xhigh** reviewer inspected source artifacts and challenged the first DeepSeek report. The parent reconciled both against actual sources and the user's decisions. Reviewer agreement is not empirical evidence.

## Review record

- Implementation follow-up: [DeepSeek packet](debate/deepseek-implementation-prompt.md), [response](debate/deepseek-implementation.md), [receipt](debate/deepseek-implementation.receipt.json). It completed successfully in 158.7 seconds using the same explicitly approved research material. The local project launcher now forwards CLI arguments; credentials were never extracted or put in the packet.
- GPT-6.1 Sol xhigh reviewed the actual native callbacks, restoration paths, source seals and first-wave fixtures. Repairs included transaction cleanup, prediction-row attribution, conditional assignment accounting, and binding inherited selectors/runners/Marlin to the runtime. Separate sampled output repeats are diagnostic; mandatory no-op fidelity compares IDs/weights on the same hidden state.
- [Sol implementation review and resolutions](debate/sol-implementation-review.md).

The implementation adjudication is ADR-06. Discovery checked/unchecked nomination is a destination association, not proof of anticipation. The primary conditional event is verification **before closure within 1024 tokens**, with all assigned branches retained; closure is a competing event, not a reason to filter the population. The cheap-controller output cost is zero, while generator output remains capped at 16384 per policy request and research retries are itemized. Native validation measures coverage and execution; it does not establish treated dose transfer. Reviewer suggestions do not create additional sweeps.

- [DeepSeek round 1 prompt](debate/deepseek-round1-prompt.md), [response](debate/deepseek-round1.md), [execution receipt](debate/deepseek-round1.receipt.json).
- [GPT source-grounded counterargument](debate/gpt61-counterargument.md).
- [DeepSeek round 2 prompt](debate/deepseek-round2-prompt.md), [response](debate/deepseek-round2.md), [execution receipt](debate/deepseek-round2.receipt.json).

Round 1 completed in 77.06 seconds; round 2 in 66.40 seconds. Raw CLI event/state logs remain private in HOME. The saved reports below are critiques with errors preserved, not protocols to execute.

## What the debate changed

| Issue | Initial criticism or proposal | Source-grounded adjudication |
|---|---|---|
| Hidden versus routing information | DeepSeek described grouped probe scores as routing evidence | Corrected: they are hidden-state prefix probes. Clean R3-D is the distinct positive routing evidence |
| Repeating probe validation | Round 1 proposed repairing response-cluster inference again | Rejected as a required repair: grouped folds, inner layer selection and response-bootstrap already exist. New controls need a distinct claim |
| Missing common evidence | Round 1 treated the probe as the only positive result | Corrected using seven-model clean readout and GPT/Qwen3.6 anticipation; DeepSeek withdrew the claim |
| Number of models | Early proposals centered one model or a second replication | User rejected this: all seven models, including causal verification, are required |
| Semantic precision threshold | Round 1 introduced 70% held-out precision | Rejected: existing gate uses apparent precision with held-out uncertainty/coverage; no new threshold adopted |
| Causal success | Round 1 required significant accuracy improvement | Rejected: a substantive transition effect is control; usefulness is a separate joint accuracy/cost question |
| Token endpoint | Round 1 added prompt tokens to the primary endpoint | Rejected: preserve generator plus controller output tokens; itemize other expenditures separately |
| Fidelity | Round 1 suggested all arms must share one execution | Corrected: within-execution native/executed fidelity differs from matched-prefix, separately generated causal arms |
| Null interpretation | Both reviewers warned against interpreting zero-crossing intervals as no effect | Accepted: report scope, precision and missingness; current FRESH benefit remains unestablished |
| Completed labels | Both distinguished inventory closure from semantic validity | Accepted: 187,524 reconciled identities do not resolve 560 unknown FRESH semantic pairs |
| Conditional/context work | Reviewers treated extra context complexity as conditional | Accepted within existing K=1 fallback and prespecified gates, without silently deleting frozen contracts |

## Remaining disagreements resolved for this wiki

**An unchanged reader rerun cannot fix the full recognizer's support.** DeepSeek round 2 still proposes rerunning the corrected reader to determine whether some decisions can qualify. That may measure precision on supported decisions, but the saved outcome-blind precheck already proves the fixed frame cannot pass the full current contract: apparent support is 0/0/4 for three required decisions. The final v2 receipt explicitly blocks dispatch when a required decision is missing. Advancing on a subset would need a declared protocol amendment. The wiki therefore puts feasibility/design resolution before another GPU audit, rather than treating that rerun as the route to complete qualification.

**Actual reader inference is not a CPU-only metadata audit.** Round 2 places reader precision/uncertainty measurement in a CPU-only stage. Counts and prompt validation are available through bounded metadata checks; model-generated semantic ratings require the existing separately priced inference workflow. No new reader workload follows from the review.

**Holdout status depends on exposure and adaptation.** The pilot obtained no semantic ratings. Inspection of proposal counts is not inspection of semantic outcomes. DeepSeek's blanket claim that the 16 prefixes automatically require replacement is too strong. Record exactly what covariates and labels were seen and how redesign used them; preserve outcome blindness or declare a suitable fresh audit/amendment where needed. Do not claim independence after tuning on held-out semantic outcomes.

**The existing Qwen protocol is not a seven-model price.** A compact common verification design is a useful prospective proposal. It must preserve existing Qwen results and frozen comparisons, specify equivalent controls under each native router, bind all seven models and obtain measured workload authorization. Neither model review authorizes an operator-grid reduction or seven inference campaigns.

## Resulting shared direction under ADR-06 (preserved)

The paper follows **all-model readable information → model-specific conditional targets → measured routing engagement → substantive verification effects or scoped limits → joint utility**. Verification goes first. Every model must appear, and missing support or instrumentation remains visible. Smaller-budget and preloading hypotheses are secondary, as the user accepted. ADR-06 and the implemented campaign now supersede the earlier prospective feasibility step; current advancement uses the adopted gates and measured cards in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Retained reader format and protected handoff review — 2026-10-08

The first retained GPT eligibility measurement exposed a stopped, complete JSON object inside one Markdown fence and a separate truncated draw. Sol required the recovery to be declared as a decoding interpretation amendment before first recognizer/target selection, applied uniformly to all seven models. It cannot count as success under the original parser or turn the truncated draw into an agreeing pair. The adopted annex accepts only plain JSON or one complete outer lowercase `json` fence around the entire final suffix, then uses the original content validator. Every raw draw remains preserved and every derived semantic result must bind its complete decision ledger.

Sol also found two resume risks in the proposed controller handoff: saved rescue receipts alone cannot establish live protection, and a saved release intent may precede actual scheduler release. The final transfer verifies both actual released rescues before retiring the old chain; a still-held primary with a release intent rechecks them. A release already performed before its receipt is reconciled without repeating the scheduler mutation. The72 passing tests cover these cases and interruption during partial cancellation and final closeout. The observed handoff reuses the same accepted chain and changes no scientific allocation or reader draw. Readiness, annex and actual transfer receipts are recorded in [IMPLEMENTATION.md](IMPLEMENTATION.md); this review does not establish semantic support or a causal effect.

## Full grader failure review — 2026-10-08

GPT-6.1 Sol xhigh reviewed the three disagreements in the completed common audit, the original blind labels and the frozen grading/analysis code. All nine draws stopped normally, parse as plain JSON and pass fixed-vote, seed, frame and invocation validation. The disagreements concern whether short-run context makes `MR=MC` equivalent to `MR=SRMC`, whether the1650/1700 reference comparison should remain uncertain, and whether context supplies an omitted `10^-18` scale. The review does not adjudicate those references as independent human truth.

Sol recommends preserving the original FAIL and using the existing fallback: no production J1, unknown residual accuracy in worst-case bounds, and continued all-seven work through independent semantic/control gates. The parent independently validated all100 receipts,300 draws and the exact qualification join. [Saved review](debate/sol-j1-qualification-review.md) and [observed closeout](../../report/verification-study-v3/J1_QUALIFICATION_CLOSEOUT-r1.json) preserve the conclusion. No scientific source, label, threshold or submission changed for this review.

## Full GPT recognition failure review — 2026-10-08

GPT-6.1 Sol xhigh independently reproduced GPT's complete256-draw decision ledger and recognition FAIL under the declared format annex. The parent did the same provenance and qualification checks. Execution completed, and the annex recovered75 fenced stopped draws, but196/256 draws are usable and only84/128 prefixes have a usable pair. Apparent eligibility confirmation is22.41% across29 families; already-checked confirmation is18.75% across8 families. Both fail70%, and already-checked also fails the10-family requirement.

The reviewers distinguish a reader-defined measurement/support limit from an operational failure or a causal null. Held-out support is present; unconfirmed judgments and a degenerate interval do not establish absence of checking. The existing runner correctly closes GPT before nomination/treatment and continues the other six cells. [Saved review](debate/sol-gpt-recognition-review.md), [full closeout](../../report/verification-study-v3/GPT_ELIGIBILITY_CLOSEOUT-r1.json). No source, gate, draw or allocation changed for this review.

## Full Nemotron recognition failure review — 2026-10-08

Parent and GPT-6.1 Sol xhigh independently reproduced Nemotron's full256-draw ledger and qualification FAIL. Eligibility confirmation is25% across34 apparent families and already-checked confirmation10% across10. Both support minima pass, but both precision gates fail. The audit has191 usable stopped draws,37 truncated and28 content-invalid stopped draws;97 complete stopped fences were recovered under the existing annex. Agreeing readers also reject proposals. No definite operational defect was found.

This limits the current cheap prefix controller before nomination or treatment. It does not measure a causal null, and the held-out zero interval for unconfirmed checked proposals does not establish absence of verification. The unchanged runner records Nemotron `LIMIT` while the remaining five models continue; GPT retains its separate limit. [Saved review](debate/sol-nemotron-recognition-review.md), [full closeout](../../report/verification-study-v3/NEMOTRON_ELIGIBILITY_CLOSEOUT-r1.json). No new draws, gate changes or scientific allocations followed from the review.

## Full Qwen3-30B recognition failure review — 2026-10-08

Parent and GPT-6.1 Sol xhigh independently reproduce the original full256-draw ledger and qualification FAIL. Both apparent support minima pass, but confirmation10.10%/33.33% across33/12 families fails both precision gates. The193 usable stopped draws yield80 usable pairs;39 capped and24 invalid draws remain unconfirmed. Agreeing stopped readers also reject proposals. No definite execution or interpretation defect was found.

Preserve the controller's model-specific `LIMIT` before nomination/treatment, with causal/utility outcomes unmeasured. The four remaining checkpoints continue; GPT and Nemotron retain their earlier limits. [Saved review](debate/sol-qwen330b-recognition-review.md), [full closeout](../../report/verification-study-v3/QWEN330B_ELIGIBILITY_CLOSEOUT-r1.json). The reporting helper seals exact evidence without dispatch or scientific writes.

## Full Gemma recognition failure review — 2026-10-08

Parent and GPT-6.1 Sol xhigh independently reproduce the full256-draw ledger and qualification FAIL. Both support minima pass, but25.52%/41.67% confirmation across32/12 apparent families fails both precision gates. There are209 usable stopped draws,30 capped and17 invalid draws, with91 usable pairs. Agreeing stopped readers also reject proposals. No definite execution or interpretation defect was found.

Preserve Gemma's recognition `LIMIT` before nomination/treatment, with causal/utility outcomes unmeasured. GPT, Nemotron and Qwen3-30B retain their earlier limits; Qwen3.5, Qwen3.6 and GLM continue. [Saved review](debate/sol-gemma-recognition-review.md), [full closeout](../../report/verification-study-v3/GEMMA_ELIGIBILITY_CLOSEOUT-r1.json). No new draw, source change or scientific allocation followed from this review.

### Qwen3.6 full discovery advancement review — 2026-10-08

[Sol's source-grounded stage review](debate/sol-qwen36-discovery-review.md) independently verifies all48 original receipt/archive hashes,128-prefix/public-problem joins, accepted native cards and terminal accounting, largest retained input and actual protected successor/rescue dependencies. It confirms discovery/audit advancement only, with recognition, causal effects and utility unmeasured. Two reporting gaps found before sealing were corrected: explicit native receipt-to-card/submission/slice joins, and a read-only in-memory health snapshot. The first full reporting run then exposed a missing in-memory seal and wrote no closeout; sealing was corrected before the successful immutable report. No frozen source, scientific job, original draw or acceptance gate changed. The adopted direction remains ADR-06; this review adds observed stage evidence rather than another research plan.

## Analysis-first and memory objective — 2026-10-08

The user explicitly requested subagent debate, including DeepSeek V4.1 Flash with maximum reasoning. Two additional real calls used the existing configured provider, ephemeral read-only sessions and private HOME event logs. No credential was extracted or put into a research packet. Configuration records the requested model; it does not independently attest backend weights. The request for no external validators was respected: no new human/semantic validation study or replacement grader was added.

| Review | Bound record | Observed execution |
|---|---|---|
| Analysis-first contribution critique | [Prompt](debate/deepseek-analysis-first-20261008-prompt.md), [response](debate/deepseek-analysis-first-20261008.md), [receipt](debate/deepseek-analysis-first-20261008.receipt.json) | Exit 0, 68.176 seconds, `deepseek-flash`, max reasoning |
| Adversarial memory-design follow-up | [Prompt](debate/deepseek-memory-design-20261008-prompt.md), [response](debate/deepseek-memory-design-20261008.md), [receipt](debate/deepseek-memory-design-20261008.receipt.json) | Exit 0, 87.281 seconds, same requested configuration |

Local source reviewers separately audited the result corpus, the package/code gap and the manuscript. Their [source adjudication](debate/package-analysis-review-20261008.md) and [result audit](../../report/model-analysis-review-20261008/AUDIT_NOTES.md) take precedence over review suggestions where actual artifacts resolve a claim.

| Issue | Adjudication |
|---|---|
| Another correctness-prediction or generic stability study | Reuse completed GPT B4 and the narrower Qwen family-crossfit nominations. They do not fill every all-seven retention cell, but cannot be called absent. |
| Five missing anticipation caches | This means five missing equivalent clean next-class results, not missing all-seven observational profiles; they are not required for static pruning. |
| Noninferiority and latency prerequisites | No margin is required for the user's joint accuracy/memory estimate. Latency is secondary. A later negligible-loss claim would need a meaningful margin and adequate precision. |
| Static versus online policies | Retrospective labels and declared correctness features can train a static keep-set. Online state-dependent actions need prefix-only inputs. Evaluation outcomes never select the policy. |
| Strong baselines | Frequency and REAP are distinct. Counts cannot recreate expert-output magnitudes. New collection is conditional on the exact claimed comparison and retained-input inventory. |
| Memory saving | Masking is insufficient. Real omitted storage, quantization/ID/native-selector fidelity and measured runtime bytes are necessary. Header bytes supply static accounting only. |
| Seven-model scope | One initial architecture can stage engineering, but cannot replace the all-seven analysis/usefulness matrix. Missing resident cells remain explicit. |
| Authorization | Neither reviewer agreement, stable profiles nor a source seal authorizes computation. Concrete resource cards and the existing publication gate remain required. |
| Numerical transcription | Nemotron's CLEAN interval upper endpoint is −0.11819; −0.11619 in the first packet was an error. Source-exported paper/wiki values are correct. |

The adopted current direction is ADR09: all-seven analysis and a reusable local package, followed by training-only retention design and separately qualified accuracy/memory evaluation. The user additionally selected a **common percentage reduction in expert bytes across models**, rather than a fixed local-GPU capacity. No numerical percentage or new workload budget has yet been selected. Existing unfavorable evidence is preserved, while the paper leads with model analysis and the new positive descriptive concentration result.

The new curve was independently checked against integer-reconstructed counts for GPT and GLM. This verifies its calculation, not a causal pruning benefit. Original sources, existing jobs and all previous reviews remain preserved.

**User correction after the debate:** the package is the paper's eventual output; current work must focus on results. API and general deployment suggestions above are therefore deferred. Minimal code for source analysis, physical memory measurement and the frozen held-out comparison remains relevant to the paper.
