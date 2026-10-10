# Independent scientific critique, round 1

**Scope.** This review uses only the supplied reports, cross-checks and paper description. I did not inspect artifacts, rerun analyses or verify receipts independently. I distinguish **observed**, **engineering-qualified**, **hypothesis**, and **missing** evidence.

## 1. Defensible contribution hierarchy

**Defensible now — observational measurement.** The response-grouped nested probe analysis is the strongest scientific result. On 38 responses/3,087 units, routing/state features predict episode labels beyond positional nuisance: macro-AUROC .852/.780/.788 for Qwen/GPT/Gemma, with gains over position of +.286/.209/.221. This repairs a serious legacy leakage/confounding problem and is defensible as an observational claim: routing contains information about coarse reasoning-state descriptions in these models and this corpus. It does *not* establish manipulability, intervention layers, causal transitions or generalization.

**Defensible now — engineering and measurement protocol.** The frozen 32 validation/96 final families, two seeds, 16,384-token protocol, 1,024-token transition endpoint, source seals, dense-recovery completion (187,136 valid, 388 unusable, zero unresolved among 187,524 identities), and the 13-case backend qualification are valuable reproducibility assets. They are not scientific effects. The 16k request took 2,637.33 seconds and 136,231,573 capture bytes; this is an engineering price input, not a utility result.

**Defensible now — bounded causal non-result.** Across A/B/C/FRESH, no contrast in the reported family strictly excludes zero. FRESH’s candidate-to-verify force-native effect is +5.5pp, simultaneous CI [-4.5,+15.5]; force-random +11pp, [-0.5,+23.5]. The honest claim is: under the tested caps, horizons, readers and interventions, no reliable causal advantage was demonstrated. That is weaker than “no effect,” because intervals are wide and semantic missingness is substantial.

**Not yet defensible.** Causal control of reasoning transitions; “best layer to intervene on”; expert specialization; paired-verification or second-pulse benefits; context clusters as natural kinds; smaller-budget optimization; SSD expert prefetch benefits; joint accuracy–token efficiency frontier; generalization to new math populations; and any claim equating seven-class sentence descriptions with substantive reasoning-transition utility. The private *Rational Manifolds* ladder — observation → information beyond nuisance → dose → substantive displacement → utility — is a useful paper skeleton, but its MFA regions, eight episode types and residual-space causal claims do not transfer to sparse-MoE expert IDs, seven labels, or this intervention design.

**Recommended paper hierarchy.** A defensible paper can be: (1) a measurement/confounding study showing state information beyond nuisance and documenting recognition limits; (2) a bounded causal attempt showing why the tested interventions do not establish control; (3) a protocol/infrastructure contribution; and only conditionally (4) a joint utility/cost result. The durable wiki should encode exactly this hierarchy: immutable claim IDs, status, source revision, family freeze, endpoint, contrast, estimator, missingness treatment, decision and hash. Narrative reports should not be the primary evidence store.

## 2. Five most consequential mistakes or hidden assumptions

1. **Treating semantic labels as qualified intervention opportunities.** Sentence/episode labels and prefix-only recognition of actionable states are different objects. The proposal evidence is decisive: 48 apparent discovery prefixes give complete_unchecked=0, complete_checked=0, identified_approach=4; the held-out 16 give 1,0,2, against ten eligible families. Serialization repair cannot solve this. If proposal rules are changed after inspecting those held-out counts, that held-out set is no longer clean held-out evidence. A fresh disjoint held-out set or an explicit protocol amendment is required. Until then, there is no identifiable treatment population.

2. **Inferring manipulable layers from predictive probes.** The legacy whole-trace router–length correlation (~−.45) attenuates within absolute positions; response grouping fixes leakage but remains observational. High AUROC or strong routing-label association does not imply that perturbing that layer changes behavior. Predictive relevance and causal sensitivity can diverge sharply under redundancy. A layer-selection policy must require (a) nuisance-controlled predictive stability under response/family holdout, and (b) verified executed dose/displacement at the same hidden state, and (c) a causal screen against matched random. Max-AUROC scanning on validation/final data would be selection on outcomes.

3. **Reading wide zero-crossing intervals as evidence of no effect.** “No contrast excludes zero” supports “not demonstrated,” not “ruled out.” FRESH’s force-random interval reaches −0.5pp to +23.5pp; that is compatible with meaningful benefit and small harm. Different horizons, readers and seven-class descriptions cannot be pooled. The central question permits well-supported limits, but a limit claim requires a specified estimand and missingness bounds. Current designs do not justify an equivalence or no-effect claim.

4. **Assuming complete accounting repairs selection and missingness.** Dense recovery completing all 187,524 identities removes an inventory problem, not dependence: completed labels are not independent questions. FRESH retains 560 unknown substantive pairs; semantic qualification abstains; opportunity selection is based on observed native routing/prefixes. These mechanisms are unlikely to be missing completely at random. Exclusion can bias toward recognizable or high-confidence cases. The wiki should report treatment-policy coverage, abstention rate, reader disagreement, and worst-case bounds for every claim, not just valid-label counts.

5. **Underestimating dependence, power and post-selection.** Only 38 responses underpin the positive probe result; unit-level intervals around 3,087 units may be too narrow if resampling is not response/family-respecting. The frozen families are good design, but 32 validation families with two seeds and binary endpoints have limited power for small-to-moderate effects. The incumbent (forced L28/E9+E189; +1 L24/E159 bias) is explicitly unproven and may be selected on earlier data. Multiple simultaneous contrasts help error control but not power. Native cross-run variation (mean expert-set turnover .1824) also means matched comparisons must be bound within execution, not inferred across separate forced-token runs; the passing inactive-path check is same-hidden-state engineering fidelity, not cross-run equivalence.

## 3. Minimal closure plan using current assets

The minimal coherent route may close as a **limits/measurement paper without new GPU work**. If the lead wants to preserve a causal-control question, only the conditional sequence below is scientifically clean. No submission is authorized by this review; any GPU stage requires measured component prices and separate approval.

**Stage 0 — harden the only positive result (CPU-only, existing data).**
- **Question:** Does the response-grouped probe gain over nuisance survive response-level resampling and length/position-matched controls?
- **Reused inputs:** existing 38 responses/3,087 units; legacy leakage audit; frozen model/source identities.
- **Endpoint/controls:** incremental macro-AUROC and Brier score over position + token length; response/family-clustered bootstrap; leave-one-response-out; shuffled-label and position-only controls.
- **Decision:** if stable, this is the main observational contribution; if not, demote to exploratory.
- **Stopping rule:** one locked reanalysis; no new labels or outcomes.

**Stage 1 — semantic recognizability (GPU, proposed only; measured price required).**
- **Question:** Can the required prefix-only decisions meet the frozen ten-family and 70% discovery/held-out standard?
- **Reused inputs:** corrected v3 reader over the frozen 64 prefixes, existing family splits, two readers; no validation/final families.
- **Endpoint/controls:** per-decision eligible-family count, apparent precision, coverage and held-out uncertainty; abstention, lexical/position-only baselines; reader disagreement.
- **Decision:** only passing decisions enter intervention. Failure is a documented recognition limit and stops the causal route.
- **Stopping rule:** one preflighted load; no threshold/proposal tuning on the observed held-out 16; no adaptive rescue.

**Stage 2 — conditional causal/utility test (GPU, proposed only, after Stage 1 and authorization).**
- **Question:** Does one qualified policy change family-level final accuracy and total token cost versus native and matched random?
- **Reused inputs:** frozen 32 validation families, seeds 0/1, existing incumbent comparator, 16k cap, 1k transition endpoint; final 96 only if the frozen validation rule passes.
- **Endpoint/controls:** strict/J1 accuracy and total tokens (prompt + controller + generated reasoning + retries), seed-averaged, equal family weights, paired simultaneous family-bootstrap intervals; native, matched random in the same execution, incumbent.
- **Decision:** positive control only if the beneficial accuracy contrast excludes zero; cost is reported jointly, not required to improve. Otherwise report bounded uncertainty, not equivalence.
- **Stopping rule:** at most two qualified policies; no post hoc arms/thresholds; stop after validation if none qualifies.

**Best metrics and layer logic.** Use macro-AUROC plus calibration and incremental gain over position/length for observational claims; use final correctness plus total tokens for utility. Report dose and substantial displacement separately, and exclude near-zero-dose ratios. For layer selection, rank by nuisance-controlled incremental predictive stability, then require same-hidden-state dose/displacement and matched-random causal screening. Never select a layer on validation or final outcomes. The frozen L28/E9+E189 and L24/E159 targets remain useful comparators, not demonstrated optima.

## 4. Context clustering, smaller budgets, SSD prefetch

**Context clustering: later, not main.** It is currently an unqualified moderator. It belongs in the main paper only if it passes the existing nested family gains and ten-positive association rules, with context labels treated as descriptive.

**Smaller budgets: secondary endpoint now, separate policy later.** Token cost is central, so report it jointly. Optimizing the default thinking budget is a different intervention and should not be mixed with the transition-control claim unless it shares the same frozen estimand and controls.

**SSD prefetch: later/systems.** It is motivated by route stability and expert locality, but current evidence measures engineering execution, not cache hit rate, latency, energy or accuracy-cost utility. It should not be a main-paper causal contribution.

## 5. Hard questions to the project lead

1. If every causal interval includes zero and the semantic gate fails, is the intended publication a measurement/limits paper, or must you keep searching for a positive effect?
2. What is the estimand: effect among all requests, among recognizable opportunities, or among intervened opportunities? The current evidence supports different conclusions for each.
3. Why six fine-grained semantic decisions when held-out support is 1/0/2? Is the actual claim coarse answer emission/closure rather than fine transition control?
4. Which layer-selection evidence is outcome-independent? Were L28/E9+E189 and L24/E159 chosen on data later used to describe them?
5. How will 560 FRESH unknowns and semantic abstentions be bounded, and what missingness assumptions are acceptable?
6. What minimum detectable accuracy effect is plausible with 32 validation families, two seeds and family-equal weighting?
7. Will any proposal-rule or threshold change require a fresh held-out set, or will the observed held-out 16 be reused?
8. Is total token cost measured through closure, including controller, retries and failed generations? What is the bound when cost is missing?
9. Are matched-random and native controls guaranteed to share execution, RNG and prefix state?
10. What is the stopping rule for GPU spending if one further semantic qualification fails?