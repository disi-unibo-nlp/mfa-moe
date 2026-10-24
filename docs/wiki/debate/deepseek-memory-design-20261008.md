**1. Substantive errors and overrequirements in review1**

- It silently changes the objective from *reduce resident expert memory while measuring accuracy jointly* to a routing-observability paper. The memory end point is deferred until after another stability study; that is backwards.
- A noninferiority margin is not needed for honest joint estimation. Report paired Δaccuracy, Δlog-loss if useful, and Δresident expert bytes with family-clustered intervals. Predeclare a margin only if the claim is “retention” or “no meaningful loss.” Review1 imposes the margin before the claim exists.
- Latency and throughput are secondary. They should be measured if the implementation already exposes them cheaply, but they should not gate the first memory claim. Peak resident memory and actual expert-parameter bytes are the primary resource end points.
- Profile stability is analysis completeness, not submission authorization. Only explicit user authorization, live entitlement, and a measured price authorize computation. Stable cards and unstable cards alike require that gate.
- Review1 demands five missing clean next-state caches and independent label generation. Neither is necessary for the memory-first question. Existing all-seven observation summaries and per-expert profiles can support frequency/layer accounting; GPT and Qwen3.6 transition results are narrower secondary evidence. Do not relabel observation summaries as anticipation caches or generate five new caches merely to complete a routing story.
- It proposes another benchmark-correctness prediction analysis. Historical R3-E B4 already covers that estimand with 508 questions/495 families and a precision study. A new accuracy analysis should estimate accuracy under frozen retention masks, not repeat correctness prediction.
- It treats mask-only screening and structural deletion as if one must subsume the other. Masking can screen routing/fallback semantics but frees no parameter bytes. Actual deletion, quantized storage, or offload is required only for the resident-memory claim.
- It lets “actual weight removal, native-k preservation, fallback dispatch, p50/p95 latency, and noninferiority across modes” all become prerequisites. That is a campaign, not a bounded first result.

**2. One central question and package contract**

**Central question:** *At one predeclared resident expert-memory budget per checkpoint, what held-out benchmark accuracy is obtained by a frozen state-associated expert-retention mask compared with a strong frequency/REAP-style mask, and what is the measured joint accuracy–resident-memory point across all seven checkpoints?*

The contribution is not frequency pruning or routing diagnostics as such; EAC-MoE, REAP, MoNE, and Yoon et al. already occupy those spaces. The narrower positive contribution is a controlled, all-seven, resource-accounted test of whether existing automated-state associations add retention value over strong frequency/activation baselines at equal measured resident bytes. The paper reports the estimated frontier, not a promise of improvement.

**Already implemented:** `correlation-generate/forward/analyze/dynamics`, `moe-guiding`; artifact-bound class/routing/combined analysis; transitions/dwell/re-entry; per-expert tables; grouped predictions; matched contrasts; candidate cards; resource estimates; the 32,529-attempt historical benchmark corpus with 42 scored model/benchmark cells; all-seven present-class readout; GPT/Qwen3.6 next-class evidence.

**Missing:** a canonical parameter/storage manifest (expert tensor names, shapes, dtypes, bytes, shared experts, router storage); a family-keyed join from checkpoint/attempt/question to expert profiles and class states; a frozen-budget mask API; a loader adapter that actually changes resident expert storage; a paired accuracy-plus-memory evaluator; and an availability audit for REAP-style per-expert activation/output norms. All-seven next-state caches are not a package prerequisite.

**Contract:** `source audit → candidate ledger → frozen mask → mask-only semantics check → resident apply → paired evaluation`. Outputs carry statuses: observational, mask-screened, resident-measured, or undetermined. The package never auto-deletes experts or reports a memory saving from a mask.

**3. Two next computations**

**A. Saved-data retention ledger—only if the package cannot already emit it.**
- **Claim:** for each checkpoint, at one frozen budget, an auditable mask and projected resident-byte delta exist, with state-associated rare experts separated from frequency effects.
- **Reuse:** existing routing/profile/class-transition summaries, historical benchmark metadata for question/family identity only, and checkpoint parameter metadata. Do not use benchmark correctness to select experts.
- **Exact missing input:** the canonical family-keyed join and actual per-expert storage manifest. State-conditioned summaries must be used only where they exist; otherwise mark that arm unavailable.
- **Estimator:** family-held-out conditional usage lift for each expert relative to frequency, layer, position, and matched-random nulls; mask overlap with frequency/REAP-style scores; projected bytes from the manifest at the frozen budget. This is selection/resource analysis, not an accuracy estimate.
- **Units/split:** question families; attempts/tokens clustered within family. Freeze train/validation/test families before scoring. Selection uses train/validation only.
- **Comparisons:** global frequency, layer-balanced frequency, layer/frequency-matched random, and REAP-style if exact activation norms exist. If they do not, do not reconstruct REAP from selection counts.
- **Uncertainty:** family bootstrap for stability and overlap, FDR or simultaneous intervals across experts within model.
- **Cost/decision:** saved-data/CPU analysis only after artifact availability and join integrity pass. If support is inadequate, do not generate five caches; retain the frequency-only result.

**B. One bounded resident-pruning qualification/evaluation path.**
- **Claim/exhibit:** paired held-out accuracy and measured resident expert bytes for the frozen state-aware mask versus frequency/REAP-style mask and unmodified checkpoint at the same budget.
- **Reuse:** masks from A, historical benchmark questions and scoring provenance, the qualified loader.
- **Exact missing input:** actual modified-model outputs and measured bytes. This cannot be inferred from routes or masks.
- **Estimator:** per-question-family paired Δaccuracy and Δresident expert bytes; optionally Δlog-loss. Report peak resident memory separately from parameter bytes.
- **Units/split:** family is the independent unit; repeated attempts are clustered. Candidate selection/tuning occurs only on non-evaluation families; evaluation runs once with frozen masks.
- **Comparisons:** unmodified model, frequency/REAP-style mask, frequency/layer-matched random mask. Preserve native-k and fallback semantics or report the change explicitly.
- **Uncertainty:** paired family bootstrap, simultaneous across the seven model estimates rather than treating them as independent replications.
- **Cost prerequisites:** live entitlement, measured baseline load/evaluation price, and measured peak memory; qualify one checkpoint first. Expand to other checkpoints only after each measured cost is authorized. Do not invent the budget or claim 96 families are sufficient; choose the budget from target capacity/measured baseline before outcomes and justify support from observed family counts.

**4. Rare state-associated experts and loader necessity**

A rare expert justifies retention beyond global frequency only when it is below the frequency cutoff but shows a reproducible, prefix-only, family-held-out conditional lift for a state relative to frequency/layer/position-matched nulls, and that lift is incremental beyond expert load and nuisance features. Qwen3.6 L28/E189 and E9 are candidates, not established retention wins. The observation must be made on selection families only; benchmark correctness, test questions, and future outcomes cannot enter the score. Split shared questions consistently across models.

Mask-only screening can test hook behavior, fallback semantics, and whether frozen candidates destabilize routing. It is scientifically useful as a screening step, but it does not demonstrate resident-memory reduction. Actual structural/quantized/offload implementation is necessary for the memory claim because only it changes resident bytes. Deletion, quantization, and offload must be reported as distinct modes. The minimum package boundary is mask selection plus a truthful parameter/resource ledger; a separately qualified loader is the minimum boundary for claiming deactivation usefulness.

**5. Stop/continue decision**

Continue to resident evaluation only if A yields a frozen mask at one budget, observed family-level support is adequate without asserting an unverified sample-size rule, the state-aware signal is not explained by frequency/layer/position, and a loader plus measured price/entitlement exist. Stop if the join, storage manifest, loader, or price is missing: deliver the all-seven observational/resource ledger and mask-only screening, explicitly without an accuracy or memory-reduction claim.

Regardless of sign, report the joint estimate and Pareto points: Δaccuracy with intervals, Δresident bytes, mask overlap, and decision status. If state-aware retention does not beat frequency/REAP, the package defaults to the strong baseline and the paper presents the calibration result—not a failure-centered thesis. For October 15, prioritize A across all seven and one qualified resident evaluation; all seven remain in the design, with unmeasured resident cells marked as such rather than silently dropped.