**Bottom line.** The defensible positive paper is a seven-model, within-model audit of MoE routing as a correlate of automatically extracted visible reasoning-episode classes, delivered with a reusable analysis package that separates readout, prediction, and intervention. The package is a supporting artifact, not evidence that deactivation works. Expert deletion remains an unmeasured deployment hypothesis.

## 1. Strongest supportable contribution

**Established, observational.** Across all seven checkpoints, present-class routing readout improves over the saved baseline, with model-specific simultaneous intervals excluding zero, and all seven lexical-null excess intervals exclude zero. Label scope is explicit: automated visible episode classes (Read, Analyze, Plan, Implement, Explore, Verify, Monitor), not latent cognition. Next-class anticipation is established only for GPT and Qwen3.6. Qwen3.6’s conditional L28/E189 and E9 nominations show matched next-Verify membership differences, but they remain correlational candidate hypotheses. Historical FRESH and legacy steering show engagement and caution: different selector rankings target differently, and some promotion policies severely harm accuracy.

**Hypotheses, not established.** Stable expert profiles or transitions can rank safe-to-remove experts; phase-aware routing can guide intervention timing; benchmark correctness can be improved by constraining routes; the same profile logic transfers across architectures. None follows from current evidence.

**CL contribution.** “Routing-state observability across heterogeneous MoE checkpoints”: a reproducible method for extracting architecture-aware route statistics, evaluating them against nuisance baselines, and reporting expert/transition cards with automated-label limits. Its scientific finding is that routing carries phase information across all seven models, with narrower next-phase evidence in two. Its package contribution is standardized, provenance-preserving analysis—not an automatic pruning method.

## 2. Three analyses worth doing within one week

**A. Benchmark-integrated routing readout (saved-data first).**
- **Claim/exhibit:** Model-specific out-of-sample incremental association between routing-state features and benchmark correctness/score, shown as a forest plot or table alongside present-class readout.
- **Reusable source:** Sep29 A/B summaries, benchmark tables, source-level availability/join audit. Use the older 4,647-attempt cohort only as historical sensitivity, never pooled with A.
- **Missing input:** Audited per-attempt joins for routing features, benchmark score, scoring provenance, and question identity; raw grouped results are incomplete in places.
- **Estimand:** Paired change in held-out log-loss/AUC or score from adding routing features to nuisance and history baselines.
- **Baseline/control:** length, position, token entropy/confidence, lexical features, class priors, past text/history, label-shuffled null.
- **Independent unit / held-out selection:** question or question family, not token or attempt; nested grouped CV with feature/profile selection inside training folds.
- **Uncertainty/multiplicity:** bootstrap or model-based intervals clustered by question; simultaneous across models and feature families. Do not treat seven checkpoints as seven independent samples.
- **Decision rule:** If the join yields adequate observed independent families and informative intervals, do not generate. Otherwise weaken the claim or generate only the exact missing cell after verified entitlement and measured price.

**B. Transition and expert-profile stability under matched nulls.**
- **Claim/exhibit:** Held-out stability and calibration of candidate experts and transition windows, with matched-null effects; candidate cards for L28/E189, E9, and any other pre-specified candidates.
- **Reusable source:** Qwen3.6’s 48 families, 6,905 records, 42 transitions; saved trajectories/turnover; completed discovery rosters; historical engagement results only as external context.
- **Missing input:** Equivalent transition caches for the other five models. If absent, do not fit a partial recognizer or claim all-seven next-class.
- **Estimand:** Leave-family-out AUC/precision for next-Verify membership; rank stability; incremental effect over nuisance features.
- **Baseline/control:** random experts/windows matched on frequency, layer, and position; past-label and lexical history; label shuffle. Features must not contain future labels.
- **Independent unit / held-out selection:** family, with windows clustered within family; nomination only on training families.
- **Uncertainty/multiplicity:** family-clustered uncertainty and simultaneous/FDR control across candidate experts and models; sparse labels reported with missingness bounds.
- **Decision rule:** Stable candidates authorize a bounded deactivation pilot; unstable or null candidates forbid a deletion claim. No generation unless an all-seven transition claim is essential.

**C. Bounded deactivation feasibility pilot (only after B).**
- **Claim/exhibit:** Paired quality/resource table by model and mode: masking selection, weight removal, native-k, fallback dispatch.
- **Reusable source:** package harness, candidate list from B, untouched benchmark tasks.
- **Missing input:** Actual modified-model outputs and measured resources; this cannot be inferred from route profiles.
- **Estimand:** Δ accuracy/log-loss versus unmodified; Δ actual freed parameter bytes, peak resident memory, p50/p95 latency, and throughput.
- **Baseline/control:** unmodified model; random frequency/layer-matched removal; candidates selected on disjoint families.
- **Independent unit / held-out selection:** independent question/family; evaluation tasks never used for candidate selection.
- **Uncertainty/multiplicity:** pre-specified noninferiority margin, paired clustered uncertainty, simultaneous across modes/models.
- **Decision rule:** This necessarily needs new generation. Run only if B yields a small bounded candidate set and price, entitlement, and measured support have been verified. Otherwise package capability may remain, but utility is unmeasured.

## 3. Stress test: transitions and profiles

Automated labels are the ceiling. If labels were produced from the same evidence used as predictors, current-label prediction is partly circular; independent label generation or label-blind controls are needed.

Nuisances are strong: length, position, token frequency, episode length, task format, lexical overlap, entropy/confidence, and expert load can mimic “state” or “profile.” Require incremental performance and matched nulls.

Sparse-label selection is nonrandom: missing sentences, truncation, and broken adjacency can select easy episodes. Report eligibility, missingness bounds, and sensitivity; do not silently drop unusable draws.

Shared-question dependence is pervasive: tokens and windows repeat within questions, questions repeat across models, and models are related architecturally. Cluster by question/family and use hierarchical or clearly descriptive cross-model summaries; model-specific intervals are not a joint test.

Expert profiles may reflect frequency, position, layer, or routing normalization, especially with shared experts and different selector spaces. Compare within model or verified comparable spaces. Expert engagement is not necessity.

Benchmark correctness and reasoning-label outcomes are different estimands. Label predictability does not imply benchmark improvement, and benchmark correlation does not imply control over reasoning.

## 4. What would establish useful deactivation

Useful deactivation requires all of the following: a declared mode; candidate experts selected without evaluation data; quality measured jointly on held-out independent questions; actual weight removal, not merely selection masking; native active-set size and normalization semantics preserved or explicitly changed; fallback dispatch specified; measured freed parameter bytes and peak resident memory; and p50/p95 latency/throughput on the target hardware and batch regime. Compare against unmodified and frequency/layer-matched random removal. Report a Pareto curve, but call no point useful without both quality and resource evidence.

Masking selection alone saves no parameter memory. Removing weights can save memory but may change routing, native k, load balance, or fallback cost. Fallback dispatch may preserve quality while retaining memory or worsening latency. The minimum defensible package boundary is an audit-and-simulation harness that emits candidate cards and “supported/unsupported/undetermined” decisions. Only a v1 harness with paired noninferiority and resource measurements can support “deactivate this set.”

## 5. What not to promise

Do not promise latent cognitive states, human-like reasoning phases, cross-architecture equivalence, seven independent replications, causal routing control, safe expert deletion, benchmark generalization, or memory/latency gains from masks. The strongest adversarial reviewer argument is: “Your automated labels are circular proxies, profiles are frequency/position artifacts selected on 48 families, and no actual deletion experiment connects routing to accuracy or resources.” The answer needs label-blind, held-out, nuisance-controlled stability evidence plus paired deactivation measurements with actual memory and latency. Without those, narrow the thesis to routing correlates with automated visible phase labels, and present the package as a measurement tool whose intervention claims are explicitly unmeasured.