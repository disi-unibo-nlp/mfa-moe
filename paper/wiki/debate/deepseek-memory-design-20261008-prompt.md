# Independent adversarial review: memory-oriented MoE analysis package

Use no tools; review only the supplied material. The user explicitly requests DeepSeek V4.1 Flash/max debate at api.deepseek.com. No credentials or full private reference documents are supplied. Do not submit jobs or make changes.

Your task is to challenge the first review and arrive at a smaller, source-aware design that serves the explicit user objective: REDUCE RESIDENT EXPERT MEMORY FOOTPRINT, MEASURING ACCURACY JOINTLY. No external human validators and no failure-centered fallback paper. All seven checkpoints remain in scope. Results needed by October15. Do not turn correlation into causality, omit unfavorable evidence, require a new annotation campaign, or suggest unpriced generic GPU runs.

New source findings since the first packet:
- A complete historical/card-profile benchmark corpus already exists for ALL SEVEN,32529 attempts total,4647/model,1547 shared questions. Saved42 model/benchmark cells have strict/adjudicated accuracy and question-bootstrap intervals. Caps/runtime profiles differ, so pooled scores are descriptive rather than an equal-budget ranking. The earlier accepted J1 audit is98/100 with2 Minerva errors; the new v3 grader FAIL does not retroactively invalidate this differently bound historical scorer. The seven-model baseline table can be imported without generating another benchmark.
- Existing package entry points include correlation-generate, correlation-forward, correlation-analyze, correlation-dynamics, moe-guiding. Dynamics has artifact-bound class/routing/combined analysis, gap-aware transitions/dwell/re-entry, per-expert tables, grouped predictions, matched expert contrasts, candidate cards and resource estimates. The package task is integration around reusable modules, not constructing every analysis anew.
- Historical R3-E B4 correctness-prediction analysis already exists:508 questions/495families, primary held-out log-loss change-.000729, nominal interval[-.005313,+.004384], with a precision study. Do not propose repeating it merely as a new routing-correctness experiment. A stronger/new test needs an exact missing estimand and adequate support.
- Prior deactivation is also real:Qwen-only13families/156branches,256-token counterbalanced negative-bias/force-off screen,144/156 semantic pairs. Force-off target-native -3.85pp, simultaneous[-23.08,+11.54]. It uses the fully loaded model, all branches capped256, and measures neither answer accuracy nor resident-memory reduction. It cannot establish useful weight pruning but should not be rerun under another name.
- Active all-seven v3 supports positive singleton pulses and rejects negative bias. Deactivation/structural pruning needs a separately qualified operator/loader; do not modify sealed sources under active jobs.
- Available all-seven observation summaries and per-expert frequency profiles are not the same as five missing clean next-state anticipation caches. Distinguish these.

Literature checked on primary pages (abstract-level grounding, not a reproduction):
- EAC-MoE, Chen et al., ACL2025, https://aclanthology.org/2025.acl-long.633/ combines router selection calibration for quantization with task-conditioned expert frequency pruning.
- REAP, Lasby et al., arxiv2510.13999v3, https://arxiv.org/abs/2510.13999 ranks router-weighted expert activation norms and evaluates structured pruning on generative tasks. Norms/per-expert outputs may not exist in our old captures; availability must be checked, not fabricated from selection counts.
- MoNE, arxiv2507.00390 uses frequency and output variance with lightweight replacements.
- When Are Experts Misrouted?, Yoon et al., arxiv2605.07260 https://arxiv.org/abs/2605.07260 studies fixed-model equal-compute route alternatives on reasoning trajectories (includes GPTOSS/Qwen330B), and router-only adaptation.
These papers mean frequency pruning and routing diagnostics alone cannot be advertised as novel state-of-the-art pruning. A state-conditioned/transition-informed allocation report plus controlled test against appropriate existing baselines may be a narrower contribution.

Required response (<=1600words):
1. List substantive errors/overrequirements in review1. In particular, a noninferiority margin is unnecessary for honest joint accuracy/memory estimation unless retention is claimed; latency/throughput are secondary for this user's first memory claim; profile stability is not submission authorization. Distinguish analysis completeness from online controller qualification.
2. Propose ONE central research question and package contract, with explicit already-implemented/missing pieces and positive, testable contribution beyond existing pruning papers.
3. Specify at most TWO next computations: one saved-data analysis (if actually missing), one bounded resident-pruning qualification/evaluation path. Each must name source reuse, exact missing input, estimator, independent units, comparisons including a strong available pruning baseline, split discipline, and measured-cost prerequisites. Prefer one predeclared retention budget before any sweep; do not invent its numerical value or claim96 families adequate.
4. Explain what source-derived observation would justify retaining rare state-associated experts beyond global frequency; how to avoid selecting on correctness/evaluation leakage; whether actual structural/quantized loader implementation is scientifically necessary vs a separately described mask-only screening step.
5. Give an explicit stop/continue decision that leads to a useful reported contribution regardless of effect sign, while respecting the user's analysis-first framing and Oct15 deadline.

# Source summary from first review
# Analysis-first MoE package and Computational Linguistics contribution review

You are an independent scientific reviewer, configured as DeepSeek V4.1 Flash with max reasoning. Use no tools and inspect no other files. Review only this supplied research summary. The user explicitly requested this review through the existing api.deepseek.com provider. No credentials, raw private manuscripts, or private reference PDFs are included. Do not submit jobs, write code, or change files. Your recommendations are critiques, not experimental results or authorization.

## User's latest direction

We target Computational Linguistics, with results needed by October 15, 2026 (today October 8). All seven previously studied MoE models remain required. No external human validators. The user rejects a negative-result paper fallback as the present research focus and requests careful model analysis first: sound correlations, useful state transitions, expert usefulness, and integration of already computed math benchmark results. Ultimately a reusable package should accept a local MoE model and benchmarks, run analyses that reveal supported patterns, and help users decide whether to deactivate experts or constrain routing locally. The user has explicitly selected **reduce expert memory footprint, measuring accuracy jointly** as the first deployment objective. Intervention support is a goal, but a package/report is not itself scientific evidence that deletion or control is useful. We are converting the paper to CL format and retrieving existing results before defining new work. Fresh runs need a necessary missing input, exact claim/exhibit, independent support and uncertainty, controls, and measured bounded price. We preserve recorded failures as limitations; we are not asking for a failure-centered paper or changing gates to find positives.

## Existing evidence and exact scope

Seven checkpoints: GPT-OSS-20B, Nemotron-3.5-Lightning-30B-A3B-NVFP4, Qwen3.5-35B-A3B-GPTQInt4, Qwen3.6-35B-A3B-FP8, Gemma-4-26B-A4B-NVFP4, GLM-4.7-Flash, Qwen3-30B-A3B. Routed layer counts 24/23/40/40/30/46/48; expert pools/native k 32/4,128/6,256/8,256/8,128/8,64/4,128/8. Shared experts and selector semantics vary; logit/probability/sigmoid/correction-bias spaces are not interchangeable. Seven checkpoints, with related Qwen architectures, are not seven independent benchmark samples.

Latest Sep29 cross-4 observational inventory contains all seven A and B roots. A has 1547 one-per-question attempts/model with1545-1547 scored and median observed sentence labeling shares36.36/24.29/8.61/7.93/18.12/16.44/17.81 percent. B has895 attempts total but51 distinct questions in the cross-model union. Sparse missing sentences break adjacency. A/B summaries include benchmark scores, routing/hidden/position summaries, class transitions and expert profiles; source-level availability and table integration are being checked. The older paper cohort has4647 attempts/model and includes32 attempts for each AIME24/AIME25/AMC23 question; do not mix its accuracy with A's sampling contract. Saved rescoring changed many previously incorrect grades. Current paper sources have benchmark tables, but all-seven versus three-model and scoring-provenance distinctions require reconciliation.

Clean R3-D present-class routing readout is already positive for all seven, each522 questions/509 duplicate families. Candidate-minus-baseline log-loss differences and simultaneous CI7:
GPT -0.07858[-0.09567,-0.06179]; Nemotron -0.13721[-0.15628,-0.11619]; Qwen3.5 -0.14090[-0.15359,-0.12781]; Qwen3.6 -0.13899[-0.15276,-0.12601]; Gemma -0.14866[-0.16136,-0.13692]; GLM -0.13679[-0.15302,-0.12035]; Qwen3-30B -0.09427[-0.10929,-0.08100]. The saved procedure has model-specific simultaneous intervals; it is not a joint causal test. All seven lexical-null excess intervals also exclude zero. Labels are automated visible reasoning-episode classes (Read, Analyze, Plan, Implement, Explore, Verify, Monitor), not latent cognitive ground truth.

Clean next-class anticipation exists for GPT (509 questions/10209 adjacent pairs, gain -0.02073 nats[-0.02548,-0.01601]) and Qwen3.6 (494/2492, -0.05570[-0.06944,-0.04154]), with positive contained-prefix sensitivities. Baselines use past text/history and training-only estimates of current class. Equivalent completed prepared caches/results were not found for the other five. Present-class readout, next-class prediction, and token-scale online intervention timing are distinct.

Grouped hidden-state probes on38 external DeepSeek-R1 SAT responses/3087 units exist for Qwen/GPT/Gemma, with response-disjoint nested layer selection and AUROC0.852/0.780/0.788. These are teacher-forced external-text hidden-state results, not seven-model native routing results. Raw grouped result availability is incomplete in this checkout; tables/source hashes exist. No repeat is justified merely for fresh provenance.

Whole-trace confidence/correctness directions vary with model and dataset. A saved late-boundary association survives one adjustment (d=-0.2983,q=.004879), whereas no feature survives correction in the common reasoning range. A routing-gap/length association near-.45 approaches zero at matched absolute positions. Cross-4 pooled adjusted Explore fraction r=-.06241[-.08800,-.03674], same direction7/7, is descriptive; pooling does not automatically account for shared questions. Confidence and entropy are algebraically dependent. No blanket recommendation to suppress Explore or low-confidence experts follows.

Qwen3.6 conditional L28/E189 and E9 are nominated by matched next-Verify membership differences+.08118/+.06296 in48 discovery families, positive in40/44 families,483 matched positive windows. Features include completed source text and target is a future episode label. Saved native profiles include48 traces/6905 records/42 transitions. Historical A/B intervention inclusion changes+.1376/+.2940 demonstrate engagement. Historical FRESH123 families/134 starts/3480 assignments includes verification and commitment, with2920 valid reader pairs/560 unknown. Force-native verification +5.5pp simultaneous95%[-4.5,+15.5], force-random +11pp[-.5,+23.5]. Missingness bounds respectively[-16.5,+24.5] and[-7.5,+30.0]. No FRESH contrast excludes zero in its28-comparison family. Dense historical labeling is fully reconciled187524 identities,187136 valid/388 unusable, zero unresolved, with saved trajectories/expert turnover; do not redo it.

Rescored legacy global steering has1395 paired attempts/465 problems. Some strong promotion policies severely harm accuracy; weaker actions have no established improvement. Different selector rankings produce different targets. These are evidence and cautionary controls, not the intended new paper headline. Historical short capped continuations do not measure original-prompt policy utility.

Current verification-v3: all seven backend adapters pass14/14 engineering cases. Five48-family native discovery rosters and128-prefix frames are complete, two ongoing. Four full256-draw candidate-recognizer audits fail before nomination/treatment: eligible/already-checked confirmation22.41/18.75,25/10,25.52/41.67,10.10/33.33 percent for GPT/Nemotron/Gemma/Qwen3-30B. Requirement70percent and10 apparent families each. Two fixed Qwen3.8-27B draws, unknown/disagreement unconfirmed. No independent human calibration and none will be required in the revised research plan. Existing reader-relative claim scope must be explicit. Shared J1 equivalence audit97/100 reference matches fails its per-stratum gate; strict grades and unknown residual bounds remain available. Current causal/utility outcomes are unmeasured. Sealed scientific sources and active jobs remain preserved; user requested analysis of potential next Slurm experiments, not unlimited new submissions.

## Review task

1. Give the strongest positive, scientifically supportable analysis/package contribution for CL, using actual existing evidence and explicit automated-label scope. Separate already established results from hypotheses and new tests. Do not make negative findings the organizing thesis, hide them, or promise expert usefulness from correlations.
2. Identify at most three precise missing analyses/experiments worth doing within one week. For each: exact paper exhibit/claim, reusable source, missing input, estimand, baseline/control, independent sampling unit, held-out selection, uncertainty/multiplicity, and a decision rule for whether new generation is necessary. Do not invent costs or promise sample sufficiency; tell us what prices/support must be verified. Prefer saved-data analysis where it can answer the claim.
3. Stress-test whether state transitions and expert profiles can give user-relevant guidance without relying on external annotators. Include nuisance baselines, sparse-label selection bias, shared-question dependence, architecture grouping, and the distinction between benchmark score and reasoning-label outcome.
4. What exactly would establish useful expert deactivation for local deployment? Distinguish masking selection, removing weights, maintaining native k, fallback dispatch, quality, parameter memory and measured latency. Recommend the minimum defensible package boundary.
5. State what the paper/package must NOT promise, and give a concise adversarial argument a reviewer could make plus the evidence needed to answer it. Keep the reply under1800 words and prioritize rather than listing every possible experiment.


# First reviewer response to challenge
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