# steering: decisions

## whole-reasoning-design: Whole-reasoning routing efficiency
- status: active
- decided by: user (2026-10-20)
- applies to: Whole-reasoning GPT/Qwen efficiency experiment and corrected main/strong CPU analyses.
- not to: Further 512-token rescue work, manuscript edits, pushes, outcome-driven tuning.
- chosen: Alignment v3 control-feasible global sign-stable RD nomination with exact layer/quintile/role controls at both doses; original prompts, dual score override at epsilon 0.01, prompt and reasoning steering until closure, native and native_repeat. Continuous 64-slot admission with native and efficiency cells before other promote cells. Relative full-allowance canaries, two GPT and seven Qwen shards, immutable releases, fresh stores and six simultaneous bounded contrasts. Monitor deadlines/errors never stop production; actual CPU usage remains bounded separately.
- rejected: Full-top-k promotion, new 512-token generation, routine dose sweeps, mixed doses per model, treating the exploratory sign-transfer link as a preregistered replication.
- evidence: User's Alignment v2, 2026-10-20: "This message replaces that plan and the 2026-10-20 'final implementation plan'." Exact endpoints, debate, gates and authorization are recorded in report/whole-reasoning-20261020/README.md. New-work ceiling 72 GPU-hours plus retained r3 engineering 0.346111; CPU ceiling 16 core-hours. Science cutoff 2026-10-23 05:30 Europe/Rome.
- keywords: whole, efficiency, routing, bootstrap, nomination, SteerMoE, rescue
- supersedes: rescue-population

### Alignment v3 amendments (user, before any whole GPU outcome)

User: "adopt your control-feasible nomination. Do not use tertiles" and "This
session is the only owner and submitter." Keep all v2 scientific parameters
except the explicit v3 amendments in report/whole-reasoning-20261020-r3/README.md.
The absolute-RD greedy walk reserves distinct controls jointly across roles;
all 35 main and 35 half GPT promote selections must equal the saved DP proposal.
All Qwen and GPT suppress targets retain exact ranks. Record retained RD per
cell and native-frequency, changed-membership, L1 and deactivated-weight checks
by model/arm. Targets remain the larger frequency perturbation (reviewed GPT
means: controls 0.290 vs targets 0.375). Generic disruption historically
lengthened reasoning and cut accuracy, so the residual mismatch works against
an efficiency claim. The proposal and historical means are documented in
report/whole-reasoning-20261020-r2/CONTROL_FEASIBLE_RD_BOTH_DOSES.json and the
user's Alignment v3 review; they are not whole-experiment results.

The original whole r3 canary measured an instrumentation bottleneck before
any unit completed:26.5785 aggregate tokens/s across48GPT rows. The engineering
correction batches the same row-wise equations, per-layer additive counters
and full audit records; no scientific setting changes. All failed qualification
costs remain charged. Production walltimes may be shortened below their
approved165/240-minute maxima at admission so every committed maximum plus
actual terminal costs fits72GPU-hours. This is budget admission, not an
outcome-driven dose or nomination choice. See the original r3 PERFORMANCE_HALT
and the batched continuation's PRIOR_USAGE.json.

GPT61581237 later failed the unchanged strict rank-capture hash check. Its
failing rank payloads were not retained, and a second release in cleanup masked
the first exception. A separate15-minute/two-GPU engineering command now saves
both payloads losslessly, records consumed release state and preserves the first
error. It cannot qualify or submit production. Full allocated CPU validation
precedes it; actual diagnosis and repaired allocated qualification are still
required. Every charge remains inside the original72/16ceilings, together with
the existing Qwen graph's maximum reservations. No scientific parameter changes.

Canaries retain the same eight families/48 rows with each production allowance.
Compare with same-run native/native_repeat: cap-rate excess >0.15, geometric
length ratio >1.40, repetition excess >0.10 fail. GPT45/Qwen90-minute canaries
and 2 GPT165/7 Qwen240-minute two-GPU shards reserve 71.5 GPU-hours. No shard
is held. The half-dose fallback must fit remaining charged usage and maxima.
The monitor cannot stop GPUs at its own deadline, on a CPU-budget exit or on
an exception. Scheduler/quota exceptions log and retry after 60 seconds.
Extraction is queued with production, analysis after both extractions.

Non-nominated cells are not_in_design and excluded from every composite;
efficiency-directed cells must all be nominated. Extraction failures are
reported explicitly, sentence coverage failures make that unit unknown, and
reasoning closed before a cap is point-observed while fixed-budget accuracy
remains zero. The exploratory link uses shared 10,000 draws/seed20261015, which
differs from the preregistered 2,000/20261017. Corrected main analysis records
every bounded activate:log_total row and its reason.

### Frozen endpoints (user, verbatim)

**Predeclared efficiency-directed cells** (declared 2026-10-20 before any whole-arm outcome; only the promote side
derives from the pre-registration; the suppress side is a new hypothesis):
- GPT: promote Verify, promote Monitor, suppress Read/Analyze/Plan/Explore.
- Qwen: promote Verify, promote Monitor, suppress Analyze/Plan/Implement.

**Primary: 6 contrasts with one shared simultaneous critical value.**
- Per model, average equally over its efficiency-directed cells. Pooled = 0.5·GPT + 0.5·Qwen.
- Token benefit = −[log(1 + reasoning tokens)_target − log(1 + reasoning tokens)_random].
- Strict-accuracy difference, target − random. A capped run counts as incorrect for this fixed-budget endpoint.
- Two endpoints × {GPT, Qwen, pooled} = 6 contrasts.
- Inference: 10,000 benchmark-stratified family bootstrap draws, seed 20261015, equal class/direction weights within
  model, max-|t| simultaneous envelope over all point AND bound components.
- **Bounds.**
  - Capped run with reasoning still open: log reasoning ∈ [log observed, log total_allowance], using each request's actual allowance, not a
    hard-coded 32,768.
  - Missing run: [log 1, log allowance] for tokens and [0,1] for accuracy.
  - Token-benefit lower bound = −(target_upper − random_lower).
- **Decision rule.** "Efficient steering" only if the simultaneous lower bound of token benefit is > 0 AND the
  simultaneous lower bound of the accuracy difference is > −5 pp.
  - Tokens shorter but accuracy bound below −5 pp: speed–accuracy trade-off.
  - Target ≈ random but both differ from native: generic disruption.
  - Never claim equivalence from a null interval.

**Mandatory companions** (same bootstrap):
- target−native and random−native for both endpoints;
- native_repeat−native as the noise reference;
- cap rate, answer presence, 16-gram repetition rate and total tokens, by arm.

**Pre-registered link: exploratory, not a replication.** Use this wording: "The frozen preregistered test
(activation arms, central strength, log total tokens, main design) is not estimable in this design. We report an
exploratory transfer of the frozen beta_tok signs to the promote arms." Report it with log(total tokens) (the
historical comparand) and log(reasoning tokens) (the declared new endpoint).

In the corrected **main-study** analysis, restore the pre-registered `activate:log_total` as the link's primary and
keep `log_reasoning` as a declared secondary.

**Mechanism (secondary).**
- Lexical C10 class occupancy over the generated reasoning, fixed denominator: target−random and target/random−native
  for the steered class.
- Signed 7×7 response matrices.
- Per-layer dose for every unit: active prompt and generated positions, changed-membership fraction,
  native vs executed target weight, L1.
- Full per-token captures only for 32 pre-selected audit units per model.

**Statistical planning.**

Alignment v3 supersedes the between-family fallback below: use the main-study
native_repeat−native within-family differences d, sigma_w=SD(d)/sqrt(2), then
composite SD=sigma_w*sqrt(2/m). Accuracy comes from main-study driver outcomes
or re-scored labels, never legacy trace is_correct. Mark transport SUSPECTED;
no gate uses this planning value. Preserve cross-model family covariance in
the pooled calculation. The following original v2 expectations are historical.

Allocated CPU61577861 found no main-study native_repeat assignments/outcomes
for either model (129native, zero repeats per manifest). SD_MDE.json in the
whole r3 report therefore records zero paired observations and unavailable
SUSPECTED SD/MDE values. The implemented paired formula is retained; no
between-family estimate, rescue-window estimate or invented replicate is
substituted. This evidence limitation does not block the authorized canaries.
- Composite family-level SDs from main-study native outcomes: about 0.19 / 0.16 log-tokens and 0.10 / 0.07 accuracy
  (GPT / Qwen). Recompute these and record them in a receipt before production. If you cannot reproduce them, use and
  report the native-only SDs.
- MDE at 129 families with conservative 6-contrast planning (z ≈ 3.48):
  - GPT about 6% tokens and 3.0 pp;
  - Qwen about 5% and 2.3 pp;
  - pooled about 4% and 1.7 pp.
- These transport from main-study data, so mark them SUSPECTED.

## gentle-routing-nudge: Outcome-selected gentle routing nudge
- status: superseded by class-conditioned-outcome-design
- decided by: user (2026-10-22)
- applies to: The outcome-steering experiment (outcome-steering-r1) on gpt, qwen330b and nemotron; extension to the other models with cohort-B captures only if the three-model result is positive.
- not to: Changes to the finished whole-reasoning releases r7-r17 or their analysis; manuscript claims before a sealed receipt.
- chosen: Select experts by outcome on campaign-v3 cohort B (per-attempt usage per reasoning token, z-scored within question, mean wrong-minus-right over mixed-outcome questions, sign stable under leave-one-question-out). Exclude the shared core (top 5% native frequency per layer) and the most execution-specific experts (union over folds of the top 10% positive Implement/Analyze risk differences). Take stable positive pairs until 1.2% of native routing mass (5 to 40 pairs; fill to 5 by score with an unstable_fill flag if needed). Families whose question is in cohort B use the set selected without that question. Subtract delta in {0.4, 0.8, 1.6} from the selected experts' router logits at every prompt and reasoning token, keeping native top-k and renormalisation. Matched random controls with the same layers, frequency quintiles and exclusions. Reuse the base releases' native and native_repeat rows. Length-limit baseline: cut each family's native reasoning at 50/70/85% of its length, append the closing suffix and generate only the answer (allowance min(2048, remaining)).
- rejected: Class-risk-difference nomination (the 87-cell whole design found no helpful experts); forced on/off membership; excluding every expert with any positive Implement/Analyze link (empties the control pools); fresh native reruns.
- evidence: User answers 2026-10-22 (3 models first, length-limit baseline, three strengths 0.4/0.8/1.6, top-10% execution protection, reuse natives); analysis in /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/claude-review/why-steering-20261022/; chen2026beyond ("Beyond the Trace") reports shorter and more accurate GPT-OSS-20B output with five outcome-selected experts at a -0.8 logit shift.
- keywords: outcome, nudge, steering, budget, early-exit, routing

### Extension rule (declared 2026-10-22, before any nudge outcome)

The user asked to extend to every model if the three-model result is good, and to schedule every
step automatically. A CPU job after the r1 analysis applies `moe_exp.outcome_steering.decide`: extend
when some first-wave model and delta is shorter than matched random experts (bounded simultaneous
token interval above zero) with accuracy against unedited routing not below -1 point, or more
accurate than random experts (accuracy interval above zero). If met, it launches
outcome-steering-20261022-r2 for gemma, glm, qwen35 and qwen36 (LFM2.5 has no cohort-B captures),
GLM with more shards because it decodes about four times slower.

## class-conditioned-outcome-design: Class-conditioned outcome steering design (r2)
- status: active
- decided by: user (2026-10-23)
- applies to: The second outcome-steering campaign (outcome-steering-r2, attempt-selection-by-routing): native captures on all eight models, class-conditioned expert selection, the GPT-OSS nudge sweep on question half A, the automatic pick and its confirmation on half B, and the attempt-selection analysis. The other seven models' sweeps are decided jointly after the GPT-OSS half-B receipt.
- not to: Release r1 and its analysis (kept unchanged, reported as the broad-edit test); the whole-reasoning releases r7-r20; GPT-OSS low/high reasoning effort (a later GPT-only extension); manuscript claims before a sealed receipt.
- chosen: Captures: the same 300 questions for every model (aime24 30, aime25 30, amc23 40, olympiad 100, minerva 50, math500 50; digest-ranked within benchmark, one question per family, no outcome read), 6 native attempts each, medium effort, original prompt tokens, 32,768-token completion cap on every row of r2, native top-k ids and weights of every reasoning token at every routed layer. Labels: lexical readout v1 (TF-IDF), out-of-fold by FOLDS.json; no judge. Score per class c: within-question z-scored usage on tokens labelled c, mean of wrong minus mean of right, over questions with both outcomes; capped or unknown attempts excluded. Sets: one per class (7) plus a class-blind set (GPT-OSS: Beyond the Trace's five pairs L12-E23, L12-E12, L12-E24, L16-E11, L20-E0; other models: the plain outcome set). Set rule from related work: candidates only in the three routed layers at relative depth 1/2, 2/3, 5/6 (n//2, 2n//3, 5n//6; 12/16/20 on GPT-OSS, Beyond the Trace); best positive pairs until they carry 1.2% of those layers' native gate weight per reasoning token, at most 10 pairs (the paper's own definition, confirmed by the ELM grounding check of 2026-10-23: "1.2% of the gated mass across the three recorded layers", gate weight accumulated over routed assignments; the slot share is reported beside it); committee experts (top 5% native use per layer, Illusion of Specialization) and the top-10% Implement/Analyze experts never nudged; logit shift with native top-k and renormalisation (SteerMoE-style operator). Each evaluation family uses sets selected without its own question. Arms: target nudge and two disjoint matched random draws (same layer, native-frequency quintile) per set and shift; shifts 0.4, 0.8, 1.2 at every prompt and reasoning token; unedited arm = the 6 capture attempts; stop-early baseline at 50/70/85% of a native attempt. Halves: the 300 questions split by a benchmark-stratified digest into A and B. Pick rule (frozen now): among half-A cells with accuracy against native no worse than -5 points (point estimate) take the most negative tokens-against-mean-random point estimate; if none qualifies, the best accuracy-against-random estimate. Half B confirms that cell and the class-blind set at its best shift. Primary reported contrast: tokens against the mean of the two random draws with accuracy against native noninferior within 5 points, on half B; every half-A cell is exploration; no success threshold gates anything. Attempt selection on the captures: routing consensus, outcome-profile and class-mix scores against majority vote, random, shortest and longest.
- rejected: r1's all-layer set of 20-40 experts (acts like a random edit); the leave-one-question-out stability filter (passes as many pairs under shuffled labels as under real ones); cohort B as selection data (37 or fewer questions); the 'middle third' window (excludes two of the paper's five experts); a slot-share target (the paper's five carry 5.35% of their layers' top-k slots and 0.67% of all slots on cohort B; the paper measures gate weight, so the slot share is only reported); a judge pass on the captures; an automatic gate before the other seven models.
- evidence: User answers in the grilling session of 2026-10-23 (Q1-Q25) and the plan approval of the same day; ELM grounding check /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/claude-review/outcome-r2-grounding-20261023/report.md (Beyond the Trace App. A.6: gate-weight share in layers 12/16/20, Low effort for selection and edit, 12 evaluation problems not stated to be held out, shifts 0.3/0.8 at every token, equal-dose random sham; SteerMoE, RICE, Illusion of Specialization, Command A+ modularity and causal-audit designs tabulated); paper/wiki/OUTCOME_STEERING_REVIEW.md; footprint check on /leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs/outcome-steering/gpt/USAGE.npz (paper's five: 0.0535 of window slots, 0.0067 of all slots).
- keywords: outcome, nudge, steering, class, capture, attempt selection, Beyond the Trace
- supersedes: gentle-routing-nudge
