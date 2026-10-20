# D-0002: Whole-reasoning routing efficiency
- status: active
- date: 2026-10-20
- decided_by: user
- supersedes: D-0001
- scope: Whole-reasoning GPT/Qwen efficiency experiment and corrected main/strong CPU analyses.
- not_scope: Further 512-token rescue work, manuscript edits, pushes, outcome-driven tuning.
- chosen: Alignment v3 control-feasible global sign-stable RD nomination with exact layer/quintile/role controls at both doses; original prompts, dual score override at epsilon 0.01, prompt and reasoning steering until closure, native and native_repeat. Continuous 64-slot admission with native and efficiency cells before other promote cells. Relative full-allowance canaries, two GPT and seven Qwen shards, immutable releases, fresh stores and six simultaneous bounded contrasts. Monitor deadlines/errors never stop production; actual CPU usage remains bounded separately.
- rejected: Full-top-k promotion, new 512-token generation, routine dose sweeps, mixed doses per model, treating the exploratory sign-transfer link as a preregistered replication.
- evidence: User's Alignment v2, 2026-10-20: "This message replaces that plan and the 2026-10-20 'final implementation plan'." Exact endpoints, debate, gates and authorization are recorded in report/whole-reasoning-20261020/README.md. New-work ceiling 72 GPU-hours plus retained r3 engineering 0.346111; CPU ceiling 16 core-hours. Science cutoff 2026-10-23 05:30 Europe/Rome.
- keywords: whole, efficiency, routing, bootstrap, nomination, SteerMoE, rescue
- paths: src/moe_exp/study, src/moe_exp/final_routing, tests/study, scripts/study

## Alignment v3 amendments (user, before any whole GPU outcome)

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

## Frozen endpoints (user, verbatim)

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

