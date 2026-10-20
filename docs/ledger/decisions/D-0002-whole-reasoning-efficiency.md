# D-0002: Whole-reasoning routing efficiency
- status: active
- date: 2026-10-20
- decided_by: user
- supersedes: D-0001
- scope: Whole-reasoning GPT/Qwen efficiency experiment and corrected main/strong CPU analyses.
- not_scope: Further 512-token rescue work, manuscript edits, pushes, outcome-driven tuning.
- chosen: Original saved prompts, cross-fitted global sign-stable inclusion risk differences, dual score override at epsilon 0.01, prompt and reasoning steering until closure, matched random, native and native_repeat. Six simultaneous bounded accuracy/token-benefit contrasts with equally weighted models and efficiency cells. Immutable release code and fresh bound stores. Mandatory canaries, declared half-dose fluency fallback, projection and hard budget stops.
- rejected: Full-top-k promotion, new 512-token generation, routine dose sweeps, mixed doses per model, treating the exploratory sign-transfer link as a preregistered replication.
- evidence: User's Alignment v2, 2026-10-20: "This message replaces that plan and the 2026-10-20 'final implementation plan'." Exact endpoints, debate, gates and authorization are recorded in report/whole-reasoning-20261020/README.md. New-work ceiling 72 GPU-hours plus retained r3 engineering 0.346111; CPU ceiling 16 core-hours. Science cutoff 2026-10-23 05:30 Europe/Rome.
- keywords: whole, efficiency, routing, bootstrap, nomination, SteerMoE, rescue
- paths: src/moe_exp/study, src/moe_exp/final_routing, tests/study, scripts/study

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
  - Capped run: log reasoning ∈ [log observed, log total_allowance], using each request's actual allowance, not a
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
- Composite family-level SDs from main-study native outcomes: about 0.19 / 0.16 log-tokens and 0.10 / 0.07 accuracy
  (GPT / Qwen). Recompute these and record them in a receipt before production. If you cannot reproduce them, use and
  report the native-only SDs.
- MDE at 129 families with conservative 6-contrast planning (z ≈ 3.48):
  - GPT about 6% tokens and 3.0 pp;
  - Qwen about 5% and 2.3 pp;
  - pooled about 4% and 1.7 pp.
- These transport from main-study data, so mark them SUSPECTED.

