# Corrected whole-reasoning release

Continuation of Alignment v2, authorized by the user's “keep going”. The first
whole release `abbcffd769d324eaaa7ddfd38a8198418749bcc7` failed allocated CPU
validation before discovery or GPU loading. Preserve its [terminal receipt](../whole-reasoning-20261020/TERMINAL.json),
all seven native job records, code archive and logs. It used zero GPU-hours and
0.407778 CPU core-hours. These costs carry into this release; the 72/16 limits
do not reset.

The correction updates the incomplete-sentence fixture to the declared endpoint,
resolves historical fixture data through STUDY_DATA_ROOT, returns immediately
after failed validation, carries prior usage into accounting and projection,
and retains paired native-family covariance in the suspected SD/MDE fallback.
Ten focused regressions passed. All scientific doses, families, seeds, gates
and endpoints below remain the pre-outcome decision. No manuscript edits or pushes.

---

# Whole-reasoning efficiency experiment

Authorized by the user's **Alignment v2** message on 2026-10-20, before any
whole-arm outcome. This replaces the 2026-10-20 final implementation plan and
the unfinished r4 512-token rescue. No r3 stage is resubmitted. Historical r1,
r2, r3 and the unsubmitted r4 preparation remain intact. Decision D-0002 and
feature F-0003 govern this run. No manuscript changes or pushes are authorized.

New work has a hard 72 GPU-hour ceiling and 16 CPU-only core-hour ceiling.
The failed r3 engineering 0.346111 GPU-hours remains charged separately, as do
all older failures. The native queue was empty at the initial readback on
2026-10-20 (login05.leonardo.local, lmolfett). Native r3 accounting: CPU61554184
COMPLETED 0:0, 637 seconds, four CPUs; GPU61554207 FAILED 1:0, 312 seconds,
two GPUs; GPU61554215 FAILED 1:0, 311 seconds, two GPUs. Original evidence:
[r3 terminal](../final-routing-rescue-20261020-r3/TERMINAL.json),
[r1 provenance](../final-routing-rescue-20261019-r1/GENERATION_PROVENANCE_AUDIT.json).

Science cutoff: **2026-10-23 05:30 Europe/Rome**. Target generation finish:
October 21 morning, analysis at midday. Prompt qualification deadline:
2026-10-20 22:30 Europe/Rome. If qualification has not passed by then, freeze
the generated-only manifest and describe it as “generated-reasoning-only; not
a SteerMoE whole-token replication”.

## Evidence behind the replacement

The independent review supplied these measurements of the historical 512-token
rescue: answer completion approximately 46% GPT / 10% Qwen; mean strict-accuracy
bound widths 0.64 / 0.90; first sentence tokens 0–13 identical to native in
64.5% / 79.8%, and target/random identical in 59.6% / 81.9%; median first
divergence 10–60 tokens. These are review-provided historical measurements,
not observations from the new experiment. Its roughly 0.77–0.78 target weight
share, matched by random, is historical dose evidence. Fixed 512-token censoring
cannot establish full-solution token efficiency.

The legacy full-top-k promotion filled GPT's four slots at each of layers
15/19/21/23 and eight Qwen slots at six layers. The historical writeup reports
GPT accuracy 77.85% to 16.20% and mean length 6,497 to 22,489 tokens, and Qwen
70.5% to 0.8% with every run capped; see
[guiding_results.tex](../../paper/6a510c6db0d9b08aa438c142/guiding_results.tex), lines 101–126.
This file is evidence only and is not edited by this task.

[SteerMoE](https://arxiv.org/html/2509.09660v1), section 3, Table A.2 and
appendix A.1, ranks model-wide expert inclusion differences, combines positive
activation and negative deactivation, and changes log-softmax scores to original
row extrema plus/minus epsilon 0.01. It applies steering to prompt and generated
tokens. We use the operation-specific published counts from Alignment v2:
GPT promote 10/50, suppress 0/20; Qwen promote 15/0, suppress 0/500. Half doses
5/25 and 0/10, or 8/0 and 0/250, are a fluency fallback only. Our discovery
class-versus-complement comparisons differ from its minimal paired prompts.

The frozen link receipt declares a log-total-token treatment effect and frozen
`tok_reason` association predictors; the old effects implementation used
`activate:log_reasoning`. The corrected main link restores `activate:log_total`
and keeps log reasoning as secondary, without changing the frozen predictors.

R3's socket failure was a native 107-byte ZMQ IPC limit. Commit `91e3309`
provides the short WORK path and remains part of this release. Historical
receipts hash the live tree; new workers instead import from a materialized
release commit with an explicit `RELEASE_ROOT` and `PYTHONPATH`.

## Endpoints and analysis — Alignment v2 section 4, verbatim

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

## Fixed design and implementation conventions

All 129 ALLOCATION families are planned in both models. Original saved native
prompt token IDs are used with zero generated prefix, native sampler and native
family seed. Native_repeat has a separate deterministic seed. Every family is
ordered native, native_repeat, then class/direction target/random pairs; shards
partition complete families. There are 3,096 planned GPT and 2,838 planned Qwen
UIDs per dose. A nomination drop remains an explicit planned unknown in analysis.

For each outer fold, nomination uses only the other four discovery folds and
all four inner refits. Pair support must be at least two families on both sides
in each fit. Native-frequency quintiles are training-only and deterministic,
including tie-breaking by expert identity. Controls match role, physical layer,
quintile and count, exclude current-cell targets and the equivalently requested
opposite-polarity selections, and may be reused across classes.

The new endpoint uses log1p throughout its point and bound components. Every
capped run retains log1p(actual allowance) as its upper endpoint, as declared.
Unknown point entries are
compatibility midpoints, never observed zero effects. Occupancy uses each
family's common original allowance as a fixed denominator, retaining ended and
unknown mass. Native-repeat token noise is raw repeat-minus-native log1p length;
primary and native companions express shorter tokens as positive benefit.

Thirty-two production audit UIDs are selected per model/dose before outcomes;
all other units retain only additive per-layer counters. Full audit arrays are
stored losslessly as compressed NumPy float32 weights and uint8 identities,
with separate hashes, to fit the personal WORK quota. Canary outputs have new
UIDs and never enter analysis; they additionally have full captures for debugging.

The eight canary families cover the six benchmarks (one per benchmark, then two
additional families), selected by a fixed family hash, without looking
at whole outcomes or selecting short historical solutions. Their 32 steered
units are promote Verify and suppress the first declared suppress class, each
with target/random; native and native_repeat references are additional units.
The 4,096-token cap and <=2/32 cap gate are applied exactly as requested, even
though full-solution means exceed this cap. Failure can therefore reflect the
short canary allowance; it does not authorize tuning or bypassing the gate.

Production has two GPT and five Qwen shards, two GPUs/16 CPUs/120 GiB and
04:50:00 each, with the fifth Qwen held until a production allocation ends.
Production depends afterok on the passing canary. Extraction depends afterany
on its model's shards; analysis depends afterany on both extractions. One
bounded one-core supervisor checks native accounting every 45 seconds, the
45/75-minute fluency gates, quarter-pair projection and the absolute ceiling.
It stops at 71.6 new GPU-hours to retain cancellation/accounting-latency margin.
Native Lustre project quota 20149688 is checked on allocated CPUs and during
supervision; a 14-GiB remaining-space stop and pre-write audit checks protect
the inherited 10-GiB WORK reserve and concurrent output margin.
All failures and load times count. Fixed CPU maximums: validation 2 core-hours,
supervision 9, two extractions 2 combined, analysis/corrected historical analyses
2 combined; bounded local development must fit the remaining one core-hour.

## Debate record — Alignment v2 section 9

Adopted unanimously: whole-reasoning efficiency; no new 512-token generation;
cross-fitted global stable RD; epsilon 0.01; stop at closure; matched random,
native and native_repeat; co-primary accuracy/reasoning with bounds; immutable
release and short sockets; abort rules and no efficacy-driven retuning.

Resolved: original prompt; absolute-position prompt scope with declared
generated-only fallback; dual promote; operation-specific published counts;
mandatory canary; cell-local control exclusions; exploratory link; pooled
equal model weights. The hybrid Qwen15/500 and density-scaled25/833 proposal
are excluded. Not adopted: routine two-dose sweeps, prefill-free primary,
deactivation-only Qwen promote.

Open risks: class/complement is not minimal paired prompting; transported MDEs
are unverified; length inflation may hit the ceiling, leaving bounded partial
blocks. The canary's short cap may itself prevent production. No null interval
establishes equivalence.

## Observed execution

No whole-design Slurm job has been submitted at this initial record. Job IDs,
native readbacks, selected doses, gates and final accounting are added only
after they are observed. See JOBS.json, ACCOUNTING.json, per-model CANARY files,
PROJECTION.json, extraction receipts and analysis/EFFECTS.json when present.
