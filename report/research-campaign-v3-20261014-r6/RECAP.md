# MoE experiment recap — 2026-10-14

The latest operations amendment preserves the original scientific seal and
reuses its evidence. [Operations preflight](OPS_R1_PREFLIGHT.json),
[protected milestone](ops_r1/milestone/PROTECTION.json) and
[authorization](OPS_R1_AUTHORIZATION.json) supersede the older queue snapshot
in [RECAP_EVIDENCE.json](RECAP_EVIDENCE.json). Scientific production is **not
launched**. Canonical manuscript work is deferred; publication is by your manual push.

## Done

- Implemented and sealed r6: token-ID handling, original-assignment recovery, component failure isolation, CPU priority, read-only forecasts, immutable task tables and indexed/checkpointed export.
- **111 tests passed** in job 60488160. The actual seven-model import completed, preserving seven ancestor campaigns and 17,288 component references. Original source/mask/execution identities were checked for **69,459 non-REAP generations and 346 compact shapes**.
- Reused all seven repaired derivations, seven native memory fixtures and six REAP qualifications. GPT’s REAP limitation is preserved. GPT **81/81** and Nemotron **77/77** coordination pairs are complete; native context manifests are recovered for all seven models.
- Audited the **64 original reader assignments**: all were rejected before engine admission and may recover under the same identities, prompts and seeds. **They have not been rerun.**
- Added local-only delivery preparation and seven passing policy tests; strengthened ignore rules for bulk cell banks, runtime data and caches. No pushes are performed.
- Added the separately bound operations adapter: real proof-gated launch, corrected external-ID reader cleanup, report-only completion and compact commits on private local refs. **20 operational tests pass**, including the actual installed abort method; tiny installed tokenizer/input/sampler checks pass. Implementation commits: `83a45da`, `67ee376`.

## Findings so far

- Existing CLEAN readouts show that routing contains information about the seven annotated reasoning classes; this concerns visible annotated reasoning, not a demonstrated latent mechanism.
- At 75% expert retention, state balancing improves minimum held-out class coverage in **five models**; labelled frequency does better for **Qwen3.5/Qwen3.6**. Coverage does not establish accuracy or memory savings. [Source-bound existing figures](../../paper/6a510c6db0d9b08aa438c142/generated/best_results_20261009/README.md).
- Qualified native process residence, summed over the two GPU workers:

| Model | Native residence (GiB) | REAP verdict |
|---|---:|---|
| GPT-OSS | 32.84 | method limitation |
| Nemotron | 37.23 | qualified |
| Qwen3.5 | 44.35 | qualified |
| Qwen3.6 | 57.84 | qualified |
| Gemma | 38.26 | qualified |
| GLM | 91.93 | qualified |
| Qwen3-30B | 75.57 | qualified |

These are fixed-context native baselines with the registered 8-GiB KV capacity, measured separately under each model’s profile. **No new compact accuracy–memory curve or causal steering effect has been measured.** [Fixture and coverage evidence](PARTIAL_VALIDATION_EVIDENCE.json).

## In the queue now

| Job | Observed state | Intended output or purpose |
|---|---|---|
| 60509618 | Pending, October 15 at 23:59:59 Europe/Rome | Report-only milestone from current validated evidence, or the completed r5 report plus r6 partial evidence if validation fails. |
| 60509619 | Pending, dependency | Primary reporting owner; one bounded full-export recovery, with manuscript and remote writes disabled. |
| 60509620 | Pending, dependency | First failure owner; compact fallback after full-export recovery is exhausted. |
| 60509621 | Pending, dependency | Final failure owner; bounded compact fallback and explicit terminal ownership. |

**No scientific generation, saved-sequence or coordination job is queued.**
All four replacements were accepted/read back before retiring only old pending
predecessors `60454552/60462231/60462233/60462236`, each with zero runtime
and accounting `Start=None`. The existing 28-hour reporting reservation persists.

## Current blocker and next runs

- Job **60488160 failed after import**, because its CPU fixture assigned the installed vLLM processor’s read-only `tokenizer` property. The versioned candidate r5 reuses the corrected renderer fixture and validates the new operational policy. Full retained-reader and 243,747-row export validation remain unverified. The original [failure receipt](VALIDATION_EXTENSION_TERMINAL.json) and preflights remain preserved.
- The user has now authorized **one additional four-CPU/24-GiB/three-hour attempt**, at most 12 local hours / 15 with margin, with no automatic retry. Prior validation used **2.39 local accounting hours**; the original 42,050-hour ceiling is preserved. This attempt has not yet been submitted at this recap revision.
- **After validation passes:** retained reader pricing will test the real sampling/annotation path and its cost; five CPU coordination pilots/shards will measure the remaining **741 pairs**; successive compact qualifications and retained pricing assignments will measure actual storage release, within-execution fidelity and paired strict accuracy; **10,822 saved-sequence forwards** and **35 calibration folds** will supply missing REAP/steering inputs. These are planned, **not queued**.
- Complete measured admission with **25% margin** must precede production of the full memory curves, central steering and balanced sensitivity/supporting analyses. Full-study affordability is still unmeasured; no scope reduction or replacement draws are authorized.
