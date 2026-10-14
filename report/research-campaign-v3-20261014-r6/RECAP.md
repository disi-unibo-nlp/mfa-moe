# MoE experiment recap — 2026-10-14

Observed 2026-10-14T11:02:01.185305+00:00. This is a status snapshot; [queue and evidence](RECAP_EVIDENCE.json) bind the observations. Scientific production is **not launched**. Paper work is deferred; publication is by your manual push.

## Done

- Implemented and sealed r6: token-ID handling, original-assignment recovery, component failure isolation, CPU priority, read-only forecasts, immutable task tables and indexed/checkpointed export.
- **111 tests passed** in job 60488160. The actual seven-model import completed, preserving seven ancestor campaigns and 17,288 component references. Original source/mask/execution identities were checked for **69,459 non-REAP generations and 346 compact shapes**.
- Reused all seven repaired derivations, seven native memory fixtures and six REAP qualifications. GPT’s REAP limitation is preserved. GPT **81/81** and Nemotron **77/77** coordination pairs are complete; native context manifests are recovered for all seven models.
- Audited the **64 original reader assignments**: all were rejected before engine admission and may recover under the same identities, prompts and seeds. **They have not been rerun.**
- Added local-only delivery preparation and seven passing policy tests; strengthened ignore rules for bulk cell banks, runtime data and caches. No pushes are performed.

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
| 60454552 | Pending, October 15 at 23:59:59 Brussels time | A durable local milestone report/PDF of completed and missing evidence; it cannot establish missing scientific effects. |
| 60462231 | Pending, user hold | Primary report/publication controller; held to honor manual publication. No scientific measurement while held. |
| 60462233 | Pending, user hold | First failure owner; retained but held. No scientific measurement while held. |
| 60462236 | Pending, user hold | Second failure owner; retained but held. No scientific measurement while held. |

**No scientific generation, saved-sequence or coordination job is queued.** No predecessor was retired and no replacement controller was released.

## Current blocker and next runs

- Job **60488160 failed after import**, because the CPU acceptance fixture assigned the installed vLLM processor’s read-only `tokenizer` property. This is a fixture implementation error, not a measured model failure. The corrected [unsubmitted runner](../../scripts/research_campaign_v3_r6_validation_candidate_r4.py) uses `renderer.tokenizer`; its small installed-interface check passes. Full retained-reader and 243,747-row export validation remain unverified. [Failure receipt](VALIDATION_EXTENSION_TERMINAL.json), [preflight](VALIDATION_CPU_FIXTURE_PREFLIGHT.json).
- The approved extra attempt is exhausted. Another CPU allocation requires explicit authorization. Validation has used **2.39 local accounting hours in total**; the original 42,050-hour ceiling is preserved.
- **After validation passes:** retained reader pricing will test the real sampling/annotation path and its cost; five CPU coordination pilots/shards will measure the remaining **741 pairs**; successive compact qualifications and retained pricing assignments will measure actual storage release, within-execution fidelity and paired strict accuracy; **10,822 saved-sequence forwards** and **35 calibration folds** will supply missing REAP/steering inputs. These are planned, **not queued**.
- Complete measured admission with **25% margin** must precede production of the full memory curves, central steering and balanced sensitivity/supporting analyses. Full-study affordability is still unmeasured; no scope reduction or replacement draws are authorized.
