# W-0002: Whole nomination and an actual control-feasible GPT cell
- date: 2026-10-24
- commit: 6bb8d88b807fca6ebc0c2aa9dccd23a577f96f4f
- source: report/whole-reasoning-20261020-r3-batched/deterministic/gpt/WHOLE_NOMINATIONS.json (sha256 11e2324f), gpt/WHOLE.json (8b78678e), gpt/WHOLE_HALF.json (99764d19), qwen36/WHOLE.json (83a966ae), NOMINATION_AUDIT.json (215cdfb7), the DP reference CONTROL_FEASIBLE_RD_BOTH_DOSES.json (e3577e2f); every value below comes from a run on 2026-10-24 of the code at this commit over those saved files
- paths: src/steering/selection.py, src/steering/nominations.py, src/steering/whole.py, tests/steering/test_whole.py
- keywords: whole, risk difference, quintile, feasibility, controls, nomination

**Input:** The GPT-OSS-20B Read nomination of the whole-reasoning release for held-out fold 0 (main dose,
promote), rebuilt from discovery folds 1, 2, 3 and 4 only. No whole-arm outcome is read anywhere in this
trace. This is the retrace of the walkthrough first written on 2026-10-20 at commit 16cf599, when the code
was `final_routing.selection` and `study.whole`; the modules are now `steering.selection` and `steering.whole`.

## 1. Global nomination (`nominations.discovery_inputs`, `selection.global_pairs`)

The discovery inputs are 1,547 GPT attempts of 1,486 question families: a counts array
`[attempt, class, layer, expert]` of shape (1547, 9, 24, 32) and token denominators (1547, 9). A pair
(layer, expert) is a candidate for a class when its risk difference between the class and all other
classes has support from at least two families on both sides and keeps its sign in the full fit and in all
four inner refits that leave one more fold out.

| Fold 0, class Read | Value |
|---|---|
| training folds | [1, 2, 3, 4] |
| sign-stable positive pairs / negative pairs | 363 / 322 |
| strongest positive pair | layer 8, expert 29, +0.18327 |
| strongest negative pair | layer 9, expert 15, -0.19581 |
| activations and deactivations requested (GPT) | 10 and 50, both available and qualified |

`global_pairs` over all five folds and seven classes returns 35 records, and they equal the saved
`WHOLE_NOMINATIONS.json` records exactly (`True`).

## 2. Capped ranking (`selection.select_pairs`, no controls)

`whole.dose_counts('gpt', ..., 'main', 'promote')` gives (10, 50). The activation cap is one pair per layer,
the deactivation cap is ceil(50/24) = 3. Walking the candidates in order of absolute risk difference, the
exact-rank deactivation set has total absolute RD 5.992919759516524. The opposite-polarity exclusion (the
10 negative activations and 50 positive deactivations the other direction would pick) has 60 pairs.

## 3. Reserve distinct controls (`selection.select_pairs` with `frequency=` and `excluded=`)

Controls must come from the same layer and the same native-frequency quintile as their target. At layer 14
the most frequent quintile holds the experts [9, 20, 3, 26, 13, 25]. The exact-rank deactivation targets at
that layer are [3, 25, 20] and the opposite exclusion there is [4, 13, 28], so expert 13 is also taken.
That leaves only [9, 26] to serve three targets. The greedy walk charges a new target its own pool slot plus
one distinct control and skips the target that cannot leave one.

## 4. Match the selected roles (`selection.layer_matched_controls`, called by `whole.cell_actions`)

| Layer 14, deactivation | Experts |
|---|---|
| targets after the walk | [3, 25] |
| matched random controls | [9, 20] |

Across the cell, 2 deactivation pairs change against the exact-rank set, all 10 activations are unchanged
(`activation_changed` False) and all 50 deactivations have controls under the same layer, quintile and
exclusions. Retained absolute RD is 5.933801208792859, a fraction of 0.9901352674329091. Mean native
frequency of the steered experts is 0.361816186798991 for the targets and 0.2953428770321593 for the random
controls. The rebuilt cell equals the cell stored in `gpt/WHOLE.json` (`True`).

## 5. Qualification of the whole release (`whole.cell_actions` for every cell, the comparisons of `whole.audit_nominations`)

All cells of the deterministic release were rebuilt from the saved nominations and compared with what the
release stored. For GPT the nominations themselves were recomputed too; for Qwen3.6 the saved candidate
lists were used (recomputing them did not finish inside the 550 s wall limit set for this login-node run).

| Check | GPT-OSS | Qwen3.6 |
|---|---|---|
| cells rebuilt (2 doses x 5 folds x classes) | 110 | 100 |
| qualified | 110 | 100 |
| equal to the cell in the saved manifest | 110 | 100 |
| equal to the record in `NOMINATION_AUDIT.json` | 110 | 100 |
| promote cells whose targets and controls equal the saved DP reference | 70 of 70 | not in the DP reference |
| rows planned = executable, per dose | 3,096 | 2,838 |

The unit tests `tests/steering/test_whole.py` and `tests/steering/test_stream_batch.py` ran on the login
node in the vLLM environment: 32 tests, OK, 59 s. They include the sign-survival, half-availability and
control-matching tests behind steps 1 to 4.

## Worth noticing

- The new code reproduces every nomination receipt of 2026-10-20 without a difference: 35 GPT records, 210
  cells and 70 DP matches. The move from `final_routing` and `study` into `steering` changed no result.
- Control capacity decides the nomination before any outcome. In this cell expert 20 is replaced to keep
  distinct controls, at a cost of 1 percent of the retained RD; the lowest retained fraction over the GPT main promote cells is 0.9426.
- Targets still sit on more frequently used experts than their controls (0.362 against 0.295), so the
  matching is by quintile, not by value.
- The release's original allocated qualification (job 61581053, 190 tests) is a Slurm stage of every release
  (`python -m steering.whole validate`); the login-node run above is a spot check of the same code, not a replacement.
