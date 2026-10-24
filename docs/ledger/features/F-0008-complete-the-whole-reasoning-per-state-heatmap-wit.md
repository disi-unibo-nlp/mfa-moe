# F-0008: Complete the whole-reasoning per-state heatmap with the missing suppress cells
- status: active
- date: 2026-10-23
- request: "SBatche all the remaining runs to compelte Figure 9. I don't want not-run cells"
- purpose: The base releases suppress only the states tied to longer reasoning, so 25 of the 112 model-by-state cells of the per-state heatmap (Fig. 9, whole_states) were never run. Run them at each model's base strength on the same families and seeds, joined to the base rows by family, so every cell has a targeted-minus-random estimate.
- acceptance: A sealed complement release runs the 25 suppress cells (6,372 rows: gpt Im/Ve/Mo, nemotron Ex/Ve/Mo, qwen35 Re/Im/Ve/Mo, qwen36 Re/Ex/Ve/Mo, gemma Im/Ve/Mo, glm Ex/Pl/Ve/Mo, qwen330b Ve/Mo, lfm25 Ve/Mo); a new combined EFFECTS_WHOLE.json (r7-r17 plus the complement) has a cell_effects entry for every model x operation x state at base strength; primary, companions and dose response are unchanged in definition (complement cells are not efficiency-directed); GPU work before 2026-10-24 06:00 Europe/Rome within a 610 GPU-hour ceiling.
- keywords: whole-reasoning, complement, suppress, heatmap, figure 9, whole_states, missing cells
- paths: src/steering, scripts/steering, tests/steering
- decisions: D-0002

## Tasks

- [x] F-0008.1 whole.complement_manifest builds the missing suppress cells from the parent's nominations and native rows; whole_effects joins a source's `complements` by family and keeps them out of arm levels; tests in tests/study/test_whole_complement.py; evidence: 4 new tests pass; dry run on r7/r8 data built 25 cells, all qualified, 6,372 rows, row fields identical to base suppress rows.
- [ ] F-0008.2 Seal and launch the complement release with its combined analysis and staged paper export; evidence: report/whole-reasoning-20261021-r18/JOBS.json.
- [ ] F-0008.3 Supervise to terminal accounting; report per-cell results and remaining unknown rows.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|
