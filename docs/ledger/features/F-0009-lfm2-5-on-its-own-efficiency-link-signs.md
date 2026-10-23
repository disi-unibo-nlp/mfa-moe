# F-0009: LFM2.5 on its own efficiency-link signs
- status: active
- date: 2026-10-23
- request: "If this is true it measn we are mossing some experiments. Please check that and launch what we need to make the comparisong even"; on how LFM's own signs enter the paper: "Replace the main result".
- purpose: The whole-reasoning steering chose each model's efficiency-directed states from the seven-model efficiency-link pre-registration (driver/EFFICIENCY_LINK_PREREG.json), which had no native table for LFM2.5, so LFM used the pooled signs. Give LFM its own native signs everywhere (primary, dose response, pooled row, both link tests) so the eight models are compared on the same rule.
- acceptance: driver/EFFICIENCY_LINK_V2.json covers 8 models with the frozen estimator (2,000 draws, seed 20261017), its seven pre-registered models equal the prereg's cells; every LFM cell directed under its own signs has target and random rows at every strength in r7-r18 (no new steering); a combine-only release r19 rewrites EFFECTS_WHOLE.json with the V2 link (seven models unchanged); the driver link test reruns on analysis-final pairs with 56 cells and reproduces the published 49-cell test on the prereg; paper tables and text name LFM's after-the-fact estimate.
- keywords: efficiency link, LFM2.5, lfm25, pooled signs, directions, WHOLE_LINK, relink, r19, EFFICIENCY_LINK_V2
- paths: src/moe_exp/study, scripts/study, tests/study, scripts/paper
- decisions: D-0002

## Tasks

- [x] F-0009.1 Check the claim: whole.directions gave lfm25 the prereg's pooled signs ("lfm25 (no native lexicon table)"); evidence: src/moe_exp/study/whole.py directions at 6978e0b; quick read-only estimate on lfm25/derived/sentences.parquet (96,137 deduplicated sentences, 1,486 families): Read 0, Analyze +, Plan +, Implement 0, Explore +, Verify -, Monitor - versus pooled + + + + + - -, so LFM's directed set is a subset of the pooled one and no GPU run is missing.
- [x] F-0009.2 efficiency_link: full LFM native table (lexical-link-r2/), eight-model receipt EFFICIENCY_LINK_V2.json, driver link test analysis-final-link-v2/EFFICIENCY_LINK_TEST.json; evidence: jobs 61936851 (table + link, seven models equal the prereg) and 61937963 (driver test, point estimates of analysis-final reproduced); LFM tok_reason signs Read 0, Analyze +, Plan +, Implement + (CI low +0.0003), Explore +, Verify -, Monitor -; LFM acc signs all 0 (one trace per question).
- [x] F-0009.3 whole.LINK (WHOLE_LINK) drives directions and the exploratory link; combine records the link it used; exporters read the analysis's link; tests/study/test_whole_models.py covers LFM's own signs; evidence: commit 5a4b169; job 61936428 ran 104 tests/study/test_whole*.py tests, OK.
- [x] F-0009.4 Seal and launch the relink release (whole_relink: allocated tests, combine after r18's combine 61927867); evidence: r19 launch failed before submitting (report/whole-reasoning-20261021-r19/LAUNCH_FAILED.json); r20 commit 9d576da, tests 61937253, combine 61937255, staged export 61938212 (report/whole-reasoning-20261021-r20/JOBS.json).
- [ ] F-0009.5 Check coverage and the r19 versus r18 comparison: every LFM directed cell present at every strength; seven models' primary and dose response unchanged; report LFM and pooled changes.
- [ ] F-0009.6 Paper: rerun export_routing_results.py and export_whole_results.py on the new receipts, edit the LFM pooled-sign sentences (appendix efficiency link, method, results), verify tables, build.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|
| Driver link test, token Spearman rho [95% CI] | analysis-final pairs, central activation | 49 cells: -0.111 [-0.283, 0.121] | 56 cells: -0.104 [-0.269, 0.106], not supported | driver/analysis-final-link-v2/EFFICIENCY_LINK_TEST.json |
| Driver link test, token sign agreement | same | 17/41 (p 0.349) | 19/47 (p 0.243) | same |
| LFM directed cells with target+random rows | r7 main, r9 quarter, r10 half manifests | - | 6/6 at each strength | report/whole-reasoning-20261021-r{7,9,10}/lfm25/WHOLE*.json |
