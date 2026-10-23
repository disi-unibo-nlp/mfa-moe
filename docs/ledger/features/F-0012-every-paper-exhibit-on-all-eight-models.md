# F-0012: Every paper exhibit on all eight models
- status: active
- date: 2026-10-23
- request: "In the last codex session I discsussed about the missing fld data in a figure. Please check that, also check in general if there are some misalignments between models. Even experimetns that were run only on some models. Everything should be run over eerything. Check what is pending and already scheduled and what requires scheduling."; choices on 2026-10-23: Fig. 6 "Qwen3.6 rebuild + narrow", run "Strong dose, 4 models" and "Fig. 9 follow-up", remove the old global-steering table and the Qwen3.6-only historical study; the probe rebuild was not chosen.
- purpose: Every result the paper reports covers the same eight models (gpt, nemotron, qwen35, qwen36, gemma, glm, qwen330b, lfm25), with no unestimated cells caused by missing runs or sparse labels.
- acceptance: Fig. 6 and the state-transitions table have every row and interval estimated for all eight models; the strong-dose appendix table has eight model rows from a fresh eight-model analysis-strong receipt; Fig. 9 / whole_occupancy have no not-run cell that a follow-up release could fill; the two-model global-steering table and the Qwen3.6-only historical interventions are gone from the manuscript.
- keywords: eight models, coverage, strong dose, state transitions, figure 6, n/e, next sentence labels, Qwen3.6 509 families, complement follow-up, r21
- paths: src/moe_exp/study
- decisions:

## Tasks

- [x] F-0012.1 Audit model coverage of every manuscript exhibit and report what is queued and what needs scheduling; evidence: 2026-10-23 audit: Fig. 6 LFM Read row, LFM Verify persistence and Qwen3.6's three transition intervals n/e; next-state 2 models (F-0011 queued); strong dose 4 models; probes 3; verification qualification 7; global steering 2; Qwen3.6-only prefix interventions 1; Fig. 9 complement r18 1/75 shards ran by 15:30.
- [ ] F-0012.2 Qwen3.6 native dynamics on the full 1,486 families: rebuild its sentence table from the existing campaign-v3 labels (102,820 judged sentences on 1,547 attempts; the current table holds 35,484 on 522 questions), CPU only.
- [ ] F-0012.3 Narrow next-sentence labels on all eight models: for every labelled Read or Verify sentence whose next sentence in the same segment is unlabelled, label that sentence with each model's original frozen judge settings; the new labels enter only transition estimates (occupancy and dwell keep the random sample). User-approved estimate about 40 GPU-h.
- [ ] F-0012.4 Recompute native dynamics for all eight models, regenerate native_dynamics.json, Fig. 6 and the state-transitions table; report remaining n/e cells.
- [ ] F-0012.5 Strong-dose arm for qwen35, qwen36, glm, lfm25 (user-approved about 110 GPU-h): GLM test shard first, then the rest of GLM; evidence so far: STRONG.json planned 2026-10-23 from snapshot releases/37506fb (qwen35 1806, qwen36 1806, glm 1806, lfm25 1568 rows); jobs qwen35 61969933 61969946, qwen36 61969945 61969948, lfm25 61969950 61969951 61969952, GLM test 61969953.
- [ ] F-0012.6 Strong-dose outcomes per new model and a fresh eight-model analysis-strong; widen the strong-model roster in export_routing_results.py and verify_cl_tables.py; regenerate routing_strong.tex.
- [ ] F-0012.7 Fig. 9 follow-up release after the r18 cutoff (2026-10-24 06:00 Rome): run r18's never-started and interrupted rows with a later cutoff, combine on EFFICIENCY_LINK_V2 (as r20), staged export; user-approved up to about 285 GPU-h.
- [ ] F-0012.8 Paper: remove the two-model global-steering table and the Qwen3.6-only historical interventions (queued to the Codex paper session 2026-10-23), then integrate F-0012.4, F-0012.6 and F-0012.7 results.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|
