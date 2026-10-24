# F-0011: Next-state prediction from routing for all eight models
- status: active
- date: 2026-10-23
- request: "we need to improve that figure and experiment. WE can just consider all stences and classifucy the nex state. For all models. So it wuld be like an extension of figure 6 but not divided by class. Determine what experiments we need and sbatch them as well asap"
- purpose: Fig. 7 (next_state) reports the anticipatory routing gain only for GPT-OSS and Qwen3.6, with a retrospective relative-progress control and a 32-token sensitivity. Replace it with one next-state classification over all adjacent labelled pairs for all eight models: current state only (Fig. 6), plus text and position, plus routing.
- acceptance: For each of the eight models, held-out cross-entropy of the next state for markov, text and routing on every adjacent labelled pair of the 522 readout questions (no length filter, no relative progress); routing gain with family-cluster 95% intervals and its excess over five token-permutation nulls; receipts report/final-routing-20261015-r1/<model>/NEXT_STATE.json and driver/NEXT_STATE.json; CPU only.
- keywords: next state, anticipation, ARG, figure 7, next_state, routing gain, transition, markov
- paths: src/routing, scripts/routing, tests/routing
- decisions:

## Tasks

- [x] F-0011.1 Check the data: adjacent labelled pairs in the 522 readout questions; evidence: gpt 10,936, nemotron 6,096, qwen35 2,812, qwen36 2,531, gemma 5,191, glm 4,556, qwen330b 6,018, lfm25 6,313 (campaign-v3 dynamics sentences.parquet; LFM derived sentences.parquet).
- [x] F-0011.2 moe_exp.study.next_state: pair data from the present-state readout loaders, markov/text/routing fits through readout_analysis; evidence: commits e69bd17..6521595; login smoke test on Qwen3.6 (2,531 pairs, 41,511 source tokens; untuned fold 0: markov 1.655, text 1.681, routing 1.610 nats); jobs 61946688 61946690 61946692 61946694 61946696 61946698 61946700 61946702 + collect 61946704 from snapshot releases/6521595 (boost_usr_prod, 8 CPUs, 40 GB, 3 h, CPU only).
- [ ] F-0011.3 Collect driver/NEXT_STATE.json and report the eight routing gains.
- [ ] F-0011.4 Paper: new Fig. 7 (eight models, three nested predictors) and its text; hand to the paper session.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|
