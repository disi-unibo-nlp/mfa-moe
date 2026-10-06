# Clean B4 precision interpretation (2026-10-02)

The registered clean B4 analysis covers 509 assigned questions in 496 frozen duplicate families; 508 questions have an observed correctness outcome. Its added class-conditioned routing block has estimated predictive gain **−0.000729 nats per question**, nominal family-bootstrap 95% interval **[−0.005313, +0.004384]**, and nominal simultaneous interval **[−0.006747, +0.005289]**. This does not meet the registered 0.005-nat gain threshold. These numbers come from the sealed `r3e-cv-clean-v1/full.json` result (Slurm job 59198802, SHA-256 `f0894f8decfc8c6ea09d7490f6c041857293c90de2cd3ce361e078f50b76ca36`). They concern correctness prediction in GPT source data, not Qwen routing interventions.

The separate precision exercise fixed the same features, family folds, nested estimator and 508 scored questions. It generated 200 independent outcome replicates at each of five Bernoulli KL targets. A fold-specific baseline was fitted without that fold's original outcomes, and a positive residual PA signal was calibrated using held-out *covariates and train-fitted probabilities*, never held-out original correctness labels. This makes the scored-population oracle KL equal its stated target. Simulated outcomes are conditionally independent given features and do not include residual family effects. The exercise therefore measures behavior of **this estimator under this generator**, not a universal minimum detectable effect. All 1,000 assigned replicates completed in the sealed `r3e-b4-precision-v3/summary.json` (aggregation Slurm job 59203686, SHA-256 `1da9aff85c10a7ac81fac3e96c4eb5a9d8df984add739e261f4c35df25426feb`).

| Held-out oracle KL target (nats/question) | Mean fitted gain (nats/question) | Registered advance | Nominal 95% interval coverage of simulated fitted-algorithm mean |
|---:|---:|---:|---:|
| 0 | −0.000432 | 0/200 (0%) | 183/200 (91.5%) |
| 0.001 | +0.000117 | 0/200 (0%) | 177/200 (88.5%) |
| 0.003 | +0.000571 | 3/200 (1.5%) | 173/200 (86.5%) |
| 0.005 | +0.000912 | 9/200 (4.5%) | 164/200 (82.0%) |
| 0.010 | +0.001412 | 11/200 (5.5%) | 179/200 (89.5%) |

“Registered advance” requires a positive Holm-safe one-sided result, a positive nominal interval, at least four of five repeat gains positive, and fitted gain ≥0.005 nats/question. At the largest simulated target, its 11/200 rate has Wilson 95% Monte Carlo interval [3.10%, 9.58%]. Fitted gain is smaller than oracle KL; finite-sample learning and model misspecification are possible contributors. The distinct oracle-KL coverage diagnostic must not be interpreted as coverage of fitted predictive gain.

The observed 82–91.5% empirical coverage falls below the nominal 95%, with 200 simulations at each target. The bootstrap holds cross-fitted predictions fixed while resampling evaluation families, so it does not fully represent variation from refitting the learner under new outcomes. This is a plausible explanation, not an identified decomposition of the coverage error. The paper should label the original intervals **nominal family-bootstrap intervals** and disclose this simulation result. A larger independent outcome evaluation or an interval method that also refits the learner is needed for a calibrated 95% claim. Non-rejection of B4 does not establish equivalence, absence of a conditional signal, or accuracy retention.

The preserved v2 simulation calibrated KL on the generator's training folds only and overshot the intended held-out KL targets. Its sealed summary (`r3e-b4-precision-v2/summary.json`, job 59202682, SHA-256 `a3e5c995cab8bb7724cf15e8b130888e4033be7b19c81e1106c2add258386c83`) remains a sensitivity to that training-calibration choice. It is not pooled with or silently substituted for v3.
