# Exploratory clean correctness diagnostics after registered B4 — 2026-10-02

The sealed 509-question registered clean B4 result `r3e-cv-clean-v1/full.json` estimates an added PA-block gain of −0.000729 nats per question with family-clustered 95% interval [−0.005313, +0.004384]. Its 28-column class-conditioned marginal baseline has a measured negative gain of −0.026720 against controls. The registered primary remains unchanged and cannot be redefined after this observation.

To localize this baseline failure, evaluate the following fixed candidates on the same 509-question cohort with the same five repeats, five outer frozen-family folds and three inner folds:

1. Controls C alone.
2. C + PA, C + PAlex, and C + {PA, PAlex, PA_sd}.
3. C + the registered 28-column marginal block M, and C + M + {PA, PAlex, PA_sd} as a replication check.
4. C + a three-component marginal PCA block M3, and C + M3 + {PA, PAlex, PA_sd}.

The PCA mean, scaling and loadings are fitted only on each training set; scoring rows receive the frozen transform. Ridge penalties come from the fixed nine-point grid 10^−4 through 10^4, selected only within the three inner training folds. Report every candidate, family-clustered and simultaneous intervals, and five-repeat stability. This is explicitly exploratory reuse of the primary cohort and supports diagnosis or an independent follow-on proposal, not a newly confirmed correctness gain. The confirm-connected-family exclusion sensitivity is separately frozen and will be reported even if this diagnostic is unfavorable.

If the clean PA features fail to improve controls directly, assess adequacy of the lexical reference and measurement rather than declaring routing irrelevant. If M3 removes the baseline harm while preserving predictive signal, propose independent validation of that fixed architecture. The causal routing-action study remains the path to steering claims; predictive diagnostics do not substitute for randomized semantic and utility outcomes.
