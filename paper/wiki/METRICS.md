# Metrics and layer selection

There is no single correlation metric whose maximization implies useful steering. Use a small set of quantities with distinct roles, apply the same questions to all seven models, and preserve each model's routing semantics. The existing analyses already demonstrate why these roles cannot be collapsed.

## Layer selection in the current interview

[ADR14/L1](DECISIONS.md#adr-20261008-14-joint-research-design-from-existing-evidence)
requires analysis at every routed layer, then one discovery-nominated site in
each early/middle/late routed-depth third for every model and target class.
Q19 nominates those sites and expert sets from current-class association;
source-conditioned next-class association is measured separately as a core
transition analysis. Q24 makes the difference in current-class expert inclusion
rate the primary empirical score; routing-weight occupancy and relative
enrichment are companions. Report support and stability, rank layer candidates
by excess allocation in their class-associated sets, and keep set-size/support
cutoffs configurable. Q25 reuses the existing five full-universe family folds;
each evaluated family's nominations use only the other four folds.
Under Q16, new nomination uses
empirical routing/activation statistics and explicit rules; no new classifier,
learned steering vector or model training is authorized. Class readability, association
stability and intervention effectiveness must be measured separately.

| Model | Routed layers analysed | Native zero-based IDs |
|---|---:|---|
| GPT-OSS | 24 | 0–23 |
| Nemotron | 23 | 1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51 |
| Qwen3.5 | 40 | 0–39 |
| Qwen3.6 | 40 | 0–39 |
| Gemma | 30 | 0–29 |
| GLM | 46 | 1–46 |
| Qwen3-30B | 48 | 0–47 |

The maps come from each `W/v3_analysis/results-r2/<model>/A/provenance.json`
([D05](EXPERIMENT_INVENTORY.md#d05-routing-captures-count-archives-and-common-population));
the existing selection receipts and class/layer coverage CSV retain native IDs.
Relative depth is an analysis coordinate, not a replacement for those IDs.
For example, Nemotron routed position 12 is physical layer 29, and GLM routed
position 0 is physical layer 1. A gap of one in routed depth need not span one
physical decoder layer.

Existing enrichment profiles and held-out mask coverage cover all layers, but
the clean R3-D class readout combines layers. It cannot supply an unfitted
per-layer incremental-loss ranking. That absent fitted-predictor analysis is
not adopted under the user's no-training correction. GPT B1 separately has controlled pair
information; it does not choose a seven-model intervention layer. Reuse those
distinct measurements and calculate only missing, agreed empirical statistics.
Choose sites on discovery families, report support/stability and deterministic
ties or unsupported cells, and keep final causal outcomes out of selection.

## Empirical routing statistics in the current design

### Empirical nomination quantities

For a discovery family with observed class-C and other-class tokens, estimate
the native inclusion rate of expert e at layer l separately in those two
populations. The Q24 score is their difference, then a family-balanced summary.
“Other classes” means the other six observed labels, not unlabelled tokens.
Report the two rates and supporting families along with the difference. A class
missing from a family is missing support, not a zero inclusion rate. Q27 adopts
equal benchmark weights, then equal family weights within each benchmark, in
discovery as well as evaluation. Report absent benchmark/class support explicitly.

For a nominated set T, the layer score is its excess selected-slot fraction:
`sum(delta[l,e,C] for e in T) / k[l]`, evaluated on the declared common-support
population. This puts the set-level difference on an allocation-fraction scale.
Show expert-set size/fraction and per-family uncertainty/stability; the exact
eligibility, set-size/cutoff and tie rules belong to the indexed configuration.
Rank only inside the accepted depth third and discovery folds. Companion
weight-occupancy differences retain native units; relative enrichment must show
the baseline rate and zero-support cases rather than inflating rare experts.

The pruning score is a separate object: existing state-balanced selection
averages supported class-conditional allocation shares. It is not the new
class-versus-rest steering score. Preserve that distinction when reusing masks
or comparing equal and unequal layer budgets.

### Empirical full-top-k coordination specification for review

Represent a token at layer l by its entire selected set `S[l,t]`, keeping expert
identities layer-qualified. For layers l and m, a normalized all-slot joint is
`P[e,f] = mean(1[e in S[l,t]] * 1[f in S[m,t]]) / (k[l]*k[m])` on the stated
weighted population. Its marginals are selection-slot shares. This uses every
selected expert pair; it is not a replacement of the native mixture with a
single dominant expert. Binary co-inclusion probabilities before normalization
are also useful for interpretable expert-pair effects.

Proposed direct outputs are the joint and marginal tables, excess co-selection,
MI/JSD summaries relative to the product of the same-stratum marginals, and
class/source-transition contrasts with support. Whole-set re-pairing within
declared family/class/benchmark/position/lexical strata is a possible empirical
null: it preserves each layer's selected sets while breaking aligned
cross-layer choices. Its exchangeability, movable support and within-sequence
dependence need explicit treatment; singleton strata do not supply a null.
This is a proposed statistical specification, not a completed new computation.

A product-marginal or re-paired reference also removes generic coupling. A
change in class-conditional MI alone therefore does not isolate irreducible
class-specific interaction. The proposed comparison for final review has two
parts: retain those descriptive joints, then compare supported binary-pair
coupling across individual reasoning classes, beside each expert's inclusion
contrast. No fitted predictor is needed.

For layer-qualified experts, let `A = 1[e in S[l,t]]` and
`B = 1[f in S[m,t]]`. Within class c and a declared nuisance stratum z, retain
the weighted four-cell table `p[c,z,a,b]`. For two observed classes c and d:

```text
lambda[c,z] = log(p[c,z,1,1] * p[c,z,0,0]
                  / (p[c,z,1,0] * p[c,z,0,1]))
theta[c,d,z] = lambda[c,z] - lambda[d,z]
```

The zero-contrast null permits shared pair coupling and different individual
inclusion rates across classes; it asks whether coupling changes on the
odds-ratio scale. Homogeneity of odds ratios is an established interaction
question in [stratified contingency-table analysis](https://www.sciencedirect.com/science/article/abs/pii/S0167715298001631).
This is our proposed application to native expert co-selection, not a result
from that reference. Compare individual classes on common support. Pooling the
other six classes can change an odds ratio through their mixture, so the
class-versus-rest nomination score and this interaction comparison use distinct
contrasts. Standardize within-stratum contrasts with common declared weights;
do not silently replace them with a pooled-table odds ratio.

Pair panels, nuisance strata and eligibility rules use discovery folds. Retain
native layer/expert identities and held-out support for every nominated cell.
Require supported finite odds ratios; zero or unstable cells remain explicit,
without an undocumented pseudocount or a favorable replacement pair. Uncertainty
resamples whole families, preserving their sentences, benchmark occurrences and
paired observations together, then reapplies the benchmark/family weighting.
The statistical motivation is [cluster bootstrap inference](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2007.00593.x),
not independent token resampling. For fixed supported contrasts, centered
bootstrap errors can provide asymptotic calibration; this is not an exact
margin-preserving randomization test. Bind simultaneous comparison panels and
the treatment of fold-dependent nominations before execution. Do not interpret
variation across evaluation folds as statistical-run SD.

This comparison supports observed class-dependent pair association. It does
not establish improved prediction or remove unmeasured lexical confounding.
Preserve the stronger existing fitted GPT B1 evidence under its original scope.
Neither these pair summaries nor full-top-k inputs establish higher-order path
sufficiency, causal mediation or directed expert communication. The normalized
joint uses all selected slots but remains a pairwise projection, not a
distribution over whole multi-layer routing sets. No learned path model is adopted.

## Dynamics in the current interview

Immediate class response and subsequent dynamics are equally important (Q15).
Use the reference's local/global and trajectory questions as described in
[REFERENCE.md](REFERENCE.md#october-8-interview-dynamics-and-depth). Proposed
dynamic quantities to freeze are time-resolved class occupancy, transition
counts/probabilities, observed dwell and re-entry, and native route turnover or
weight displacement during and after the pulse. Their windows and uncertainty
are configurable grids under Q17, with no single setting yet adopted. Treat closure as an explicit outcome and annotation gaps
as missingness. Conditional transition matrices describe the trajectories each
arm produces; assignment-level contrasts are needed for causal dynamics.

Cross-layer structure and temporal dynamics are different axes. Selected expert
sets at the same token describe depth coordination; ordered tokens/sentences at
a fixed layer describe time. A top-1 projection is not the full top-k mixture.
Q20 adopts full native top-k co-selection as the primary new coordination
representation; the existing dominant-expert path results remain reusable
comparisons. Joint token-aligned selections require joint captures, not products
of sentence marginals. The direct statistic and null above are proposed for
final shared-design confirmation.
The existing [depth/path studies](EXPERIMENT_INVENTORY.md#x13-cross-layer-routing-and-depth-path-studies)
must be reused before adding a library. [pathpyG's official documentation](https://www.pathpy.net/0.2.0-dev/tutorial/paths_higher_order/)
provides higher-order path models and order selection; its use would still
require a justified representation and family-held-out comparison. No package
has been installed or adopted by this review.

## Preserved metrics from earlier studies

The table below describes the earlier fitted-readout and substantive-verification
contracts. Current Q16 permits reuse of their results, not new probe/controller
training; Q11 targets seven observable classes. The revised empirical/causal
metrics are in the [current matrix](NEXT_EXPERIMENTS.md#measurement-and-reporting-contract-under-review).

| Question | Primary quantity | Necessary comparison or control | What it does not establish |
|---|---|---|---|
| Is an episode readable? | Held-out log-loss gain; grouped macro-AUROC for the existing probes | Family/response-disjoint fitting, position and lexical/past-history baselines | A causal actuator or qualified online detector |
| Does routing add information about the next operation? | Candidate-minus-baseline next-class log loss | Strictly available prefix, source-before-destination alignment, family-nested selection | Substantive verification or fine token-scale timing |
| Which expert is conditionally associated with an operation? | Family-balanced layer/expert selection-rate difference; executed-weight occupancy when observed | Same source condition, nuisance controls, support, held-out sign stability | Utility of promoting that expert |
| Did the intervention execute? | Inclusion change for bias/force; aligned selected-weight displacement for reweighting | Native versus executed allocation at the same hidden state, on intended opportunities | Downstream reasoning benefit |
| Did reasoning behavior change? | Assigned-arm difference in blinded substantive-success probability | Native and matched random; frozen horizon, all assignments, missing-reader bounds | Correct final answers or lower end-to-end cost |
| Is the policy useful? | Paired accuracy difference and complete policy-cost difference | Original prompts, equal budgets, frozen independent families, simultaneous intervals | Accuracy retention from nonsignificance |
| Is allocation temporally organized? | Lagged overlap/turnover and changes in normalized usage | Composition-preserving shuffled-order control, matched windows | Expert prefetch latency or causal thought dynamics |
| Could expert preloading help? | Future-set recall at declared lookahead, cache misses/bytes under fixed capacity | Frequency, recency and oracle bounds using the same workload | SSD latency without an actual cost model or measurement |

## Exact objects and normalization

Let `S[l,t]` be the native selected expert set at layer `l`, with native size `k`,
and `w[l,t,e]` the executed mixture weight under the source model's contract.
Do not assume unit weight mass across architectures or archives. If a probability
vector is needed, define a separate `p = w / sum(w)` only when the weights are
nonnegative with positive total mass, and retain the original scale. IDs always
include model/checkpoint and native layer.

- **Selection frequency:** `f[l,e] = mean_t 1(e in S[l,t])`. It lies in [0,1] and sums to `k` within a layer. For an entropy/JSD over selections, use `f/k`, not the unnormalized vector. Current v2 profiles have shape `(N,40,256)` and sum to eight per layer; this shape is Qwen3.6-specific, not a seven-model interface.
- **Conditional selection difference:** difference between per-family selection frequencies in matched positive and comparison windows, then an equal-family average. Preserve support counts on each side. Do not divide by rare activation frequency by accident: legacy accuracy-lift and rate-difference selectors ranked different experts.
- **Selected-weight occupancy:** mean executed weight on an expert, assigning zero when absent. Selection counts alone cannot recover it.
- **Set overlap and turnover:** `|S_t intersect S_(t-lag)| / k` and `1 - overlap` for fixed-k sets. This is not Jaccard overlap, whose denominator is the union. Declare lag/window and whether tokens or windows are compared.
- **Weight displacement:** `0.5 * sum_e |w_executed[e] - w_native[e]|`, after aligning IDs and inserting zeros, is half-L1 displacement in native weight units. Call it total variation only for separately normalized probability vectors; normalization discards total-mass changes. Selected-mixture weights are not a full-router probability distribution. Retain tensor-parallel ranks and exclude padded/inactive layers.
- **Selection fragility:** the native selected/unselected boundary gap in the correct score space. Families with correction biases can differ from naive raw-score top-k. Negative saved gaps need interpretation through the adapter, not automatic deletion.
- **Full-router entropy/confidence:** useful descriptive baselines when captured and semantically valid. Confidence is an affine transform of entropy; selected/rest mean-probability gap is an affine transform of top-k mass. Do not count equivalent quantities as separate evidence.
- **Allocation variability:** entropy of mean allocation minus mean per-token entropy. Permuting token order leaves it unchanged. It is not evidence for temporal transitions.

Historical “velocity” and “acceleration” fields use multiple definitions. The dense FRESH artifact supports selection-frequency changes in fixed 64-token windows, but not full gate-TV motion. Older motion summaries and v2 gate-TV definitions must keep their original formulas, units, lag and normalization. Never merge them because the column names resemble one another. [Metric implementation](../../src/moe_exp/correlation_pipeline/dynamics/routing.py), [motion amendment](../../report/experimental-resume-v1/ROUTING_MOTION_SUPPLEMENT.md), [dense source](../../report/routing-study-v1/dense-recovery-v3/TRAJECTORY_RESULT.json)

## Preserved verification-first selection pipeline

This is the earlier conditional-verification contract, retained for interpreting
its results. Its fixed support thresholds, fitted predictors and causal screen
are not defaults for ADR14's all-class, no-training, indexed-threshold design.

The requested layer-selection pipeline should formalize the existing proposal/family-fold machinery, not restart an unconstrained metric hunt.

1. **Bind the model contract.** Checkpoint, actual native selector, k, expert pool, shared experts, router-layer numbering, available ID/weight/score fields and exact token ownership.
2. **Freeze the scientific target and eligible discovery population.** Verification first. Labels of a destination are offline targets; the online input remains the prefix. Group duplicate families across models before any split.
3. **Estimate conditional expert differences.** Match source conditions, absolute position/run age and benchmark/difficulty where available. Select features, expert targets and layers only inside training folds. Keep unseen or low-support cells explicit.
4. **Require stability and support before nomination.** Reuse declared ten-positive/ten-comparison-family and fold-direction rules where that contract applies. Report uncertainty and rank stability; do not name every small positive coefficient a specialist.
5. **Qualify the operator and dose.** Confirm actual model-native inclusion or weight displacement and inactive fidelity. Random targets need the same layer/count support and declared exposure match; report realized dose mismatch rather than quietly relaxing tolerances.
6. **Use a frozen causal screen to choose the actuator.** Nuisance-controlled prediction nominates it; assigned semantic behavior tests it. Keep negative-direction actions as necessity/interference diagnostics. Discovery selection does not make the effect confirmatory.
7. **Freeze before independent validation/final outcomes.** The result is a model-specific candidate with stated uncertainty, not a universal best layer. Use relative depth for cross-model plots only; do not translate expert indices between checkpoints.

Legacy readable layers and the final Qwen L28 pair can be reused as documented starting evidence. Their earlier selection limitations remain. More all-layer capture is warranted only where the existing capture cannot answer the frozen selection question.

## Correlation and inference rules

For descriptive outcomes, use within-benchmark estimates and question/family-balanced summaries. Spearman association is a description of monotonic relation, not a confounder adjustment. Report absolute-position or fixed-window controls and complete versus capped populations separately. Avoid adjusting causal effects for post-treatment length, which the intervention can change.

For new policy evaluation, keep all assigned requests. Average seeds within family and give families equal weight. Jointly resample shared families across arms and across models when making paired cross-model claims; do not count seven executions of one problem as seven independent problems. Existing studies retain their original estimators, including legacy attempt weighting and its equal-problem sensitivity.

Report model-specific estimates and intervals first, then common-direction and heterogeneity summaries on comparable populations. Do not equate “significant in model A, not significant in B” with a significant model difference. Any additional across-model combined inference needs a frozen estimand and multiplicity plan; seven post hoc selected successes are not generalization.

Accuracy and generator-plus-controller output tokens are the existing policy endpoints. Report prompt processing, GPU time, capture overhead, offline readers/grading, loads and retries separately and include operational failures in the declared expenditure. Do not silently replace the endpoint by a sum of all these unlike quantities. No acceptable accuracy-loss margin is frozen, so retention/noninferiority is unavailable. Missing grades get explicit worst-case accuracy sensitivity; missing cost can have an unbounded upper limit under the current protocol.

## Statistical replication and GPU variation

Q26 explicitly prohibits repeat-only experiments to check GPU reproducibility,
including same-seed token/route comparisons. Exact sampled-run replay is not a
qualification criterion. Existing source/execution evidence and within-execution
invariants remain the engineering basis when implementations change.

The user anticipates later multiple runs for mean and standard deviation.
Each planned run needs an explicit replicate identity and declared seed, even
when a seed is reused. Compute the same benchmark/family-weighted estimand in
each run; report the mean and sample SD across those run estimates. A single run
does not supply run SD. Do not use five-fold variation as a substitute: folds
contain different evaluation families and candidate statistics. Family-clustered
uncertainty, paired differences and unresolved-outcome bounds remain separate;
repeated executions of one family are not additional independent families. The
number and scope of replicates are deferred, not automatic resubmission authority.

## Budget and preloading hypotheses

**Lower thinking budgets:** use existing exact token traces first to quantify natural finish/cap rates and the measurement needed at a lower cap. Cropping a long trace is not an evaluated budget-aware policy, and an absent final answer after cropping is not its measured accuracy. A new comparison requires common caps for native, policy and controls, a declared answer/grading protocol and full ITT accounting. Do not choose the cap on final outcomes.

**Preloading:** distinguish predicting native expert use (a cache/scheduling intervention) from constraining which experts the model may select (a model intervention). Existing ID traces can support a bounded replay of locality and lookahead; weights or latency must not be invented. Cross-model expert pools and footprints differ. Claim an SSD serving benefit only after a resource-specific transfer/cache/latency evaluation, with quality measured if native routing is restricted. This is a secondary branch, as the user accepted.
