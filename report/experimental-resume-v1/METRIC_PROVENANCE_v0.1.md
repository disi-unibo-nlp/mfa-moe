# Metric provenance and decision use

This mapping separates literature-inspired methods from this study's own
operational endpoints. The linked arXiv papers are primary research sources;
peer-reviewed venue status is not asserted for every source.

| Measurement | Literature source | Implementation and limit |
|---|---|---|
| Paired expert activation risk difference | [SteerMoE, §3.1](https://arxiv.org/html/2509.09660v2) | Difference in top-k membership rates between matched native behavioral contrasts. Average within frozen family, then equally across families. The observed contrast proposes experts; it is not a causal effect. The short sparse pulse design is our adaptation. |
| Random expert controls | [SteerMoE, Appendix A.1.6](https://arxiv.org/html/2509.09660v2) | Several fixed expert sets with identical layer/count support, native exposure within ±10%, balanced assignment and measured dose. Exposure/dose matching are additional constraints here. |
| Same-position alternative-route NLL | [When Are Experts Misrouted?](https://arxiv.org/html/2605.07260v1) | Replay the same prediction prefix and compare raw target-token loss under alternate fixed-k routes. Our inclusion/exclusion of a sparse expert set, semantic target/non-target matching and syntax controls are adaptations, not that paper's full route search or router training. |
| Causal expert-importance audit | [From Observation to Intervention](https://arxiv.org/html/2606.10703v1) | Compare observational proposal scores with functional loss changes and controls. That paper's single-token ablation results motivate avoiding utilization/gate-weight-only selection; they do not establish what works on Qwen3.6 reasoning transitions. |
| Loss-based counterfactual routing | [CoR](https://arxiv.org/html/2604.14246v1) | Additional precedent for functional expert assessment. CoR's variable expert count differs from our top-k=8 replacement screen. |
| nPMI association with reasoning markers | [RICE, §2](https://arxiv.org/html/2505.14681v1) | Proposed secondary ranking only. RICE scores marker association and multiplies weights of selected cognitive experts; marker association alone is neither semantic verification detection nor causal expert usefulness. |

Full-prefix eligibility, detector confusion/coverage, substantive transition
completion, dwell, re-entry and loops are **study-specific operational measures**.
The primary local endpoint is completion of the frozen semantic trajectory in
1,024 tokens after enrollment, excluding the triggering sentence. A check must
evaluate an original constraint or computation; verification wording alone is
insufficient. The four-prefix engineering pilot used a separate 256-token cap
and yielded zero jointly accepted starts. It cannot identify semantic control.

The clean B1 held-out class-associated routing gain, measured in nats per token
pair, is a statistical endpoint from the legacy analysis. Its current clean
result is on the **GPT source routing dataset** (509 questions/496 families),
not the Qwen3.6 intervention model. It motivates testing class-related routing
information; it does not establish model transfer or causal semantic experts.

Routing motion is our diagnostic extension. The native 64-token supplement
defines velocity as total-variation displacement of window gate distributions,
vector acceleration as half the L1 norm of their second difference, and
turnover as one minus top-eight set overlap divided by eight. These native
descriptive quantities are distinct from the v0.1 intervention analysis's
per-token velocity and signed change of that scalar. Preserve the versioned
definitions; do not combine their scales or call either an optimal sequence.

Accuracy and total token cost test utility on disjoint original-prompt families
at the 16,384-token scout endpoint. Include emitted tokens, controller-side
calls and injected tokens, with prefill/compute expenditure reported separately.
No minimum favorable effect is imposed. Similar observed accuracy is not
retention without a chosen noninferiority margin, and savings with an accuracy
interval spanning harm leave the tradeoff unresolved.

Family-grouped splits, intention-to-treat accounting, cluster bootstrap,
simultaneous intervals and explicit missing/failure receipts are statistical
and experimental controls adopted for this study. They are not attributed to
a particular steering paper. Engineering score-API parity and numerical
native repeats qualify measurement; they do not establish semantic efficacy.

The decision sequence is: valid pre-action opportunity → executed routing
change → arm-blind substantive transition effect → accuracy/cost effect.
Use expert contrasts and functional screens for selection, and motion for
diagnosis. Literature precedent supplies testable designs, not a guarantee
that these interventions will control reasoning trajectories.
