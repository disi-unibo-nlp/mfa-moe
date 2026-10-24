# What the reference paper contributes to this design

The user-supplied private manuscript is **Rational Manifolds: A Nonlinear Atlas for LLM Reasoning**, a 19-page ICLR 2027 submission. It was read locally from [the supplied PDF](../../report/iclr2027_conference_anonym_private.pdf). It supplies methodological inspiration, not evidence that our MoE routing method works. Do not upload or publish the PDF as part of the wiki.

## October 8 interview: dynamics and depth

The user explicitly requires trajectory dynamics to be **equally important**
to the immediate causal class response, and asks that this reference inform the
design ([ADR14, Q15/L1](DECISIONS.md#adr-20261008-14-joint-research-design-from-existing-evidence)).
The parent reread PDF pages 1–9 and the selection appendix on pages 17–19 locally
with Ghostscript; no upload or new scientific computation was performed.

| Reference idea and location | Relevant use in our revised study | Measurement boundary |
|---|---|---|
| Sentence representations become walks on a directed graph; §3.1, pp. 4–5 | Describe observed seven-class transitions, occupancy, persistence and re-entry; measure how assigned steering changes the subsequent trajectory | Historical missing labels break adjacency. Seven text classes are not MFA regions; a row-normalized graph alone does not prove a first-order Markov process. |
| Most regions contain multiple episode types; suppressing an associated region can increase its class share; §4.1/Fig. 4, p. 7 | Report the full target-by-observed-class response and both intervention signs, including compensation and spillover, rather than only the targeted diagonal | Class association does not establish one-expert/one-function or predict the sign of its intervention. |
| Local and global functional directions differ; §4.4, pp. 9–10 | Compare pooled expert/class associations with source-state/progress-conditioned associations on discovery families, and report context-dependent intervention effects where supported | Routing allocations and residual-space geometry are different representations. No new hidden-state collection or MFA fit is implied. |
| Multi-depth atlases, but interventions fixed at relative depth 0.5; §3.2, p. 6 | Analyse every routed layer, then nominate early/middle/late sites per class/model using explicit discovery statistics and compare their held-out effects | Preserve native layer IDs. A routed-depth index can skip physical decoder layers; middle depth is not established as universally optimal. |
| Perturbation exposure and answer effects reported together; Figs. 2/5 and §4.2 | Measure actual expert selection/weight displacement during the pulse, subsequent routing recovery, class dynamics, continuation accuracy and tokens | Nominal score shifts, actual routing engagement and behavioral effects are different quantities; ratios to tiny exposure can be unstable. |
| Candidate selection separated from test outcomes; §3.2 and Appendix B | Compute empirical layer/expert nominations on discovery families; retain held-out effects and uncertainty | Q16 excludes new training. Q7 requests removal curves without a required favorable point; the reference's gain-selected region heuristics do not override either instruction. |

The immediate and dynamic endpoints still need their exact pulse/horizon,
missingness and simultaneous-inference contract. Transition graphs conditioned
on states reached after treatment are mechanism descriptions; causal dynamics
should also use assignment-level occupancy/event/count contrasts with terminal
and unknown cases retained. This avoids calling differences among treatment-
selected survivors a causal transition effect.

## Comparable questions, different interventions

This comparison retains the historical v2/v3 scopes where explicitly named.
The current design uses Q18 signed score offsets on empirical expert sets;
its pulse/observation windows are indexed settings, still to be defined.

| Aspect | Reference manuscript | This project |
|---|---|---|
| Representation | Mean sentence residual activations; unsupervised mixture of factor analyzers, K=50 and rank 25 | Model-native expert allocations and optional hidden-state readouts; v2 context clusters use ID composition |
| Functional labels | Eight episode types, including Answer | Seven reasoning episode classes; answer/closure tracked separately |
| Structure | Directed region graph with row-normalized transitions | Contiguous class transitions, native expert profiles, prospective feature/label timing |
| Action | Residual reconstruction displacement between source/target regions | Selection-score bias, fixed-k forced inclusion, or executed selected-mixture reweighting |
| Selection | Single-region screen followed by combined selected regions | Conditional discovery targets followed by operator/dose/semantic checks |
| Layer | A fixed middle layer | Model-specific observed layer/expert candidates; no universally best layer established |
| Data separation | 1,000 training problems for atlas/graph, 250 validation for region/strength choice, official test outcomes; SuperGPQA supplies GPQA train/validation | Historical cohorts and prior exposure retained explicitly; independent family seals for new validation/final work |
| Budget | 2,048-token evaluation | Historical budgets differ; new v2 scientific cap 16,384, early semantic endpoint 1,024 |
| Controls | CAA, Overclocking, covariance-matched noise and reported perturbation exposure | Native, matched random, incumbent and appropriate timing/operator controls, with actual allocation/weight dose |

The reference applies `h' = h + alpha * (rho_target(h) - rho_source(h))`, with a running current-sentence mean and reasoning-only intervention. This is not expert dispatch. Its middle layer, region count, labels and favorable budget do not identify our actuator or optimal cap. See its methods and intervention sections, main results, and appendix details (PDF pp. 3–11, 17–19).

## What we should reuse

Its useful experimental logic is: map representation to interpretable behavior, test individual candidates before combinations, measure what an intervention actually changes, then evaluate accuracy and tokens on selection-disjoint data. Add our explicit information-time, missingness and native-router semantics checks. A transition graph is a description; a graph-guided intervention must still beat relevant controls.

We already have clean routing readouts, two-model next-class results, conditional
candidates, historical readers and random-target interventions. The current
revision adds an explicit all-layer selection rule, all-class bidirectional
causal scope and equally important dynamics. Its remaining measurements must
be identified from existing artifacts; the stopped verification campaign's
opportunity gates are not automatically its new scientific contract.

## What its results do not justify

The authors report multi-functional regions, cases where rerouting away from a type increases its observed share, and the difficulty of turning attribution into control. Brevity promotion loses more than 10% relative accuracy in 6/23 settings. These observations support caution about a one-region/one-operation or one-expert/one-operation interpretation.

One table point example is Qwen3-32B/MATH joint promotion: accuracy 25.7% → 52.8%, tokens 1,951 → 1,466. These are the reference's point estimates, under its 2k budget and selection procedure, not an expected effect size or power input for our MoE study. Paired accuracy uncertainty is not shown in that table. Some draft prose is also stronger than the accompanying numbers: the Olmo/MATH promotion-accuracy token row is 2,001 → 1,929, about 3.6%, rather than a fifth. Treat the manuscript as a design example requiring scrutiny, not an unquestioned standard of proof.

The credible distinction for our paper is **native sparse allocation, exact routing engagement, and the cross-model relation between readable episode information and causal control**. Context conditioning is a hypothesis about effect heterogeneity; it is not evidence that we reproduced residual-space manifolds. SSD preloading is a separate systems question absent from these intervention results.
