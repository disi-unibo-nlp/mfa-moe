# Research glossary

| Term | Meaning in this project | Do not confuse with |
|---|---|---|
| Problem identity | Canonical dataset and source-problem ID, independent of model and attempt | One generated response |
| Duplicate family | Connected set of known exact/near-duplicate problems, frozen before splits | A benchmark or model family |
| Attempt or assignment | One declared arm/seed/prefix or original-prompt request, including failures and no-fires | An independent statistical unit |
| Reasoning episode class | One of Read, Analyze, Plan, Implement, Explore, Verify, Monitor assigned to visible text | A measured latent cognitive state |
| Observed state transition | A class change or repetition between original consecutive labelled sentences in the same trace segment, with family support reported | Adjacency created by dropping unlabelled sentences; complete latent dynamics |
| Routing-defined state | A grouping learned from routing features under a stated representation and training population | An independently annotated reasoning operation; correlation with its own defining features is not external validation |
| Substantive verification | A visible check of an original constraint or computation under the frozen rubric | Saying “let me verify” or receiving a Verify class label |
| Substantive commitment | An identified approach followed by a concrete committed step under the rubric | Generic intention or planning language |
| Prefix opportunity | A condition recognized using only original problem and already emitted tokens | A retrospective span selected using future text |
| Native routing | The model's actual selection semantics at a particular hidden state | Top-k of raw scores when correction biases or other rules apply |
| Selection frequency | Fraction of token positions selecting a layer-qualified expert | Gate probability, executed weight, or compute time |
| Selected weight | Weight actually used for an executed expert mixture | A probability over all experts |
| Expert identity | At least model, checkpoint, layer and expert index | An expert index pooled across layers/models |
| Target nomination | Discovery-only association used to propose an intervention | Evidence that the expert causes the behavior |
| Discovery statistics | Empirical counts, rates, associations and explicit rules computed without using final evaluation families | Training a new predictor, model, adapter or learned steering vector |
| Current-class association | Routing during a labelled sentence related to that sentence's class | Source-only routing associated with the next sentence's class |
| Source-conditioned next-class association | Routing in sentence t related to the original adjacent label at t+1, comparing destinations from the same source class | Using destination routing or retrospective future text as a deployable prefix input |
| Bias | Alter selection scores before native top-k | Membership-preserving reweighting |
| Routing deactivation | Suppress nominated layer-qualified experts in the selection procedure while retaining the model's allocated weights; the exact score operator is declared | Physical weight removal, output-zeroing after selection, or guaranteed memory/compute savings |
| Output ablation | Zero selected expert contributions after the native selection under an explicit normalization rule | Excluding experts before selection and replacing them with other native top-k candidates |
| Intervention strength | A declared control such as the number of targeted experts or calibrated score displacement; its definition is fixed before evaluation | Treating expert-count levels and score amplitudes as the same treatment |
| Conditional continuation utility | Final-answer accuracy and new/total token cost when completing a declared shared prefix under assigned interventions | Original-question policy utility or benchmark performance from fresh original-prompt runs |
| Forced inclusion | Ensure specified experts enter the executed set while preserving the declared k | A small logit perturbation |
| Reweighting | Change the executed selected-mixture weights under a specified support rule | Expert selection frequency |
| Dose or target engagement | Realized membership/weight displacement during intended action opportunities | Nominal bias magnitude or substantive success |
| No-op fidelity | Native and executed paths agree at the same hidden state when inactive | Bit-identical outputs from separate runs |
| Matched random control | Random target identities with specified layer/support/exposure and measured dose | Automatically equal causal strength |
| Anticipation | Predicting an outcome strictly after the available prefix | Classifying the just-observed destination |
| Composition | Which experts occurred and how often | Their temporal order |
| Cross-layer routing pattern | Joint layer-qualified expert selections for the same aligned token, retaining the declared native selected sets or weights | An expert identity shared across layers, or observed pairwise causal communication |
| Temporal routing pattern | Ordered selections at one fixed layer across aligned token positions or windows | Moving through model depth at one token; aggregate usage entropy |
| Dominant-expert path | A declared one-expert-per-layer projection of a token's routing; historical depth studies choose the largest executed selected weight | The full native top-k mixture or the only computational route |
| Marginal routing counts | Aggregated expert occurrences in a stated layer/class/attempt population | Joint expert co-selection, token order or cross-layer paths |
| Context in v2 | Discovery-fit clustering of native expert-ID composition with K=1 fallback | Hidden-activation MFA regions in the reference |
| Causal control | An assigned intervention changes the specified outcome versus appropriate controls | An observational association or a manipulation check alone |
| Utility | Original-prompt accuracy and complete policy cost under a fixed execution contract | Short capped continuation behavior |
| Policy token cost | Generator plus controller output tokens in the declared protocol | Offline annotation cost, input processing or wall-clock latency; report those separately |
| Cap | Administrative stopping at a fixed budget | Natural completion or a failed reasoning state |
| Closure | Observed reasoning termination; competing outcome for a future reasoning transition | Window truncation or missing labels |
| ITT | Retain all assigned requests, including nonfires, caps, errors and unscored cases | Analysis only among successful interventions |
| Simultaneous interval | Uncertainty adjusted over the frozen comparison family | A nominal interval selected after looking at many contrasts |
| Missing-outcome bounds | Effects consistent with unresolved outcomes under stated extreme assignments | A confidence interval or reader reliability estimate |
| Incumbent | Previously frozen, unproven policy used as a comparator | A proven winner or optimal policy |
| Qualified | Passed the exact named measurement or execution contract | Globally valid or authorized for new workloads |
| Reader draw | One seeded judgment from the fixed semantic reader checkpoint | An independent annotator or independent problem |
| Confirmation fraction | Family-weighted share of proposed decisions receiving two usable positive reader judgments; unknowns and disagreements count as unconfirmed | Independently validated semantic precision or evidence that every unconfirmed proposal is wrong |
| Optimistic unknown completion | Recompute the finite-audit fraction with every unresolved/disagreeing pair positive, retaining agreeing negatives; conditional on those reader labels | A confidence interval, correction for reader bias, new draw, or replacement qualification gate |
| Controller qualification limit | The named opportunity detector and measurement procedure did not pass its prespecified support/confirmation gate | A measured null effect of an intervention that was never applied |
| Frozen | Bound to a declared source, assignment, code or analysis version | Impossible to amend prospectively |
| Analysis configuration | Window, lag or statistic applied to compatible retained outputs | A change to the model's executed action or sampled continuation |
| Execution configuration | Source identity, model/runtime, seed, action/mask, decoding and stopping contract that determines a requested execution | A plot setting; native data do not contain every counterfactual execution |
| Threshold index | Stable row identifier in a versioned configuration table, mapped explicitly to analysis and execution dependencies | An unrecorded numeric cutoff or permission to run every grid combination |
| Reuse classification | Per-result status: existing output, derivable from saved input, compatibility unresolved, or new outcome required | Treating an uninspected archive as missing, or a new schema as a need to regenerate |
| Keep-set | Layer-qualified expert identities selected under a declared training rule and resource budget | A proven safe deletion policy |
| Labelled selection coverage | Fraction of original labelled token-layer-expert slots whose IDs fall in a specified set | All-token coverage, complete native top-k containment, selected-weight mass or rerouted behavior |
| Static expert retention | One keep-set frozen before evaluation and unchanged across test requests | Online state-conditioned routing, which needs prefix-only inputs |
| Serialized expert bytes | Tensor payload and associated quantization metadata in checkpoint headers, with scope/aliases specified | Actual GPU resident bytes or measured freed memory |
| Resident expert removal | An implementation actually omits or releases expert storage and records surviving ID mapping | Masking selection while every expert tensor stays allocated |
| Joint accuracy–memory point | Assigned accuracy with uncertainty/unknown bounds alongside measured storage and runtime memory | Accuracy retention inferred from a nonsignificant difference |
| Configured result cell | One registered model/population/method/setting comparison row; several rows may alias one execution | A completed run or an independent sample |
| Execution alias | Multiple comparison rows referencing the same compatible execution identity | Permission to combine incompatible baselines or count replicas as independent families |
| Report artifact completeness | A saved report has enumerated results, missing cells and status at its stated cutoff | Scientific completion, successful paper publication, or a campaign-root completion receipt |
| Publication completion | A verified immutable report snapshot and its intended current-paper selection have been committed | A partial set of copied figures or an output report's internal completion flag |
| Operational dependency | A prerequisite imposed by the scheduler graph | Necessarily an input required to compute the downstream scientific operator |
| Results-first scheduling | Prioritize outcome-critical qualified work while preserving frozen assignments, controls, complete admission and required analyses | Dropping expensive comparisons, choosing favorable outcomes or bypassing qualification |
