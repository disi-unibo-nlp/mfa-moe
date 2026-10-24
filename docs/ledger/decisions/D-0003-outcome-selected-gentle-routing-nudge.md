# D-0003: Outcome-selected gentle routing nudge
- status: active
- date: 2026-10-22
- decided_by: user
- supersedes: 
- scope: The outcome-steering experiment (F-0004) on gpt, qwen330b and nemotron; extension to the other models with cohort-B captures only if the three-model result is positive.
- not_scope: Changes to the finished whole-reasoning releases r7-r17 or their analysis; manuscript claims before a sealed receipt.
- chosen: Select experts by outcome on campaign-v3 cohort B (per-attempt usage per reasoning token, z-scored within question, mean wrong-minus-right over mixed-outcome questions, sign stable under leave-one-question-out). Exclude the shared core (top 5% native frequency per layer) and the most execution-specific experts (union over folds of the top 10% positive Implement/Analyze risk differences). Take stable positive pairs until 1.2% of native routing mass (5 to 40 pairs; fill to 5 by score with an unstable_fill flag if needed). Families whose question is in cohort B use the set selected without that question. Subtract delta in {0.4, 0.8, 1.6} from the selected experts' router logits at every prompt and reasoning token, keeping native top-k and renormalisation. Matched random controls with the same layers, frequency quintiles and exclusions. Reuse the base releases' native and native_repeat rows. Length-limit baseline: cut each family's native reasoning at 50/70/85% of its length, append the closing suffix and generate only the answer (allowance min(2048, remaining)).
- rejected: Class-risk-difference nomination (the 87-cell whole design found no helpful experts); forced on/off membership; excluding every expert with any positive Implement/Analyze link (empties the control pools); fresh native reruns.
- evidence: User answers 2026-10-22 (3 models first, length-limit baseline, three strengths 0.4/0.8/1.6, top-10% execution protection, reuse natives); analysis in /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/claude-review/why-steering-20261022/; chen2026beyond ("Beyond the Trace") reports shorter and more accurate GPT-OSS-20B output with five outcome-selected experts at a -0.8 logit shift.
- keywords: outcome, nudge, steering, budget, early-exit, routing
- paths: src/steering/outcome, src/steering/operators.py, scripts/steering, src/steering

## Extension rule (declared 2026-10-22, before any nudge outcome)

The user asked to extend to every model if the three-model result is good, and to schedule every
step automatically. A CPU job after the r1 analysis applies `moe_exp.outcome_steering.decide`: extend
when some first-wave model and delta is shorter than matched random experts (bounded simultaneous
token interval above zero) with accuracy against unedited routing not below -1 point, or more
accurate than random experts (accuracy interval above zero). If met, it launches
outcome-steering-20261022-r2 for gemma, glm, qwen35 and qwen36 (LFM2.5 has no cohort-B captures),
GLM with more shards because it decodes about four times slower.
