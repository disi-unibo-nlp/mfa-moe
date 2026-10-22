# Outcome-selected gentle steering (outcome-steering-20261022-r1)

Nudges a small set of experts chosen by outcome (used more in wrong than right attempts of the same
question, campaign-v3 cohort B) by subtracting a delta from their router logits at every prompt and
reasoning token, against matched random experts, and compares both with unedited routing (the base
releases' native runs) and a stop-thinking-earlier baseline cut from those native runs.
Models: gpt, qwen330b, nemotron; deltas [0.4, 0.8, 1.6]; length fractions [0.5, 0.7, 0.85].
Code: src/moe_exp/outcome_steering (README there). Ledger F-0004, D-0003.
GPU work stops at 2026-10-23 23:00 Rome.
