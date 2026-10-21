# Whole-reasoning strength sweep r10

Runs one more steering strength per model (gpt main, glm main, qwen36 half, qwen35 half, qwen330b half, gemma half, nemotron half, lfm25 half).
Only the efficiency-directed cells run (push up the step types tied to shorter reasoning, push down
those tied to longer), target and matched random experts, for every question. There are no unsteered
rows or canaries: unsteered rows come from each model's base release, and breakdown (rows stuck at the
length limit, looping) is reported as an outcome. Each model reuses its base release's nominations, so
experts and seeds match across strengths. Approved by the user on 2026-10-21 (plan: three strengths,
quarter, half and full). Own ceilings: 430 GPU-hours, 200 CPU core-hours;
GPU work stops at 2026-10-23 23:00 Europe/Rome. Released by the external reviewer (Claude).
