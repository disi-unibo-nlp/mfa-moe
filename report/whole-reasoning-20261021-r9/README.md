# Whole-reasoning strength sweep r9

Runs one more steering strength per model (gpt quarter, qwen36 quarter, qwen35 quarter, qwen330b quarter, gemma quarter, glm quarter, nemotron quarter, lfm25 quarter).
Only the efficiency-directed cells run (push up the step types tied to shorter reasoning, push down
those tied to longer), target and matched random experts, for every question. There are no unsteered
rows or canaries: unsteered rows come from each model's base release, and breakdown (rows stuck at the
length limit, looping) is reported as an outcome. Each model reuses its base release's nominations, so
experts and seeds match across strengths. Approved by the user on 2026-10-21 (plan: three strengths,
quarter, half and full). Own ceilings: 430 GPU-hours, 150 CPU core-hours;
GPU work stops at 2026-10-23 23:00 Europe/Rome. Released by the external reviewer (Claude).
