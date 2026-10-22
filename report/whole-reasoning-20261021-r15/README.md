# Whole-reasoning resume r15

Continues the rows its parent release or its top-ups left partial when a job reached its walltime or
crashed, after all their jobs ended. Rows keep the parent UIDs, seeds and actions; each saved partial is
copied into this release's own store (receipts in <model>/resume/) and re-prefilled through the same
hooks, so nothing is redrawn and no source store is written. A row interrupted again stays unknown.
Rows per model: gpt 127, qwen36 364, qwen35 418, qwen330b 222, gemma 297, glm 493, nemotron 38, lfm25 348.
GPU work stops at 2026-10-23 23:00 Rome. Released by the external reviewer (Claude).
