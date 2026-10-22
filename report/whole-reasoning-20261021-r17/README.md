# Whole-reasoning resume r17

Continues the rows its parent release or its top-ups left partial when a job reached its walltime or
crashed, after all their jobs ended. Rows keep the parent UIDs, seeds and actions; each saved partial is
copied into this release's own store (receipts in <model>/resume/) and re-prefilled through the same
hooks, so nothing is redrawn and no source store is written. A row interrupted again stays unknown.
Rows per model: qwen36 171, qwen35 54, qwen330b 24, gemma 57, glm 238, nemotron 17, lfm25 239.
GPU work stops at 2026-10-23 23:00 Rome. Released by the external reviewer (Claude).
