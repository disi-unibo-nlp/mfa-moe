# Whole-reasoning top-up r12

Runs the never-started rows of every shard of its parent release, after all its jobs ended (shards that
reached their 4-hour walltime, chiefly the slow LFM2.5 and GLM, and rows the first top-up r11 never
reached). Rows keep the parent UIDs, seeds and actions; started rows stay with their release, complete or
unknown. Rows per model: qwen36 102, qwen35 87, qwen330b 165, glm 59, lfm25 1219.
GPU work stops at 2026-10-23 23:00 Rome. Released by the external reviewer (Claude).
