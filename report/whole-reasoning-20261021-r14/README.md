# Whole-reasoning top-up r14

Runs the never-started rows of every shard of its parent release, after all its jobs ended (shards that
reached their 4-hour walltime, chiefly the slow LFM2.5 and GLM, and rows the first top-up r11 never
reached). Rows keep the parent UIDs, seeds and actions; started rows stay with their release, complete or
unknown. Rows per model: lfm25 448.
GPU work stops at 2026-10-23 23:00 Rome. Released by the external reviewer (Claude).
