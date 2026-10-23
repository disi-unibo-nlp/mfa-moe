# Whole-reasoning complement (whole-reasoning-20261021-r18)

The base releases suppress only the states tied to longer reasoning. This release suppresses the
other states at each model's base strength (GPT and GLM half, the others full) on the same families
and seeds, so the per-cell results cover both operations for all seven states. The new cells are
not efficiency-directed and never enter the primary or the dose response. Its combine job writes
the final combined analysis of r7-r17 plus these cells to analysis-combined/EFFECTS_WHOLE.json.
Shards: gpt 3, nemotron 4, qwen35 12, qwen36 12, gemma 6, glm 27, qwen330b 3, lfm25 8. GPU work stops at 2026-10-24 06:00 Rome.
