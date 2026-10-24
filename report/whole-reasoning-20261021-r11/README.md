# Whole-reasoning top-up r11

Twelve base shards (r7: Qwen3-30B, Gemma x3, LFM2.5, Qwen3.5 x3, GLM x2; r8: GPT x2) failed between 16:11
and 16:20 on 2026-10-21: a full audit write met the WORK space reserve while WORK was almost full (space was
freed at 16:19). A thirteenth, GLM shard 2/24, failed later on an engine RPC timeout after 3h43.
This release runs only the rows of those shards that never started, at the base strength,
with the parent UIDs, seeds and actions (TOPUP.json). Rows that had started stay with the parent: complete,
or unknown if unfinished; none is redrawn. The combined analysis (SOURCES.json) joins each top-up to its
parent by UID. GPU work stops at 2026-10-23 23:00 Rome. Released by the external reviewer (Claude).
