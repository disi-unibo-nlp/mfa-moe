# Dense trajectory analysis input v1

`scripts/experimental_resume/analyze_dense_trajectory_v1.py` runs on a CPU Slurm
node after 1,024-token mechanism generation and independent offline measurement.
It accepts one sealed JSON document with `schema: dense-trajectory-analysis-input-v1`,
`horizon: 1024`, `population`, `ordered_transitions` (one or two frozen registered
names), explicit `comparison_pairs`, `bindings`, `assignments`, and `observations`.
`bindings` has seven lowercase SHA256 strings: `generation_manifest_sha256`,
`generation_summary_sha256`, `tokenizer_config_sha256`,
`sentence_label_binding_sha256`, `semantic_rating_binding_sha256`,
`blind_map_sha256`, and `rubric_sha256`. The upstream join must verify each source
seal and map before creating this input. The `sha256`
is SHA256 of the other JSON fields using sorted keys, compact separators, UTF-8,
and `ensure_ascii=False`.

Each assignment has `uid`, `question`, `family`, `arm`, `seed`, and
`trigger_sentence_index`. Each observation repeats the first five identity fields,
then has `status`, binary `correct`, `injected_token_count`, and
`emitted_token_ids`. Every assigned UID needs one observation, including nonfires,
early finishes, caps, failures, and unscored results. `correct` is zero for failed
or unscored accuracy. The token endpoint is emitted plus injected tokens.

For all generated rows other than `failure`, including `nonfire`, each observation
additionally has
`reasoning_closed`, `reasoning_token_end`, `token_owner`, `routed_positions`, and
`sentences`. The reasoning end is the first Qwen3.6 `</think>` token position or
the length of the emitted continuation. `token_owner` has one integer sentence
index or null per reasoning token. `routed_positions` must enumerate exactly every
reasoning token position. These arrays bind the label spans to emitted and routed
token rows. The producer must bind the source tokenizer, raw arrays, and decoder
hashes in its upstream sealed frame; this analyzer checks their joined structure.

Sentence rows have contiguous post-trigger `sentence_index`, one `segment`,
half-open branch-relative `token_start` and `token_end`, `complete`, `label`,
`label_finish_reason`, and two arm-blind `behavior_votes` from `reader0` and
`reader1`. Each vote has `reader`, `finish_reason`, `transition` (registered name
or null), and `substantive` (boolean). A partial final sentence never counts.
A class cannot count as a semantic event without both independent reader votes
agreeing on the substantive behavioral transition. Missing or discordant ratings
remain in the ITT table with semantic success zero and an explicit measurement
status. Triggering sentences are excluded. No sparse gaps, changed reasoning
segments, crossing token spans, post-closure labels, or over-horizon tokens are
accepted.

The output includes a receipt for every assignment, within-request seven-class
transition/dwell/re-entry/loop summaries, and frozen paired family-clustered ITT
intervals from `analysis.paired_itt`. All three endpoints—semantic completion,
correctness, and token cost—are reported for each explicit comparison pair.
This is an LLM audit of one frozen local template; it does not establish latent
reasoning or a universal optimal sequence.
