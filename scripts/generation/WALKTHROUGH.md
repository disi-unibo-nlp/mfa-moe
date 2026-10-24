# generation: walkthroughs

## generation-trace: Generation, one saved GPT-OSS trace
- commit: e63f4d8dc28cc751351c09cdb329d323de567c6b
- source: results/correlation_pipeline/card-v3/gpt/generation/openai--gpt-oss-20b/math500/generation_shards/math500_test_algebra_1004.json__sample_00.json (sha256 facdb9e6…; also line 0 of traces.jsonl in the same folder); every value below comes from a run on 2026-10-25 of the code at this commit on that file, with the HTTP call to the server replaced by the saved reply
- code: src/generation/generate_answers.py, src/generation/client.py, src/generation/model_profiles.py, src/generation/serve_flags.py, src/generation/benchmarks.py, src/generation/scoring.py, src/models/token_replay.py, scripts/generation/generate_answers.sbatch, tests/generation/test_generate_contract.py, tests/generation/test_serve_flags.py

**Input:** One saved trace of the primary corpus, GPT-OSS-20B on MATH-500, problem
`math500_test/algebra/1004.json`, attempt 0:
> If $2^8=4^x$, what is the value of $x$?
Traces like this one are made by `sbatch scripts/generation/generate_answers.sbatch openai/gpt-oss-20b math500 0 1
OUT_DIR`, which sources `scripts/slurm/env.sh`, starts the server (`start_server`, which runs
`scripts/serve/serve_model.sh`) and calls `python -m generation.generate_answers --model openai/gpt-oss-20b
--datasets math500 --shard-index 0 --num-shards 1 --base-url $URL --output-dir OUT_DIR`. The model is given by its
Hugging Face id; `models.roster.resolve` turns it into the short key `gpt` (`args.profile`), which is the key of
its card profile and is recorded in every trace. This walkthrough rebuilds, on the CPU, everything such a call
decides for this attempt and compares it with the saved file.

### 1. Profile (`model_profiles.CARD_PROFILES['gpt']`, `serve_flags.serve_argv`)

The sampler and the server settings come from the model's own card, not from a command-line flag.

| Field | Value |
|---|---|
| model, revision | `openai/gpt-oss-20b`, 6cee5e81ee83917806bbde320786a8fb61efebee |
| sampler | temperature 1.0, top_p 1.0, top_k 0 (off), no min_p, no penalties |
| output cap `max_tokens` | none: bounded by the 131,072-token context |
| reasoning effort, template kwargs | medium, none |
| attention, speculation, extra server args | TRITON_ATTN, none, none |
| requests in flight (`max_num_seqs`) | 16 |

`python -m generation.serve_flags openai/gpt-oss-20b` turns the same profile into the arguments of `vllm serve`
(one per line; `serve_model.sh` runs them): the model and `--revision`, the flags every server shares (two GPUs,
bfloat16 weights and KV cache, prefix caching, 8,192 batched tokens, GPU memory 0.90), `--reasoning-parser
openai_gptoss`, `--attention-config {"backend":"TRITON_ATTN"}`, `--max-model-len 131072` and `--max-num-seqs 16`.
No speculation flag is added.

### 2. Prompt (`benchmarks.sample_variant`, `benchmarks.generation_messages`)

One user message and no system prompt. The messages equal the saved `generation_messages` (`True`):

```
Please reason step by step, and put your final answer within \boxed{}.
Question: If $2^8=4^x$, what is the value of $x$?
```

### 3. Contract (`generate_answers._job_contract`)

Everything that can change the output is hashed: the sampler, context, server settings, the vLLM release (0.29.0,
without its `+cu129` tag), the scoring and replay versions and the seed `SEED + 1000 * example_index + sample`
= 20260925 + 1000 * 77 + 0 = 20337925. The server address is left out, so a resumed job may use a new port.
Row 77 of the MATH-500 loader is this problem (`math500_test/algebra/1004.json`, same prompt as the saved one).

| Hash | Rebuilt | Equal to the saved one |
|---|---|---|
| `example_sha256` | 8080d96389af18f75c615b90f72e126de64e0cdf511bd2e882051663e3f33839 | yes |
| `generation_sha256` | 94e9382d5d2ff79538415c429ca2534e58fbdcd9b9659c5abeeeb505a7737c99 | yes |

The 20 keys of `generation_config` equal the saved ones (`profile` is `gpt`). A finished attempt is reused only
when both hashes match, so this equality is what lets an interrupted job resume.
`tests/generation/test_generate_contract.py` and `tests/generation/test_serve_flags.py` pass (32 tests, 0.56 s:
12 and 20); the first holds the same `generation_sha256` for the first MATH-500 attempt of all seven models and
checks that the id on the command line resolves to the short key that is hashed, the second pins the server
flags of the seven models and the judge.

### 4. Request (`client.generate_completion`)

The post call was intercepted to capture the payload; the reply was filled from the saved trace
(`demo:` reply, real payload). Payload sent to `http://127.0.0.1:PORT/v1/chat/completions`:

```json
{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Please reason step by step, ... $2^8=4^x$, what is the value of $x$?"}],
 "max_tokens": null, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "seed": 20337925, "stream": false,
 "return_token_ids": true, "include_reasoning": true, "reasoning_effort": "medium"}
```

`top_k` 0 is sent as -1 (off); `min_p` and the penalties are left out because the card sets none.
`return_token_ids` makes the server return the exact prompt and completion ids that the next step keeps.

### 5. Token replay (`models.token_replay.make_token_replay`, called by `generate_answers._generate_trace`)

| | Value |
|---|---|
| prompt token ids | 104 |
| completion token ids | 263 |
| tokenizer hash | 23fd21da5b96c283… |
| first completion tokens | `<|channel|>` `analysis` `<|message|>` `We` ` need` ` to` (200005, 35644, 200008, 2167, 1309, 316) |
| last completion tokens | `}.\n` `\` `]` `<|return|>` (40066, 59, 60, 200002) |
| offsets of the first three tokens | [0, 11], [11, 19], [19, 30] |

The rebuilt `token_replay` and the rebuilt `cot_text` (720 characters) equal the saved ones. Later stages replay
these ids and offsets, and never render the chat template again.

### 6. Scoring (`scoring.score_completion_detailed`)

Only the final answer (`assistant_content`) is scored, never the reasoning, so a half-finished reasoning
trace cannot score by accident. The last `\boxed{}` is `4`, the gold answer is `4`: `model_answer` 4,
`is_correct` True, method `math_verify`, status `ok`. All equal the saved record.

### 7. The saved record

| Field | Value |
|---|---|
| `finish_reason` | stop |
| `usage` | prompt 104, completion 263, total 367 |
| `completion_tokens_details.reasoning_tokens` | 0 |
| `effective_max_tokens` | 130,968 (131,072 - 104) |

The file name is `math500_test_algebra_1004.json__sample_00.json` in
`OUT_DIR/openai--gpt-oss-20b/math500/generation_shards/`. The rebuilt record equals the saved one in every field
except `steps`: the saved file holds 6 strings there, the rebuilt one leaves it empty. All of `metadata` is equal.

### Worth noticing

- gpt-oss reports `reasoning_tokens` = 0 for this 263-token trace, although 88 of its tokens are reasoning:
  `labeling.sample_stratified.reasoning_tokens` counts 88 from the saved token replay (source `token_replay`), and
  the capture manifest of the same trace says `reasoning_tokens` 88. That is why length strata are counted from the
  saved token replay.
- The whole trace is one analysis paragraph of 88 tokens, then the final answer in the `final` channel; the
  reasoning ends where `<|channel|>final<|message|>` begins (character 239). The 94 tokens before that marker are
  the 3 channel-header tokens, the 88 reasoning tokens and `<|end|><|start|>assistant`.
- The seed encodes the problem position (77), so attempt 0 of problem 77 has the same seed in any rerun.
- Changing anything in the contract (a sampler value, the vLLM release, the scoring version) changes
  `generation_sha256` and makes every saved shard stale, which is the intended behaviour.
- The reply of step 4 is the only demo value; nothing here needed a GPU.
