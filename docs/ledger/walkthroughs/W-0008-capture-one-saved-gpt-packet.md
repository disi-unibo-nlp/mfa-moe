# W-0008: Capture, one saved GPT-OSS packet
- date: 2026-10-24
- commit: 610ed2a6b1f5089b828de755c0eb951df4df7e08
- source: /leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3: manifests/gpt/A/0.json (binding 120fe778), capture/gpt/A/0/9c56fb4509d14ce5960d184c2fe9a45606ccf4be932c02f0d4aabf33f88213ce.{complete.json,json.gz,pt.gz} and result.json (sha256 ba0330c3), and the accepted cohort in /leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/acceptance/capture-gpt-A.json; every value below comes from a run on 2026-10-24 of the code at this commit on those files
- paths: src/capture/manifest.py, src/capture/reduce.py, src/capture/gate.py, src/capture/dynamics/classes.py, src/models/router_adapters.py, scripts/capture/gate.sh, tests/capture/test_capture_reduce.py, tests/models/test_streaming_routing_extraction.py
- keywords: capture, router, packet, manifest, gate, gpt-oss, expert counts

**Input:** The cohort-A capture of the trace that the two earlier walkthroughs followed (W-0006 generated it,
W-0007 labelled it): GPT-OSS-20B on MATH-500 `algebra/1004`, 263 completion tokens, capture id
9c56fb4509d14ce5960d184c2fe9a45606ccf4be932c02f0d4aabf33f88213ce. The capture itself needs two GPUs
(`sbatch scripts/capture/capture.sbatch gpt MANIFEST OUTPUT_DIR`, `python -m capture.run`) and its files are dated
2026-09-27; this walkthrough reads what it left, rebuilds what can be rebuilt on a CPU, and runs the acceptance gate.

## 1. Manifest (`manifest.load_manifest`, `manifest.read_location`)

The frozen manifest of GPT cohort A, shard 0, is checked against its own binding hash (120fe778…), the cohort
file and the annotation inputs it names. It lists 1,547 attempts and 6,087,185 tokens; its spec has
`capture_version` 2, 24 routed layers, 32 experts and top-k 4. The row of this trace: 263 completion tokens,
367 input tokens, 88 reasoning tokens. The source trace line is read back by byte offset and its size, time
and line hash are checked; its trace digest 68d9059a… equals the row's `trace_sha256`.

## 2. Marker and files (`manifest.resume_records`)

A trace counts as captured only when its marker `<id>.complete.json` lists two files whose sizes and hashes
still match. This marker names `…pt.gz` (61,095 bytes, sha256 ee788d61…) and `…json.gz` (54,819 bytes,
294a6ba7…), 115,914 bytes in all, written by job 58767921 in 1.69 s. Both verify.

## 3. Tensors and packet (`<id>.pt.gz`, `<id>.json.gz`)

| Tensor | Shape and type | Meaning |
|---|---|---|
| `ids` | (24, 263, 4), int16 | experts the router really dispatched, per layer and token |
| `weights` | (24, 263, 4), float16 | their routing weights |
| `native_boundary_gaps` | (24, 263), float32 | selection margin; minimum 0.0, mean 0.1208 |
| `reasoning_tokens` | (88,), int32 | the tokens that belong to the reasoning |

Layer 0, token 0 dispatched experts [31, 21, 22, 6] with weights [0.5195, 0.168, 0.1572, 0.1543] (sum 0.999 in
fp16). The packet holds the trace, 286 feature values (schema 3), 24 routing-dynamics window records, the
cohort-A views (19 scopes; label coverage 8 selected, 8 labelled, 0 unknown), the expert counts, the dispatch audit and the manifest binding.

## 4. Reductions rebuilt from the saved ids (`reduce.counts_for_tokens`, `marker_counts`, `expert_class_counts`)

`reduce_capture` needs the raw router signal and hidden states, which are not saved, so the whole reduction
cannot be rerun. The parts that need only ids, text and labels can:

| Packet field | Rebuilt value | Equal to the packet |
|---|---|---|
| `native_expert_counts` | 24 layers x 32 experts, each layer sums to 263 x 4 = 1,052 | yes |
| reasoning tokens of `saved_layout` | 88, the same indices as the tensor | yes |
| `marker_counts` (`<think>`, `</think>`, `Alternatively`) | 0 occurrences each, status `unavailable_no_occurrence` | yes |
| `expert_class` (`normalize_annotation`, `sentence_table`) | 88 labelled tokens: Read 14, Analyze 9, Implement 52, Monitor 13 | yes |

The class counts come from the 8 labels of W-0007: a token counts for the sentence that owns it by largest
character overlap, so each reasoning token is counted once.

## 5. The hooks behind the ids (`models.router_adapters.stream_router_signals`)

GPT-OSS is captured with Hugging Face transformers (`backend: hf` in the packet), with the forward pass in
native eager attention, query chunks of 128 (`attention_execution: native_eager_query_chunked`). The hooks read
the experts the router dispatched, not a top-k of raw scores, and keep the absolute layer index; the packet's
`native_dispatch` audit is True for layers 0 to 23. `tests/models` (the streaming extraction against the full
forward pass on a tiny model): 15 passed in 88.6 s. `tests/capture`: 45 passed, 22 skipped (they need torch).

## 6. Gate (`gate.gate`, run as `scripts/capture/gate.sh MANIFEST_DIR OUTPUT.json`)

The gate checks the cohort as a whole: the shard list equals `ready.json`, the verification receipt passed,
every trace has its two files with matching sizes and hashes, no trace is missing or double, and the shard
`result.json` agrees with the markers and with the planned totals. On the login node it hashed the 3.2 GB of
GPT cohort A in 12.9 s:

| Field | Rebuilt | Accepted result of 2026-09-27 |
|---|---|---|
| status | complete | complete |
| attempts, tokens | 1,547; 6,087,185 | 1,547; 6,087,185 |
| `ready_sha256` | ab012743… | ab012743… |
| `result.json` sha256 | ba0330c3… | ba0330c3… |

The saved acceptance has one more key, `imports`, from the frozen bundle that made it; today's gate writes
`code` (the commit and the hash of `src/`) instead. Every other field is equal.

## Worth noticing

- The packet is trusted through a chain of hashes (manifest binding, marker, two file hashes, the cohort
  result, the gate), not because it can be recomputed: the reduction of features, windows and views cannot be
  replayed without a new GPU capture.
- The 263 replayed tokens are exactly the 263 tokens W-0006 saved; only 88 of them are reasoning, the other
  175 (closing markers and the final answer) lie outside the views.
- One more reason for the checks: a layer-0 weight vector sums to 0.999, not 1, because weights are stored in fp16.
- The gate is cheap (13 s for 3.2 GB) but refuses to run outside Slurm as a command (`python -m capture.gate`
  checks `SLURM_JOB_ID`); the function was called directly here.
- The capture code of the repository is the last version of the bundles that ran; `src/capture/README.md` notes
  that a rerun records a `capture_code_sha256` that differs from the historical receipts.
