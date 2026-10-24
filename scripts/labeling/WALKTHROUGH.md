# labeling: walkthroughs

## labeling-trace: Labeling, one saved sentence
- commit: e63f4d8dc28cc751351c09cdb329d323de567c6b
- source: sentence steps 1 to 5: the GPT-OSS trace math500_test/algebra/1004.json__sample_00 (results/correlation_pipeline/card-v3/gpt/generation/openai--gpt-oss-20b/math500) and its cohort-A annotation inline in /leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/manifests/gpt/A/0.json (annotation sha256 fdb7a363); step 6: the saved pilot label run results/correlation_pipeline/routing-dynamics-v1/pilot/label-checkpoints (472 checkpoint files, annotations.json, summary.json) and the pilot traces in results/correlation_pipeline/routing-dynamics-v1/pilot/generation; every value below comes from a run on 2026-10-25 of the code at this commit on those files
- code: src/labeling/spans.py, src/labeling/batch_predictor.py, src/labeling/label_sentences.py, src/labeling/gepa/optimize_judge.py, scripts/labeling/label_sentences.sbatch, tests/labeling/test_label_sentences_resume.py

**Input:** Sentence 4 of the trace that the generation walkthrough (generation-trace) followed, the fifth of eight
reasoning sentences of GPT-OSS-20B's answer to `If $2^8=4^x$, what is the value of $x$?`:
> So 8 = 2x → x = 4.
Its saved label is `Implement`. The judge is Qwen3.8-27B with the frozen GEPA program, normally run with
`sbatch scripts/labeling/label_sentences.sbatch TRACE_ROOT OUT_DIR` (which starts the judge server with
`start_server $JUDGE_MODEL`, that is `scripts/serve/serve_model.sh`, and calls
`python -m labeling.label_sentences --trace-root TRACE_ROOT --output-dir OUT_DIR --base-url $URL`). No judge was
called here: steps 1 to 5 rebuild what the judge is given, and step 6 resumes a finished label run from its
checkpoints.

### 1. Sentence units (`spans.sentence_spans`, `spans.trace_digest`)

The saved text (`cot_text`) has 720 characters; the segmenter cuts its reasoning (characters 30 to 214, after
`<|channel|>analysis<|message|>`) into 8 units with exact character offsets. The units equal those of the saved
annotation. The trace digest is 68d9059a… and equals the `trace_sha256` that the capture manifest froze for the
same trace.

| index | start | end | text |
|---|---|---|---|
| 0 | 30 | 57 | We need to solve 2^8 = 4^x. |
| 1 | 58 | 72 | Since 4 = 2^2. |
| 2 | 73 | 99 | So 4^x = (2^2)^x = 2^(2x). |
| 3 | 100 | 126 | So equation: 2^8 = 2^(2x). |
| **4** | **127** | **145** | **So 8 = 2x → x = 4.** |
| 5 | 146 | 159 | So answer: 4. |
| 6 | 160 | 178 | Provide reasoning. |
| 7 | 179 | 214 | We'll then output boxed: \boxed{4}. |

### 2. Which sentences are labelled (`spans.selected_sentence_indices`)

The sampler writes `metadata.sentence_selection` into the traces it hands to the labeller; the capture manifest
keeps that value for this trace: selection `campaign-v3`, stratum `{state: correct, cohort: A}`, indices 0 to 7.
Set on the trace, it makes `selected_sentence_indices` return 0 to 7. A trace this short has every sentence
selected, so all 8 are labelled; longer traces carry a sample of their sentences.

### 3. Judge inputs (`batch_predictor.question_of`, `batch_predictor.judge_inputs`)

The judge sees the question, the sentence and one neighbour on each side, never the rest of the trace.

```json
{"problem_statement": "Please reason step by step, and put your final answer within \\boxed{}.\nQuestion: If $2^8=4^x$, what is the value of $x$?",
 "previous_sentence": "So equation: 2^8 = 2^(2x).", "sentence": "So 8 = 2x → x = 4.", "next_sentence": "So answer: 4."}
```

The first sentence gets `<START OF RESPONSE>` as its previous sentence and the last gets `<END OF RESPONSE>` as its next.

### 4. Prompt (`batch_predictor.load_program_and_adapter`, `render_conversation`, `batch_payload`)

`gepa.optimize_judge.EpisodeJudge` loads the frozen program (`selected_program_20260827_173300.json`, sha256
467510e4fc1759bf…) and DSPy's `ChatAdapter` renders it. The conversation has two messages:

| Message | Size | Content |
|---|---|---|
| system | 36,578 characters | the field list, the output format and the GEPA instructions (all seven state names occur in it) |
| user | 443 characters | the four fields above, then "Respond with the corresponding output fields, starting with `[[ ## label ## ]]`" |

All conversations of a batch go in one request to `/v1/chat/completions/batch`: model `Qwen/Qwen3.8-27B`, n 1,
`max_tokens` 16,384, temperature 1.0, top_p 0.95, top_k 20, min_p 0.0 and chat-template kwargs
`enable_thinking` True, `reasoning_effort` low, `preserve_thinking` False.

### 5. Answer (`batch_predictor.parse_completion`, `label_sentences.outcome_record`)

The saved annotation of unit 4 is `{index: 4, start: 127, end: 145, text: "So 8 = 2x → x = 4.", label: "Implement"}`. The
eight labels of the trace are Read, Analyze, Implement (four times) and Monitor (twice). The raw reply of a
known label is not kept. `demo:` replies run through the real parser:

| Reply | Parsed |
|---|---|
| `[[ ## label ## ]]` `Implement` `[[ ## completed ## ]]` | Implement |
| the same with `Implementation` | `LabelParseError: invalid label 'Implementation'; expected one of Read, ...` |

An unreadable answer is stored as `unknown` together with its raw text and the failure, and never drops its
batch neighbours. `demo:` a batch of three replies through `classify_batch_outcomes` (the post call replaced): the
first gives `{"label": "Implement"}`, the `Implementation` reply gives `status` unknown with `failure.kind`
`label_parse`, and a reply cut at the length limit (`finish_reason` length) gives unknown with kind `non_stop`,
although its text contains `Implement`. `outcome_record` turns the first into the row plus `label`, the others into
the row plus `status`, `raw_completion` and `failure`.

### 6. Checkpoint binding (`label_sentences.enumerate_items`, `label_sentences.run_part`)

The pilot label run (Qwen3.6-35B-A3B-FP8 traces, 64 MATH-500 attempts) has 30,187 sentences in 472 checkpoint
files of 64 (the last holds 43). A checkpoint is bound to the hash of all items and to the judge settings:

| Check | Result |
|---|---|
| `enumerate_items` on the pilot traces: identity, unit and inputs of all rows | 30,187 of 30,187 equal the saved ones |
| `items_sha256` rebuilt / in the checkpoint | 38d6acd7ab832058… / the same |
| `run_part` on a copy of the checkpoints with a judge that raises if called | complete, 0 judge calls |
| `binding_sha256` and `annotations_sha256` of the new summary | c2d0af67…, ceaea640…: equal the saved summary |
| `annotations.json` written by that resume | byte for byte the saved file |
| the same run with temperature 0.7 in the config | `stale checkpoint` |
| judge and sampling keys of today's `checkpoint_config` vs the pilot's | all 13 equal (judge model, program sha256, decoding) |
| keys only the pilot has / only today's has | `pilot_manifest_sha256`, `revision` / `source_plan_sha256` |

The pilot's labels: Implement 8,901, Analyze 7,313, Verify 4,258, Monitor 3,448, Explore 2,905, Read 1,857,
Plan 1,505. The four test files for resume, include-traces, splitter version 2 and choosing the sentences
(`test_label_sentences_resume.py`, `test_label_sentences_include_traces.py`, `test_splitter_v2.py`,
`test_choose_sentences.py`): 26 passed, 13 subtests passed, 0.29 s.

### Worth noticing

- The judge's system prompt is 36.6 k characters for a 443-character question: the frozen program carries
  the whole annotation guide, and every sentence sent to the judge repeats it (the servers run with prefix
  caching, one of the shared flags of `generation/serve_flags.py`).
- The same units, trace digest and offsets are used by sampling, labelling and capture. The digest 68d9059a… in
  step 1 is the one the capture manifest froze, so a change of the segmenter would break all three.
- The pilot's checkpoints use other plan keys (`pilot_manifest_sha256`, `revision`) than today's
  `source_plan_sha256`. The rows and every judge setting still match today's code, and a changed setting is refused.
- Resume is a pure check: with all checkpoints valid the judge is never needed, and a mismatch stops before any request.
