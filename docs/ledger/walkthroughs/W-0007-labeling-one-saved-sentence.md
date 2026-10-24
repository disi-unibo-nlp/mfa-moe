# W-0007: Labeling, one saved sentence
- date: 2026-10-24
- commit: 610ed2a6b1f5089b828de755c0eb951df4df7e08
- source: sentence steps 1 to 5: the GPT-OSS trace math500_test/algebra/1004.json__sample_00 (results/correlation_pipeline/card-v3/gpt/generation/openai--gpt-oss-20b/math500) and its cohort-A annotation inline in /leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/manifests/gpt/A/0.json (annotation sha256 fdb7a363); step 6: the saved pilot label run results/correlation_pipeline/routing-dynamics-v1/pilot/label-checkpoints (472 checkpoint files, annotations.json, summary.json); every value below comes from a run on 2026-10-24 of the code at this commit on those files
- paths: src/labeling/spans.py, src/labeling/batch_predictor.py, src/labeling/annotation_batch.py, src/labeling/gepa/run.py, scripts/labeling/label.sbatch, tests/labeling/test_annotation_batch_resume.py
- keywords: labeling, sentence, spans, judge, dspy, checkpoint, annotation

**Input:** Sentence 4 of the trace that the generation walkthrough (W-0006) followed, the fifth of eight
reasoning sentences of GPT-OSS-20B's answer to `If $2^8=4^x$, what is the value of $x$?`:
> So 8 = 2x → x = 4.

Its saved label is `Implement`. The judge is Qwen3.8-27B with the frozen GEPA program, normally run with
`sbatch scripts/labeling/label.sbatch TRACE_ROOT OUT_DIR` (which starts `scripts/serve/judge.sh` and calls
`python -m labeling.annotation_batch`). No judge was called here: steps 1 to 5 rebuild what the judge is
given, and step 6 resumes a finished label run from its checkpoints.

## 1. Sentence units (`spans.sentence_spans`, `spans.trace_digest`)

The saved text (`cot_text`) has 720 characters; the segmenter cuts its reasoning into 8 units with exact
character offsets (the first starts at 30, after `<|channel|>analysis<|message|>`). The trace digest is 68d9059a… and
equals the `trace_sha256` that the capture manifest froze for the same trace.

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

## 2. Which sentences are labelled (`spans.selected_sentence_indices`)

The trace carries `metadata.sentence_selection` from the sampler: selection `campaign-v3`, stratum
`{state: correct, cohort: A}`, indices 0 to 7. A trace this short has every sentence selected, so all 8 are
labelled; longer traces carry a sample of their sentences.

## 3. Judge inputs (`batch_predictor.question_of`, `batch_predictor.judge_inputs`)

The judge sees the question, the sentence and one neighbour on each side, never the rest of the trace.

```json
{"problem_statement": "Please reason step by step, and put your final answer within \\boxed{}.\nQuestion: If $2^8=4^x$, what is the value of $x$?",
 "previous_sentence": "So equation: 2^8 = 2^(2x).", "sentence": "So 8 = 2x → x = 4.", "next_sentence": "So answer: 4."}
```

The first sentence gets `<START OF RESPONSE>` as its previous sentence and the last gets `<END OF RESPONSE>` as its next.

## 4. Prompt (`batch_predictor.load_program_and_adapter`, `render_conversation`, `batch_payload`)

`gepa.run.EpisodeJudge` loads the frozen program (`selected_program_20260827_173300.json`, sha256
467510e4fc1759bf…) and DSPy's `ChatAdapter` renders it. The conversation has two messages:

| Message | Size | Content |
|---|---|---|
| system | 36,578 characters | the field list, the output format and the GEPA instructions (all seven state names occur in it) |
| user | 443 characters | the four fields above, then "Respond with the corresponding output fields, starting with `[[ ## label ## ]]`" |

All conversations of a batch go in one request to `/v1/chat/completions/batch`: model `Qwen/Qwen3.8-27B`, n 1,
`max_tokens` 16,384, temperature 1.0, top_p 0.95, top_k 20, min_p 0.0 and chat-template kwargs
`enable_thinking` True, `reasoning_effort` low, `preserve_thinking` False.

## 5. Answer (`batch_predictor.parse_completion`, `annotation_batch.outcome_record`)

The saved annotation of unit 4 is `{index: 4, start: 127, end: 145, text: "So 8 = 2x → x = 4.", label: "Implement"}`. The
eight labels of the trace are Read, Analyze, Implement (four times) and Monitor (twice). The raw reply of a
known label is not kept. `demo:` replies run through the real parser:

| Reply | Parsed |
|---|---|
| `[[ ## label ## ]]` `Implement` `[[ ## completed ## ]]` | Implement |
| the same with `Implementation` | `LabelParseError: invalid label 'Implementation'; expected one of Read, ...` |

An unreadable answer is stored as `unknown` together with its raw text and the failure, and never drops its
batch neighbours.

## 6. Checkpoint binding (`annotation_batch.enumerate_items`, `annotation_batch.run_part`)

The pilot label run (Qwen3.6-35B-A3B-FP8 traces, 64 MATH-500 attempts) has 30,187 sentences in 472 checkpoint
files of 64 (the last holds 43). A checkpoint is bound to the hash of all items and to the judge settings:

| Check | Result |
|---|---|
| `enumerate_items` on the pilot traces: identity, unit and inputs of all rows | 30,187 of 30,187 equal the saved ones |
| `items_sha256` rebuilt / in the checkpoint | 38d6acd7ab832058… / the same |
| `run_part` on a copy of the checkpoints with a judge that raises if called | complete, 0 judge calls |
| `binding_sha256` and `annotations_sha256` of the new summary | c2d0af67…, ceaea640…: equal the saved summary |
| the same run with temperature 0.7 in the config | `stale checkpoint` |
| judge and sampling keys of today's `checkpoint_config` vs the pilot's | all 13 equal (judge model, program sha256, decoding) |
| keys only the pilot has / only today's has | `pilot_manifest_sha256`, `revision` / `source_plan_sha256` |

The pilot's labels: Implement 8,901, Analyze 7,313, Verify 4,258, Monitor 3,448, Explore 2,905, Read 1,857,
Plan 1,505. `tests/labeling` (resume, include-traces, splitter version 2, cohort segmentation): 24 passed.

## Worth noticing

- The judge's system prompt is 36.6 k characters for a 443-character question: the frozen program carries
  the whole annotation guide, and every sentence sent to the judge repeats it (the servers run with prefix
  caching, `scripts/serve/args.sh`).
- The same units, trace digest and offsets are used by sampling, labelling and capture. The digest 68d9059a… in
  step 1 is the one the capture manifest froze, so a change of the segmenter would break all three.
- The pilot's checkpoints use other plan keys (`pilot_manifest_sha256`, `revision`) than today's
  `source_plan_sha256`. The rows and every judge setting still match today's code, and a changed setting is refused.
- Resume is a pure check: with all checkpoints valid the judge is never needed, and a mismatch stops before any request.
