# src/labeling

Cuts reasoning into sentences, picks which sentences to label, and labels them with the judge. A label is one of seven states: Read, Analyze, Plan, Implement, Explore, Verify, Monitor.

| File | What it does |
| --- | --- |
| `spans.py` | The sentence units of a reasoning text and their offsets into the exact tokens the model saw. |
| `sample_cohorts.py` | Picks the sentences to label for one model: cohorts A and B, in four parts. |
| `sample_stratified.py` | Picks 100,000 sentences stratified by correctness and reasoning length (used for LFM2.5). |
| `annotation_batch.py` | Labels the selected sentences of one part with the judge, 64 at a time, resuming after each batch. |
| `batch_predictor.py` | The judge client: builds the prompt from the frozen GEPA program and reads the answers. |
| `contexts.py`, `renderer.py`, `renderer_child.py`, `reader.py` | The judge's inputs (the question, the sentence and one neighbour on each side) and the process that renders and parses with the original DSPy program. |
| `lfm_inputs.py` | The LFM2.5 side: its native task list, its sampled sentences and the judge inputs for them. |
| [`gepa/`](gepa/README.md) | The one-off optimisation of the judge prompt. |

## Sentences

`spans.py` defines the sentence segmentation for the whole project. It splits on punctuation and newlines, keeps LaTeX and decimal points whole, and records exact character offsets. Labels, sampling plans and captures all key off these units and the `trace_digest` of the text, so its output is frozen. A change to the segmentation invalidates every existing label set: bump the version of every consumer first. The first splitter (version 1) is still used to read older annotations (`capture/sentence_schema.py`), and `annotation_batch --include-traces` relabels the traces that the version 2 fix changed.

## Choose the sentences

```bash
scripts/labeling/sample.sh gpt          # CPU job; python -m labeling.sample_cohorts --model gpt --gen-root ... --out ...
```

Input: the finished traces of one model (`--gen-root`, the folder holding `<dataset>/traces.jsonl`). Output in `--out`: `cohorts.json`, `parts.json` and four part roots `labels/part-0X/generation/<checkpoint>/<dataset>/traces.jsonl`, where each trace line lists its chosen sentences in `metadata.sentence_selection`.

- Cohort A: one attempt per question, drawn uniformly at random, labelled with a shared budget of sentences (every sentence of an attempt is equally likely, with a floor per attempt). Most analyses of native routing use it.
- Cohort B: AIME 2024, AIME 2025 and AMC 2023 questions that have at least two successes and two failures, with up to four of each, and a fixed number of labelled sentences per attempt. It supports analyses that compare right and wrong attempts of one question.
- Each attempt ends in one of four states: correct, finished wrong, capped (hit the length limit) or unscored. They are never merged.
- Traces that cannot be segmented are excluded and listed in `cohorts.json` (at most 0.5 percent).

`sample_stratified` is the other sampler:

```bash
PYTHONPATH=src python -m labeling.sample_stratified --generation-dir GEN_DIR --generation-model MODEL_ID --output-dir OUT [--dry-run]
```

It keeps one completion per source problem, splits each benchmark into four equal-count length quartiles, spreads 100,000 sentences over the cells benchmark x correct/incorrect x quartile in proportion to their supply, and cuts four parts of exactly 25,000. Seed 42. The plan is hashed; the same plan into the same folder is a no-op and a different plan is refused. The reasoning length is the server's `reasoning_tokens`, or, where the server reports 0 (gpt-oss), the count derived from the saved token replay. A benchmark whose traces all have the same length is refused.

## Label

```bash
sbatch scripts/labeling/label.sbatch TRACE_ROOT OUT_DIR       # one part; python -m labeling.annotation_batch ...
```

The judge is Qwen3.8-27B, served by `scripts/serve/judge.sh`, with the frozen program `results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json`. Thinking is on with low reasoning effort, temperature 1.0, top-p 0.95, top-k 20, up to 16,384 new tokens. Each sentence is judged with its question and one neighbouring sentence on each side.

Output in `OUT_DIR`: `annotations.json` (one record per sentence, in file order), `summary.json` and `checkpoints/batch-NNNNNN.json`. A rerun with the same inputs skips the finished batches. A changed judge setting changes the checkpoint key, so stale labels are recomputed rather than reused. An answer that cannot be read is stored as `unknown` with its raw text; it never drops its neighbours.

`python -m labeling.annotation_batch --plan-only ...` prints what would be labelled and stops.

Paper use: the labels define the states in the readout, selection, occupancy, transition, anticipation and steering exhibits. The sampling is described in the appendix on native sentence sampling and token alignment.

Tests: [`tests/labeling`](../../tests/labeling/README.md).
