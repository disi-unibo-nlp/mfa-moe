"""Generate OUT/README.md from validation_report.json and the parquet schemas."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

OUT = Path("/scratch_local/claude-133943/-leonardo-work-IscrC-MIOSR-lmolfett-mfa-moe-repo/"
           "88d89532-1e37-4900-bace-03c5cfadb6b4/scratchpad/lexicon/data")
R = json.load(open(OUT / "validation_report.json"))
MODELS = "gpt nemotron qwen35 qwen36 gemma glm qwen330b".split()
CLASSES = ["Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor"]

NATIVE_COLS = {
    "model": "source model key (the model that GENERATED the text, not the judge)",
    "row_id": "`<model>|part-0X|<row_index>`; unique; = position in the part's annotations.json",
    "annotation_part": "label part (part-00..03) the row was labelled in",
    "annotation_row_index": "row index inside that part's annotations.json = `receipts.row_index`",
    "context_hash": "LABEL_INDEX context digest of the 4 judge inputs + program + reader config; use for dedup/grouping",
    "n_rows_with_context": "number of rows in this model's table that share `context_hash` (>1 = exact duplicate judge input)",
    "label": "judge label (Read/Analyze/Plan/Implement/Explore/Verify/Monitor) from the annotation record; NULL if the judge output was unusable (`label_status=unknown`)",
    "label_status": "`ok` or `unknown`",
    "unknown_kind": "failure kind for unknown rows (non_stop / label_parse / request_rejected ...), else NULL",
    "index_label": "label stored in LABEL_INDEX `labels` (first receipt per context; NULL when the context has conflicting labels)",
    "sentence_text": "the exact sentence the judge labelled (= annotation `inputs.sentence` = `unit.text`)",
    "prev_sentence": "exact `previous_sentence` the judge saw (the immediately preceding unit of the FULL trace, or `<START OF RESPONSE>`)",
    "next_sentence": "exact `next_sentence` the judge saw (immediately following unit, or `<END OF RESPONSE>`)",
    "prev_is_start": "prev_sentence == `<START OF RESPONSE>` (sentence is the first unit)",
    "next_is_end": "next_sentence == `<END OF RESPONSE>` (sentence is the last unit)",
    "problem_sha": "sha256 of the judge `problem_statement`; text in `problems_native.parquet`",
    "trace_digest": "`spans.trace_digest` of the trace = annotation `identity.trace_sha256`; trace identity",
    "dataset": "benchmark (math500, aime24, aime25, olympiad, amc23, minerva)",
    "problem_id": "`<source_problem_id>__sample_NN` (attempt identity within the model)",
    "source_problem_id": "benchmark question id",
    "question": "`<dataset>|<source_problem_id>`; key of FOLDS.json `question_families`",
    "sample_id": "attempt index for the question",
    "family_id": "FOLDS.json family hash (near-duplicate questions share a family); the unit that must not be split across folds",
    "fold_campaign": "frozen campaign 5-fold (FOLDS.json `folds[family]`), population independent (0..4)",
    "fold_readout5": "fold from `study.readout_analysis.family_folds(questions, 5, 20260929, question_families)` over THIS model's labelled questions; depends on the population (see caveats)",
    "sentence_index": "unit index in the FULL trace unit list (spans v2)",
    "n_units_in_trace": "number of units in the trace",
    "char_start": "char offset of the unit in the trace `cot_text`",
    "char_end": "end char offset",
    "n_chars": "char length of the sentence",
    "n_tokens": "number of completion tokens owned by the sentence (`dynamics.classes.saved_layout`); NULL if no saved offsets",
    "reasoning_tok_start": "rank of the sentence's first token among the trace's reasoning tokens (readout `tok_start`)",
    "n_reasoning_tokens": "number of reasoning tokens in the trace (readout `n_reasoning`)",
    "rel_pos": "reasoning_tok_start / n_reasoning_tokens, in [0,1] (identical to the readout `rel_pos`)",
    "rel_pos_sentence": "sentence_index / (n_units_in_trace - 1)",
    "abs_bin": "readout absolute-position bin of reasoning_tok_start, edges 1024/2048/4096/8192/16384 (0..5)",
    "cohort": "campaign-v3 cohort(s) of the trace: `A`, `B` or `A+B` (a few traces are enrolled in both with identical selected sentences; each sentence is labelled once)",
    "stratum": "cohort stratum: correct / finished_wrong / capped / unscored",
    "termination": "`stop` or `length` (generation hit the cap)",
    "truncated": "termination == `length`",
    "completion_tokens": "completion tokens of the whole attempt",
    "trace_inclusion_probability": "cohort-A inclusion probability of the trace (design weight 1/p); NULL for traces only in cohort B",
    "arm_inclusion_probability_b": "cohort-B arm inclusion probability (NULL for traces only in cohort A)",
    "n_selected_in_trace": "number of sentences sampled from this trace for labelling",
    "strict_correct": "math_verify correctness (outcomes_v3, corpus card-v3); NULL for capped/unscored",
    "adjudicated_correct": "strict OR judge-equivalent correctness (outcomes_v3 `adjudicated_correct`); NULL for unscored",
    "adjudication_status": "outcomes_v3 adjudication status",
    "stored_is_correct": "`is_correct` stored in the trace record (kept for audit; prefer the two columns above)",
    "outcome_joined": "trace found in outcomes_v3 (always true)",
    "context_label_conflict": "row's context has conflicting labels in the original annotations (index label NULL although the row has a label)",
}
STEER_COLS = {
    "study": "A / B / C / FRESH (see below)", "run_id": "generated-dense run directory name",
    "uid": "continuation id (one generated continuation = one row of ARM_MAP)", "prefix_uid": "id of the native prefix the continuation started from",
    "blind_id": "HMAC sentence id used by the labelling job; unique per sentence",
    "question": "`<dataset>|<source_problem_id>`", "dataset": "benchmark", "source_problem_id": "question id",
    "family_id": "FOLDS.json family hash (equals the ARM_MAP `family`)", "fold_campaign": "frozen campaign fold of the family",
    "question_in_native_qwen36": "question appears in the native qwen36 labelled set (native_qwen36.parquet)",
    "family_in_native_qwen36": "family appears in the native qwen36 labelled set",
    "prefix_trace_sha256": "trace digest of the native trace the prefix came from (A/B/C only; NULL for FRESH: manifest does not store it)",
    "prefix_trace_in_native_qwen36": "that trace is in the native qwen36 labelled cohort (A/B/C: always true; NULL for FRESH)",
    "prefix_attempt_id": "manifest attempt id of the prefix source", "prefix_trigger_sentence_index": "unit index of the triggering sentence in the source trace",
    "prefix_tokens": "tokens in the native prefix",
    "transition": "registered class transition the prefix was selected for: candidate_to_verify / approach_to_commit",
    "target_class": "class the intervention is meant to promote: Verify for candidate_to_verify; `Plan|Implement` (commitment, no single class) for approach_to_commit",
    "intended_event": "plain-language intended event", "arm": "arm name inside the study (native, native_duplicate, bias, force, reweight, once, repeat, pair, expert9, expert189, random_*)",
    "role": "native / target / random", "arm_group": "native / targeted / random (random = matched random-expert control)",
    "policy": "action policy name (`zero` for native arms); resolves to layer/experts/kind in the manifest",
    "action_kind": "none / bias (add to router logits before selection) / force (force experts into the selected set) / reweight (reweight selected experts)",
    "magnitude": "bias size (1.0), force 0.0, reweight 1.0; NULL for native", "layers": "JSON list of intervened layers",
    "layer": "single intervened layer or NULL", "experts_json": "JSON `[[layer,[expert ids]]]` of the action",
    "n_pulses": "1 or 2 (repeat pulse)", "slots": "JSON pulse start offsets (0 and optionally 512)", "random_set": "index of the random expert set for random arms",
    "seed": "generation seed (0/1)", "execution_position": "position in the arm order", "horizon": "max generated tokens (1024; 256 for study C)",
    "hit_cap": "continuation reached the horizon", "finish_reason": "stop / length", "closed_reasoning": "reasoning closed (think end) inside the window",
    "reasoning_end": "token index where reasoning ended or the window end", "n_emitted_tokens": "tokens generated in the continuation",
    "prompt_len": "prompt tokens (incl. prefix)",
    "dose_active_rows": "router rows with the action active at the main layer (rank 0)", "dose_actual_target_hits": "target-expert selections with the action",
    "dose_native_target_hits": "target-expert selections the native routing would have had", "dose_membership_changes": "rows whose selected set changed",
    "dose_target_mass": "sum of normalised executed weight on the target experts", "dose_weight_l1": "L1 weight displacement",
    "pulse_segments": "JSON `[[start,end],...]` generated-token windows where the action was active (from `action_dose.segments`); native arms get the virtual window [[0,256]]",
    "virtual_pulse": "pulse_segments is the virtual window (native arms; no intervention happened)",
    "phase": "position of the sentence START relative to pulse_segments: in_pulse1 / between_pulses / in_pulse2 / post_pulse",
    "in_pulse": "sentence token span overlaps any pulse segment", "tokens_since_first_pulse_end": "token_start minus end of the first segment (negative = inside/before)",
    "tokens_since_last_pulse_end": "token_start minus end of the last segment", "after_first_pulse_end": "token_start >= end of the first segment",
    "in_window_0_256": "arm-agnostic: sentence overlaps generated tokens [0,256)", "in_window_512_768": "arm-agnostic: overlaps [512,768) (the repeat-pulse window)",
    "sentence_index": "unit index inside the continuation", "n_sentences_in_window": "units in the continuation",
    "is_first_sentence": "first unit (its prev_sentence is the native triggering sentence)", "is_last_sentence": "last unit",
    "char_start": "char offset in the continuation text", "char_end": "end char offset", "n_chars": "char length",
    "token_start": "first generated-token index owned by the sentence (0 = first continuation token)", "token_end": "end token index",
    "n_tokens": "tokens in the sentence", "rel_pos_window": "token_start / n_emitted_tokens", "complete": "sentence ends with terminal punctuation/newline or reasoning closed (false = cut by the horizon)",
    "sentence_text": "sentence labelled by the dense judge", "prev_sentence": "exact previous_sentence input", "next_sentence": "exact next_sentence input",
    "next_is_end_of_response": "next_sentence == `<END OF RESPONSE>` (reasoning closed)", "next_is_end_of_window": "next_sentence == `<END OF OBSERVED WINDOW>` (horizon cut; sentinel the native judge never saw)",
    "prev_is_native_trigger": "prev_sentence is the last native-prefix sentence", "problem_sha": "sha256 of the problem statement; text in `problems_steered.parquet`",
    "label": "dense-judge label; NULL if unlabelled or unknown", "label_status": "ok / unknown (judge output unusable, mostly the last window sentence) / not_labelled (FRESH only)",
    "label_finish_reason": "judge finish reason", "label_generated_tokens": "judge output tokens", "is_labelled": "label is a valid class",
    "judge_protocol": "judge protocol tag (differs from the native protocol)",
}
CONT_COLS = {"study": "study", "run_id": "run", "uid": "continuation id", "prefix_uid": "prefix id", "question": "question", "family_id": "family",
             "transition": "transition", "arm": "arm", "role": "native/target/random", "policy": "policy name", "policies": "JSON list of policy names per pulse",
             "action_kind": "none/bias/force/reweight", "magnitude": "operator magnitude", "experts_json": "JSON experts", "layers": "JSON layers",
             "slots": "JSON slots", "pulse_segments": "JSON active windows", "virtual_pulse": "native virtual window", "random_set": "random set id",
             "seed": "seed", "execution_position": "arm order position", "horizon": "token horizon", "hit_cap": "hit the horizon",
             "finish_reason": "finish reason", "closed_reasoning": "reasoning closed", "reasoning_end": "reasoning end token", "n_emitted_tokens": "tokens generated",
             "n_sentences": "units in the continuation", "n_sentences_labelled": "units with a valid label",
             "missing_telemetry_ranks": "JSON ranks lacking telemetry", "inactive_native_checks": "JSON native-equivalence checks of inactive layers",
             "action_dose_json": "full `action_dose` record (per TP rank, per policy, per layer)"}


def table(cols, path):
    schema = pq.read_schema(OUT / path)
    missing = [n for n in schema.names if n not in cols]
    assert not missing, (path, missing)
    return "\n".join(f"| `{n}` | {t} | {cols[n].replace('|', chr(92) + '|')} |" for n, t in zip(schema.names, schema.types))


lines = []
w = lines.append
w("# Lexicon feasibility data: judge-labelled reasoning sentences (native + steered)\n")
w("Built 2026-10-17 by read-only extraction from the campaign-v3 label parts and the generated-dense steering runs. "
  "Nothing in the repo or in any source directory was modified. Scripts: `../scripts/` "
  "(`build_native.py`, `build_steered.py`, `build_extras.py`, `validate.py`, `make_readme.py`); validation output: `validation_report.json`. "
  "This directory is temporary session scratch.\n")
w("Purpose: test whether a cheap lexical (text-only) classifier trained on the Qwen3.8-27B judge's seven Schoenfeld labels "
  "(Read, Analyze, Plan, Implement, Explore, Verify, Monitor) can replace the judge on sentences generated after routing interventions.\n")
w("## 1. Files\n")
w("| file | rows | content |\n|---|---:|---|")
for m in MODELS:
    w(f"| `native_{m}.parquet` | {R['native'][m]['rows']:,} | every judge-labelled sentence of `{m}` (campaign-v3 labels) |")
w(f"| `steered_A.parquet` | {R['steered']['A']['rows']:,} | study A (expert pair/singles), {R['steered']['A']['labelled']:,} labelled |")
w(f"| `steered_B.parquet` | {R['steered']['B']['rows']:,} | study B (once vs repeat pulse), {R['steered']['B']['labelled']:,} labelled |")
w(f"| `steered_C.parquet` | {R['steered']['C']['rows']:,} | study C (bias/force/reweight, 256-token horizon), {R['steered']['C']['labelled']:,} labelled |")
w(f"| `steered_FRESH.parquet` | {R['steered']['FRESH']['rows']:,} | FRESH comparison; only {R['steered']['FRESH']['labelled']:,} sentences labelled (rest `not_labelled`) |")
w(f"| `steered_all_labelled.parquet` | {R['steered_all_labelled']['rows']:,} | union of the labelled rows of A, B, C, FRESH |")
w("| `steered_continuations.parquet` | one per continuation | arm/policy/segments/dose per generated continuation (full `action_dose` JSON) |")
w(f"| `judge_parity_200.parquet` | {R['judge_parity_200']['rows']} | same-input pairs: historical (native-protocol) label vs dense-protocol label; agreement {R['judge_parity_200']['agreement']:.3f} on {R['judge_parity_200']['covered']} covered |")
w("| `problems_native.parquet`, `problems_steered.parquet` | | `problem_sha` -> problem statement text (the judge `problem_statement`) |")
w("| `validation_report.json` | | counts and all consistency checks |\n")

w("## 2. What the judge saw (IMPORTANT for a text-only classifier)\n")
w("### 2.1 Native labels (all `native_*.parquet`)\n")
w("Program: `results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json` "
  "(sha256 `467510e4fc1759bf2833e7bc8f5a9d7937c52f2d31f530168776788a590b7a07`), a DSPy `EpisodeJudge` program with ZERO demos and a 32.7k-character GEPA-optimised "
  "instruction: an ordered decision sequence (Monitor first, then Verify, Explore, ...) with many lexical cues and 'Critical Nuance' rules "
  "(\"Hmm/Okay/Let me think\" -> Monitor, \"check/verify\" -> Verify, \"maybe/perhaps/alternatively\" -> Explore, final-answer markers -> Monitor).\n")
w("Input fields per sentence (rendered by DSPy ChatAdapter, sent through vLLM `/v1/chat/completions/batch`): "
  "`problem_statement`, `previous_sentence`, `sentence`, `next_sentence` and nothing else.\n")
w("- `problem_statement` = the full user prompt (for math: `Please reason step by step, and put your final answer within \\boxed{}.\\nQuestion: ...`); stored via `problem_sha` + `problems_native.parquet`.")
w("- `previous_sentence` / `next_sentence` = exactly ONE neighbouring unit on each side, taken from the FULL list of units of the trace (not only from the sampled units); edge sentinels `<START OF RESPONSE>` / `<END OF RESPONSE>`.")
w("- NOT seen by the judge: earlier/later history beyond +-1 unit, the model's final answer, the gold answer, position in the trace, length, correctness, routing.")
w("- Units are `spans.sentence_spans` v2 units: split after `.`/`!`/`?` followed by whitespace, or at newlines, with `$...$`, `$$...$$`, `\\(..\\)`, `\\[..\\]` math protected, inside `reasoning_ranges` (reasoning channel only). Labelled sentences are a stratified SUBSET of each sampled trace (`n_selected_in_trace`), so a row's neighbours need not be rows of the table.")
w("- Decoding (identical `config` apart from port and plan hash; checked in the first and last checkpoint binding of all 28 parts): judge `Qwen/Qwen3.8-27B`, thinking on, `reasoning_effort=low`, temperature 1.0, top_p 0.95, top_k 20, min_p 0, max_tokens 16384, batch 64, ONE stochastic draw per sentence. Non-stop or unparseable outputs would be stored as `unknown` (counts below). Labels are therefore noisy samples, not a deterministic function of the inputs; the only internal noise estimate is the set of exact-duplicate contexts (`n_rows_with_context > 1`) with conflicting labels (`context_label_conflict`).")
w("- Consequence for a lexical model: it sees at most the sentence (and optionally prev/next text and position features). The judge additionally read the problem statement; Read-vs-Analyze depends on whether content restates the problem, Plan-vs-Implement on the NEXT sentence, Verify-vs-Monitor on the PREVIOUS sentence. Offer variants: sentence only; prev|cur|next concatenation (the `judge_context` mode of `study/readouts.py`); plus `rel_pos`, `n_tokens`. The existing hashed n-gram featuriser is `src/moe_exp/study/readout_text.py`.\n")
w("### 2.2 Steered labels (`steered_*.parquet`) - a DIFFERENT protocol\n")
w("Same four input fields and the same GEPA instruction text (plus `Return only the one label. Do not include an explanation.`), but a new direct offline vLLM chat "
  "(`scripts/experimental_resume/label_dense_discovery.py`, `generated_dense_pipeline_v1.py`): system = instruction, user = "
  "`Problem Statement: ...\\nPrevious Sentence: ...\\nSentence: ...\\nNext Sentence: ...\\nLabel:`, temperature 0, thinking on with `reasoning_effort=low`, max_tokens 1024, TP2, no DSPy adapter. "
  f"Parity audit against the historical labels on 200 fixtures (`judge_parity_200.parquet`): coverage {R['judge_parity_200']['covered']}/200, agreement {R['judge_parity_200']['agreement']:.3f}. "
  "Edge handling differs: the first continuation sentence's `previous_sentence` is the native triggering sentence of the prefix; the last sentence's `next_sentence` is `<END OF RESPONSE>` when reasoning closed, else `<END OF OBSERVED WINDOW>` (never seen by the native judge); the last sentence is often cut by the 1024/256-token horizon (`complete=false`) and its label is more often `unknown`.\n")

w("## 3. Native table: row counts\n")
w("Rows are `receipts` rows (every annotation record), so identical judge inputs appear several times; `unique_contexts` = rows of the LABEL_INDEX `labels` table.\n")
w("| model | rows | traces | questions | unique contexts | rows in duplicate ctx | unknown | conflict ctx (rows) |\n|---|---:|---:|---:|---:|---:|---:|---:|")
for m in MODELS:
    r = R["native"][m]
    w(f"| {m} | {r['rows']:,} | {r['traces']:,} | {r['questions']:,} | {r['unique_contexts']:,} | {r['rows_in_duplicate_contexts']:,} | {r['unknown']} | {r['conflict_contexts']} ({r['conflict_rows']:,}) |")
tot = sum(R["native"][m]["rows"] for m in MODELS)
w(f"| **total** | **{tot:,}** | | | | | | |\n")
w("Rows per label:\n")
w("| model | " + " | ".join(CLASSES) + " |\n|---|" + "---:|" * len(CLASSES))
for m in MODELS:
    w(f"| {m} | " + " | ".join(f"{R['native'][m]['label_counts'][c]:,}" for c in CLASSES) + " |")
tl = {c: sum(R["native"][m]["label_counts"][c] for m in MODELS) for c in CLASSES}
w("| **total** | " + " | ".join(f"**{tl[c]:,}**" for c in CLASSES) + " |\n")
w("Trace-level design/outcome counts (distinct traces):\n")
w("| model | A-only / B-only / A+B traces | truncated traces (rows) | strict correct / incorrect / NA | adjudicated correct / incorrect / NA | strict-wrong but adjudicated-right | stored `is_correct` != strict |\n|---|---|---|---|---|---:|---:|")
for m in MODELS:
    r = R["native"][m]
    w(f"| {m} | {r['cohort_traces'].get('A', 0)} / {r['cohort_traces'].get('B', 0)} / {r['cohort_traces'].get('A+B', 0)} | {r['truncated_traces']} ({r['truncated_rows']:,}) | "
      f"{r['strict_correct_traces']} / {r['strict_incorrect_traces']} / {r['strict_na_traces']} | "
      f"{r['adjudicated_correct_traces']} / {r['adjudicated_incorrect_traces']} / {r['adjudicated_na_traces']} | "
      f"{r['strict_vs_adjudicated_disagree_traces']} | {r['stored_is_correct_vs_strict_disagree_traces']} |")
w("")

w("## 4. Steered tables: counts\n")
w("Single source model: Qwen3.6-35B-A3B (= native model `qwen36`). Continuations start from a native prefix (ending at a triggering sentence of the registered `transition`), "
  "are generated with a 256-token router intervention window at generated-token offsets [0,256) (and [512,768) for 2-pulse arms) on layer 28 (candidate_to_verify) or 24 (approach_to_commit), "
  "then run to the horizon. Native/native_duplicate arms are the same-harness no-intervention controls; `random_*` arms apply the same operator to a matched random expert set.\n")
w("| study | what varies | horizon | continuations | sentences | valid labels | unknown | not labelled | questions | families |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
what = {"A": "target expert 9 / 189 / pair (bias +1, layer 28)", "B": "bias pulse once vs repeated (2 pulses)", "C": "operator: bias / force / reweight",
        "FRESH": "bias, repeat, force, reweight, expert9/189 + matched random controls; 134 native prefixes, 2 seeds"}
hor = {"A": 1024, "B": 1024, "C": 256, "FRESH": 1024}
for s in ("A", "B", "C", "FRESH"):
    r = R["steered"][s]
    w(f"| {s} | {what[s]} | {hor[s]} | {r['continuations']:,} | {r['rows']:,} | {r['labelled']:,} | {r['unknown']} | {r['not_labelled']:,} | {r['questions']} | {r['families']} |")
w("")
for s in ("A", "B", "C", "FRESH"):
    r = R["steered"][s]
    w(f"- **{s}** arms (continuations): " + ", ".join(f"{k} {v}" for k, v in r["arms"].items()) + ". Label counts: " + ", ".join(f"{c} {r['label_counts'][c]:,}" for c in CLASSES) + ". "
      f"Phase rows: " + ", ".join(f"{k} {v:,}" for k, v in r["phase_rows"].items()) + ".")
w("")
import pandas as pd
_f = pd.read_parquet(OUT / "steered_FRESH.parquet", columns=["uid", "arm_group", "phase", "is_labelled"])
_g = _f.groupby("arm_group").is_labelled.agg(["sum", "size"])
_p = _f.groupby("phase").is_labelled.agg(["sum", "size"])
w("FRESH label coverage: the dense labelling job was only partly run (350 batches over 24 shards); "
  f"{int(_f.is_labelled.sum()):,} of {len(_f):,} sentences ({_f.is_labelled.mean():.1%}) have a label, covering {_f[_f.is_labelled].uid.nunique():,} of {_f.uid.nunique():,} continuations. "
  "Measured coverage by arm group: " + ", ".join(f"{k} {int(v['sum']):,}/{int(v['size']):,} ({v['sum'] / v['size']:.1%})" for k, v in _g.iterrows()) +
  "; by phase: " + ", ".join(f"{k} {v['sum'] / v['size']:.1%}" for k, v in _p.iterrows()) +
  ". Coverage is therefore approximately uniform over arms and phases, but it is not a designed stratified sample.\n")

w("## 5. Join recipes\n")
w("```python\nimport pandas as pd\nn = pd.read_parquet('native_qwen36.parquet')          # keep per-row data; dedup with context_hash when needed\n"
  "d = n.drop_duplicates('context_hash')                   # unique judge inputs (rows with conflict -> index_label NULL)\n"
  "# folds: use fold_campaign (frozen) or fold_readout5 (readout recipe); never split a family_id across train/test\n"
  "train = n[n.fold_campaign != 0]; test = n[n.fold_campaign == 0]\n"
  "s = pd.read_parquet('steered_all_labelled.parquet')\n"
  "# leakage-safe steered eval: train only on native qwen36 rows from families that are NOT in the steered set\n"
  "fam = set(s.family_id); train = n[(n.model == 'qwen36') & ~n.family_id.isin(fam)]\n"
  "# full text of problem statements\nprob = pd.read_parquet('problems_native.parquet').set_index('problem_sha').problem_statement\n```\n")
w("Source join keys (for audits): native row = (`annotation_part`, `annotation_row_index`) in `campaign-v3/labels/<model>/part-0X/annotations.json` = (`source`,`row_index`) in `LABEL_INDEX.sqlite.receipts` of `report/research-campaign-v3-20261014-r6/<model>/`; "
  "trace = `trace_digest` in `campaign-v3/cohorts/<model>/labels/part-0X/generation/<dir>/<dataset>/traces.jsonl`; cohort row = `cohorts.json` (`trace_sha256`); correctness = `outcomes_v3-99bf8d59506e2893.parquet` (corpus `card-v3`, key model+dataset+problem_id); "
  "family/fold = r6 `FOLDS.json` (`question_families`, `folds`). Steered sentence = `blind_id` in `<run>/ARM_MAP.json` (`records[].sentences[]`) = `<run>/BLIND_FRAME.json` (inputs) = `<run>/labels-shard-*/batches/*.json` (labels); manifest in `report/experimental-resume-v1/`.\n")

w("## 6. Recommended evaluation hygiene / caveats\n")
for t in [
    "**Folds.** `fold_campaign` is the frozen population-independent 5-fold over families. `fold_readout5` replicates the readout code (`OUTER=5`, `FOLD_SEED=20260929`, families sorted by `digest('forum-v1|seed|family')`, fold = rank % 5) over each model's own labelled families; it only equals the readout folds if the readout population has the same family set. The two fold columns are unrelated (about 20% agreement).",
    "**Duplicates.** Exact duplicate judge inputs are frequent in repetition loops (gemma: 21% of rows share a context). Group by `context_hash` or deduplicate before computing agreement; conflicting duplicates (`context_label_conflict`) are a lower bound on label noise.",
    "**Label noise.** Native labels are single temperature-1.0 draws. Cross-protocol agreement of two judge runs on 200 same-input fixtures is 0.874 (qwen36 text, deterministic vs stochastic protocol); treat it as a reference scale for label-noise-limited agreement, not as a proven ceiling.",
    "**Steered leakage.** All steered questions belong to the native qwen36 labelled question set (`question_in_native_qwen36`), and the A/B/C prefixes are prefixes of labelled native qwen36 traces (`prefix_trace_in_native_qwen36`). Train the lexical model on families disjoint from the steered families.",
    "**Single model, few questions.** Steered data exist only for qwen36, so cross-model transfer under intervention cannot be tested. Studies A, B, C cover only 8, 13 and 13 questions (families); FRESH covers 123 (11,179 labelled sentences). Cluster bootstraps/CV by `family_id`; repeated seeds, arms and prefixes of one question are not independent.",
    "**Pulse flags for native arms.** Native and native_duplicate arms have no intervention; their `pulse_segments`/`phase`/`in_pulse` use the virtual window [[0,256]] (`virtual_pulse=True`) so they can be compared position-for-position with treated arms.",
    "**Intervention nesting.** `random_*` arms are the matched random-expert controls of the targeted arms of the same study; `native` and `native_duplicate` are two no-intervention generations from the same prefix and seed index; their texts differ in most cases (identical in 3 of 16 prefix-seed pairs of study A), so native-vs-native_duplicate gives a same-harness null for shift measures.",
    "**Sampling design.** Native rows are a stratified sample (cohort A: `trace_inclusion_probability`; cohort B: `arm_inclusion_probability_b`), not all sentences; weight by 1/p for population estimates. Cohort B questions are all questions of cohort A, and a few traces are in both (`cohort == 'A+B'`).",
    "**Correctness.** Two definitions (`strict_correct`: math_verify; `adjudicated_correct`: adds LLM equivalence). Most capped (`truncated`) traces have NULL strict correctness (a few carry a value). The trace's stored `is_correct` is kept only for audit.",
    "**Protocol mismatch.** Steered labels use the dense protocol (T=0, low effort, 1024 tokens, different edge sentinels). A drop in lexical accuracy from native to steered conflates intervention shift with protocol shift; use `judge_parity_200` and the steered native arms (`arm_group == 'native'`) to separate them.",
    "**dtypes.** Nullable booleans (`strict_correct`, `adjudicated_correct`, `stored_is_correct`) are read by pandas as `object` with `None`; use `.astype('boolean')`. `fold_campaign`/`fold_readout5` are int8, `abs_bin` is nullable Int8.",
    "**Small intervention effect.** Over the 27,746 labelled steered sentences the judge class mix is almost identical across arm groups (e.g. Verify 12.3% native, 12.6% random, 12.8% targeted), so steering moves class shares by at most about one percentage point; a lexical classifier is therefore mainly tested against text-level shift, and detecting an intervention effect needs the clustered, paired design of the original studies.",
    "**Window edges.** Last sentences of horizon-cut continuations are incomplete (`complete=false`) and carry a sentinel `next_sentence` the native judge never saw.",
]:
    w("- " + t)
w("")
w("## 7. Known gaps (not recoverable, stated rather than guessed)\n")
for t in [
    "Native judge raw completions / reasoning are not stored in `annotations.json` (checkpoint files exist but were not extracted); per-sentence sampling seeds are unknown.",
    "The exact rendered DSPy prompt is not stored (DSPy is not installed in the available environments); the field layout and instructions are documented above and in the program JSON.",
    "FRESH: the prefix trace identity (`prefix_trace_sha256`) is not in the manifest, so `prefix_trace_in_native_qwen36` is NULL; question-level membership is available. Only 11,179 of 187,524 sentences have labels (arm/phase coverage ~6%, see section 4).",
    "The `Plan|Implement` target class for approach_to_commit is an interpretation of 'commitment'; the registered event is defined by the transition detector, not by a single class.",
    "`dose_*` columns use TP rank 0 at the first intervened layer; the full per-rank record is in `steered_continuations.parquet`.",
    "No unlabelled steered text was extracted. Candidates (not extracted): verification-study-v3 (7 models, ~50 generations each with `emitted_text`; reader = substantive-verification ratings, not classes) under `/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/candidate-verification-v3-20261007-r4/<model>/discovery/`; mechanism-1024 / micro-screen / eligible-immediate semantic frames (qwen36) under `.../steering-v1/runs/routing-control-v1/`; `results/moe_guiding/rtx3070` (legacy generations).",
    "The final-routing study (`report/final-routing-20261015-r1`) and research-campaign r4-r6 contain no judge-labelled steered continuations (receipts/validation only; r6 qwen330b holds 64 reader-pricing labels of native redraws).",
]:
    w("- " + t)
w("")
w("## 8. Schemas\n")
w("### 8.1 `native_<model>.parquet`\n")
w("| column | dtype | description |\n|---|---|---|")
w(table(NATIVE_COLS, "native_gpt.parquet"))
w("\n### 8.2 `steered_<study>.parquet` / `steered_all_labelled.parquet`\n")
w("| column | dtype | description |\n|---|---|---|")
w(table(STEER_COLS, "steered_A.parquet"))
w("\n### 8.3 `steered_continuations.parquet`\n")
w("| column | dtype | description |\n|---|---|---|")
w(table(CONT_COLS, "steered_continuations.parquet"))
w("\n### 8.4 `judge_parity_200.parquet`\n")
w("columns: question, family_id, dataset, problem_id, sentence_index, trace_digest, sentence_text, prev_sentence, next_sentence, historical_label (original DSPy-protocol label), dense_label (dense protocol), dense_finish_reason, dense_generated_tokens.\n")
w("## 9. Source paths\n")
for t in [
    "Labels + judge inputs: `/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/labels/<model>/part-0{0..3}/annotations.json`",
    "Context index: `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/research-campaign-v3-20261014-r6/<model>/LABEL_INDEX.{sqlite,json}` (symlinks to FAST r5-r2)",
    "Traces / cohorts: `.../campaign-v3/cohorts/<model>/{cohorts.json,labels/part-0X/generation/}`; original card-v3 traces under `repo/results/correlation_pipeline/card-v3/<model>/`",
    "Outcomes: `.../campaign-v3/outcomes/outcomes_v3-99bf8d59506e2893.parquet`; folds: `repo/report/research-campaign-v3-20261014-r6/FOLDS.json`",
    "Judge program: `repo/results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json`; code: `src/moe_exp/correlation_pipeline/{annotation_batch,annotation_partition,batch_predictor,batch_judge,spans}.py`, `src/moe_exp/research_campaign_v3_r6/annotation.py`",
    "Steered runs: `/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/generated-dense/dense-generated-v1-{9b714fe1091f,9c3dc95fda9c,65e402b7e4fe}-8e0bff870bed` and `generated-dense-fresh-metadata-recovery-v1/dense-generated-v1-6b5aa286e46c-a77287237f3a`; manifests `repo/report/experimental-resume-v1/OVERNIGHT_{DISCOVERY_A,DISCOVERY_B,DISCOVERY_C,FRESH_COMPARISON}_MANIFEST_v*.json`; dense judge `repo/scripts/experimental_resume/{label_dense_discovery,qualify_dense_judge,generated_dense_pipeline_v1}.py`; parity `dense-judge-parity/results-8e939bee-2341bac1/PARITY.json`",
]:
    w("- " + t)
(OUT / "README.md").write_text("\n".join(lines) + "\n")
print("README written", len("\n".join(lines)))
