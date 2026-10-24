"""Union of labelled steered rows, merged native problems table, judge parity table. Writes only under OUT_DIR.

Usage: python -m routing.lexical_data.build_extras OUT_DIR   (the folder the builders wrote into)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

if len(sys.argv) != 2:
    sys.exit('usage: python -m routing.lexical_data.build_extras OUT_DIR')
OUT = Path(sys.argv[1])
PAR = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/"
           "routing-control-v1/dense-judge-parity")
MODELS = "gpt nemotron qwen35 qwen36 gemma glm qwen330b".split()

# 1) union of labelled steered rows
frames = [pd.read_parquet(OUT / f"steered_{s}.parquet") for s in ("A", "B", "C", "FRESH")]
allrows = pd.concat(frames, ignore_index=True)
lab = allrows[allrows.is_labelled].reset_index(drop=True)
lab.to_parquet(OUT / "steered_all_labelled.parquet", compression="zstd", index=False)
print("steered_all_labelled", len(lab), lab.groupby("study").size().to_dict())

# 1b) per-continuation table (dose, segments, arm/policy definition)
cont = pd.concat([pd.read_parquet(OUT / f"_cont_{s}.parquet") for s in ("A", "B", "C", "FRESH")], ignore_index=True)
cont.to_parquet(OUT / "steered_continuations.parquet", compression="zstd", index=False)
print("steered_continuations", len(cont))
for p in OUT.glob("_cont_*.parquet"):
    p.unlink()

# 2) problems tables -> two merged files
nat = pd.concat([pd.read_parquet(OUT / f"_problems_{m}.parquet").assign(model=m) for m in MODELS])
nat = nat.groupby(["problem_sha", "dataset", "source_problem_id", "problem_statement"], as_index=False).agg(
    models=("model", lambda s: ",".join(sorted(set(s)))))
nat.to_parquet(OUT / "problems_native.parquet", compression="zstd", index=False)
st = pd.concat([pd.read_parquet(OUT / f"_problems_steered_{s}.parquet") for s in ("A", "B", "C", "FRESH")])
st = st.drop_duplicates("problem_sha")
st.to_parquet(OUT / "problems_steered.parquet", compression="zstd", index=False)
print("problems native", len(nat), "steered", len(st), "steered problem texts also in native:",
      st.problem_sha.isin(set(nat.problem_sha)).mean())
for p in OUT.glob("_problems_*.parquet"):
    p.unlink()

# 3) same-input judge-protocol pairs (GEPA/DSPy label vs dense-v1 label)
fx = json.load(open(PAR / "FIXTURES.json"))["fixtures"]
pr = json.load(open(PAR / "results-8e939bee-2341bac1/PARITY.json"))["records"]
rows = []
for a, b in zip(fx, pr, strict=True):
    assert a["identity"] == b["identity"]
    i = a["inputs"]
    rows.append(dict(question=a["question"], family_id=a["family"], dataset=a["identity"]["dataset"],
                     problem_id=a["identity"]["problem_id"], sentence_index=a["identity"]["sentence_index"],
                     trace_digest=a["identity"]["trace_sha256"], sentence_text=i["sentence"],
                     prev_sentence=i["previous_sentence"], next_sentence=i["next_sentence"],
                     historical_label=b["historical_label"], dense_label=b["new_label"],
                     dense_finish_reason=b["finish_reason"], dense_generated_tokens=b["generated_tokens"]))
par = pd.DataFrame(rows)
par.to_parquet(OUT / "judge_parity_200.parquet", compression="zstd", index=False)
ok = par[par.dense_label.notna() & (par.dense_finish_reason == "stop")]
print("parity", len(par), "covered", len(ok), "agreement", (ok.dense_label == ok.historical_label).mean())
