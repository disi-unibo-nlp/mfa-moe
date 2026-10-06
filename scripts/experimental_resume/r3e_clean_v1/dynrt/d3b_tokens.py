"""D3b stage 1: labelled-token routing tables for B1 (GPT A, dense Qwen3.6 pilot) and A2 (pilot).

One row per token of a labelled sentence (all reasoning tokens for the pilot, whose sentences are all
labelled). Alignment is that of D1 / v3an: `saved_layout` ownership on the saved token replay offsets and the
`dynrt.extract.match_labels` rules. No outcome column is ever read.

  python -m dynrt.d3b_tokens pilot [--limit N] [--workers W]  -> results/pilot/tokens/
  python -m dynrt.d3b_tokens gpt   [--limit N] [--workers W]  -> results/B1/tokens/

Outputs (directory): tokens.npz (ids uint8 [N, L, k], cls, att, rank, sent, tokid, comp), sentences.parquet,
attempts.parquet, provenance.json. The token rows are ordered by (attempt, reasoning-token rank).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CLASSES, FAST, RESULTS, V3R2, W, load_split, sha256_file
from .extract import match_labels

PILOT_DIR = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/results/correlation_pipeline/"
                 "routing-dynamics-v1/pilot")
PILOT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
PILOT_DATASET = "math500"
PILOT_TRACES = PILOT_DIR / "generation/Qwen--Qwen3.6-35B-A3B-FP8/math500/traces.jsonl"
PILOT_ANN = PILOT_DIR / "annotations/Qwen--Qwen3.6-35B-A3B-FP8/math500/annotations.jsonl"
PILOT_TENSORS = PILOT_DIR / "forward/unsloth--Qwen3.6-35B-A3B/math500/tensors"
PILOT_MANIFEST = PILOT_DIR / "pilot.json"
PILOT_SHAPE = dict(num_experts=256, num_layers=40, top_k=8)
OUT_PILOT = RESULTS / "pilot" / "tokens"
OUT_GPT = RESULTS / "B1" / "tokens"
GPT_COLUMNS = ["attempt_id", "dataset", "problem_id", "source_problem_id", "sample_id", "trace_sha256",
               "completion_tokens", "tensor_path", "source_location", "question"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- alignment

def align_attempt(trace_dict: dict, label_records: list, expected_digest: str) -> dict:
    """Sentence units, owners and matched labels of one trace (D1 alignment rules)."""
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.spans import reasoning_ranges, trace_digest
    from moe_exp.schemas import TraceRecord

    trace = TraceRecord(**trace_dict)
    if trace_digest(trace) != expected_digest:
        raise ValueError("trace digest differs from the expected digest")
    layout = saved_layout(trace)
    if layout["token_count"] is None:
        raise ValueError("trace without token replay offsets")
    units, owners = layout["units"], layout["unit_tokens"]
    ranges = reasoning_ranges(trace)
    label_of_unit, audit = match_labels(units, label_records)
    replay = trace.metadata["token_replay"]
    return dict(units=units, owners=owners, reasoning=list(layout["reasoning_tokens"]),
                token_count=int(layout["token_count"]), ranges=ranges, label_of_unit=label_of_unit,
                audit=audit, token_ids=np.asarray(replay["completion_token_ids"], dtype=np.int64))


def token_rows(al: dict) -> dict:
    """Per-sentence and per-token arrays of the labelled sentences with at least one owned token."""
    units, owners, ranges = al["units"], al["owners"], al["ranges"]
    reasoning = np.asarray(al["reasoning"], dtype=np.int64)
    n_r = len(reasoning)
    comp_to_rank = np.full(al["token_count"], -1, dtype=np.int64)
    comp_to_rank[reasoning] = np.arange(n_r)
    idx_rows = [i for i in sorted(al["label_of_unit"]) if owners[i]]
    tokenless = len(al["label_of_unit"]) - len(idx_rows)
    comp, sent, cls, seg_rows = [], [], [], []
    tok_start = np.zeros(len(idx_rows), np.int64)
    tok_end = np.zeros(len(idx_rows), np.int64)
    n_tok = np.zeros(len(idx_rows), np.int64)
    checks = dict(tokens_outside_reasoning=0, non_ascending=0, noncontiguous=0, tokenless_labelled=tokenless)
    for s, i in enumerate(idx_rows):
        toks = np.asarray(owners[i], dtype=np.int64)
        rank = comp_to_rank[toks]
        if (rank < 0).any():
            checks["tokens_outside_reasoning"] += 1
            raise ValueError("owned token outside the reasoning set")
        if len(toks) > 1 and not (np.diff(toks) > 0).all():
            checks["non_ascending"] += 1
        tok_start[s], tok_end[s], n_tok[s] = rank.min(), rank.max() + 1, len(toks)
        checks["noncontiguous"] += int(tok_end[s] - tok_start[s] != len(toks))
        comp.append(toks)
        sent.append(np.full(len(toks), s, np.int64))
        cls.append(np.full(len(toks), CLASSES.index(al["label_of_unit"][i]), np.int8))
        start = units[i]["start"]
        seg_rows.append(next(j for j, (a, b) in enumerate(ranges) if a <= start < b))
    comp = np.concatenate(comp) if comp else np.zeros(0, np.int64)
    order = np.argsort(comp_to_rank[comp], kind="stable")
    return dict(
        sentence_index=np.asarray(idx_rows, np.int32), segment=np.asarray(seg_rows, np.int16),
        s_cls=np.array([CLASSES.index(al["label_of_unit"][i]) for i in idx_rows], np.int8),
        tok_start=tok_start.astype(np.int32), tok_end=tok_end.astype(np.int32),
        n_tokens=n_tok.astype(np.int32), comp=comp[order], rank=comp_to_rank[comp][order].astype(np.int32),
        sent=np.concatenate(sent)[order] if sent else np.zeros(0, np.int64),
        cls=np.concatenate(cls)[order] if cls else np.zeros(0, np.int8), checks=checks, n_reasoning=n_r)


# ------------------------------------------------------------------------------ workers

def _load_ids(task: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """Expert-id tensor [L, T, k] (and the tensor's reasoning-token set when it has one)."""
    if task["loader"] == "gz":
        from v3an.extract import _load_tensors
        ids, weights, gaps, reasoning = _load_tensors(task["tensor_path"])
        del weights, gaps
        return ids, reasoning
    import torch
    torch.set_num_threads(1)
    return torch.load(task["tensor_path"], map_location="cpu", weights_only=True).numpy(), None


def extract_attempt(task: dict) -> dict:
    """Worker: one attempt -> token and sentence arrays plus checks."""
    t0 = time.time()
    ids, ref_reasoning = _load_ids(task)
    L, T, k = ids.shape
    E = int(task["num_experts"])
    aid = task["attempt_id"]
    if (L, k) != (int(task["num_layers"]), int(task["top_k"])):
        raise ValueError(f"{aid}: tensor layout {ids.shape} differs from spec")
    if int(ids.min()) < 0 or int(ids.max()) >= E:
        raise ValueError(f"{aid}: expert id out of range")
    al = align_attempt(task["trace"], task["labels"], task["trace_sha256"])
    if al["token_count"] != T:
        raise ValueError(f"{aid}: replay token count {al['token_count']} != tensor tokens {T}")
    if ref_reasoning is not None and list(al["reasoning"]) != ref_reasoning.tolist():
        raise ValueError(f"{aid}: replayed reasoning-token set differs from the capture tensor")
    tr = token_rows(al)
    comp = tr["comp"]
    sel = np.asarray(ids[:, comp, :]).transpose(1, 0, 2).astype(np.uint8)
    tokid = al["token_ids"][comp].astype(np.int32)
    tr["checks"]["all_reasoning_labelled"] = int(len(comp) == tr["n_reasoning"])
    return dict(attempt_id=aid, tr=tr, ids=sel, tokid=tokid, audit=al["audit"],
                seconds=time.time() - t0, n_reasoning=tr["n_reasoning"], completion_tokens=T)


# -------------------------------------------------------------------------------- tasks

def pilot_tasks(limit: int | None = None) -> tuple[list[dict], dict]:
    """One task per pilot trace; hashes and identity checks tie tensor, trace and annotation together."""
    man = json.loads(PILOT_MANIFEST.read_text())
    expected = {(t["identity"][2], int(t["identity"][3])): (t["trace_sha256"], int(t["reasoning_tokens"]),
                                                            int(t["sentences"])) for t in man["traces"]}
    ann = {}
    for line in open(PILOT_ANN):
        r = json.loads(line)
        if r["status"] != "complete":
            raise ValueError("incomplete pilot annotation")
        ann[(r["problem_id"], int(r["sample_id"]))] = r
    split = load_split()
    tasks, prov_rows = [], []
    for line in open(PILOT_TRACES):
        r = json.loads(line)
        key = (r["problem_id"], int(r["sample_id"]))
        if key not in expected:
            continue
        sha, n_reasoning, n_sent = expected[key]
        a = ann[key]
        if a["trace_sha256"] != sha:
            raise ValueError(f"annotation digest differs from the manifest for {key}")
        if any(u.get("label") not in CLASSES for u in a["units"]) or len(a["units"]) != n_sent:
            raise ValueError(f"pilot annotation of {key} is not complete")
        safe = r["problem_id"].replace("/", "_").replace("\\", "_")
        files = sorted(PILOT_TENSORS.glob(f"{r['dataset']}_{safe[:100]}_*_experts.pt"))
        if len(files) != 1:
            raise ValueError(f"tensor file of {key}: {len(files)} matches")
        side = json.loads(files[0].with_name(files[0].name.replace("_experts.pt", "_extraction.json"))
                          .read_text())
        if side["config"]["identity"] != [PILOT_MODEL, r["dataset"], r["problem_id"], r["sample_id"]]:
            raise ValueError(f"tensor sidecar identity differs for {key}")
        fwd = hashlib.sha256(json.dumps(
            {"prompt": r["prompt"], "system_prompt": r["system_prompt"],
             "generation_messages": r["generation_messages"], "cot_text": r["cot_text"],
             "token_replay": r["metadata"]["token_replay"]}, sort_keys=True,
            ensure_ascii=False).encode("utf-8")).hexdigest()
        if side["config"]["trace_sha256"] != fwd:
            raise ValueError(f"forward trace digest differs for {key}")
        if side["config"].get("layer_indices") != list(range(PILOT_SHAPE["num_layers"])):
            raise ValueError("unexpected pilot layer indices")
        question = f"{r['dataset']}|{r['source_problem_id']}"
        labels = [(u["index"], u["start"], u["end"], u["label"]) for u in a["units"]]
        tasks.append(dict(attempt_id=f"{r['problem_id']}#{r['sample_id']}", question=question,
                          dataset=r["dataset"], tensor_path=str(files[0]), loader="pt", trace=r,
                          trace_sha256=sha, labels=labels, expected_reasoning=n_reasoning,
                          expected_sentences=n_sent, split=split[question],
                          completion_tokens=int(r["metadata"]["usage"]["completion_tokens"]),
                          **PILOT_SHAPE))
        prov_rows.append(dict(problem_id=r["problem_id"], tensor=files[0].name))
    if len(tasks) != len(expected):
        raise ValueError(f"pilot traces found {len(tasks)} of {len(expected)}")
    tasks.sort(key=lambda t: t["attempt_id"])
    if limit:
        tasks = tasks[::max(1, len(tasks) // limit)][:limit]
    prov = dict(dataset="pilot", manifest_sha256=sha256_file(PILOT_MANIFEST),
                annotations_sha256=sha256_file(PILOT_ANN), traces_sha256=sha256_file(PILOT_TRACES),
                spec=PILOT_SHAPE, n_tasks=len(tasks))
    return tasks, prov


def gpt_tasks(limit: int | None = None) -> tuple[list[dict], dict]:
    """GPT cohort-A tasks (outcome columns are never loaded)."""
    from v3an.data import load_labels, read_location
    ready = json.loads((FAST / "manifests/gpt/A/ready.json").read_text())
    spec = ready["spec"]
    meta = dict(num_experts=int(spec["experts"]), num_layers=len(spec["layers"]), top_k=int(spec["top_k"]))
    att = pd.read_parquet(V3R2 / "gpt/A/attempts.parquet", columns=GPT_COLUMNS)
    att = att.sort_values("attempt_id").reset_index(drop=True)
    if limit:
        att = att.iloc[::max(1, len(att) // limit)].iloc[:limit].reset_index(drop=True)
    labels, label_prov = load_labels("gpt")
    tasks, missing = [], 0
    for r in att.itertuples(index=False):
        lab = labels.get((r.dataset, r.problem_id, int(r.sample_id), r.trace_sha256))
        if lab is None:
            missing += 1
            lab = []
        tasks.append(dict(attempt_id=r.attempt_id, question=r.question, dataset=r.dataset,
                          tensor_path=r.tensor_path, loader="gz",
                          trace=read_location(json.loads(r.source_location)), trace_sha256=r.trace_sha256,
                          labels=lab, completion_tokens=int(r.completion_tokens), split="", **meta))
    tasks.sort(key=lambda t: -Path(t["tensor_path"]).stat().st_size)
    prov = dict(dataset="gpt-A", spec=meta, n_tasks=len(tasks), attempts_without_labels=missing,
                label_files=label_prov,
                attempts_parquet_sha256=sha256_file(V3R2 / "gpt/A/attempts.parquet"),
                ready_sha256=sha256_file(FAST / "manifests/gpt/A/ready.json"))
    return tasks, prov


# ------------------------------------------------------------------------------- driver

def run(kind: str, out: Path, *, workers: int, limit: int | None) -> dict:
    t0 = time.time()
    out = Path(out)
    if not str(out.resolve()).startswith(str(W)):
        raise ValueError("outputs must stay under the dynamics-routing workspace")
    if out.exists():
        raise FileExistsError(f"output exists (never overwritten): {out}")
    tmp = out.parent / f".{out.name}.tmp-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    tasks, prov = pilot_tasks(limit) if kind == "pilot" else gpt_tasks(limit)
    log(f"d3b tokens {kind}: {len(tasks)} attempts, {workers} workers")
    results = []
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {v: os.environ.get(v) for v in thread_vars}
    os.environ.update({v: "1" for v in thread_vars})
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(extract_attempt, t): t["attempt_id"] for t in tasks}
            for i, f in enumerate(as_completed(futs), 1):
                results.append(f.result())
                if i % 25 == 0 or i == len(tasks):
                    log(f"extract: {i}/{len(tasks)} done, {time.time() - t0:.0f}s")
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    results.sort(key=lambda r: r["attempt_id"])
    info = {t["attempt_id"]: t for t in tasks}
    questions = sorted({t["question"] for t in tasks})
    q_index = {q: i for i, q in enumerate(questions)}
    ids_all, cls_all, att_all, rank_all, sent_all, tok_all, comp_all = [], [], [], [], [], [], []
    sent_rows, att_rows, audits = [], [], []
    n_sent = 0
    for a, res in enumerate(results):
        tr, t = res["tr"], info[res["attempt_id"]]
        n = len(tr["cls"])
        ids_all.append(res["ids"])
        cls_all.append(tr["cls"])
        att_all.append(np.full(n, a, np.int32))
        rank_all.append(tr["rank"])
        sent_all.append((tr["sent"] + n_sent).astype(np.int32))
        tok_all.append(res["tokid"])
        comp_all.append(tr["comp"].astype(np.int32))
        ns = len(tr["s_cls"])
        sent_rows.append(pd.DataFrame(dict(
            att=np.full(ns, a, np.int32), sentence_index=tr["sentence_index"], cls=tr["s_cls"],
            segment=tr["segment"], tok_start=tr["tok_start"], tok_end=tr["tok_end"],
            n_tokens=tr["n_tokens"])))
        n_sent += ns
        att_rows.append(dict(att=a, attempt_id=res["attempt_id"], question=t["question"],
                             q=q_index[t["question"]], dataset=t["dataset"], split=t["split"],
                             n_reasoning=res["n_reasoning"], completion_tokens=res["completion_tokens"],
                             n_labelled_tokens=n, n_sentences=ns))
        audits.append(dict(attempt_id=res["attempt_id"], seconds=res["seconds"], **res["audit"],
                           **tr["checks"]))
    ids = np.concatenate(ids_all)
    np.savez_compressed(tmp / "tokens.npz", ids=ids, cls=np.concatenate(cls_all), att=np.concatenate(att_all),
                        rank=np.concatenate(rank_all), sent=np.concatenate(sent_all),
                        tokid=np.concatenate(tok_all), comp=np.concatenate(comp_all))
    pd.concat(sent_rows, ignore_index=True).to_parquet(tmp / "sentences.parquet")
    att_df = pd.DataFrame(att_rows)
    att_df.to_parquet(tmp / "attempts.parquet")
    meta = pd.DataFrame(audits)
    meta.to_csv(tmp / "extract_meta.csv", index=False)
    checks = {c: int(meta[c].sum()) for c in ("tokens_outside_reasoning", "non_ascending", "noncontiguous",
                                              "tokenless_labelled", "offset_mismatch", "out_of_range",
                                              "unknown_label", "conflicting")}
    checks["attempts"] = len(att_df)
    checks["labelled_tokens"] = int(len(ids))
    checks["sentences"] = int(n_sent)
    if kind == "pilot":
        checks["all_reasoning_labelled_attempts"] = int(meta["all_reasoning_labelled"].sum())
        exp_r = {t["attempt_id"]: t["expected_reasoning"] for t in tasks}
        exp_s = {t["attempt_id"]: t["expected_sentences"] for t in tasks}
        checks["reasoning_tokens_mismatch_manifest"] = int(sum(
            r["n_reasoning"] != exp_r[r["attempt_id"]] for r in results))
        checks["sentences_mismatch_manifest"] = int(sum(
            len(r["tr"]["s_cls"]) != exp_s[r["attempt_id"]] for r in results))
    bad = [k for k in ("tokens_outside_reasoning", "non_ascending", "noncontiguous", "tokenless_labelled",
                       "offset_mismatch", "out_of_range", "conflicting", "sentences_mismatch_manifest")
           if checks.get(k)]
    if kind == "pilot" and checks["all_reasoning_labelled_attempts"] != len(att_df):
        bad.append("all_reasoning_labelled")
    checks["failed_checks"] = bad
    prov.update(host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"), limit=limit,
                time=time.strftime("%Y-%m-%dT%H:%M:%S"), seconds=time.time() - t0, checks=checks,
                code_sha256={p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("d3b_*.py"))},
                tokens_sha256=sha256_file(tmp / "tokens.npz"))
    (tmp / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    tmp.rename(out)
    log(f"d3b tokens {kind}: {len(ids)} tokens, {n_sent} sentences -> {out} in {time.time() - t0:.0f}s")
    log(json.dumps(checks, default=str))
    return checks


def load_tokens(directory: Path) -> dict:
    """Load a token table directory: npz arrays, sentence and attempt tables."""
    d = Path(directory)
    z = np.load(d / "tokens.npz")
    return dict(ids=z["ids"], cls=z["cls"].astype(np.int64), att=z["att"].astype(np.int64),
                rank=z["rank"].astype(np.int64), sent=z["sent"].astype(np.int64), tokid=z["tokid"],
                comp=z["comp"], sentences=pd.read_parquet(d / "sentences.parquet"),
                attempts=pd.read_parquet(d / "attempts.parquet"),
                provenance=json.loads((d / "provenance.json").read_text()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=("pilot", "gpt"))
    p.add_argument("--out", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    a = p.parse_args(argv)
    out = a.out or (OUT_PILOT if a.kind == "pilot" else OUT_GPT)
    checks = run(a.kind, out, workers=a.workers, limit=a.limit)
    return 1 if checks["failed_checks"] else 0


if __name__ == "__main__":
    sys.exit(main())
