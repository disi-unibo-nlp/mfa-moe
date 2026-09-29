"""Outcome sidecar v3: corrected, adjudication-aware outcomes for every attempt.

Each build writes a new content-addressed file outcomes_v3-<sha16>.parquet (never
overwritten) and updates the pointer outcomes_v3.latest.json. Keeps strict (math_verify) correctness, J1-adjudicated correctness, a Minerva numeric-
tolerance sensitivity variant and termination as separate fields; never overwrites the
source traces. J1 verdicts are applied only when the J1 acceptance receipt exists; before
that, residual answers are 'pending'.

  python -m moe_exp.correlation_pipeline.outcomes --campaign DIR --rescored PARQUET
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SCORER_CONTRACT = 3


_FRAC = re.compile(r"(-?)\s*\\[dt]?frac\s*\{\s*([-+]?[\d.]+)\s*\}\s*\{\s*([-+]?[\d.]+)\s*\}")
_SCI = re.compile(r"([-+]?\d*\.?\d+)\s*(?:\\times|\\cdot|x)\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?")
_PLAIN = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?")


def _frac_value(m: re.Match) -> float:
    value = float(m.group(2)) / float(m.group(3))
    return -value if m.group(1) else value


def _num(text: str | None) -> float | None:
    """First numeric value of an answer: fractions, a x 10^b, e-notation, plain numbers.

    The earliest match wins, so "v = \\frac{3}{4}" is 0.75 rather than the 3 inside it;
    at the same position a fraction beats scientific notation, which beats a plain number.
    """
    if not text:
        return None
    t = text.replace("\\,", "").replace("\\!", "").replace("$", "").strip()
    found = []
    for rank, (pattern, convert) in enumerate((
            (_FRAC, _frac_value),
            (_SCI, lambda m: float(m.group(1)) * 10 ** int(m.group(2))),
            (_PLAIN, lambda m: float(m.group().replace(",", ""))))):
        m = pattern.search(t)
        if m:
            found.append((m.start(), rank, m, convert))
    if not found:
        return None
    _, _, m, convert = min(found, key=lambda f: f[:2])
    try:
        return convert(m)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def tolerance_match(boxed: str | None, gold: str | None, rel: float = 0.01) -> bool | None:
    b, g = _num(boxed), _num(gold)
    if b is None or g is None:
        return None
    if g == 0:
        return abs(b) < 1e-9
    return abs(b - g) / abs(g) <= rel


def load_verdicts(campaign: Path, pattern: str) -> dict[str, str]:
    """item key -> verdict, only for items judged (keys from the items files)."""
    out = {}
    for items_file in sorted((campaign / "j1").glob(pattern)):
        verdict_file = items_file.with_name(items_file.name.replace("_items.jsonl", "_verdicts.jsonl"))
        if not verdict_file.exists():
            continue
        verdicts = {}
        for line in verdict_file.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            verdicts[row["item_id"]] = row["verdict"]
        for line in items_file.open():
            item = json.loads(line)
            if item["item_id"] in verdicts:
                out[f"{item['corpus']}|{item['model']}|{item['problem_id']}"] = verdicts[item["item_id"]]
    return out


def adjudicate(strict, termination, verdict, j1_accepted):
    if termination == "length" and not strict:
        return False, "no_answer" if strict is None else "capped"
    if strict is True:
        return True, "math_verify_true"
    if strict is None:
        return None, "unscored"
    if not j1_accepted or verdict is None:
        return None, "pending"
    return {"EQUIVALENT": (True, "judge_equivalent"), "NOT_EQUIVALENT": (False, "judge_not_equivalent")}.get(
        verdict, (None, "uncertain"))


def build(args) -> None:
    import pandas as pd
    campaign = Path(args.campaign)
    j1_accepted = (campaign / "acceptance" / "j1.json").exists() and \
        json.loads((campaign / "acceptance" / "j1.json").read_text()).get("status") == "complete"
    rows = []
    # historical corpus (re-scored in the main thread)
    verdicts = load_verdicts(campaign, "hist_items.jsonl")
    hist = pd.read_parquet(args.rescored)
    corpus = pd.read_parquet(Path(args.rescored).with_name("corpus.parquet"),
                             columns=["model", "problem_id", "completion_tokens"])
    hist = hist.merge(corpus, on=["model", "problem_id"], how="left")
    for r in hist.itertuples():
        model = "qwen35" if r.model == "qwen35gptq" else r.model
        termination = r.finish_reason
        strict = None if pd.isna(r.mv_correct) else bool(r.mv_correct)
        adjudicated, status = adjudicate(strict, termination, verdicts.get(f"historical|{model}|{r.problem_id}"), j1_accepted)
        rows.append(dict(corpus="historical", model=model, dataset=r.dataset, problem_id=r.problem_id,
                         source_problem_id=r.problem_id.rsplit("__sample_", 1)[0], trace_sha256=None,
                         completion_tokens=None if pd.isna(r.completion_tokens) else int(r.completion_tokens),
                         sample_id=r.sample_id, termination=termination, strict_correct=strict,
                         adjudicated_correct=adjudicated, adjudication_status=status,
                         minerva_tolerance_correct=(tolerance_match(r.boxed, r.gold) if r.dataset == "minerva" else None),
                         scorer_contract_version=SCORER_CONTRACT))
    # card-v3 corpus
    verdicts = load_verdicts(campaign, "new_*_items.jsonl")
    card_root = Path(args.card_root)
    from moe_exp.utils import last_boxed
    from moe_exp.correlation_pipeline.spans import trace_digest
    from moe_exp.schemas import TraceRecord
    for model_dir in sorted(p for p in card_root.iterdir() if p.is_dir()):
        for traces in sorted((model_dir / "generation").glob("*/*/traces.jsonl")):
            for line in traces.open():
                record = json.loads(line)
                meta = record["metadata"]
                termination = meta.get("termination") or meta.get("finish_reason")
                strict = record.get("is_correct")
                adjudicated, status = adjudicate(strict, termination, verdicts.get(
                    f"card-v3|{model_dir.name}|{record['problem_id']}"), j1_accepted)
                boxed = last_boxed(meta.get("assistant_content") or "")
                rows.append(dict(corpus="card-v3", model=model_dir.name, dataset=record["dataset"],
                                 problem_id=record["problem_id"], sample_id=record.get("sample_id"),
                                 source_problem_id=record.get("source_problem_id") or record["problem_id"],
                                 trace_sha256=trace_digest(TraceRecord(**record)),
                                 termination=termination, strict_correct=strict,
                                 completion_tokens=len((meta.get("token_replay") or {}).get("completion_token_ids") or []),
                                 adjudicated_correct=adjudicated, adjudication_status=status,
                                 minerva_tolerance_correct=(tolerance_match(boxed, record.get("gold_answer"))
                                                            if record["dataset"] == "minerva" else None),
                                 scorer_contract_version=SCORER_CONTRACT))
    # continuation branches of historical capped traces
    verdicts = load_verdicts(campaign, "cont_*_items.jsonl")
    for scored in sorted((campaign / "continuation").glob("*/scored.jsonl")):
        for line in scored.open():
            r = json.loads(line)
            termination = r["finish_reason"]
            adjudicated, status = adjudicate(r["strict_correct"], termination,
                                             verdicts.get(f"continuation|{r['model']}|{r['problem_id']}"), j1_accepted)
            rows.append(dict(corpus="continuation", model=r["model"], dataset=r["dataset"],
                             problem_id=r["problem_id"], sample_id=r["sample_id"], termination=termination,
                             source_problem_id=r["problem_id"].rsplit("__sample_", 1)[0], trace_sha256=None,
                             completion_tokens=r["cumulative_tokens"], strict_correct=r["strict_correct"],
                             adjudicated_correct=adjudicated, adjudication_status=status,
                             minerva_tolerance_correct=None, scorer_contract_version=SCORER_CONTRACT))
    frame = pd.DataFrame(rows)
    frame["completion_tokens"] = frame["completion_tokens"].astype("Int64")
    frame["sample_id"] = frame["sample_id"].astype("Int64")
    out = campaign / "outcomes"
    out.mkdir(parents=True, exist_ok=True)
    staging = out / ".outcomes_v3.staging.parquet"
    frame.to_parquet(staging)
    digest = hashlib.sha256(staging.read_bytes()).hexdigest()
    path = out / f"outcomes_v3-{digest[:16]}.parquet"  # immutable: never overwritten in place
    if path.exists():
        staging.unlink()
    else:
        staging.replace(path)
    manifest = dict(rows=len(frame), j1_accepted=j1_accepted, path=str(path),
                    sha256=digest,
                    by_corpus=frame.groupby("corpus").size().to_dict(),
                    status_counts={f"{c}|{s}": int(n) for (c, s), n in
                                   frame.groupby(["corpus", "adjudication_status"]).size().items()})
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    # the pointer is the only mutable file; consumers bind to the path + sha256 it names
    latest = out / "outcomes_v3.latest.json"
    latest.with_suffix(".tmp").write_text(json.dumps(dict(path=str(path), sha256=digest,
                                                          j1_accepted=j1_accepted), indent=1))
    latest.with_suffix(".tmp").replace(latest)
    print(json.dumps(manifest, default=str)[:1500])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--rescored", required=True)
    ap.add_argument("--card-root", default="/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/results/correlation_pipeline/card-v3")
    build(ap.parse_args(argv))


if __name__ == "__main__":
    main()
