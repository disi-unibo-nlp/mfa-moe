"""J1 acceptance gate: score the judge on the frozen blind audit consensus, then publish verdicts.

Pre-agreed rule (fixed before any verdict was read): >=95/100 three-state matches with the
agent consensus, zero false EQUIVALENT on clear-wrong controls, <=2 errors in every stratum.
Writes acceptance/j1.json (status complete|failed) and, on success, hist_verdicts.jsonl
(the historical subset of the combined run) so outcomes.load_verdicts can use it; then
rebuilds the outcome sidecar. Exits non-zero on failure so nothing downstream runs.

    python j1_gate.py --audit j1/audit2_items.json --consensus j1/audit2_consensus.json \
        --consensus-sha 8605916f... --verdicts j1/j1v2_verdicts.jsonl --hist-items j1/hist_items.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

D = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3")
REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
CLIENT_PY = "/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python"
RESCORED = "/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/tables/rescored.parquet"
MIN_MATCHES, MAX_STRATUM_ERRORS = 95, 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--consensus", type=Path, required=True)
    ap.add_argument("--consensus-sha", required=True, help="sha256 of the frozen consensus file")
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--hist-items", type=Path, required=True)
    ap.add_argument("--no-rebuild", action="store_true")
    ap.add_argument("--campaign", type=Path, default=D)
    args = ap.parse_args(argv)

    if sha256(args.consensus) != args.consensus_sha:
        raise SystemExit("consensus file changed after freezing")
    audit = json.loads(args.audit.read_text())
    consensus = json.loads(args.consensus.read_text())
    if consensus["audit_sha256"] != audit["audit_sha256"] or not consensus["frozen_before_j1_verdicts"]:
        raise SystemExit("consensus does not belong to this audit set")
    judged = {}
    for line in args.verdicts.open():
        row = json.loads(line)
        judged[row["item_id"]] = row
    stratum_of = {i["item_id"]: i["stratum"] for i in audit["items"]}
    missing = [c["item_id"] for c in consensus["labels"] if c["item_id"] not in judged]
    if missing:
        raise SystemExit(f"{len(missing)} audit items have no verdict")

    errors, false_eq, per_stratum = [], [], Counter()
    for c in consensus["labels"]:
        verdict = judged[c["item_id"]]["verdict"]
        stratum = stratum_of[c["item_id"]]
        if verdict != c["verdict"]:
            per_stratum[stratum] += 1
            errors.append(dict(index=c["index"], item_id=c["item_id"], stratum=stratum,
                               consensus=c["verdict"], judge=verdict,
                               votes=judged[c["item_id"]]["votes"]))
            if stratum == "clear_wrong_control" and verdict == "EQUIVALENT":
                false_eq.append(c["index"])
    matches = len(consensus["labels"]) - len(errors)
    failures = []
    if matches < MIN_MATCHES:
        failures.append(f"matches {matches} < {MIN_MATCHES}")
    if false_eq:
        failures.append(f"false EQUIVALENT on clear-wrong controls {false_eq}")
    for stratum, n in per_stratum.items():
        if n > MAX_STRATUM_ERRORS:
            failures.append(f"{stratum}: {n} errors > {MAX_STRATUM_ERRORS}")

    all_rows = list(judged.values())
    receipt = dict(
        status="failed" if failures else "complete", audit_sha256=audit["audit_sha256"],
        consensus_sha256=args.consensus_sha, verdicts_sha256=sha256(args.verdicts),
        matches=matches, of=len(consensus["labels"]), per_stratum_errors=dict(per_stratum),
        failures=failures, errors=errors,
        judge_verdicts_all=dict(Counter(r["verdict"] for r in all_rows)),
        votes_all=dict(Counter(v for r in all_rows for v in r["votes"])),
        checked_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        note="Agent adjudication (Claude + Codex-DeepSeek), not human ground truth.")
    campaign = args.campaign
    (campaign / "acceptance").mkdir(exist_ok=True)
    (campaign / "acceptance" / "j1.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ("status", "matches", "of", "per_stratum_errors",
                                              "failures", "judge_verdicts_all")}))
    if failures:
        return 1

    hist_ids = {json.loads(line)["item_id"] for line in args.hist_items.open()}
    unjudged = hist_ids - judged.keys()
    if unjudged:
        print(json.dumps(dict(error="judge run incomplete; resume it", unjudged=len(unjudged))))
        return 3
    target = args.hist_items.with_name("hist_verdicts.jsonl")
    tmp = target.with_suffix(".tmp")
    with tmp.open("w") as sink:
        for row in all_rows:
            if row["item_id"] in hist_ids:
                sink.write(json.dumps(row) + "\n")
    tmp.replace(target)
    print(json.dumps(dict(hist_verdicts=sum(r["item_id"] in hist_ids for r in all_rows),
                          hist_items=len(hist_ids))))
    if not args.no_rebuild:
        subprocess.run([CLIENT_PY, "-m", "moe_exp.correlation_pipeline.outcomes",
                        "--campaign", str(campaign), "--rescored", RESCORED],
                       check=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
        print((campaign / "outcomes" / "outcomes_v3.latest.json").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
