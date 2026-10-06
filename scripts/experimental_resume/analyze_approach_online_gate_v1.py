"""Discovery-family audit for a prospective query-every-approach-fire gate.

The gate has no learned score threshold: it asks the native semantic side
model at every complete-sentence approach lexical fire through the fixed 16k
utility horizon.  Its online use still requires execution qualification.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
AGREEMENT = REPO / "report/experimental-resume-v1/FULL_PREFIX_NATIVE_VETO_AGREEMENT_v1.json"
BURDEN = REPO / "report/experimental-resume-v1/NATIVE_VETO_ALL_FIRE_BURDEN_v0.json"
POOL = BASE / "JOINT_QWEN_NATIVE_EXACT_POOL_v1.json"
OUT = REPO / "report/experimental-resume-v1/APPROACH_ONLINE_GATE_DISCOVERY_v1.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({key: val for key, val in value.items() if key != "sha256"}):
        raise ValueError(f"changed sealed input: {path}")
    return value


def interval(numerator, denominator, columns, seed=20261002, draws=20000):
    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(columns), size=(draws, len(columns)))
    sums = np.asarray(columns, dtype=np.int64)[take].sum(axis=1)
    den = sums[:, denominator]
    valid = den > 0
    value = sums[valid, numerator] / den[valid]
    return {"method": "discovery-family cluster bootstrap, fixed seed, 20000 resamples",
            "resamples_with_denominator": int(valid.sum()),
            "lower_95": float(np.quantile(value, .025)),
            "upper_95": float(np.quantile(value, .975))}


def main():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("analysis requires allocated CPU Slurm")
    agreement, burden, pool = map(sealed, (AGREEMENT, BURDEN, POOL))
    if (agreement["schema"] != "full-prefix-native-veto-agreement-v1"
            or burden["schema"] != "native-veto-all-fire-burden-v0"
            or pool["schema"] != "joint-qwen-native-exact-pool-v1"
            or agreement["sha256"] != pool["agreement_sha256"]
            or burden["counts"].get("approach_to_commit|all_fires") != 42):
        raise ValueError("discovery gate inputs or coverage changed")
    rows = [row for row in agreement["records"] if row["transition"] == "approach_to_commit"]
    fires = [row for row in rows if row["v22_fired"]]
    nonfires = [row for row in rows if not row["v22_fired"]]
    if len(rows) != 83 or len(fires) != 35 or len(nonfires) != 48:
        raise ValueError("frozen stratified approach sample changed")
    status = Counter()
    by_family = defaultdict(lambda: np.zeros(8, dtype=np.int64))
    # Columns: fire, Qwen accepted, Qwen rejected, Qwen split, native accepted,
    # Qwen accepted and native accepted, Qwen rejected and native accepted,
    # Qwen accepted and native not accepted.
    for row in fires:
        q, n = row["qwen_status"], row["native_status"]
        status["qwen_" + q] += 1
        status["native_" + n] += 1
        item = by_family[row["family"]]
        item[0] += 1
        item[1] += q == "accepted"
        item[2] += q == "rejected"
        item[3] += q == "split"
        item[4] += n == "accepted"
        item[5] += q == "accepted" and n == "accepted"
        item[6] += q == "rejected" and n == "accepted"
        item[7] += q == "accepted" and n != "accepted"
    for row in nonfires:
        by_family[row["family"]]  # include sampled families with zero fire
    if len(by_family) != 48 or status != {
            "qwen_accepted": 12, "qwen_rejected": 21, "qwen_split": 2,
            "native_accepted": 9, "native_rejected": 24, "native_split": 2}:
        raise ValueError("approach start audit statuses changed")
    matrix = list(by_family.values())
    rate = lambda num, den: {"numerator": num, "denominator": den,
                             "estimate": num / den if den else None}
    rates = {
        "qwen_two_reader_accepted_among_lexical_fires": {
            **rate(12, 35), "interval": interval(1, 0, matrix, seed=20261002)},
        "qwen_two_reader_rejected_among_lexical_fires": {
            **rate(21, 35), "interval": interval(2, 0, matrix, seed=20261003)},
        "native_two_reader_accepted_among_lexical_fires": {
            **rate(9, 35), "interval": interval(4, 0, matrix, seed=20261004)},
        "qwen_accepted_among_native_accepted": {
            **rate(6, 9), "interval": interval(5, 4, matrix, seed=20261005)},
        "qwen_rejected_among_native_accepted": {
            **rate(3, 9), "interval": interval(6, 4, matrix, seed=20261006)},
        "native_accepted_among_qwen_accepted": {
            **rate(6, 12), "interval": interval(5, 1, matrix, seed=20261007)},
    }
    selected = [row for row in pool["records"] if row["uid"] in set(pool["selected_global_uids"])]
    approach = [row for row in selected if row["transition"] == "approach_to_commit"]
    early = {str(limit): sum(row["prefix_tokens"] <= limit for row in approach)
             for limit in (128, 256, 512, 1024)}
    if len(approach) != 5 or early["512"] != 5:
        raise ValueError("global exact-ID early approach pool changed")
    body = {
        "schema": "approach-online-gate-discovery-v1",
        "job_id": os.environ["SLURM_JOB_ID"],
        "agreement_sha256": agreement["sha256"],
        "burden_sha256": burden["sha256"],
        "joint_pool_sha256": pool["sha256"],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rule": "At each completed approach_to_commit lexical fire through the 16384 output-token horizon, ask two independent unsteered native-model start-only questions on the actual current prefix; accept only two strict natural-stop true answers; ambiguous/failure abstains; stop at reasoning closure. This is a discovery proposal, not qualified utility policy.",
        "rated_stratified_windows": {"fires": 35, "nonfires": 48,
                                     "families": 48,
                                     "nonfire_qwen_accepted": sum(row["qwen_status"] == "accepted" for row in nonfires),
                                     "fire_statuses": dict(status)},
        "rates": rates,
        "all_native_discovery_burden": {
            "all_approach_fires": 42,
            "within_16384_prefix": burden["counts"]["approach_to_commit|online_context_eligible"],
            "one_query_prompt_tokens": burden["tokens"]["approach_to_commit|prompt_tokens_once"],
            "two_query_prompt_tokens": 2 * burden["tokens"]["approach_to_commit|prompt_tokens_once"],
            "two_query_response_cap_tokens": 2 * burden["tokens"]["approach_to_commit|response_cap_tokens_once"],
            "late_8193_to_16384_fires": burden["counts"]["approach_to_commit|utility_late_8193_to_16384"],
        },
        "global_family_disjoint_exact_pool": {
            "approach_rows": 5, "prefix_tokens": sorted(row["prefix_tokens"] for row in approach),
            "at_or_before_tokens": early},
        "interpretation": "These are descriptive discovery-family, LLM-audited start rates from a stratified window sample; the cluster bootstrap is uncertainty within these families, not independent validation. One Qwen-approved nonfire among 48 cannot estimate live recall because fires/nonfires were sampled separately. Side-query burden is cached-native, not steered online execution."
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "sha256": digest(body),
                      "rates": {key: value["estimate"] for key, value in rates.items()},
                      "early": early}), flush=True)


if __name__ == "__main__":
    main()
