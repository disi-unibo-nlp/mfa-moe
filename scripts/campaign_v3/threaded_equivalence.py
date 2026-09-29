"""Scorer gate: threaded (spawn-pool) and main-thread math_verify agree on real traces."""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from moe_exp.correlation_pipeline.continuation import DATASETS, HISTORICAL
from moe_exp.correlation_pipeline.scoring import score_completion_detailed, shutdown_scoring_pool



def main() -> int:
    # spawn workers re-import __main__; everything must stay behind the guard
    rng = random.Random(20260925)
    cases = []
    for model, root in HISTORICAL.items():
        for dataset in DATASETS:
            lines = (root / dataset / "traces.jsonl").read_text().splitlines()
            for line in rng.sample(lines, 7):
                r = json.loads(line)
                cases.append(({"gold_answer": r.get("gold_answer")}, r["metadata"].get("assistant_content") or ""))
    cases = cases[:300]
    direct = [score_completion_detailed(e, answer_type="math", model_text=t) for e, t in cases]
    with ThreadPoolExecutor(8) as pool:
        threaded = list(pool.map(lambda c: score_completion_detailed(c[0], answer_type="math", model_text=c[1]), cases))
    shutdown_scoring_pool()
    mismatch = [i for i, (a, b) in enumerate(zip(direct, threaded)) if (a["is_correct"], a["method"]) != (b["is_correct"], b["method"])]
    fallback = sum(t["method"] != "math_verify" for t in threaded)
    print(json.dumps(dict(cases=len(cases), mismatches=len(mismatch), threaded_fallbacks=fallback,
                          statuses=sorted({t["math_verify_status"] for t in threaded}))))
    return 0 if not mismatch else 3


if __name__ == "__main__":
    sys.exit(main())
