"""Paired problem analysis, stratified bootstrap, frozen frontiers and standalone plots."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import (BENCHMARKS, digest, seal, verify, rows, read_json, file_hash, write_json)
from .execution import MATCH, check_record
from .policy import validate

OUTCOMES = ("accuracy", "tokens", "truncation", "latency", "wrong_to_right", "right_to_wrong", "active_rate")


def nondominated(points):
    return sorted([p for p in points if not any(
        q["accuracy"] >= p["accuracy"] and q["tokens"] <= p["tokens"]
        and (q["accuracy"] > p["accuracy"] or q["tokens"] < p["tokens"]) for q in points)],
        key=lambda p: (p["tokens"], -p["accuracy"], p["policy_id"]))


def select_frontier(points):
    frontier = nondominated(points)
    if not frontier:
        return []
    high = min(frontier, key=lambda p: (-p["accuracy"], p["tokens"], p["policy_id"]))
    low = min(frontier, key=lambda p: (p["tokens"], -p["accuracy"], p["policy_id"]))
    # One point per observed token cost; median-token frontier point, lower median for even n.
    by_cost = {}
    for p in frontier:
        by_cost.setdefault(p["tokens"], p)
    distinct = sorted(by_cost.values(), key=lambda p: (p["tokens"], -p["accuracy"], p["policy_id"]))
    middle = distinct[(len(distinct)-1)//2]
    return list(dict.fromkeys(p["policy_id"] for p in (high, low, middle)))


def bh(values):
    if not values:
        return []
    order = np.argsort(values, kind="stable")
    result, previous = [0.] * len(values), 1.
    for rank in range(len(values), 0, -1):
        index = int(order[rank-1])
        previous = min(previous, float(values[index]) * len(values) / rank)
        result[index] = previous
    return result


def paired_bootstrap(arrays, *, clusters=None, replicates=1000, seed=42):
    """arrays[benchmark] = [questions, conditions, outcomes]; shared question draws."""
    if set(arrays) != set(BENCHMARKS) or any(len(a) == 0 for a in arrays.values()):
        raise ValueError("Require every benchmark and nonempty question coverage")
    rng = np.random.default_rng(seed)
    result = {b: [] for b in BENCHMARKS}
    for _ in range(replicates):
        for benchmark in BENCHMARKS:
            array = np.asarray(arrays[benchmark])
            groups = clusters[benchmark] if clusters else [[i] for i in range(len(array))]
            selected = rng.integers(0, len(groups), len(groups))
            indices = [i for g in selected for i in groups[g]]
            result[benchmark].append(array[indices].mean(0))
    result = {b: np.asarray(values) for b, values in result.items()}
    result["equal_benchmark_mean"] = np.mean(list(result.values()), axis=0)
    return result


def load_run(path, study):
    path = Path(path)
    manifest = read_json(path/"manifest.json")
    if manifest["status"] != "complete" or not manifest.get("dispatch_verified"):
        raise ValueError("Incomplete or unverified run")
    if manifest["study_sha256"] != study["sha256"]:
        raise ValueError("Run belongs to a different frozen study")
    phase = manifest["phase"]
    if phase not in ("tuning", "test"):
        raise ValueError("Calibration cannot enter comparison")
    if phase == "test":
        selection = manifest.get("final_selection")
        if selection is None:
            raise ValueError("Final-test release receipt missing")
        verify(selection)
        if (selection["study_sha256"] != study["sha256"] or selection["stage"] != "final"
                or len(selection["selected_policy_hashes"]) > 3
                or manifest["policy"]["sha256"] not in selection["allowed_policy_hashes"]):
            raise ValueError("Final-test policy was not frozen")
    planned = {r["id"]: r for r in study["attempts"][phase]}
    records = {}
    for record in rows(path/"attempts.jsonl"):
        key = record["id"]
        if key in records or key not in planned:
            raise ValueError("Duplicate/unplanned attempt in comparison")
        check_record(record, planned[key], manifest["policy"]["sha256"])
        if record.get("infrastructure_failure"):
            raise ValueError("Infrastructure failure is not a scored completion")
        if set(record["dispatch"]) != set(map(str, range(40))):
            raise ValueError("Missing native dispatch diagnostics")
        records[key] = record
    if (set(records) != set(planned) or len(records) != manifest["expected_count"]
            or file_hash(path/"attempts.jsonl") != manifest["attempts_sha256"]):
        raise ValueError("Final comparisons require complete, unchanged planned coverage")
    failures = list(rows(path/"failures.jsonl")) if (path/"failures.jsonl").exists() else []
    if failures and file_hash(path/"failures.jsonl") != manifest["failures_sha256"]:
        raise ValueError("Infrastructure failure ledger changed")
    for f in failures:
        if f["id"] not in planned or f["input_sha256"] != digest(planned[f["id"]]):
            raise ValueError("Failure ledger inputs differ")
        if f["token_count_status"] != "exact":
            raise ValueError("Unmeasured failed-attempt token cost; no complete cost comparison")
    return manifest, records, failures


def compare(paths, study, comparisons, *, replicates=1000, output=None):
    verify(study)
    runs = [load_run(p, study) for p in paths]
    baseline, base_records, _ = runs[0]
    if baseline["policy"]["action"] != "native":
        raise ValueError("First run must be the matched native baseline")
    ids = [run[0]["policy"]["sha256"] for run in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate policy runs")
    for manifest, records, _ in runs:
        if any(manifest.get(k) != baseline.get(k) for k in MATCH):
            raise ValueError("Unmatched prompts/seeds/backend/checkpoint/concurrency/output limits")
        for key, record in records.items():
            base = base_records[key]
            if (record["prompt_token_ids"] != base["prompt_token_ids"]
                    or record["sampling_args"] != base["sampling_args"]
                    or record["input_sha256"] != base["input_sha256"]):
                raise ValueError("Paired request mismatch")
    for manifest, _, _ in runs:
        pid = manifest["policy"]["sha256"]
        if (manifest.get("registration_sha256") != comparisons["sha256"]
                and comparisons.get("existing_controls", {}).get(pid) != manifest["attempts_sha256"]):
            raise ValueError("Run predates/mismatches registration without a frozen existing-control binding")
    # Comparisons must be declared before outcomes are opened; CLI reads a sealed registration.
    verify(comparisons)
    if (comparisons["study_sha256"] != study["sha256"] or comparisons["phase"] != baseline["phase"]
            or comparisons["policy_hashes"] != ids):
        raise ValueError("Prespecified comparison registration differs")
    question_rows = {b: defaultdict(lambda: defaultdict(list)) for b in BENCHMARKS}
    total_tokens, failure_counts = [], []
    for condition, (manifest, records, failures) in enumerate(runs):
        failed_tokens = defaultdict(int)
        for fail in failures:
            failed_tokens[fail["id"]] += fail["generated_token_count"]
        total_tokens.append(sum(r["generated_token_count"] for r in records.values()) + sum(failed_tokens.values()))
        failure_counts.append(len(failures))
        for key, record in records.items():
            base = base_records[key]
            diagnostics = [d["diagnostics"] for d in record["dispatch"].values()]
            from .runtime import applies
            denominator = sum(record["dispatch"][str(l)]["diagnostics"]["reasoning_tokens"]
                              for l in range(40) if applies(manifest["policy"], l))
            active = sum(d["active"] for d in diagnostics) / denominator if denominator else 0.
            vector = [float(record["is_correct"]), record["generated_token_count"] + failed_tokens[key],
                float(record["finish_reason"] == "length"), record["latency_seconds"],
                float(not base["is_correct"] and record["is_correct"]),
                float(base["is_correct"] and not record["is_correct"]), active]
            question_rows[record["dataset"]][record["problem_key"]][condition].append(vector)
    arrays, clusters, keys = {}, {}, {}
    group_map = {k: group for group, members in study["split"]["groups"].items() for k in members}
    for benchmark, questions in question_rows.items():
        keys[benchmark] = sorted(questions)
        arrays[benchmark] = np.array([[np.mean(questions[q][c], axis=0) for c in range(len(runs))]
                                     for q in keys[benchmark]])
        groups = defaultdict(list)
        for i, q in enumerate(keys[benchmark]):
            groups[group_map[q]].append(i)
        clusters[benchmark] = list(groups.values())
    bootstrap = paired_bootstrap(arrays, clusters=clusters, replicates=replicates)
    means = {b: a.mean(0) for b, a in arrays.items()}
    means["equal_benchmark_mean"] = np.mean(list(means.values()), axis=0)
    estimates = {}
    for benchmark, mean in means.items():
        estimates[benchmark] = {}
        for c, pid in enumerate(ids):
            estimates[benchmark][pid] = {outcome: {"estimate": float(mean[c, j]),
                "ci95": np.quantile(bootstrap[benchmark][:, c, j], [.025, .975]).tolist()}
                for j, outcome in enumerate(OUTCOMES)}
    tests = []
    for family in OUTCOMES:
        j = OUTCOMES.index(family)
        family_tests = []
        for pair in comparisons["pairs"]:
            a, b = map(ids.index, pair)
            for benchmark in means:
                observed = means[benchmark][b,j] - means[benchmark][a,j]
                diff = bootstrap[benchmark][:, b,j] - bootstrap[benchmark][:, a,j]
                centered = diff - observed
                p = (1 + int(np.sum(np.abs(centered) >= abs(observed)))) / (replicates + 1)
                family_tests.append(dict(family=family, benchmark=benchmark, pair=pair,
                    delta=float(observed), ci95=np.quantile(diff, [.025,.975]).tolist(),
                    p_value=float(p), test="two_sided_centered_paired_question_bootstrap"))
        for row, adjusted in zip(family_tests, bh([r["p_value"] for r in family_tests])):
            row["bh_q"] = adjusted
        tests.extend(family_tests)
    points = []
    for pid in ids:
        value = estimates["equal_benchmark_mean"][pid]
        points.append(dict(policy_id=pid, accuracy=value["accuracy"]["estimate"], tokens=value["tokens"]["estimate"]))
    instrumentation = {}
    for pid,run in zip(ids,runs):
        sessions = defaultdict(float)
        for record in run[1].values():
            key = record.get("execution_session", "legacy_fixture")
            sessions[key] = max(sessions[key], record.get("hook_seconds_cumulative",0.))
        instrumentation[pid] = sum(sessions.values())
    report = seal(dict(schema=1, study_sha256=study["sha256"], phase=baseline["phase"],
        registration_sha256=comparisons["sha256"], policy_hashes=ids, replicates=replicates,
        estimates=estimates, comparisons=tests, points=points,
        observed_frontier=[p["policy_id"] for p in nondominated(points)],
        total_generated_tokens=dict(zip(ids,total_tokens)), infrastructure_failures=dict(zip(ids,failure_counts)),
        question_counts={b: len(a) for b,a in arrays.items()},
        instrumentation_seconds=instrumentation,
        active_rate_denominator="generated reasoning token-layer positions in layers targeted by the action",
        session_seconds={pid: run[0].get("session_seconds") for pid,run in zip(ids,runs)},
        supported_accuracy_increases=[t for t in tests if t["family"]=="accuracy" and t["bh_q"]<.05 and t["ci95"][0]>0],
        supported_token_reductions=[t for t in tests if t["family"]=="tokens" and t["bh_q"]<.05 and t["ci95"][1]<0],
        interpretation="Observed frontier is descriptive. Supported improvements require paired intervals and BH-adjusted prespecified comparisons.",
        method_notes={"steer": "outcome-based adaptation", "rice": "signed marker score, selected-weight action; normalized primary and unnormalized ablation",
                      "identity_margin": "colleague mathematics retained; margin membership changes measured",
                      "timing": "calibration-only hypotheses, no online judge"}))
    if output:
        output = Path(output)
        write_json(output/"comparison.json", report, immutable=True)
        plot(report, output)
        write_report(report, output)
    return report


def register(study, policies, pairs, phase, existing_controls=()):
    hashes = [validate(p)["sha256"] for p in policies]
    if len(set(hashes)) != len(hashes) or policies[0]["action"] != "native":
        raise ValueError("Unique policies with native first required")
    if any(a not in hashes or b not in hashes or a == b for a,b in pairs):
        raise ValueError("Invalid paired comparison")
    if len({tuple(p) for p in pairs}) != len(pairs):
        raise ValueError("Duplicate paired comparison")
    controls = {}
    for path in existing_controls:
        manifest = read_json(Path(path)/"manifest.json")
        p = manifest["policy"]
        if (manifest["status"] != "complete" or p["gate"] != "always" or p["sha256"] not in hashes
                or manifest["study_sha256"] != study["sha256"] or manifest["phase"] != phase):
            raise ValueError("Only completed matching always-active controls may predate this registration")
        controls[p["sha256"]] = manifest["attempts_sha256"]
    return seal(dict(schema=1, study_sha256=study["sha256"], phase=phase,
                     policy_hashes=hashes, pairs=pairs, outcome_families=list(OUTCOMES), existing_controls=controls))


def freeze_frontier(report, study, policies, *, stage, necessary_controls=(), static_selection=None):
    verify(report)
    if report["phase"] != "tuning" or report["study_sha256"] != study["sha256"] or stage not in ("static","final"):
        raise ValueError("Selection must use matching tuning outcomes")
    mapping = {p["sha256"]: validate(p) for p in policies}
    if set(mapping) != set(report["policy_hashes"]):
        raise ValueError("Selection policy catalog differs from comparison")
    eligible = set(mapping)
    if stage == "final":
        if static_selection is None:
            raise ValueError("Final freeze requires the prior static shortlist")
        verify(static_selection)
        if static_selection["stage"] != "static" or static_selection["study_sha256"] != study["sha256"]:
            raise ValueError("Invalid static screening lineage")
        eligible = set()
        for pid in static_selection["selected_policy_hashes"]:
            if pid not in mapping:
                raise ValueError("Missing always-active shortlisted action")
            base = mapping[pid]
            matching = [p for p in policies if p["action"] == base["action"] and p["strength"] == base["strength"]
                and p["targets"] == base["targets"] and {k:v for k,v in p["parameters"].items() if k != "timing"} == base["parameters"]]
            if {p["gate"] for p in matching} != {"always", "persistent", "unstable", "random_persistent", "random_unstable", "position_persistent", "position_unstable"}:
                raise ValueError("Final freeze requires both adaptive hypotheses and all matched controls")
            eligible.update(p["sha256"] for p in matching)
    if stage == "static" and any(p["gate"] != "always" for p in policies):
        raise ValueError("Static screening cannot include adaptive outcomes")
    candidates = [p for p in nondominated(report["points"]) if p["policy_id"] in eligible and mapping[p["policy_id"]]["action"] != "native"
                  and not mapping[p["policy_id"]]["gate"].startswith(("random_", "position_"))
                  and not (mapping[p["policy_id"]]["action"] == "identity" and mapping[p["policy_id"]]["strength"] == 0)]
    chosen = select_frontier(candidates)
    controls = set(necessary_controls)
    controls |= {p["sha256"] for p in policies if p["action"] == "native"
                 or p["action"] == "identity" and p["strength"] == 0}
    if not controls.issubset(mapping):
        raise ValueError("Necessary controls need complete matched tuning results")
    for pid in chosen:
        policy = mapping[pid]
        if policy["gate"] != "always":
            family = policy["gate"].split("_")[-1]
            required = {"always", family, "random_"+family, "position_"+family}
            for gate in required:
                matches = [p["sha256"] for p in policies if p["action"] == policy["action"]
                    and p["strength"] == policy["strength"] and p["targets"] == policy["targets"]
                    and p["gate"] == gate
                    and {k:v for k,v in p["parameters"].items() if k != "timing"}
                        == {k:v for k,v in policy["parameters"].items() if k != "timing"}]
                if len(matches) != 1:
                    raise ValueError("Adaptive candidate lacks a unique always/random/position control")
                controls.update(matches)
    return seal(dict(schema=1, study_sha256=study["sha256"], phase="tuning", stage=stage,
        static_selection_sha256=static_selection["sha256"] if static_selection else None,
        tuning_report_sha256=report["sha256"], selected_policy_hashes=chosen,
        controls=sorted(controls), allowed_policy_hashes=sorted(set(chosen)|controls)))


def plot(report, output, *, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    for point in report["points"]:
        pid = point["policy_id"]
        value = report["estimates"]["equal_benchmark_mean"][pid]
        acc, tokens = value["accuracy"], value["tokens"]
        ax.plot(tokens["estimate"], acc["estimate"], "o")
        ax.hlines(acc["estimate"], *tokens["ci95"], alpha=.6)
        ax.vlines(tokens["estimate"], *acc["ci95"], alpha=.6)
        ax.annotate(pid[:8], (tokens["estimate"], acc["estimate"]), xytext=(4,4), textcoords="offset points", fontsize=7)
    frontier = nondominated(report["points"])
    ax.plot([p["tokens"] for p in frontier], [p["accuracy"] for p in frontier], "k--", alpha=.5)
    ax.set(xlabel="Generated tokens per attempt (question and benchmark balanced)",
           ylabel="Accuracy (equal weight across six benchmarks)",
           title=("Synthetic verification fixture\n" if synthetic else "") + "Observed accuracy–token frontier\nPaired question-bootstrap 95% intervals")
    ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output/("tradeoff."+suffix), dpi=180)
    plt.close(fig)


def write_report(report, output):
    text = ["# Qwen3.6 routing tradeoffs", "", report["interpretation"], "",
            "Small AIME question counts can yield wide uncertainty even with 32 attempts per question.",
            "Matching seeds do not imply identical stochastic trajectories.", "",
            "| Policy | Accuracy | Mean tokens | Raw total tokens | Infrastructure failures |",
            "|---|---:|---:|---:|---:|"]
    for p in report["points"]:
        pid=p["policy_id"]
        text.append(f"| {pid[:12]} | {p['accuracy']:.4f} | {p['tokens']:.1f} | {report['total_generated_tokens'][pid]} | {report['infrastructure_failures'][pid]} |")
    text += ["", "Full per-benchmark intervals, paired reversals, latency and realized intervention rates are in comparison.json.",
             "Failed-attempt tokens are included only when their exact cost is known; incomplete cost ledgers are rejected.",
             "Seven-class pilot labels are interpretive context and never an online routing judge.", ""]
    (output/"report.md").write_text("\n".join(text))
