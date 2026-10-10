"""Join frozen A/B/C effects, execution-local dose, and trajectory descriptions.

Run only in a CPU Slurm step. No inference, grading, outcome selection, or
historical mutation occurs. The report is a new immutable version.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

import numpy as np

from moe_exp.routing_control.historical_analysis_v1 import describe_windows, dose_descriptions, window_contrasts
from moe_exp.routing_control.study_v1 import STATES, file_sha, require, save, sealed

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "report/routing-study-v1"


def reference(path):
    value = sealed(path)
    return {"path": str(path), "sha256": value["sha256"], "file_sha256": file_sha(path)}


def integrate():
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID") and
            not socket.gethostname().startswith("login"), "integration requires a CPU Slurm step")
    inventory = sealed(OUT / "STUDY_INVENTORY_v2.json")
    studies = {}
    for name in ("A", "B", "C"):
        source = inventory["studies"][name]
        dense_ref = source["dense"][0]["artifact"]
        require(reference(dense_ref["path"]) == dense_ref, "historical trajectory source changed")
        dense_path = Path(dense_ref["path"])
        dense = sealed(dense_path)
        mapped = sealed(dense_path.with_name("ARM_MAP.json"))
        require(mapped["sha256"] == dense["arm_map_sha256"], "trajectory arm map differs")
        manifest = sealed(source["manifest"]["path"])
        require(manifest["horizon"] == dense["horizon"], "historical horizon differs")
        dose_path = Path(source["routing_engagement"][0]["artifact"]["path"])
        telemetry = sealed(dose_path.with_name("TELEMETRY.json"))
        dose = sealed(dose_path)
        require(telemetry["sha256"] == dose["telemetry_sha256"], "telemetry seal differs")
        records_by_uid = {r["uid"]: r for r in dense["records"]}
        require(len(records_by_uid) == dense["assigned_continuations"] and
                set(records_by_uid) == {r["uid"] for r in mapped["records"]}, "lost continuation identity")
        archives = defaultdict(list)
        for record in mapped["records"]:
            archives[record["array"]["path"]].append(record)
        records = []
        for path, entries in sorted(archives.items()):
            archive_sha = file_sha(path)
            require(all(archive_sha == r["array"]["sha256"] for r in entries), "saved route archive changed")
            with np.load(path, allow_pickle=False) as archive:
                for mapped_record in entries:
                    record = records_by_uid[mapped_record["uid"]]
                    if not mapped_record["array"]["available"]:
                        records.append({"uid": record["uid"], "family": record["family"],
                            "prefix_uid": record["prefix_uid"], "transition": record["transition"],
                            "arm": record["arm"], "seed": record["seed"], "status": "UNAVAILABLE_ROUTE_CAPTURE"})
                    else:
                        records.append(describe_windows(record, mapped_record, archive[mapped_record["array"]["key"]],
                                                        horizon=dense["horizon"]))
        if name == "A":
            pairs = [("pair", "native"), ("expert189", "native"), ("expert9", "native"),
                     ("pair", "expert189"), ("pair", "expert9")]
        elif name == "B":
            pairs = [("once", "native"), ("repeat", "native"), ("repeat", "once")]
        else:
            pairs = [("bias", "native"), ("force", "native"), ("reweight", "native"),
                     ("force", "bias"), ("reweight", "bias")]
        pairs += [(arm["name"], "random_" + arm["name"]) for arm in manifest["arms"] if arm["role"] == "target"]
        studies[name] = {"manifest": source["manifest"], "dense": dense_ref,
            "telemetry": reference(dose_path.with_name("TELEMETRY.json")), "horizon": dense["horizon"],
            "assigned_continuations": dense["assigned_continuations"], "records": records,
            "window_contrasts": window_contrasts(records, pairs),
            "frozen_semantic_inference": source["semantic"]["primary"],
            "frozen_semantic_missing_sensitivity": source["semantic"]["missing_measurement_sensitivity"],
            "frozen_inclusion_inference": dose["inference"],
            "achieved_sparse_dose_by_rank": dose_descriptions(telemetry, manifest["actions"]),
            "pulse_schedule": [{"arm": a["name"], "slots": a["slots"], "width": manifest["pulse_width"]}
                               for a in manifest["arms"]],
            "new_inference": "None. Window effects are post hoc paired family-equal descriptions.",
            "substantive_semantics": "Frozen blinded reader endpoint only; class occupancy is a separate descriptor."}
    result = save(OUT / "HISTORICAL_INTEGRATION.json", {"schema": "routing-study-historical-integration-v1",
        "inventory_sha256": inventory["sha256"], "studies": studies,
        "code_files": {str(p): file_sha(p) for p in (Path(__file__),
            REPO / "src/moe_exp/routing_control/historical_analysis_v1.py")},
        "horizons_pooled": False, "original_prompt_accuracy": "UNAVAILABLE",
        "dose_normalized_accuracy_damage": "UNAVAILABLE_NO_ORIGINAL_PROMPT_ACCURACY",
        "net_correct_to_incorrect_flips_per_exposure": "UNAVAILABLE_NO_PAIRED_ANSWER_GRADES"})
    plot(result)
    write_report(inventory, result)
    print(json.dumps({"status": "INTEGRATED", "sha256": result["sha256"],
                      "assigned": {k: v["assigned_continuations"] for k, v in studies.items()}}), flush=True)


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, (name, study) in zip(axes, result["studies"].items(), strict=True):
        transitions = sorted({r["transition"] for r in study["window_contrasts"]})
        for transition in transitions:
            desired = ["Verify"] if transition == "candidate_to_verify" else ["Plan", "Implement"]
            arms = sorted({r["left"] for r in study["window_contrasts"] if r["right"] == "native"})
            for arm in arms:
                rows = [r for r in study["window_contrasts"] if r["left"] == arm and r["right"] == "native" and
                        r["transition"] == transition and r["metric"] in ["class_occupancy:" + s for s in desired]]
                points = defaultdict(float)
                for r in rows:
                    points[(r["start"] + r["end"]) / 2] += r["estimate"]
                axis.plot(sorted(points), [points[k] for k in sorted(points)], marker="o",
                          label=arm + (" / verify" if transition == "candidate_to_verify" else " / commit"))
        axis.axhline(0, color="black", lw=.7)
        axis.axvspan(0, 256, color="grey", alpha=.12)
        if name == "B":
            axis.axvspan(512, 768, color="grey", alpha=.12)
        axis.set(title=f"{name}: {study['horizon']}-token horizon", xlabel="Continuation token window midpoint",
                 ylabel="Class occupancy change vs native")
        axis.legend(fontsize=7)
    fig.suptitle("Paired descriptive class occupancy; unknown labels remain missing")
    for suffix in ("png", "pdf"):
        path = OUT / ("historical-pulse-trajectories." + suffix)
        require(not path.exists(), "historical trajectory figure already exists")
        fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(inventory, integration):
    dense = sealed(OUT / "DENSE_INVENTORY.json")
    price = sealed(OUT / "dense-recovery-v2/PRICE.json")
    candidate = sealed(OUT / "CANDIDATE_TABLE.json")
    lines = ["# Routing study: evidence and remaining execution", "",
        "Historical A/B/C/FRESH estimates and their saved 50,000-draw intervals were reproduced exactly. "
        "Each population, cap, endpoint and inference remains separate. Capped mechanism results do not establish final-answer utility.", "",
        "| Study | Cap | Assigned | Valid semantic pairs | Interpretation |", "|---|---:|---:|---:|---|"]
    for name, study in inventory["studies"].items():
        lines.append(f'| {name} | {study["horizon"]} | {study["semantic"]["assigned"]} | '
                     f'{study["semantic"]["valid_pairs"]} | Discovery mechanism evidence |')
    lines += ["", f'Dense recovery inventory: {dense["committed"]:,} committed of {dense["assigned"]:,}; '
        f'{dense["missing"]:,} missing; {len(dense["ambiguous_calls"])} old calls without committed outputs retained explicitly.', "",
        "A does not establish an advantage of paired verification experts over either singleton. B does not establish additional "
        "benefit of the second pulse. C shows an exploratory reweighting effect in its short horizon; the degenerate bias estimate "
        "does not establish equivalence. These are substantive reader outcomes, distinct from the seven-class sentence descriptions.", "",
        "| Study/scope | Contrast | Substantive success change (pp) | Frozen simultaneous 95% interval (pp) | Families |",
        "|---|---|---:|---|---:|"]
    for name, study in integration["studies"].items():
        for row in study["frozen_semantic_inference"]["contrasts"]:
            if row["endpoint"] != "both_positive" or not (row["reference"] == "native" or
                (name == "A" and row["arm"] == "pair" and row["reference"] in ("expert189", "expert9")) or
                (name == "B" and row["reference"] == "once") or (name == "C" and row["reference"] == "bias")):
                continue
            lo, hi = row["simultaneous_ci95"]
            lines.append(f'| {name}/{row["scope"]} | {row["arm"]} − {row["reference"]} | '
                         f'{100*row["estimate"]:.2f} | [{100*lo:.2f}, {100*hi:.2f}] | {row["families"]} |')
    lines += ["", "The complete saved contrasts, token effects, and missing-reader identification bounds are retained in "
        "HISTORICAL_INTEGRATION.json. Small-family and degenerate bootstrap limitations remain applicable.", "",
        "![Historical pulse trajectories](historical-pulse-trajectories.png)", "",
        "The 0–256 and 512–768 windows are the first and repeated pulse positions. Post-pulse descriptions use only exact "
        "saved expert IDs and sentence token ownership. Unknown labels break adjacency; closure competes with state exits; caps "
        "censor unobserved exits. The complete window table separately reports routing selection, class occupancy, unknown labels, "
        "observed reasoning duration and closure. Window comparisons are descriptive and carry no new significance claims.", "",
        "Bias and forced inclusion are compared using the frozen inclusion-per-intended-opportunity endpoint. Reweighting also "
        "requires executed selected-weight displacement: sparse L1/2 and target mass from saved execution-local telemetry. The "
        "dose table retains TP ranks separately and excludes padded inactive layers. Its weights are normalized executed top-k "
        "weights, not full router probabilities. Exposure, displacement and semantic effects are not treated as equivalent.", "",
        "![Global and conditional expert profiles](global-conditional-heatmap.png)", "",
        f'{sum(r["status"] == "SUPPORTED_ASSOCIATION" for r in candidate["rows"]):,} expert-condition rows meet the '
        "ten-family/three-fold direction rule for class-transition proxies. No new singleton is selected: candidate completeness "
        "and checked status still need a prefix-only semantic recognition audit. Global versus conditional signs, ranks and cosine "
        "comparisons are recorded. The prospective predictive interface fits feature selection and regularization inside family "
        "folds, but held-out predictive gain is unavailable without a qualified prefix baseline and trailing feature frame. "
        "Full router confidence and native selection-boundary gaps are unavailable.", "",
        "Unchecked-candidate verification and approach commitment remain hypotheses requiring qualified prefix opportunities. "
        "Already-checked candidate closure and failed-check revision have insufficient qualified support. No causal specialist "
        "or original-prompt accuracy claim is made.", "",
        "The separate native-finalization prompt-control diagnostic has zero of 32 outcomes. Native repeat variability and "
        "execution-local capture were audited; the v4 exact native-token repeat gate failed. It remains an engineering "
        "qualification failure. Counterfactual-NLL qualification remains failed; X3 remains deferred. Qualified free-generation "
        "analysis can proceed independently.", ""]
    source = OUT / "full-math-source/SOURCE.json"
    if source.exists():
        math = sealed(source)
        lines += ["The full seven-subject MATH test source has now been pinned and downloaded under full-math-source/. "
            "The prepared family-freeze job verifies all 500 previously used MATH-500 texts, preserves prior duplicate connections, "
            "and reserves 32 validation plus 96 final families only after excluding all previously used connected families. "
            "Submission is awaiting explicit authorization of two CPUs, 32 GiB and at most one hour. No new-family outcomes "
            "have been generated or used for selection.", ""]
    lines += ["The independent analysis implements seed averaging within family, equal family weights, paired simultaneous "
        "95% accuracy–cost intervals, missing-grade sensitivity, and the stated deterministic validation choice. No acceptable "
        "accuracy-loss margin is imposed. The frontier is unavailable until qualified policies and independent outcomes exist; "
        "a nonsignificant accuracy difference will not be reported as accuracy retention.", "",
        "| Resource component | Observed / authorized status |", "|---|---|",
        f'| Existing dense labels | {len(price["groups"])} packed jobs, maximum {price["maximum_GPU_hours"]:.2f} GPU-hours including bounded recovery; packaging qualification required |',
        "| Packaging qualification | Original job 59490750 failed before model load at the IPC path limit; shortened-path retry receipts in submissions-v4/ |",
        "| Existing utility pilots | Jobs 59405556, 59405561, 59405635 retained and reconciled |",
        "| Existing operator pilots | Jobs 59400762, 59400764, 59400766 retained and reconciled |",
        "| Offline integration | Two CPUs, at most one hour; historical authorized analysis |",
        "| Independent family freeze | 2 CPU-core-hours maximum, zero GPUs; concrete approval pending |",
        "| New target / operator / timing / evaluation | Not submitted; complete measured generation, prefill, controller, labeling, grading, load and bounded-recovery prices required |", "",
        "Policy cost includes controller-generated tokens. Processed input, controller overhead, runtime, reasoning tokens, "
        "and offline labeling/grading expenditure will be itemized separately. Accuracy damage and net correct-to-incorrect "
        "flips per exposure remain unavailable without paired final-answer grades; near-zero-dose ratios will stay descriptive.", "",
        "Machine-readable inputs: STUDY_INVENTORY_v2.json, CANDIDATE_TABLE.json, INTERVENTION_MANIFEST.json. "
        "The latter retains its earlier unavailable-source snapshot; the new source and authorization proposal are separate "
        "sealed inputs. Historical source artifacts remain unchanged. This study is incomplete while label recovery, queued "
        "pilots, prefix qualification and independent evaluation are outstanding.", ""]
    content = "\n".join(lines)
    path = OUT / "REPORT_v2.md"
    require(not path.exists(), "new report version already exists")
    path.write_text(content)
    from routing_study_v1 import accounting
    jobs = ["59490750", "59405556", "59405561", "59405635", "59400762", "59400764", "59400766",
            os.environ["SLURM_JOB_ID"]]
    retry_receipt = OUT / "submissions-v4/qualification.json"
    if retry_receipt.exists():
        jobs.append(sealed(retry_receipt)["job_id"])
    observed = accounting(jobs)
    save(OUT / "RESOURCE_LEDGER_v2.json", {"schema": "routing-study-resource-ledger-v2",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "allocation_accounting": observed,
        "historical_accounting": inventory["accounting"],
        "research_cost_scope": "Offline class labels, substantive readers, answer grading and analysis kept separate from policy costs.",
        "prospective_policy_fields": ["generated_tokens", "controller_generated_tokens", "reasoning_tokens",
                                     "processed_input_tokens", "controller_input_tokens", "controller_seconds", "wall_seconds"],
        "missing_observations": "Queued assignments and current in-flight job are not completed scientific outcomes."})
    save(OUT / "REPORT_v2_BINDING.json", {"schema": "routing-study-integrated-report-v2",
        "inventory_sha256": inventory["sha256"], "integration_sha256": integration["sha256"],
        "report_file_sha256": file_sha(path), "figures": {p.name: file_sha(p) for p in OUT.glob("historical-pulse-trajectories.*")},
        "status": "INCOMPLETE_EXECUTION_AND_QUALIFICATION"})


if __name__ == "__main__":
    integrate()
