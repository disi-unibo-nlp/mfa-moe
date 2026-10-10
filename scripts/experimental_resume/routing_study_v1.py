"""Reconciled routing ledger, conditional discovery and prospective design.

All outputs are additive. Historical effects and inference are copied with
their exact source seals. New routing associations use discovery families only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess

from moe_exp.routing_control.study_v1 import (STATES, association_table, digest, file_sha,
                                             require, save, sealed, select_singletons)
from moe_exp.routing_control.state_analysis_v1 import prefix_status, transition_summary

REPO = Path(__file__).resolve().parents[2]
HISTORY = REPO / "report/experimental-resume-v1"
OUT = REPO / "report/routing-study-v1"
ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
RUNS = ROOT / "steering-v1/runs/routing-control-v1"
DISCOVERY = RUNS / "dense-discovery"


def compute():
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID") and
            not socket.gethostname().startswith("login"), "study analysis requires allocated CPU Slurm")


def reference(path):
    value = sealed(path)
    return {"path": str(path), "sha256": value["sha256"], "file_sha256": file_sha(path)}


def accounting(jobs):
    command = ["sacct", "-X", "-nP", "-j", ",".join(sorted(set(jobs))),
               "--format=JobID%100,JobIDRaw,State%40,ExitCode,ElapsedRaw,AllocTRES%150"]
    result = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    rows = {}
    for line in result.splitlines():
        if not line.strip(): continue
        job, raw_job, state, exitcode, seconds, tres, *_ = line.split("|")
        counts = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
        rows[job] = {"raw_job_id": raw_job, "state": state, "exit_code": exitcode, "elapsed_seconds": int(seconds), "TRES": tres,
            "allocated_GPU_hours": int(seconds) * int(counts.get("gres/gpu", 0)) / 3600,
            "billing_core_hours": int(seconds) * float(counts.get("billing", 0)) / 3600}
    return {"command": command, "rows": rows}


def inventory():
    compute()
    source = HISTORY / "routing-study-closeout-v2/audit-59389450/STUDY_INDEX.json"
    old = sealed(source)
    studies, manifest_lanes, reproductions = {}, {}, []
    manifests = {"A": "OVERNIGHT_DISCOVERY_A_MANIFEST_v1.json", "B": "OVERNIGHT_DISCOVERY_B_MANIFEST_v1.json",
                 "C": "OVERNIGHT_DISCOVERY_C_MANIFEST_v2.json", "FRESH": "OVERNIGHT_FRESH_COMPARISON_MANIFEST_v2.json"}
    for name, item in old["semantic"].items():
        for key in ("artifact", "assigned_results"):
            ref = item[key]
            require(reference(Path(ref["path"])) == ref, "historical semantic artifact changed")
        manifest = sealed(HISTORY / manifests[name])
        manifest_lanes[manifest["sha256"]] = name
        records = sealed(item["assigned_results"]["path"])["records"]
        saved = sealed(item["artifact"]["path"])
        if manifest["schema"].endswith("v1"):
            import analyze_overnight_semantics_v1 as frozen_analysis
        else:
            import analyze_overnight_semantics_v2 as frozen_analysis
        reproduced = frozen_analysis.clustered_contrasts(manifest, records)
        require(reproduced == saved["primary"], "saved historical primary estimate failed reproduction: " + name)
        reproductions.append({"study": name, "primary": reproduced,
            "analysis_source_sha256": file_sha(frozen_analysis.__file__), "saved_analysis_sha256": saved["sha256"],
            "status": "EXACT_REPRODUCTION_SAVED_ESTIMATES_AND_50000_DRAW_INTERVALS"})
        studies[name] = {"semantic": item, "manifest": reference(HISTORY / manifests[name]),
                         "horizon": manifest["horizon"], "dense": [], "routing_engagement": [],
                         "utility": "UNAVAILABLE_CAPPED_MECHANISM_STUDY"}
    for group, key in (("dense_results", "dense"), ("routing_engagement", "routing_engagement")):
        for item in old[group]:
            # Keep entire saved references and version-specific interpretation.
            lane = manifest_lanes.get(item["manifest_sha256"])
            require(reference(Path(item["artifact"]["path"])) == item["artifact"], "historical secondary result changed")
            if lane in studies: studies[lane][key].append(item)
    save(OUT / "HISTORICAL_REPRODUCTION.json", {"schema": "routing-study-historical-reproduction-v1", "records": reproductions})
    import dense_recovery_v2 as recovery
    dense = recovery.inventory(recovery.SOURCE)
    save(OUT / "DENSE_INVENTORY.json", dense)
    # Historical accounting includes compact array range display IDs, which
    # cannot be passed back to sacct as exact jobs. Preserve that sealed ledger
    # and refresh only the live continuation/recovery identities.
    jobs = set()
    jobs.update(("59385734", "59385736", "59400762", "59400764", "59400766", "59405556", "59405561",
                 "59405635", "59418248", "59421662", "59428024", "59430300", "59430302"))
    jobs.update(row["job_id"] for row in dense["ambiguous_calls"])
    live_accounting = accounting(jobs)
    acc = {"historical_source": reference(source), "historical_rows": old["accounting"]["rows"],
           **live_accounting}
    # Exact array elements with ambiguous model calls must be terminal before
    # qualification or recovery. A cancelled parent alone is insufficient.
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
    ambiguous_terminal = all(any((job == attempt["job_id"] or row["raw_job_id"] == attempt["job_id"]) and
        row["state"].split(" by ")[0] in terminal for job, row in acc["rows"].items())
        for attempt in dense["ambiguous_calls"])
    native_replays = []
    for path in (RUNS / "native-finalization-v1").glob("correction-*/allocations/*/ENGINEERING_REPLAY.json"):
        value = sealed(path)
        native_replays.append({"reference": reference(path),
            "native_repeat": value["native_repeat"], "instrumented_repeat": value["instrumented_repeat"],
            "captures": value["captures"], "excluded_from_outcomes": True})
    prior_NLL = old["NLL_audit"]
    require(reference(Path(prior_NLL["path"])) == prior_NLL, "saved NLL audit changed")
    body = {"schema": "routing-study-reconciled-inventory-v1", "source_index": reference(source),
        "observed_utc": datetime.now(timezone.utc).isoformat(), "studies": studies,
        "dense_inventory": reference(OUT / "DENSE_INVENTORY.json"),
        "ambiguous_historical_calls_terminal": ambiguous_terminal,
        "native_finalization": {"assigned": 32, "outcomes_observed": 0,
            "status": "FAILED_NATIVE_TOKEN_REPEAT_QUALIFICATION", "engineering_replays": native_replays,
            "interpretation": "Separate native-routing prompt control. Engineering failure; accuracy unavailable."},
        "utility_pilot_jobs": ["59405556", "59405561", "59405635"],
        "operator_pilot_jobs": ["59400762", "59400764", "59400766"],
        "NLL_audit": prior_NLL, "X3": "DEFERRED", "accounting": acc,
        "historical_horizons_pooled": False,
        "code_files": {str(Path(__file__).resolve()): file_sha(__file__)}}
    result = save(OUT / "STUDY_INVENTORY_v2.json", body)
    require(ambiguous_terminal, "historical ambiguous calls need terminal accounting reconciliation")
    recovery.prepare(OUT / "dense-recovery-v2")
    print(json.dumps({"status": "INVENTORIED", "sha256": result["sha256"],
                      "dense_committed": dense["committed"], "dense_missing": dense["missing"]}), flush=True)


def discovery():
    compute()
    import numpy as np
    from prepare_transition_audit import label_rows
    units = sealed(DISCOVERY / "UNITS.json")
    labels, label_summary = label_rows(units)
    routes = DISCOVERY / "route-profiles-84a85c92-40c3d3b0"
    binding, summary = sealed(routes / "BINDING.json"), sealed(routes / "SUMMARY.json")
    family = sealed(HISTORY / "family-freeze.json")
    allowed = set(family["new_parent_pools"]["parent_pools"]["discovery"])
    require(binding["units_sha256"] == units["sha256"] and
            summary["binding_sha256"] == binding["sha256"], "discovery route binding differs")
    grouped = defaultdict(list)
    for unit, label in zip(units["records"], labels, strict=True):
        require(unit["family"] in allowed, "nondiscovery family in target selection")
        grouped[unit["family"]].append({**unit, "label": label["label"],
                                      "valid": label["finish_reason"] == "stop" and label["label"] in STATES})
    rows, globals_by_class, trajectories = [], defaultdict(list), []
    for fid, sentences in sorted(grouped.items()):
        receipt = sealed(routes / (fid + ".json"))
        path = routes / (fid + ".npz")
        require(receipt["binding_sha256"] == binding["sha256"] and
                file_sha(path) == receipt["npz_sha256"], "saved route array changed")
        with np.load(path, allow_pickle=False) as arrays:
            indices = arrays["sentence_index"].tolist()
            selection, weight = arrays["frequency"].astype(float) * 8, arrays["gate"].astype(float)
        require(selection.shape == weight.shape == (len(sentences), 40, 256) and
                sorted(indices) == sorted(r["sentence_index"] for r in sentences), "route/sentence alignment differs")
        require(np.allclose(selection.sum(axis=-1), 8, atol=1e-5) and
                np.allclose(weight.sum(axis=-1), 1, atol=1e-5), "native selected profile normalization differs")
        profiles = {idx: (selection[i], weight[i]) for i, idx in enumerate(indices)}
        ordered = sorted(sentences, key=lambda r: r["sentence_index"])
        trajectories.append({"family": fid, "summary": transition_summary(ordered)})
        for state in STATES:
            positives = [profiles[r["sentence_index"]][0] for r in ordered if r["valid"] and r["label"] == state]
            comparisons = [profiles[r["sentence_index"]][0] for r in ordered if r["valid"] and r["label"] != state]
            if positives and comparisons:
                globals_by_class[state].append(np.mean(positives, axis=0) - np.mean(comparisons, axis=0))
        for source, destination in zip(ordered, ordered[1:]):
            if not (source["valid"] and destination["valid"] and
                    source["sentence_index"] + 1 == destination["sentence_index"] and
                    source["segment"] == destination["segment"]):
                continue
            source_selection, source_weight = profiles[source["sentence_index"]]
            status = prefix_status(source["inputs"]["sentence"])
            for transition, positive in (("candidate_to_verify", destination["label"] == "Verify"),
                                         ("approach_to_commit", destination["label"] in ("Plan", "Implement"))):
                if transition == "candidate_to_verify" and source["label"] == "Verify": continue
                if transition == "approach_to_commit" and source["label"] != "Explore": continue
                rows.append({"family": fid, "source_state": source["label"], "prefix_status": status,
                    "transition": transition, "outcome": positive,
                    "selection": source_selection, "executed_weight": source_weight})
    selection = association_table(rows)
    weights = association_table(rows, signal="executed_weight")
    candidates = select_singletons(selection)
    table = save(OUT / "CANDIDATE_TABLE.json", {"schema": "routing-study-state-conditioned-candidates-v1",
        "source_units": reference(DISCOVERY / "UNITS.json"), "source_label_summary_sha256": label_summary["sha256"],
        "source_route_binding": reference(routes / "BINDING.json"), "discovery_families": sorted(grouped),
        "rows": selection, "executed_weight_rows": weights, "selected_new_singletons": candidates,
        "prefix_status_basis": "Current emitted source sentence only; lexical proposal, semantic audit unavailable.",
        "candidate_status": "UNAVAILABLE_PREFIX_RECOGNITION_QUALIFICATION" if not candidates else "SELECTED",
        "closure_opportunity": "UNAVAILABLE_CHECKED_COMPLETE_PREFIX_SUPPORT",
        "failed_check_revision": "UNAVAILABLE_PRESPECIFIED_SUPPORT",
        "prediction": "UNAVAILABLE_TRAILING_TEXT_POSITION_STATE_HISTORY_RUN_AGE_FRAME",
        "confidence": "UNAVAILABLE_FULL_NATIVE_GATE_PROBABILITIES_AND_SELECTION_BOUNDARY",
        "executed_weight_scope": "Saved normalized selected native gate weights; no full router distribution."})
    save(OUT / "DISCOVERY_TRANSITIONS.json", {"schema": "routing-study-discovery-transitions-v1",
        "candidate_table_sha256": table["sha256"], "records": trajectories,
        "terminal_events": "Unavailable at unobserved tails of historical selected windows; all tails censored."})
    arrays = {"global_" + state: np.mean(values, axis=0) for state, values in globals_by_class.items()}
    groups = defaultdict(list)
    for row in selection:
        groups[row["transition"], row["source_state"], row["prefix_status"]].append(row)
    for i, (key, entries) in enumerate(sorted(groups.items())):
        array = np.zeros((40, 256))
        for row in entries: array[row["layer"], row["expert"]] = row["risk_difference"]
        arrays["conditional_" + str(i)] = array
    path = OUT / "ASSOCIATION_PROFILES.npz"
    require(not path.exists(), "association profiles already frozen")
    np.savez_compressed(path, **arrays)
    # Profile similarity concerns different estimands; retain that interpretation.
    comparisons = []
    for i, (key, entries) in enumerate(sorted(groups.items())):
        destination = "Verify" if key[0] == "candidate_to_verify" else "Plan"
        if "global_" + destination not in arrays: continue
        a, b = arrays["global_" + destination].ravel(), arrays["conditional_" + str(i)].ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        from scipy.stats import spearmanr
        comparisons.append({"condition": list(key), "cosine_similarity": float(a @ b / denom) if denom else None,
            "rank_stability": float(spearmanr(a, b).statistic) if np.ptp(a) and np.ptp(b) else None,
            "nonzero_sign_changes": int(((a * b) < 0).sum()),
            "interpretation": "Global class occupancy versus source-conditioned transition association; observational estimands differ."})
    save(OUT / "GLOBAL_CONDITIONAL_COMPARISON.json", {"schema": "routing-study-profile-comparison-v1",
        "profiles_file_sha256": file_sha(path), "records": comparisons})
    plot_profiles(arrays, groups)
    manifest = save(OUT / "INTERVENTION_MANIFEST.json", {"schema": "routing-study-intervention-evaluation-manifest-v1",
        "candidate_table_sha256": table["sha256"], "targets": candidates, "assignments": [],
        "target_screen": {"maximum_targets": 2, "prefixes_per_transition": 24, "biases": [-1, 1],
                          "seeds": [0, 1], "pulse_tokens": 256, "continuation_cap": 1024},
        "operator_comparison": ["positive_bias", "forced_inclusion", "selected_weight_amplification"],
        "timing": ["first_qualified_boundary", "64_token_delay_with_no_fire_retained"],
        "composition": {"maximum_actions": 2, "requires_individual_support": True},
        "independent_evaluation": {"status": "UNAVAILABLE_FULL_MATH_TEST_SOURCE_AND_DUPLICATE_GRAPH",
            "validation_families": 32, "final_families": 96, "seeds": [0, 1], "cap": 16384,
            "maximum_final_assignments": 768, "arms": ["native", "selected", "matched_random", "incumbent"],
            "acceptable_accuracy_loss_margin": None},
        "stage_authorization": "New stages require qualified complete measured prices before resource authorization.",
        "status": "HELD_NO_QUALIFIED_NEW_SINGLETONS"})
    print(json.dumps({"status": table["candidate_status"], "conditional_expert_rows": len(selection),
                      "supported_associations": sum(r["status"] == "SUPPORTED_ASSOCIATION" for r in selection),
                      "manifest_sha256": manifest["sha256"]}), flush=True)


def plot_profiles(arrays, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    refs = [(28, 189), (28, 9), (24, 159)]
    ordered = [("global " + state, arrays["global_" + state]) for state in STATES if "global_" + state in arrays]
    ordered.extend((" / ".join(key), arrays["conditional_" + str(i)]) for i, key in enumerate(sorted(groups)))
    matrix = np.array([[profile[layer, expert] for layer, expert in refs] for _, profile in ordered])
    fig, ax = plt.subplots(figsize=(7, max(4, len(ordered) * .3)))
    limit = max(float(np.max(np.abs(matrix))), 1e-6)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(3), [f"L{l}/E{e}" for l, e in refs])
    ax.set_yticks(range(len(ordered)), [name for name, _ in ordered], fontsize=7)
    ax.set_title("Native selection associations: reference identities")
    fig.colorbar(im, ax=ax, label="Within-family risk difference")
    fig.tight_layout()
    fig.savefig(OUT / "global-conditional-heatmap.png", dpi=180)
    fig.savefig(OUT / "global-conditional-heatmap.pdf")
    plt.close(fig)


def report():
    compute()
    inventory_value = sealed(OUT / "STUDY_INVENTORY_v2.json")
    lines = ["# Routing study implementation and evidence\n",
        "This snapshot preserves each historical population, horizon, endpoint and inference. "
        "Capped mechanism effects do not establish original-prompt accuracy.\n",
        "| Study | Assigned continuations | Valid semantic pairs | Saved status |",
        "|---|---:|---:|---|"]
    for name, row in inventory_value["studies"].items():
        semantic = row["semantic"]
        lines.append(f'| {name} | {semantic["assigned"]} | {semantic["valid_pairs"]} | {semantic["status"]} |')
    dense = sealed(OUT / "DENSE_INVENTORY.json")
    lines.extend(["", f'Dense labels: **{dense["committed"]:,} committed / {dense["assigned"]:,} assigned**; '
        f'{dense["missing"]:,} missing; {len(dense["ambiguous_calls"])} preserved calls without committed output. '
        "Every missing blind identity is sealed in DENSE_INVENTORY.json.", "",
        "The native-finalization diagnostic retains all 32 outcomes as unknown. The v3 audit observed "
        "cross-execution route variability with identical tokens and faithful execution-local capture. "
        "The subsequent v4 native token repeat failed; qualification remains failed. No accuracy effect is available.", "",
        "Counterfactual-NLL qualification remains failed and X3 remains deferred. Neither qualifies target ranking.", "",
        "Utility and operator pilot assignments remain frozen and queued. New original-prompt utility is unavailable."])
    if (OUT / "CANDIDATE_TABLE.json").exists():
        candidates = sealed(OUT / "CANDIDATE_TABLE.json")
        supported = sum(r["status"] == "SUPPORTED_ASSOCIATION" for r in candidates["rows"])
        lines.extend(["", f'{supported} state-conditioned expert associations meet the ten-family/three-fold direction rule. '
            "The lexical prefix-status proposals have no qualified semantic recognition audit, so no new singleton is advanced.",
            "", "![Global and conditional reference associations](global-conditional-heatmap.png)", "",
            "Checked-candidate closure and failed-check revision remain unavailable for insufficient qualified prefix support. "
            "Temporal incremental prediction remains unavailable until its prefix baseline and trailing routing frame are qualified. "
            "Full gate confidence and selection-boundary gaps are unavailable; selected weights cannot supply them."])
    price_path = OUT / "dense-recovery-v2/PRICE.json"
    if price_path.exists():
        price = sealed(price_path)
        lines.extend(["", f'Dense recovery price: {len(price["groups"])} packed jobs; '
            f'{price["maximum_GPU_hours"]:.2f} GPU-hours including one bounded recovery load per packed job; '
            f'{price["qualification_GPU_hours"]:.2f} GPU-hour qualification allocation. '
            "This is a conservative cap-based price, separate from observed allocation costs. Full recovery waits for packaging qualification."])
    lines.extend(["", "The requested independent frontier needs 128 newly reserved families from the full MATH test source, "
        "its pinned revision, subject/level strata and a duplicate graph connected to all previous questions. "
        "The local MATH-500 and 272-question Minerva caches do not provide that source. "
        "No accuracy-loss margin is imposed. Paired family-bootstrap simultaneous intervals and missing-grade sensitivity "
        "are implemented in study_v1.py; final accuracy and token-cost effects remain unobserved.", "",
        "The three versioned inputs are STUDY_INVENTORY_v2.json, CANDIDATE_TABLE.json and INTERVENTION_MANIFEST.json. "
        "Existing source artifacts remain immutable. Newly needed stages must be priced from measured generation, "
        "controller, label and grading throughput before their resource authorization.", ""])
    path = OUT / "REPORT.md"
    require(not path.exists(), "report already frozen; create a new snapshot")
    path.write_text("\n".join(lines))
    save(OUT / "REPORT_BINDING.json", {"schema": "routing-study-report-binding-v1",
        "inventory_sha256": inventory_value["sha256"], "report_file_sha256": file_sha(path),
        "status": "INCOMPLETE_EXECUTION_AND_QUALIFICATION", "job_id": os.environ["SLURM_JOB_ID"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("inventory", "discovery", "report"))
    args = p.parse_args()
    globals()[args.stage]()


if __name__ == "__main__":
    main()
