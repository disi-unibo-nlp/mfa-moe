"""Transition-aligned observational profiles from saved native discovery only."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from moe_exp.routing_control.study_v1 import STATES, file_sha, require, save, sealed
from routing_study_v1 import compute, DISCOVERY, OUT, REPO, reference


def main():
    compute()
    from prepare_transition_audit import label_rows
    units = sealed(DISCOVERY / "UNITS.json")
    labels, summary = label_rows(units)
    candidates = sealed(OUT / "CANDIDATE_TABLE.json")
    require(candidates["source_units"]["sha256"] == units["sha256"] and
            candidates["source_label_summary_sha256"] == summary["sha256"], "saved discovery frame changed")
    routes = Path(candidates["source_route_binding"]["path"]).parent
    binding = sealed(routes / "BINDING.json")
    grouped = defaultdict(list)
    for row, label in zip(units["records"], labels, strict=True):
        grouped[row["family"]].append({**row, "label": label["label"],
            "valid": label["finish_reason"] == "stop" and label["label"] in STATES})
    records, source_refs = [], []
    for family, rows in sorted(grouped.items()):
        receipt = sealed(routes / (family + ".json"))
        path = routes / (family + ".npz")
        require(receipt["binding_sha256"] == binding["sha256"] and file_sha(path) == receipt["npz_sha256"],
                "saved native profile changed")
        source_refs.append(reference(routes / (family + ".json")))
        with np.load(path, allow_pickle=False) as archive:
            selection, weights, indices = archive["frequency"] * 8, archive["gate"], archive["sentence_index"]
        require(selection.shape == weights.shape == (len(rows), 40, 256), "sentence profiles do not align")
        profiles = {int(index): (selection[i], weights[i]) for i, index in enumerate(indices)}
        indexed = {r["sentence_index"]: r for r in rows}
        for source in sorted(rows, key=lambda r: r["sentence_index"]):
            index = source["sentence_index"]
            destination = indexed.get(index + 1)
            if not destination or not source["valid"] or not destination["valid"] or source["segment"] != destination["segment"]:
                continue
            transition = ("candidate_to_verify_proxy" if destination["label"] == "Verify" and source["label"] != "Verify" else
                "approach_to_commit_proxy" if source["label"] == "Explore" and destination["label"] in ("Plan", "Implement") else None)
            if transition is None:
                continue
            identities = [(28, 189), (28, 9)] if transition == "candidate_to_verify_proxy" else [(24, 159)]
            for offset in range(-3, 4):
                boundary_index = index + offset
                chain = [indexed.get(i) for i in range(min(index, boundary_index), max(index + 1, boundary_index) + 1)]
                if not all(r and r["valid"] and r["segment"] == source["segment"] for r in chain):
                    continue
                chosen, mass = profiles[boundary_index]
                for layer, expert in identities:
                    records.append({"family": family, "transition": transition, "source_index": index,
                        "source_state": source["label"], "destination_state": destination["label"],
                        "offset_sentences": offset, "layer": layer, "expert": expert,
                        "selection_rate": float(chosen[layer, expert]), "normalized_selected_weight": float(mass[layer, expert])})
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        for signal in ("selection_rate", "normalized_selected_weight"):
            key = row["transition"], row["offset_sentences"], row["layer"], row["expert"], signal
            groups[key][row["family"]].append(row[signal])
    profiles = [{"transition": transition, "offset_sentences": offset, "layer": layer, "expert": expert,
        "signal": signal, "mean": float(np.mean([np.mean(v) for v in values.values()])), "families": len(values),
        "events": sum(map(len, values.values())), "status": "SUPPORTED_DESCRIPTIVE_PROFILE" if len(values) >= 10 else
        "UNAVAILABLE_TEN_FAMILY_SUPPORT", "uncertainty": "No new significance claims"}
        for (transition, offset, layer, expert, signal), values in sorted(groups.items())]
    transitions = sealed(OUT / "DISCOVERY_TRANSITIONS.json")
    counts = np.zeros((7, 7), dtype=int)
    exits, reentries, loops = Counter(), Counter(), Counter()
    dwell = []
    for row in transitions["records"]:
        value = row["summary"]
        counts += np.array(value["transition_counts"])
        exits.update({(r["source"], r["destination"]): r["count"] for r in value["destination_specific_exit_counts"]})
        reentries.update(value["reentry_counts"])
        loops.update(value["loops"])
        dwell.extend({"family": row["family"], **r} for r in value["dwell"])
    total = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(counts, total, out=np.zeros(counts.shape, dtype=float), where=total > 0)
    result = save(OUT / "NATIVE_BOUNDARY_PROFILES.json", {"schema": "routing-study-native-boundary-profiles-v1",
        "source_units_sha256": units["sha256"], "source_routes": source_refs,
        "source_transition_sha256": transitions["sha256"], "records": records, "profiles": profiles,
        "transition_counts": counts.tolist(), "transition_probabilities": probabilities.tolist(),
        "destination_exit_counts": [{"source": a, "destination": b, "count": n} for (a, b), n in sorted(exits.items())],
        "dwell": dwell, "reentries": dict(reentries), "loops": dict(loops),
        "closure": "UNAVAILABLE_SELECTED_NATIVE_WINDOWS_END_BEFORE_TERMINATION",
        "alignment": "Offset 0 is the source sentence; +1 is the destination. Every intervening label must be observed and valid.",
        "token_prediction": "UNAVAILABLE_PENDING_RAW_PREDICTION_ROW_AND_PREFIX_BASELINE_QUALIFICATION",
        "interpretation": "Sentence-aligned native observational class proxies. Precursor and consequence are plotted separately; no online timing or substantive semantic claim.",
        "code_files": {str(Path(__file__)): file_sha(__file__)}})
    plot(result)
    base = OUT / "REPORT_v2.md"
    report = OUT / "REPORT_v3.md"
    require(not report.exists(), "report version already exists")
    report.write_text(base.read_text() + "\n![Transition-aligned native routing](native-transition-aligned.png)\n\n"
        "In this plot, source sentence 0 precedes destination sentence +1. Every included offset has a contiguous valid label "
        "chain; events and seeds are averaged within family before families receive equal weight. This separates native "
        "pre-transition profiles from routing in the destination. These sentence-scale class proxies do not qualify a "
        "64-token timing trigger or establish a substantive verification. Token-scale anticipation remains unavailable until "
        "raw prediction-row alignment and a prefix-only baseline are qualified.\n\n"
        "![Native state-transition diagram](native-state-transitions.png)\n\n"
        "The diagram shows up to two most frequent observed exits per state. Edge labels are destination probabilities "
        "conditional on an observed state exit. Self transitions, all exit counts, dwell, re-entry and loops remain in "
        "NATIVE_BOUNDARY_PROFILES.json. Missing labels break adjacency. Selected-window ends censor termination, so no "
        "closure probability is inferred from them.\n")
    save(OUT / "REPORT_v3_BINDING.json", {"schema": "routing-study-native-boundary-report-v1",
        "source_report_binding": reference(OUT / "REPORT_v2_BINDING.json"), "boundary_profiles_sha256": result["sha256"],
        "report_file_sha256": file_sha(report), "figures": {name: file_sha(OUT / name) for name in
            ("native-transition-aligned.png", "native-transition-aligned.pdf", "native-state-transitions.png", "native-state-transitions.pdf")},
        "status": "INCOMPLETE_EXECUTION_AND_QUALIFICATION"})
    print(json.dumps({"status": "COMPLETE_NATIVE_BOUNDARY_DESCRIPTION", "profiles": len(profiles), "events": len(records)}), flush=True)


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, transition in zip(axes, ("candidate_to_verify_proxy", "approach_to_commit_proxy"), strict=True):
        for layer, expert in sorted({(r["layer"], r["expert"]) for r in result["profiles"] if r["transition"] == transition}):
            rows = [r for r in result["profiles"] if r["transition"] == transition and r["layer"] == layer and
                    r["expert"] == expert and r["signal"] == "selection_rate" and r["families"] >= 10]
            axis.plot([r["offset_sentences"] for r in rows], [r["mean"] for r in rows], marker="o", label=f"L{layer}/E{expert}")
        axis.axvline(.5, color="grey", linestyle="--")
        axis.set(title=transition.replace("_proxy", ""), xlabel="Sentence offset from source", ylabel="Native selection rate")
        axis.legend()
    fig.suptitle("Contiguous native transition profiles, equal family weight")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("native-transition-aligned." + suffix), dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/2
    coords = {state: np.array([np.cos(angle), np.sin(angle)]) for state, angle in zip(STATES, angles, strict=True)}
    for state, point in coords.items():
        axis.text(*point, state, ha="center", va="center", bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "black"}, zorder=5)
    for state in STATES:
        exits = [r for r in result["destination_exit_counts"] if r["source"] == state and r["destination"] in STATES]
        denominator = sum(r["count"] for r in exits)
        for row in sorted(exits, key=lambda r: (-r["count"], r["destination"]))[:2]:
            a, b = coords[state], coords[row["destination"]]
            probability = row["count"] / denominator
            axis.annotate("", xy=.9*b, xytext=.9*a, arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=.10",
                "color": "#446588", "lw": 1 + 2*probability})
            center = .5*(a+b) + np.array([-(b-a)[1], (b-a)[0]])*.06
            axis.text(*center, f"{probability:.2f}", fontsize=8, ha="center", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
    axis.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal", title="Native observed exits; up to two destinations per state")
    axis.axis("off")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / ("native-state-transitions." + suffix), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
