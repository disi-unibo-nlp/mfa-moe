"""Build steered_<study>.parquet from the generated-dense steering runs (Qwen3.6 continuations).

Read-only on all sources. Writes only under OUT_DIR.
Usage: python -m routing.lexical_data.build_steered A|B|C|FRESH OUT_DIR
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from store import ANALYSIS

RUNS = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1")
MAN = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1")
RECEIPTS = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/research-campaign-v3-20261014-r6")  # folds
CAPTURE = ANALYSIS / "campaign-v3"   # saved cohorts of the seven-model capture
STUDIES = {
    "A": ("generated-dense/dense-generated-v1-9b714fe1091f-8e0bff870bed", "OVERNIGHT_DISCOVERY_A_MANIFEST_v1.json"),
    "B": ("generated-dense/dense-generated-v1-9c3dc95fda9c-8e0bff870bed", "OVERNIGHT_DISCOVERY_B_MANIFEST_v1.json"),
    "C": ("generated-dense/dense-generated-v1-65e402b7e4fe-8e0bff870bed", "OVERNIGHT_DISCOVERY_C_MANIFEST_v2.json"),
    "FRESH": ("generated-dense-fresh-metadata-recovery-v1/dense-generated-v1-6b5aa286e46c-a77287237f3a",
              "OVERNIGHT_FRESH_COMPARISON_MANIFEST_v2.json"),
}
TARGET = {"candidate_to_verify": "Verify", "approach_to_commit": "Plan|Implement"}
INTENDED = {"candidate_to_verify": "substantive verification of an unchecked candidate",
            "approach_to_commit": "commitment to an identified approach"}
PULSE_W = 256


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(study: str, out: Path) -> None:
    t0 = time.time()
    rundir, manifest_name = STUDIES[study]
    d = RUNS / rundir
    man = json.load(open(MAN / manifest_name))
    arm_map = json.load(open(d / "ARM_MAP.json"))
    frame = json.load(open(d / "BLIND_FRAME.json"))
    assert arm_map["generation_manifest_sha256"] == man["sha256"], "manifest/ARM_MAP mismatch"
    inputs = {r["blind_id"]: r["inputs"] for r in frame["records"]}
    labels = {}
    nbatch = 0
    for f in sorted(glob.glob(str(d / "labels-shard-*/batches/*.json"))):
        nbatch += 1
        for r in json.load(open(f))["records"]:
            assert r["blind_id"] not in labels
            labels[r["blind_id"]] = r
    print(f"[{study}] loaded arm_map {len(arm_map['records'])} frame {len(inputs)} labels {len(labels)} "
          f"batches {nbatch} ({time.time() - t0:.0f}s)", flush=True)

    folds = json.load(open(RECEIPTS / "FOLDS.json"))
    qfam, fold_of = folds["question_families"], folds["folds"]
    coh = json.load(open(CAPTURE / "cohorts/qwen36/cohorts.json"))["rows"]
    native_digests = {r["trace_sha256"] for r in coh}
    native_q = {f"{r['dataset']}|{r['source_problem_id']}" for r in coh}
    native_fam = {qfam[q] for q in native_q}

    actions = {a["name"]: a for a in man["actions"]}
    mrows = {r["uid"]: r for r in man["rows"]}
    horizon = arm_map["horizon"]
    run_id = rundir.split("/")[-1]

    rows, problems, conts = [], {}, []
    for rec in arm_map["records"]:
        policy, role, arm = rec["policy"], rec["role"], rec["arm"]
        act = actions.get(policy)
        if act is None:
            assert role == "native" and policy == "zero"
            kind, magnitude, layers, experts = "none", None, [], []
        else:
            kind = act.get("kind", "bias")
            magnitude = act.get("magnitude", act.get("bias"))
            layers = sorted({e[0] for e in act["experts"]})
            experts = act["experts"]
        dose = rec["action_dose"] or {}
        segs, dose_main = None, {}
        if dose:
            first = dose[sorted(dose)[0]]  # all TP ranks carry identical segments/dose (checked in README)
            segs = first["segments"].get(policy)
            if first["dose"].get(policy) and layers:
                dose_main = first["dose"][policy].get(str(layers[0]), {})
        virtual = segs is None
        if virtual:
            segs = [[0, PULSE_W]]
        n_pulses = len(rec["slots"])
        question = rec["question"]
        dataset, source_problem_id = question.split("|", 1)
        fam = qfam[question]
        assert fam == rec["family"], (question, fam, rec["family"])
        mrow = mrows[rec["prefix_uid"]]
        prefix_digest = mrow.get("trace_sha256")
        n_emitted = len(rec["emitted_token_ids"])
        sents = rec["sentences"]
        conts.append(dict(
            study=study, run_id=run_id, uid=rec["uid"], prefix_uid=rec["prefix_uid"], question=question,
            family_id=fam, transition=rec["transition"], arm=arm, role=role, policy=policy,
            policies=json.dumps(rec["policies"]), action_kind=kind, magnitude=magnitude,
            experts_json=json.dumps(experts), layers=json.dumps(layers), slots=json.dumps(rec["slots"]),
            pulse_segments=json.dumps(segs), virtual_pulse=virtual, random_set=rec["random_set"],
            seed=rec["seed"], execution_position=rec["execution_position"], horizon=horizon,
            hit_cap=rec["hit_cap"], finish_reason=rec["finish_reason"], closed_reasoning=rec["closed_reasoning"],
            reasoning_end=rec["reasoning_end"], n_emitted_tokens=n_emitted, n_sentences=len(sents),
            n_sentences_labelled=sum(1 for x in sents if labels.get(x["blind_id"], {}).get("label")),
            missing_telemetry_ranks=json.dumps(rec.get("missing_telemetry_ranks")),
            inactive_native_checks=json.dumps(rec.get("inactive_native_checks")),
            action_dose_json=json.dumps(dose, sort_keys=True) if dose else None))
        for si, s in enumerate(sents):
            bid = s["blind_id"]
            inp = inputs[bid]
            assert inp["sentence"] == s["text"]
            lab = labels.get(bid)
            ts, te = s["token_start"], s["token_end"]
            psha = sha(inp["problem_statement"])
            problems.setdefault(psha, (dataset, source_problem_id, inp["problem_statement"]))
            if ts is None:
                phase, in_pulse, since_first, since_last = None, None, None, None
            else:
                ends = [e for _, e in segs]
                if ts < segs[0][0]:
                    phase = "before_pulse"
                else:
                    phase = "post_pulse"
                    for k, (a, e) in enumerate(segs):
                        if a <= ts < e:
                            phase = f"in_pulse{k + 1}"
                            break
                        if k + 1 < len(segs) and e <= ts < segs[k + 1][0]:
                            phase = "between_pulses"
                            break
                in_pulse = any(ts < e and te > a for a, e in segs)
                since_first = ts - ends[0]
                since_last = ts - ends[-1]
            rows.append(dict(
                study=study, run_id=run_id, uid=rec["uid"], prefix_uid=rec["prefix_uid"], blind_id=bid,
                question=question, dataset=dataset, source_problem_id=source_problem_id, family_id=fam,
                fold_campaign=fold_of.get(fam),
                question_in_native_qwen36=question in native_q, family_in_native_qwen36=fam in native_fam,
                prefix_trace_sha256=prefix_digest,
                prefix_trace_in_native_qwen36=(prefix_digest in native_digests) if prefix_digest else None,
                prefix_attempt_id=mrow.get("attempt_id"), prefix_trigger_sentence_index=mrow.get("sentence_index"),
                prefix_tokens=mrow.get("prefix_tokens"),
                transition=rec["transition"], target_class=TARGET[rec["transition"]],
                intended_event=INTENDED[rec["transition"]],
                arm=arm, role=role, arm_group={"native": "native", "target": "targeted", "random": "random"}[role],
                policy=policy, action_kind=kind, magnitude=magnitude, layers=json.dumps(layers),
                layer=layers[0] if len(layers) == 1 else None, experts_json=json.dumps(experts),
                n_pulses=n_pulses, slots=json.dumps(rec["slots"]), random_set=rec["random_set"],
                seed=rec["seed"], execution_position=rec["execution_position"], horizon=horizon,
                hit_cap=rec["hit_cap"], finish_reason=rec["finish_reason"], closed_reasoning=rec["closed_reasoning"],
                reasoning_end=rec["reasoning_end"], n_emitted_tokens=n_emitted, prompt_len=rec["prompt_len"],
                dose_active_rows=dose_main.get("active_rows"), dose_actual_target_hits=dose_main.get("actual_target_hits"),
                dose_native_target_hits=dose_main.get("native_target_hits"),
                dose_membership_changes=dose_main.get("membership_changes"),
                dose_target_mass=dose_main.get("target_mass"), dose_weight_l1=dose_main.get("weight_l1"),
                pulse_segments=json.dumps(segs), virtual_pulse=virtual,
                phase=phase, in_pulse=in_pulse, tokens_since_first_pulse_end=since_first,
                tokens_since_last_pulse_end=since_last,
                after_first_pulse_end=(ts >= segs[0][1]) if ts is not None else None,
                in_window_0_256=(ts < 256 and te > 0) if ts is not None else None,
                in_window_512_768=(ts < 768 and te > 512) if ts is not None else None,
                sentence_index=s["sentence_index"], n_sentences_in_window=len(sents),
                is_first_sentence=si == 0, is_last_sentence=si == len(sents) - 1,
                char_start=s["char_start"], char_end=s["char_end"], n_chars=s["char_end"] - s["char_start"],
                token_start=ts, token_end=te, n_tokens=(te - ts) if ts is not None else None,
                rel_pos_window=(ts / n_emitted) if (ts is not None and n_emitted) else None,
                complete=s["complete"],
                sentence_text=s["text"], prev_sentence=inp["previous_sentence"], next_sentence=inp["next_sentence"],
                next_is_end_of_response=inp["next_sentence"] == "<END OF RESPONSE>",
                next_is_end_of_window=inp["next_sentence"] == "<END OF OBSERVED WINDOW>",
                prev_is_native_trigger=si == 0, problem_sha=psha,
                label=None if lab is None else lab["label"],
                label_status="not_labelled" if lab is None else ("ok" if lab["label"] else "unknown"),
                label_finish_reason=None if lab is None else lab["finish_reason"],
                label_generated_tokens=None if lab is None else lab["generated_tokens"],
                is_labelled=lab is not None and lab["label"] is not None,
                judge_protocol="dense-v1 (direct vLLM chat, T=0, effort low, max 1024)",
            ))
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"steered_{study}.parquet", compression="zstd", index=False)
    pd.DataFrame(conts).to_parquet(out / f"_cont_{study}.parquet", compression="zstd", index=False)
    pd.DataFrame([dict(problem_sha=k, dataset=v[0], source_problem_id=v[1], problem_statement=v[2])
                  for k, v in problems.items()]).to_parquet(out / f"_problems_steered_{study}.parquet",
                                                           compression="zstd", index=False)
    print(f"[{study}] DONE rows={len(df)} labelled={int(df.is_labelled.sum())} "
          f"unknown={(df.label_status == 'unknown').sum()} conts={df.uid.nunique()} ({time.time() - t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
