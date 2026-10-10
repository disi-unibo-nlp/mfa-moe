"""Regenerate paper evidence from bound receipts; never promote jobs to results."""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import file_sha, load, private_path, require, save
from .registry import MODEL_KEYS, REPO

MANUSCRIPT = REPO / "paper/6a510c6db0d9b08aa438c142"
NAMES = {"gpt":"GPT-OSS", "nemotron":"Nemotron", "qwen35":"Qwen3.5", "qwen36":"Qwen3.6",
         "gemma":"Gemma", "glm":"GLM", "qwen330b":"Qwen3-30B"}
ROW_END=chr(92)*2


def export(campaign_path, root):
    campaign=load(campaign_path)
    snapshot_path=REPO/"paper/wiki/evidence_snapshot.json"
    evidence=json.loads(snapshot_path.read_text())
    generated=private_path(MANUSCRIPT/"generated");generated.mkdir(parents=True,exist_ok=True)
    rows, statuses=[],[]
    source_bindings={str(snapshot_path):file_sha(snapshot_path)}
    status=[]
    analysis_path=root/"ANALYSIS.json"
    measured=load(analysis_path,"verification-analysis-v3") if analysis_path.exists() else None
    if measured:
        require(measured["campaign_sha256"]==campaign["sha256"],"foreign paper analysis")
        source_bindings[str(analysis_path)]=file_sha(analysis_path)
    for key in MODEL_KEYS:
        item=evidence["all_model_readout"][key]["judge_context"]
        effect=item["routing_P"];lower,upper=item["routing_simultaneous_CI7"]
        rows.append(f"{NAMES[key]} & {effect:.4f} & [{lower:.4f}, {upper:.4f}] "+ROW_END)
        path=root/key/"qualification/QUALIFICATION.json"
        result=load(path) if path.exists() else None
        state=result["status"] if result else "PENDING"
        if result:
            require(result["campaign_sha256"]==campaign["sha256"],"foreign qualification in paper export")
            source_bindings[str(path)]=file_sha(path)
        policy_path=root/key/"calibration/POLICY.json"
        policy=load(policy_path) if policy_path.exists() else None
        recognition_path=root/key/"RECOGNIZER.json"
        recognition=load(recognition_path) if recognition_path.exists() else None
        statuses.append(f"{NAMES[key]} & {state} & {recognition['status'] if recognition else 'Unmeasured'} & {policy['status'] if policy else 'Unmeasured'} "+ROW_END)
        status.append({"model":key,"backend":state,"recognizer":recognition["status"] if recognition else "NOT_MEASURED",
                       "dose_and_random":policy["status"] if policy else "NOT_MEASURED","causal_effect":"NOT_MEASURED","utility":"NOT_MEASURED"})
        if measured:
            for field,endpoints in (("causal_effect",("semantic",)),("utility",("accuracy","tokens"))):
                cells=[r for r in measured["contrasts"] if r["model"]==key and r["endpoint"] in endpoints]
                status[-1][field]="ESTIMATED" if all(r["status"]=="ESTIMATED" for r in cells) else "BOUNDED" if any(r["identified_mean_bounds"] is not None for r in cells) else "NOT_MEASURED"
    table=("% Generated from bound saved R3 results. Lower cross-entropy is better.\n"
        "\\begin{table}[t]\n\\centering\n\\begin{tabular}{lrr}\n\\toprule\n"
        "Model & Routing loss change & Simultaneous 95\\% interval "+ROW_END+"\n\\midrule\n"+
        "\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{CLEAN routing readout over the retrospective judge-context baseline. "
        "All seven intervals exclude zero under the saved seven-comparison correction. "
        "These are label-readout results, not causal intervention effects.}\n\\label{tab:seven-readout-current}\n\\end{table}\n")
    (generated/"seven_model_evidence.tex").write_text(table)
    qualification=("% Generated engineering/measurement status.\n\\begin{table}[t]\n\\centering\n"
        "\\begin{tabular}{llll}\n\\toprule\nModel & Backend & Recognizer & Dose/control "+ROW_END+"\n\\midrule\n"+
        "\n".join(statuses)+"\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Prospective verification study status. A passing backend qualifies execution only. "
        "Unmeasured or failed qualification cells do not estimate a causal null.}\n\\label{tab:verification-status}\n\\end{table}\n")
    (generated/"verification_status.tex").write_text(qualification)
    analysis_path=root/"ANALYSIS.json"
    if analysis_path.exists():
        analysis=load(analysis_path,"verification-analysis-v3")
        require(analysis["campaign_sha256"]==campaign["sha256"],"foreign paper analysis")
        source_bindings[str(analysis_path)]=file_sha(analysis_path)
        lines=[]
        for contrast in analysis["contrasts"]:
            bound=contrast["identified_mean_bounds"];interval=contrast["simultaneous_interval95"]
            if bound is None:continue
            label=contrast["endpoint"]
            if label!="tokens":bound=[v*100 for v in bound];interval=[v*100 for v in interval] if interval else None
            estimate=f"[{bound[0]:.2f}, {bound[1]:.2f}]"
            uncertainty=f"[{interval[0]:.2f}, {interval[1]:.2f}]" if interval else "Unavailable"
            lines.append(f"{NAMES[contrast['model']]} & {label} & {'/'.join(contrast['contrast']).replace('_',' ')} & {estimate} & {uncertainty} "+ROW_END)
        results=("\\begin{longtable}{lllll}\n\\toprule\nModel & Endpoint & Contrast & Mean bounds & Simultaneous 95\\% "+ROW_END+
                 "\n\\midrule\n"+"\n".join(lines)+"\n\\bottomrule\n\\end{longtable}\n")
    else:
        results="New conditional verification and original-prompt utility outcomes remain unmeasured. The status table reports qualification separately.\n"
    (generated/"verification_results.tex").write_text(results)
    provenance=save(generated/"PROVENANCE.json",{"schema":"verification-paper-provenance-v3",
        "campaign_sha256":campaign["sha256"],"source_files":source_bindings,"model_status":status,
        "numeric_source":"original saved R3 metrics; new engineering cells only from observed exact receipts"},replace=True)
    page=["# Experiment status","","Generated from the active campaign and observed receipts. Jobs and backend qualification do not establish semantic effects.","",
          "| Model | Backend | Recognizer | Dose/control | Causal effect | Utility |","|---|---|---|---|---|---|"]
    for row in status:
        page.append("| "+" | ".join(str(row[k]) for k in ("model","backend","recognizer","dose_and_random","causal_effect","utility"))+" |")
    page += ["",f"Campaign: `{campaign_path}`. Provenance: `{generated/'PROVENANCE.json'}`.","",
             "The main direction and stage contracts remain in [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md)."]
    (REPO/"paper/wiki/EXPERIMENT_STATUS.md").write_text("\n".join(page)+"\n")
    return provenance
