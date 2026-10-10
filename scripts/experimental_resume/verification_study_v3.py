"""All-seven verification campaign. Real file/main guard required by vLLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from moe_exp.verification_study.contracts import load, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","submit-qualification","evidence","qualify","run","audit-frame",
        "rate","recognition","nominate","calibrate","validate","freeze-final","outcome-frame","analyze",
        "grade-strict","grade-j1","j1-audit","assemble","terminal-source","prices","submit-task"))
    parser.add_argument("--campaign",type=Path,default=Path("report/verification-study-v3/CAMPAIGN.json"))
    parser.add_argument("--model")
    parser.add_argument("--root",type=Path,default=Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/candidate-verification-v3-20261007-r4"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--frame",type=Path)
    parser.add_argument("--ratings",type=Path)
    parser.add_argument("--qualification",type=Path)
    parser.add_argument("--policy",type=Path)
    parser.add_argument("--recognizer",type=Path)
    parser.add_argument("--nomination",type=Path)
    parser.add_argument("--request",type=Path)
    parser.add_argument("--validation",type=Path)
    parser.add_argument("--prices",type=Path)
    parser.add_argument("--deadline-epoch",type=float)
    parser.add_argument("--begin",type=int,default=0)
    parser.add_argument("--count",type=int)
    parser.add_argument("--role",choices=("eligibility","outcome"),default="eligibility")
    parser.add_argument("--test-only",action="store_true")
    args=parser.parse_args()
    root=args.root;campaign_path=args.campaign.resolve()
    if args.command=="prepare":
        from moe_exp.verification_study.prepare import prepare
        result=prepare(campaign_path.parent)
    elif args.command=="submit-qualification":
        from moe_exp.verification_study.campaign import qualification_card,evidence_card,submit
        from moe_exp.verification_study.registry import MODEL_KEYS
        jobs=[]
        for key in MODEL_KEYS:
            result=submit(campaign_path,qualification_card(campaign_path,key),root,test_only=args.test_only)
            jobs.append(result)
        result={"models":jobs,"closeout":submit(campaign_path,evidence_card(campaign_path),root,
            test_only=args.test_only,dependencies=[r["job_id"] for r in jobs] if not args.test_only else ())}
    elif args.command=="qualify":
        from moe_exp.verification_study.qualification import qualify
        result=qualify(campaign_path,args.model,args.output or root/args.model/"qualification",args.deadline_epoch)
    elif args.command=="submit-task":
        from moe_exp.verification_study.campaign import task_card,submit
        result=submit(campaign_path,task_card(campaign_path,load(args.request)),root,test_only=args.test_only)
    elif args.command=="evidence":
        from moe_exp.verification_study.paper import export
        result=export(campaign_path,root)
    elif args.command=="run":
        from moe_exp.verification_study.execution import run_shard
        result=run_shard(campaign_path,args.model,args.qualification,args.manifest,args.output,args.deadline_epoch,
            policy_path=args.policy,recognition_path=args.recognizer,begin=args.begin,count=args.count,
            validation_path=args.validation,price_path=args.prices)
    elif args.command=="audit-frame":
        from moe_exp.verification_study.execution import make_audit
        result=make_audit(campaign_path,args.model,args.source,args.output)
    elif args.command=="rate":
        from moe_exp.verification_study.execution import rate_frame
        result=rate_frame(campaign_path,args.frame,args.output,args.deadline_epoch,args.role)
    elif args.command=="recognition":
        from moe_exp.verification_study.stages import recognition
        result=recognition(campaign_path,args.model,args.frame,args.ratings,args.output)
    elif args.command=="nominate":
        from moe_exp.verification_study.execution import fit_nomination
        result=fit_nomination(campaign_path,args.model,args.frame,args.ratings,args.source,args.output)
    elif args.command=="calibrate":
        from moe_exp.verification_study.stages import calibrate
        result=calibrate(campaign_path,args.model,args.qualification,args.nomination,args.frame,args.source,args.output,args.deadline_epoch)
    elif args.command=="validate":
        from moe_exp.verification_study.stages import validation
        result=validation(campaign_path,args.model,args.policy,args.recognizer,args.source,args.output)
    elif args.command=="freeze-final":
        from moe_exp.verification_study.stages import freeze_final
        result=freeze_final(campaign_path,args.model,args.source,args.policy,args.recognizer,args.output)
    elif args.command=="outcome-frame":
        from moe_exp.verification_study.stages import outcome_frame
        result=outcome_frame(campaign_path,args.model,args.manifest,args.source,args.output)
    elif args.command=="grade-strict":
        from moe_exp.verification_study.closeout import grade_strict
        result=grade_strict(campaign_path,args.model,args.manifest,args.source,args.output,begin=args.begin,count=args.count)
    elif args.command=="grade-j1":
        from moe_exp.verification_study.closeout import grade_j1
        result=grade_j1(campaign_path,args.frame,args.qualification,args.output,args.deadline_epoch)
    elif args.command=="j1-audit":
        from moe_exp.verification_study.closeout import j1_audit
        result=j1_audit(campaign_path,args.output,args.deadline_epoch)
    elif args.command=="assemble":
        from moe_exp.verification_study.closeout import assemble
        result=assemble(campaign_path,root,args.output)
    elif args.command=="terminal-source":
        from moe_exp.verification_study.closeout import terminal_source
        result=terminal_source(args.manifest,args.source,args.output)
    elif args.command=="prices":
        from moe_exp.verification_study.pricing import collect_prices
        result=collect_prices(campaign_path,args.model,root,args.output)
    else:
        from moe_exp.verification_study.analysis import analyze
        data=load(args.source,"verification-analysis-input-v3")
        from moe_exp.verification_study.contracts import require
        require(data["campaign_sha256"]==load(campaign_path)["sha256"],"foreign primary analysis input")
        result=save(args.output,{**analyze(data["records"],load(campaign_path.parent/"FAMILIES.json")["final"],
                                      conditional_enrollment=data.get("conditional_enrollment")),
                                "campaign_sha256":data["campaign_sha256"],"input_sha256":data["sha256"]})
    print(json.dumps({k:v for k,v in result.items() if k in ("schema","sha256","status","assigned","missing","job_id","models","closeout")},indent=2))
    if result.get("status") in ("FAIL","INCOMPLETE"):
        return 1
    return 0


if __name__=="__main__":
    sys.exit(main())
