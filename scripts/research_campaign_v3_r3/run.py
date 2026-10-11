#!/usr/bin/env python3
"""Public entry point for the separately versioned unattended campaign."""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
sys.path.insert(0,str(REPO/"scripts/experimental_resume"))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=("prepare","dry-run","submit","worker","status","closeout"))
    parser.add_argument("--campaign",type=Path)
    parser.add_argument("--card",type=Path)
    parser.add_argument("--resume-from",type=Path)
    parser.add_argument("--detailed",action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--model",default="all")
    parser.add_argument("--threshold-index",type=int,default=0)
    parser.add_argument("--shard-index",type=int,default=0)
    parser.add_argument("--scheduler-test",action="store_true")
    args=parser.parse_args()
    from moe_exp.research_campaign_v3_r3.contracts import load,private_path,require,check_files,publish
    require(args.campaign is not None or args.card is not None,"--campaign or --card is required")
    value=load(args.card) if args.card else None
    if value is not None and value.get("array_cards"):
        import os
        index=int(os.environ["SLURM_ARRAY_TASK_ID"])
        require(0<=index<len(value["array_cards"]),"array index outside immutable table")
        value=value["array_cards"][index]
    root=private_path(args.campaign or value["root"])
    if args.command=="prepare":
        from moe_exp.research_campaign_v3_r3.inputs import prepare
        result=prepare(root,resume_from=args.resume_from)
        result={"campaign_id":result["campaign_id"],"root":result["root"],"sha256":result["sha256"]}
    else:
        campaign=load(root/"CAMPAIGN.json")
        check_files(campaign["code_files"])
        if args.command=="dry-run":
            from moe_exp.research_campaign_v3_r3.scheduler import Slurm,card
            scheduler=Slurm()
            snapshot=scheduler.snapshot()
            cards=[card(campaign,"controller",hold=True),card(campaign,"controller",shard=2,minutes=180,hold=True),
                card(campaign,"measure","qwen330b",minutes=120)]
            (root/"logs").mkdir(parents=True,exist_ok=True,mode=0o700)
            for c in cards:publish(root/"cards"/(c["sha256"]+".json"),c)
            checks=[scheduler.submit(c,test_only=True) for c in cards] if args.scheduler_test else []
            result={"campaign_id":campaign["campaign_id"],"code_and_source_bindings":True,
                "live_resources":snapshot,"registered_graph":load(root/"GRAPH.json"),"scheduler_checks":checks,
                "production_admitted":False,"submission_authorized":True}
        elif args.command=="submit":
            from moe_exp.research_campaign_v3_r3.controller import launch
            result=launch(campaign)
        elif args.command=="worker":
            from moe_exp.research_campaign_v3_r3.dispatch import worker
            if value is None:
                require(args.stage is not None,"worker requires --stage or --card")
                matches=[]
                for path in (root/"cards").glob("*.json"):
                    candidate=load(path)
                    for item in candidate.get("array_cards",[candidate]):
                        if (item["stage"],item["model"],item["shard_index"],item["threshold_index"])==(
                            args.stage,args.model,args.shard_index,args.threshold_index):
                            matches.append(item)
                require(matches,"worker assignment not registered by launcher")
                value=max(matches,key=lambda item:item["attempt"])
            result=worker(campaign,value)
        elif args.command=="status":
            from moe_exp.research_campaign_v3_r3.controller import status
            result=status(campaign,detailed=args.detailed)
        else:
            from moe_exp.research_campaign_v3_r3.contracts import allocated
            allocated()
            from moe_exp.research_campaign_v3_r3.export import closeout
            result=closeout(campaign)
    print(json.dumps(result,sort_keys=True,default=str,allow_nan=False))
    if args.command=='worker' and result.get('status') in ('execution_failed','unresolved_submission','upstream_unavailable'):
        sys.exit(1)


if __name__=="__main__":
    main()
