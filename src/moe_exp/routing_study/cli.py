"""prepare, calibrate, generate, compare, plus explicit verification/freeze helpers."""
from __future__ import annotations
import argparse
from pathlib import Path
from .common import read_json, write_json, rows, verify, bounded_input, seal, code_hashes, CHECKPOINT, ENGINE, SAMPLING
from .policy import static_grid, adaptive_variants, freeze
from .prepare import prepare


def main(argv=None):
    parser=argparse.ArgumentParser(description="Qwen3.6 accuracy–token routing study; never submits jobs")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--sources",nargs="+",type=Path,required=True)
    p.add_argument("--pilot",type=Path,required=True)
    p.add_argument("--duplicates",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("calibrate")
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--captures",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("generate")
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("--phase",choices=("calibration","tuning","test"),required=True)
    p.add_argument("--smoke",type=Path,required=True)
    p.add_argument("--selection",type=Path)
    p.add_argument("--registration",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--baseline-from",type=Path)
    p=sub.add_parser("compare")
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--runs",type=Path,nargs="+",required=True)
    p.add_argument("--registration",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("register")
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--policies",type=Path,nargs="+",required=True)
    p.add_argument("--existing-controls",type=Path,nargs="*",default=[])
    p.add_argument("--pairs",type=Path,help="JSON array of [control_sha256, policy_sha256]; default each vs native")
    p.add_argument("--phase",choices=("tuning","test"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("select")
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--policies",type=Path,nargs="+",required=True)
    p.add_argument("--stage",choices=("static","final"),required=True)
    p.add_argument("--static-selection",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("adapt")
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--policies",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("smoke")
    p.add_argument("--fixtures",type=Path,required=True)
    p.add_argument("--cpu-report",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("proposal")
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--smoke",type=Path,required=True)
    p.add_argument("--conditions",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("protocol")
    p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=="prepare":
        result=prepare(args.sources,args.pilot,args.output,
                       duplicate_pairs=read_json(args.duplicates) if args.duplicates else ())
    elif args.command=="calibrate":
        from .calibration import calibrate
        paths=sorted(args.captures.glob("*.json"))
        bounded_input(paths)
        result=calibrate([read_json(p) for p in paths],read_json(args.study))
        write_json(args.output/"calibration.json",result,immutable=True)
        policies=static_grid(result)
        for policy in policies:
            write_json(args.output/"policies"/(policy["sha256"]+".json"),policy,immutable=True)
        write_json(args.output/"catalog.json",seal({"policies":policies}),immutable=True)
    elif args.command=="generate":
        from .execution import execute
        result=execute(read_json(args.study),read_json(args.policy),args.phase,args.output,
            read_json(args.smoke),selection=read_json(args.selection) if args.selection else None,
            registration=read_json(args.registration) if args.registration else None,
            resume=args.resume,baseline_from=args.baseline_from)
    elif args.command=="compare":
        from .analysis import compare
        result=compare(args.runs,read_json(args.study),read_json(args.registration),output=args.output)
    elif args.command=="register":
        from .analysis import register
        policies=[read_json(p) for p in args.policies]
        pairs=read_json(args.pairs) if args.pairs else [[policies[0]["sha256"],p["sha256"]] for p in policies[1:]]
        result=register(read_json(args.study),policies,pairs,args.phase,args.existing_controls)
        write_json(args.output,result,immutable=True)
    elif args.command=="select":
        from .analysis import freeze_frontier
        result=freeze_frontier(read_json(args.report),read_json(args.study),
                               [read_json(p) for p in args.policies],stage=args.stage,
                               static_selection=read_json(args.static_selection) if args.static_selection else None)
        write_json(args.output,result,immutable=True)
    elif args.command=="adapt":
        calibration,selection=verify(read_json(args.calibration)),verify(read_json(args.selection))
        if selection["stage"]!="static" or "unavailable" in calibration["timing"]:
            raise ValueError("Need a static shortlist and supported calibration timing")
        if calibration["study_sha256"]!=selection["study_sha256"]:
            raise ValueError("Timing calibration and shortlist belong to different studies")
        timing={**calibration["timing"],"calibration":{**calibration["timing"]["calibration"],"sha256":calibration["sha256"]}}
        variants=[]
        for path in args.policies:
            policy=read_json(path)
            if policy["sha256"] in selection["selected_policy_hashes"]:
                variants.extend(adaptive_variants(policy,timing))
        if len(variants)!=6*len(selection["selected_policy_hashes"]):
            raise ValueError("Missing shortlisted static policies")
        for policy in variants:
            write_json(args.output/(policy["sha256"]+".json"),policy,immutable=True)
        result=seal({"policies":variants,"selection_sha256":selection["sha256"]})
        write_json(args.output/"catalog.json",result,immutable=True)
    elif args.command=="smoke":
        from .smoke import run
        result=run(args.fixtures,args.cpu_report,args.output)
    elif args.command=="proposal":
        from .smoke import resource_proposal
        result=resource_proposal(read_json(args.smoke),read_json(args.study),conditions=args.conditions)
        write_json(args.output,result,immutable=True)
    else:
        result=seal(dict(schema=1,status="implemented_pending_authorized_smoke_and_calibration",
            checkpoint=CHECKPOINT,engine=ENGINE,sampling=SAMPLING,code=code_hashes(),
            approved_study_gpu_hours=0,existing_pilot_budget_reserved_gpu_hours=32,
            split={"seed":42,"fractions":[.6,.2,.2],"rounding":"remainder_to_test","pilot_calibration":64,"fixtures":4},
            static_policies_without_calibration=[freeze("native"),freeze("identity",0),
                                                 freeze("temperature",.5),freeze("temperature",2)],
            temporal={"window":256,"lag":16,"stride":64,"duration":64,"cooldown":64},
            bootstrap_replicates=1000,final_repetitions={"aime24":32,"aime25":32,"amc23":32,"else":1},
            availability="Identity, SteerMoE, RICE, margin and adaptive policies require exact FP8 calibration; none inferred from NF4."))
        write_json(args.output,result,immutable=True)
        for p in result["static_policies_without_calibration"]:
            name="native" if p["action"]=="native" else ("identity-zero" if p["action"]=="identity" else "temperature-"+str(p["strength"]))
            write_json(args.output.parent/(name+".json"),p,immutable=True)
    print("Completed",args.command,"sha256="+str(result.get("sha256","unsealed_execution_manifest")))
    return 0
