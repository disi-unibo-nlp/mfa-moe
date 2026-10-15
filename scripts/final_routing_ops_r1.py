#!/usr/bin/env python3
"""Bound operational continuation of the existing final-routing journal."""
import argparse
import json
from pathlib import Path
from moe_exp.final_routing.common import REPO,verified,require
from moe_exp.final_routing_ops_r1.adapter import install


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('worker','resume','status'))
    parser.add_argument('--binding',type=Path,default=REPO/'report/final-routing-20261015-r1/BINDING.json')
    parser.add_argument('--card',type=Path)
    args=parser.parse_args();binding=verified(args.binding)
    if args.command=='status':
        from moe_exp.final_routing.cli import status
        value=status(binding['config'])
    else:
        install(binding)
        if args.command=='worker':
            from moe_exp.final_routing.cli import worker
            require(args.card is not None,'allocated worker requires a card')
            value=worker(args.binding,args.card)
        else:
            from moe_exp.final_routing_ops_r1.launch import launch
            value=launch(binding)
    print(json.dumps(value,sort_keys=True))
    return int(value.get('status') in ('execution_failed','upstream_unavailable','resource_limited','incomplete_results'))


if __name__=='__main__':raise SystemExit(main())
