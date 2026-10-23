"""Command line: ``python -m moe_exp.outcome_steering COMMAND``; see README.md."""
from __future__ import annotations

import argparse
import json

MODELS = 'gpt,qwen330b,nemotron'


def main() -> None:
    parser = argparse.ArgumentParser(prog='python -m moe_exp.outcome_steering')
    commands = parser.add_subparsers(dest='command', required=True)
    usage = commands.add_parser('usage', help='per-attempt expert usage on cohort B (CPU job; needs torch)')
    usage.add_argument('--models', default=MODELS)
    usage.add_argument('--workers', type=int, default=4)
    select = commands.add_parser('select', help='print each model\'s expert sets; writes nothing')
    select.add_argument('--models', default=MODELS)
    release = commands.add_parser('release', help='seal a release: receipts, code snapshot, launch.env')
    release.add_argument('--name', required=True)
    release.add_argument('--models', default=MODELS)
    release.add_argument('--shards', default='3', help='shards per model: N, or model:N,...')
    release.add_argument('--local-tests', type=int, required=True)
    release.add_argument('--kind', choices=('outcome', 'capture', 'sweep-A', 'sweep-B'), default='outcome',
                         help='outcome: r1 nudge release; capture: native attempts with routing saved; '
                              'sweep-A/B: r2 sweep halves (sweep-B reads PICK.json from $SWEEP_PICK)')
    derive = commands.add_parser('derive', help='class usage, r2 selection, signal table, attempt selection (CPU job)')
    derive.add_argument('--release', required=True, help='report/<capture release>')
    derive.add_argument('--models', required=True)
    derive.add_argument('--workers', type=int, default=8)
    choose = commands.add_parser('pick', help='apply the frozen pick rule to a half-A receipt and launch half B')
    choose.add_argument('--effects', required=True)
    choose.add_argument('--model', required=True)
    choose.add_argument('--name', required=True, help='half-B release name')
    choose.add_argument('--shards', required=True)
    choose.add_argument('--local-tests', required=True)
    commands.add_parser('analyze', help='write EFFECTS_OUTCOME.json (allocated CPU job, from the release)')
    decide = commands.add_parser('decide', help='apply the declared extension rule; launch the other models if met')
    decide.add_argument('--effects', required=True)
    decide.add_argument('--name', required=True)
    decide.add_argument('--models', required=True)
    decide.add_argument('--shards', required=True)
    decide.add_argument('--local-tests', required=True)
    args = parser.parse_args()

    if args.command == 'usage':
        from .select import usage as run
        for model in args.models.split(','):
            receipt = run(model, args.workers)
            print(model, receipt['shape'], receipt['sha256'])
    elif args.command == 'select':
        from .manifest import evaluation_questions
        from .select import selection
        for model in args.models.split(','):
            value = selection(model, evaluation_questions(model))
            summary = {name: {k: s[k] for k in ('mass', 'control_mass', 'unstable_fill', 'stable_positive', 'questions')}
                       | {'pairs': len(s['targets'])} for name, s in value['sets'].items()}
            print(json.dumps({'model': model, 'protected': value['protected'], 'sets': summary}, indent=1))
    elif args.command == 'release':
        import os
        from .release import capture_release, release as outcome_release, sweep_release
        models = args.models.split(',')
        shards = ({m: int(args.shards) for m in models} if args.shards.isdigit() else
                  {m: int(n) for m, n in (item.split(':') for item in args.shards.split(','))})
        if args.kind.startswith('sweep-'):
            half = args.kind[-1]
            value = sweep_release(args.name, models, shards, args.local_tests, half,
                                  os.environ['SWEEP_PICK'] if half == 'B' else None)
        else:
            seal = capture_release if args.kind == 'capture' else outcome_release
            value = seal(args.name, models, shards, args.local_tests)
        print(json.dumps(value))
    elif args.command == 'decide':
        from .decide import decide as rule
        print(json.dumps(rule(args.effects, args.name, args.models, args.shards, args.local_tests)))
    elif args.command == 'derive':
        from pathlib import Path
        from .derive import derive as run_chain
        for model in args.models.split(','):
            print(model, json.dumps(run_chain(Path(args.release), model, args.workers)), flush=True)
    elif args.command == 'pick':
        from .pick import pick_and_launch
        print(json.dumps(pick_and_launch(args.effects, args.model, args.name, args.shards, args.local_tests)))
    elif args.command == 'analyze':
        from .analysis import analyze
        print(json.dumps({'sha256': analyze()['sha256']}))


if __name__ == '__main__':
    main()
