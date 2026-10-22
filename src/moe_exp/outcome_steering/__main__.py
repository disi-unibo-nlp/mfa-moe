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
        from .release import release as seal
        models = args.models.split(',')
        shards = ({m: int(args.shards) for m in models} if args.shards.isdigit() else
                  {m: int(n) for m, n in (item.split(':') for item in args.shards.split(','))})
        print(json.dumps(seal(args.name, models, shards, args.local_tests)))
    elif args.command == 'decide':
        from .decide import decide as rule
        print(json.dumps(rule(args.effects, args.name, args.models, args.shards, args.local_tests)))
    elif args.command == 'analyze':
        from .analysis import analyze
        print(json.dumps({'sha256': analyze()['sha256']}))


if __name__ == '__main__':
    main()
