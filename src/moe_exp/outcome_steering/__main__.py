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


if __name__ == '__main__':
    main()
