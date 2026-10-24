"""Command line: ``python -m steering.outcome COMMAND``; see README.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from store import REPO

from steering.outcome.manifest import DEFAULT_MODELS as MODELS

def main() -> None:
    parser = argparse.ArgumentParser(prog='python -m steering.outcome')
    commands = parser.add_subparsers(dest='command', required=True)
    usage = commands.add_parser('usage', help='per-attempt expert usage on cohort B (CPU job; needs torch)')
    usage.add_argument('--models', default=MODELS)
    usage.add_argument('--workers', type=int, default=4)
    select = commands.add_parser('select', help='print each model\'s expert sets; writes nothing')
    select.add_argument('--models', default=MODELS)
    derive = commands.add_parser('derive', help='class usage, r2 selection, signal table, attempt selection (CPU job)')
    derive.add_argument('--release', type=Path, required=True, help='the capture release folder, report/NAME')
    derive.add_argument('--models', required=True)
    derive.add_argument('--workers', type=int, default=8)
    analyze = commands.add_parser('analyze', help='write EFFECTS_OUTCOME.json (CPU job, after the extractions)')
    analyze.add_argument('--release', type=Path, required=True, help='the release folder, report/NAME')
    choose = commands.add_parser('pick', help='apply the frozen pick rule to a half-A receipt; exit 0 when it picks, 3 when not')
    choose.add_argument('--effects', required=True, help='the sealed EFFECTS_OUTCOME.json of half A')
    choose.add_argument('--model', required=True)
    decide = commands.add_parser('decide', help='apply the declared extension rule; exit 0 when it is met, 3 when it is not')
    decide.add_argument('--effects', required=True, help='the sealed EFFECTS_OUTCOME.json')
    args = parser.parse_args()

    reports = REPO/'report'
    if args.command == 'usage':
        from steering.outcome.select import usage as run
        for model in args.models.split(','):
            receipt = run(model, reports, args.workers)
            print(model, receipt['shape'], receipt['sha256'])
    elif args.command == 'select':
        from steering.outcome.manifest import default_spec, evaluation_questions
        from steering.outcome.select import selection
        for model in args.models.split(','):
            questions = evaluation_questions(model, default_spec(model, reports))
            value = selection(model, questions, reports)
            summary = {name: {k: s[k] for k in ('mass', 'control_mass', 'unstable_fill', 'stable_positive', 'questions')}
                       | {'pairs': len(s['targets'])} for name, s in value['sets'].items()}
            print(json.dumps({'model': model, 'protected': value['protected'], 'sets': summary}, indent=1))
    elif args.command == 'derive':
        from steering.outcome.derive import derive as run_chain
        for model in args.models.split(','):
            print(model, json.dumps(run_chain(args.release.resolve(), model, args.workers)), flush=True)
    elif args.command == 'analyze':
        from steering.outcome.analysis import analyze as run
        print(json.dumps({'sha256': run(args.release.resolve())['sha256']}))
    elif args.command == 'pick':
        from steering.outcome.pick import write_pick
        receipt = write_pick(args.effects, args.model)
        print(json.dumps(receipt))
        sys.exit(0 if receipt['confirm'] else 3)  # 3: no cell to confirm (any other code is a failure)
    elif args.command == 'decide':
        from steering.outcome.decide import decide as rule
        decision = rule(args.effects)
        print(json.dumps(decision))
        sys.exit(0 if decision["extend"] else 3)  # 3: the rule is not met (any other code is a failure)


if __name__ == '__main__':
    main()
