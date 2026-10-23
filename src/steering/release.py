"""Seal a whole-reasoning style release: its receipts and a frozen copy of the code its jobs run.

  python -m steering.release complement NAME --tests N   suppress the states the base releases left out (r18)
  python -m steering.release relink NAME --tests N       the final combine again under the eight-model link (r19, r20)
  python -m steering.release outcome NAME --tests N      the outcome-selected nudge (--models, --shards)

N is the number of tests that passed locally on this commit. Run it from the repository with the committed code
clean: what runs is `git archive HEAD`, unpacked once into mfa-moe/releases/<commit>. It writes report/NAME/
(RELEASE.json, the kind file COMPLEMENT/RELINK/OUTCOME.json, SOURCES.json, README.md and the receipts they cite)
and prints the snapshot path. scripts/steering/launch.sh then submits the jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from store import REPO, WORK, file_sha, publish, require, verified

REPORTS = REPO/'report'
SNAPSHOTS = WORK/'mfa-moe/releases'
TEMPLATE = 'whole-reasoning-20261021-r7'  # pins the inputs, the sentence classifier and the nomination reference
RUN_PATHS = ('src', 'tests/steering', 'tests/routing', 'scripts/slurm', 'scripts/steering')
ALL_MODELS = ('gpt', 'nemotron', 'qwen35', 'qwen36', 'gemma', 'glm', 'qwen330b', 'lfm25')


def rome(text: str) -> float:
    """A Europe/Rome wall-clock time as a Unix time."""
    return datetime.fromisoformat(text).replace(tzinfo=ZoneInfo('Europe/Rome')).timestamp()


def complement(root: Path, args: argparse.Namespace) -> dict:
    """r18: the suppress cells of the classes the base releases left out, at each model's base strength.

    The base releases (r7; GPT r8) suppress only the classes tied to longer reasoning, so the per-cell heatmap
    of the paper has empty suppress cells. The new cells run on the same families and seeds with no native rows
    and no canary, and the release hosts a new combine of r7-r17 plus itself.
    """
    from steering.whole import manifest_path
    cutoff = args.cutoff or '2026-10-24T06:00'
    parents = {m: REPORTS/('whole-reasoning-20261021-r8' if m == 'gpt' else TEMPLATE) for m in ALL_MODELS}
    # 2-GPU shards of 4 hours, sized from the base releases' GPU time per row so that a shard's expected
    # work is about half its walltime (a row still running at the walltime stays unknown).
    shards = {'gpt': 3, 'nemotron': 4, 'qwen35': 12, 'qwen36': 12, 'gemma': 6, 'glm': 27, 'qwen330b': 3, 'lfm25': 8}
    sweep = {m: verified(parents[m]/m/'SELECTED.json')['dose'] for m in ALL_MODELS}
    # The combine: every source of r17, with this release joined to each model's base source.
    sources = verified(REPORTS/'whole-reasoning-20261021-r17/SOURCES.json')['sources']
    for model in ALL_MODELS:
        source = next(s for s in sources if s['model'] == model and Path(s['root']) == parents[model]
                      and s['dose'] == sweep[model])
        source.setdefault('complements', []).append(str(root))
    # The complement reads each parent's base manifest (natives) and nominations outside the snapshot.
    inputs = {}
    for model in ALL_MODELS:
        for path in (manifest_path(model, sweep[model], parents[model]), parents[model]/model/'WHOLE_NOMINATIONS.json'):
            inputs[str(path)] = file_sha(path)
    return dict(
        kind_file={'models': {m: {'parent': str(parents[m])} for m in ALL_MODELS}},
        template=verified(REPORTS/TEMPLATE/'RELEASE.json'), models=ALL_MODELS, shards=shards, sweep=sweep,
        cutoff=rome(cutoff), cpu_ceiling=200., sources=sources, inputs=inputs,
        scope='tests/steering/test_whole*.py; the allocated validation runs tests/steering, '
              'tests/steering/outcome and tests/routing in full',
        readme='The base releases suppress only the states tied to longer reasoning. This release suppresses the\n'
               "other states at each model's base strength (GPT and GLM half, the others full) on the same families\n"
               'and seeds, so the per-cell results cover both operations for all seven states. The new cells are\n'
               'not efficiency-directed and never enter the primary or the dose response. Its combine job writes\n'
               'the final combined analysis of r7-r17 plus these cells to analysis-combined/EFFECTS_WHOLE.json.\n'
               f'Shards: {", ".join(f"{m} {shards[m]}" for m in ALL_MODELS)}. GPU work stops at {cutoff} Rome.\n')


def relink(root: Path, args: argparse.Namespace) -> dict:
    """r19, r20: the combined analysis of the parent's sources again, with another efficiency link. No steering.

    The releases r7-r18 chose each model's efficiency-directed cells from the seven-model pre-registration, so
    LFM2.5 used the pooled signs. The eight-model receipt gives LFM its own signs; its directed cells are a subset
    of the pooled ones and every one of them already ran at every strength.
    """
    parent_name = 'whole-reasoning-20261021-r18'
    link = REPORTS/'final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json'
    require(verified(link)['models'] == list(ALL_MODELS), 'the link receipt does not cover the eight models')
    return dict(
        kind_file={'parent': str(REPORTS/parent_name), 'link': str(link), 'link_sha256': verified(link)['sha256'],
                   'reason': 'LFM2.5 uses its own native efficiency-link signs (eight-model receipt) '
                             'instead of the pooled seven-model signs'},
        template=verified(REPORTS/parent_name/'RELEASE.json'), models=ALL_MODELS, shards={}, sweep={}, cutoff=None,
        cpu_ceiling=24., sources=verified(REPORTS/parent_name/'SOURCES.json')['sources'],
        inputs={str(link): file_sha(link)},
        scope='tests/steering/test_whole*.py; the allocated run repeats tests/steering, tests/steering/outcome '
              'and tests/routing in full',
        readme=f'Re-analysis only, no steering. The combined analysis of {parent_name} (r7-r18) is rerun with the\n'
               'eight-model efficiency link (driver/EFFICIENCY_LINK_V2.json), in which LFM2.5 has its own native\n'
               'signs instead of the pooled seven-model ones. Its directed cells are a subset of the pooled ones,\n'
               'all of which ran at every strength, so only LFM2.5 and the pooled rows can change. The seven other\n'
               'models keep their pre-registered signs. Output: analysis-combined/EFFECTS_WHOLE.json.\n')


def outcome(root: Path, args: argparse.Namespace) -> dict:
    """The outcome-selected gentle nudge (steering.outcome; ledger F-0004, D-0003).

    --models: models to steer. --shards: 2-GPU shards (4 hours each) per model, N or model:N,...
    """
    from steering.outcome.manifest import default_spec
    from steering.outcome.select import BULK
    models = tuple((args.models or 'gpt,qwen330b,nemotron').split(','))
    shards = args.shards or '3'
    count = ({m: int(shards) for m in models} if shards.isdigit() else
             {m: int(n) for m, n in (item.split(':') for item in shards.split(','))})
    cutoff = args.cutoff or '2026-10-23T23:00'
    deltas, fractions, answer_tokens = (0.4, 0.8, 1.6), (0.5, 0.7, 0.85), 2048  # D-0003
    # The selection reads these outside the snapshot: per-attempt usage and the class nominations.
    inputs = {str(path): file_sha(path) for model in models
              for path in (BULK/model/'USAGE.json', BULK/model/'USAGE.npz', REPORTS/TEMPLATE/model/'WHOLE_NOMINATIONS.json')}
    return dict(
        kind_file={'models': {m: {**default_spec(m, REPORTS), 'deltas': list(deltas), 'fractions': list(fractions),
                                  'answer_tokens': answer_tokens} for m in models}},
        template=verified(REPORTS/TEMPLATE/'RELEASE.json'), models=models, shards=count,
        sweep={m: 'main' for m in models}, cutoff=rome(cutoff), cpu_ceiling=60., sources=None, inputs=inputs,
        scope='tests/steering/outcome plus tests/steering/test_stream_batch.py; the allocated validation runs '
              'both suites in full',
        readme='Nudges a small set of experts chosen by outcome (used more in wrong than right attempts of the same\n'
               'question, campaign-v3 cohort B) by subtracting a delta from their router logits at every prompt and\n'
               "reasoning token, against matched random experts, and compares both with unedited routing (the base\n"
               "releases' native runs) and a stop-thinking-earlier baseline cut from those native runs.\n"
               f'Models: {", ".join(models)}; deltas {list(deltas)}; length fractions {list(fractions)}.\n'
               'Code: src/steering/outcome (README there). Ledger F-0004, D-0003.\n'
               f'GPU work stops at {cutoff} Rome.\n')


KINDS = {'complement': (complement, 'COMPLEMENT.json'), 'relink': (relink, 'RELINK.json'), 'outcome': (outcome, 'OUTCOME.json')}


def snapshot(root: Path, commit: str) -> tuple[Path, Path]:
    """Archive the commit into the release and unpack it once under mfa-moe/releases/<commit>."""
    archive = root/'SOURCE.tar.gz'
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output='+str(archive), commit, 'src', 'scripts',
                    'configs', 'tests', 'docs/ledger', 'pyproject.toml', 'requirements', 'AGENTS.md'],
                   cwd=REPO, check=True)
    folder = SNAPSHOTS/commit
    if not folder.exists():
        folder.mkdir(parents=True, mode=0o700)
        subprocess.run(['tar', '-xzf', str(archive), '-C', str(folder)], check=True)
        (folder/'RELEASE_COMMIT').write_text(commit+'\n')
    return archive, folder


def seal(kind: str, name: str, tests: int, args: argparse.Namespace) -> Path:
    """Write the receipts of a new release and materialize its code snapshot; return the snapshot."""
    require(not subprocess.run(['git', 'status', '--porcelain', '--', *RUN_PATHS], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout, 'commit the code first')
    root = REPORTS/name
    require(not root.exists(), f'{root} exists')
    build, kind_file = KINDS[kind]
    spec = build(root, args)
    template, shards = spec['template'], spec['shards']
    cutoff = spec['cutoff'] or template['cutoff']  # a relink release keeps its parent's cutoff
    # A 2-GPU shard of 4 hours is 8 GPU-hours; the ceiling is the whole plan plus 10 hours of slack.
    gpu_ceiling = 8.*sum(shards.values())+10 if shards else 0.
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    root.mkdir(mode=0o700)
    publish(root/kind_file, spec['kind_file'])
    publish(root/'LOCAL_VALIDATION.json', {'passed': True, 'tests': tests, 'commit': commit, 'scope': spec['scope']})
    publish(root/'RESOURCE_PLAN.json', {
        'models': list(spec['models']), 'sweep': spec['sweep'], 'shards': shards, 'new_gpu_hour_ceiling': gpu_ceiling,
        'cpu_core_hour_ceiling': spec['cpu_ceiling'], 'cutoff': cutoff, 'no_budget_reset': True})
    if spec['sources'] is not None:
        publish(root/'SOURCES.json', {'sources': spec['sources']})
    (root/'README.md').write_text(f'# Whole-reasoning {kind} ({name})\n\n'+spec['readme'])

    archive, folder = snapshot(root, commit)
    files = {str(p): file_sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    # The template's pinned inputs that live in a snapshot are the same files in this one.
    inputs = {k.replace(template['release_root'], str(folder)): v for k, v in template['inputs'].items()}
    inputs.update(spec['inputs'])
    for path, expected in inputs.items():
        require(file_sha(Path(path)) == expected, 'input mismatch: '+path)
    publish(root/'RELEASE.json', {
        'commit': commit, 'release_root': str(folder), 'models': list(spec['models']), 'sweep': spec['sweep'],
        'shards': shards, 'cutoff': cutoff, kind_file.removesuffix('.json').lower()+'_sha256':
        verified(root/kind_file)['sha256'], 'source_archive': {'path': str(archive), 'sha256_file': file_sha(archive)},
        'code_files': files, 'inputs': inputs, 'classifier': template['classifier'],
        'nomination_reference': template['nomination_reference'],
        'local_validation_sha256': verified(root/'LOCAL_VALIDATION.json')['sha256'],
        'resource_plan_sha256': verified(root/'RESOURCE_PLAN.json')['sha256'],
        'new_gpu_hour_ceiling': gpu_ceiling, 'cpu_core_hour_ceiling': spec['cpu_ceiling'], 'engineering_only': False,
        ('parent_release_sha256' if kind == 'relink' else 'template_release_sha256'): template['sha256']})
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(prog='steering.release', description=__doc__.split('\n\n')[0])
    parser.add_argument('kind', choices=sorted(KINDS))
    parser.add_argument('name')
    parser.add_argument('--tests', type=int, required=True, help='tests that passed locally on this commit')
    parser.add_argument('--models', help='outcome only; default gpt,qwen330b,nemotron')
    parser.add_argument('--shards', help='outcome only: N per model, or model:N,...; default 3')
    parser.add_argument('--cutoff', help='complement, outcome: no GPU row starts after this Rome time, e.g. 2026-10-24T06:00')
    args = parser.parse_args()
    print(seal(args.kind, args.name, args.tests, args))


if __name__ == '__main__':
    main()
