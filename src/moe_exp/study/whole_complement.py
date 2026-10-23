"""Seal a whole-reasoning complement release: the suppress cells the base releases left out.

The base releases (r7; GPT r8) suppress only the classes tied to longer reasoning, so the per-cell
heatmap of the paper has empty suppress cells. A complement release runs those cells at each model's
base strength on the same families and seeds (whole.complement_manifest), with no native rows and no
canary, and hosts a new combined analysis of r7-r17 plus itself. Run it through
scripts/study/whole_complement.sh, which then submits the ordinary whole-reasoning job graph.

    python -m moe_exp.study.whole_complement --name whole-reasoning-20261021-r18 --local-tests 107
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from moe_exp.final_routing.common import file_sha, publish, require, verified
from moe_exp.outcome_steering.release import whole_usage

REPO = Path(__file__).resolve().parents[3]
REPORTS = REPO/'report'
TEMPLATE = 'whole-reasoning-20261021-r7'      # pinned inputs, classifier and nomination reference
FINAL = 'whole-reasoning-20261021-r17'        # its SOURCES.json lists every release of the final combine
PARENTS = {'gpt': 'whole-reasoning-20261021-r8'}  # base release of each model; r7 for the others
MODELS = ('gpt', 'nemotron', 'qwen35', 'qwen36', 'gemma', 'glm', 'qwen330b', 'lfm25')
# 2-GPU shards of 4 hours, sized from the base releases' GPU time per row so that a shard's expected
# work is about half its walltime (a row still running at the walltime stays unknown).
SHARDS = {'gpt': 3, 'nemotron': 4, 'qwen35': 12, 'qwen36': 12, 'gemma': 6, 'glm': 27, 'qwen330b': 3, 'lfm25': 8}
CUTOFF = 1792814400  # 2026-10-24 06:00 Europe/Rome: no GPU work starts after it
RUN_PATHS = ('src/moe_exp', 'tests/study', 'tests/outcome_steering', 'scripts/study')


def release(name: str, local_tests: int, models=MODELS, shards=SHARDS) -> dict:
    """Write every receipt of a new complement release and materialize its code snapshot.

    Args:
        name: Release directory under report/ (must not exist).
        local_tests: Tests that passed locally on this commit (the allocated validation reruns all).
        models, shards: Models to complete and their 2-GPU shards.
    """
    require(not subprocess.run(['git', 'status', '--porcelain', '--', *RUN_PATHS], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout, 'commit the code first')
    root = REPORTS/name
    require(not root.exists(), f'{root} exists')
    parents = {m: REPORTS/PARENTS.get(m, TEMPLATE) for m in models}
    sweep = {m: verified(parents[m]/m/'SELECTED.json')['dose'] for m in models}
    root.mkdir(mode=0o700)
    template = verified(REPORTS/TEMPLATE/'RELEASE.json')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    gpu_ceiling, cpu_ceiling = 8.*sum(shards[m] for m in models)+10, 200.

    publish(root/'PRIOR_USAGE.json', {'earlier_whole_usage_snapshot': whole_usage(), 'snapshot_time': time.time(),
                                      'budget_reset': False,
                                      'separate_budget': f'{gpu_ceiling} GPU-h and {cpu_ceiling} CPU core-h'})
    publish(root/'LOCAL_VALIDATION.json', {'passed': True, 'tests': local_tests, 'commit': commit,
                                           'scope': 'tests/study/test_whole*.py; the allocated validation runs '
                                                    'tests/study and tests/outcome_steering in full'})
    publish(root/'RESOURCE_PLAN.json', {'models': list(models), 'sweep': sweep,
                                        'shards': {m: shards[m] for m in models},
                                        'new_gpu_hour_ceiling': gpu_ceiling, 'cpu_core_hour_ceiling': cpu_ceiling,
                                        'cutoff': float(CUTOFF), 'no_budget_reset': True})
    publish(root/'PREFLIGHT.json', {
        'host': os.uname().nodename, 'user': 'lmolfett', 'account': 'iscrc_miosr', 'time': time.time(),
        'template_release_sha256': template['sha256'],
        'balance': subprocess.run(['/cineca/bin/saldo', '-b'], capture_output=True, text=True).stdout})
    # COMPLEMENT.json makes whole.prepare build the missing suppress cells of each parent.
    publish(root/'COMPLEMENT.json', {'models': {m: {'parent': str(parents[m])} for m in models}})
    # The final combine: every source of r17, with this release joined to each model's base source.
    sources = verified(REPORTS/FINAL/'SOURCES.json')['sources']
    for model in models:
        source = next(s for s in sources if s['model'] == model and Path(s['root']) == parents[model]
                      and s['dose'] == sweep[model])
        source.setdefault('complements', []).append(str(root))
    publish(root/'SOURCES.json', {'sources': sources})
    (root/'README.md').write_text(
        f'# Whole-reasoning complement ({name})\n\n'
        'The base releases suppress only the states tied to longer reasoning. This release suppresses the\n'
        'other states at each model\'s base strength (GPT and GLM half, the others full) on the same families\n'
        'and seeds, so the per-cell results cover both operations for all seven states. The new cells are\n'
        'not efficiency-directed and never enter the primary or the dose response. Its combine job writes\n'
        'the final combined analysis of r7-r17 plus these cells to analysis-combined/EFFECTS_WHOLE.json.\n'
        f'Shards: {", ".join(f"{m} {shards[m]}" for m in models)}. GPU work stops at 2026-10-24 06:00 Rome.\n')

    archive = root/'SOURCE.tar.gz'
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output='+str(archive), commit, 'src', 'scripts',
                    'configs', 'tests', 'docs/ledger', 'pyproject.toml', 'requirements', 'AGENTS.md'],
                   cwd=REPO, check=True)
    snapshot = REPO.parent/'releases'/commit
    if not snapshot.exists():
        snapshot.mkdir(mode=0o700)
        subprocess.run(['tar', '-xzf', str(archive), '-C', str(snapshot)], check=True)
        (snapshot/'RELEASE_COMMIT').write_text(commit+'\n')
    files = {str(p): file_sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    inputs = {k.replace(template['release_root'], str(snapshot)): v for k, v in template['inputs'].items()}
    # The complement reads each parent's base manifest (natives) and nominations outside the snapshot.
    for model in models:
        dose = sweep[model]
        for path in (parents[model]/model/('WHOLE.json' if dose == 'main' else f'WHOLE_{dose.upper()}.json'),
                     parents[model]/model/'WHOLE_NOMINATIONS.json'):
            inputs[str(path)] = file_sha(path)
    for path, expected in inputs.items():
        require(file_sha(Path(path)) == expected, 'input mismatch: '+path)
    value = publish(root/'RELEASE.json', {
        'commit': commit, 'release_root': str(snapshot), 'models': list(models), 'sweep': sweep,
        'shards': {m: shards[m] for m in models}, 'cutoff': float(CUTOFF),
        'complement_sha256': verified(root/'COMPLEMENT.json')['sha256'],
        'source_archive': {'path': str(archive), 'sha256_file': file_sha(archive)},
        'code_files': files, 'inputs': inputs, 'classifier': template['classifier'],
        'nomination_reference': template['nomination_reference'],
        'preflight_sha256': verified(root/'PREFLIGHT.json')['sha256'],
        'local_validation_sha256': verified(root/'LOCAL_VALIDATION.json')['sha256'],
        'resource_plan_sha256': verified(root/'RESOURCE_PLAN.json')['sha256'],
        'prior_usage': verified(root/'PRIOR_USAGE.json'), 'new_gpu_hour_ceiling': gpu_ceiling,
        'cpu_core_hour_ceiling': cpu_ceiling, 'engineering_only': False, 'template_release_sha256': template['sha256']})
    (root/'launch.env').write_text(
        f"MODELS={','.join(models)}\nSWEEP={','.join(f'{m}:{sweep[m]}' for m in models)}\n"
        f"SHARDS={','.join(f'{m}:{shards[m]}' for m in models)}\nGPU_CEILING={gpu_ceiling}\nCPU_CEILING={cpu_ceiling}\n"
        f"CUTOFF={CUTOFF}\nCOMMIT={commit}\n")
    return {'release': name, 'commit': commit, 'sha256': value['sha256'], 'snapshot': str(snapshot)}


def main() -> None:
    parser = argparse.ArgumentParser(prog='python -m moe_exp.study.whole_complement')
    parser.add_argument('--name', required=True)
    parser.add_argument('--local-tests', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(release(args.name, args.local_tests)))


if __name__ == '__main__':
    main()
