"""Seal a whole-reasoning complement release: the suppress cells the base releases left out.

The base releases (r7; GPT r8) suppress only the classes tied to longer reasoning, so the per-cell
heatmap of the paper has empty suppress cells. A complement release runs those cells at each model's
base strength on the same families and seeds (whole.complement_manifest), with no native rows and no
canary, and hosts a new combined analysis of r7-r17 plus itself. Run it through
scripts/study/whole_complement.sh, which then submits the ordinary whole-reasoning job graph.

    python -m moe_exp.study.whole_complement --name whole-reasoning-20261021-r18 --local-tests 107

A complement release that could not start or finish all its shards before its science cutoff gets a
follow-up (--continue-from): a release that runs every row the parent never started and continues every
row it left PARTIAL (whole.continuation_manifest), with the parent's UIDs, seeds and actions and a later
cutoff, after all the parent's shard jobs have ended. Its combined analysis reuses the source chain of an
earlier release (--chain, which has the newer efficiency link) with the follow-up added to each source's
complements. --plan prints what a follow-up would run and size, and seals nothing.

    python -m moe_exp.study.whole_complement --continue-from whole-reasoning-20261021-r18 --plan
    python -m moe_exp.study.whole_complement --continue-from whole-reasoning-20261021-r18 \
        --name whole-reasoning-20261021-r21 --local-tests 160 --cutoff 2026-10-25T06:00
"""
from __future__ import annotations

import argparse
from datetime import datetime
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

from moe_exp.final_routing.common import file_sha, load, publish, require, verified
from moe_exp.outcome_steering.release import whole_usage
from . import whole
from .whole_relink import materialize

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
# A follow-up continues the complement and extends the source chain of the relink that follows it.
CHAIN = 'whole-reasoning-20261021-r20'
LINK = REPO/'report/final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json'
SHARD_GPU_HOURS = 8.  # one 2-GPU shard for its 4-hour walltime
CEILING_MARGIN = 10.  # GPU-hours above the shards' walltimes, as in the complement release
APPROVED_GPU_HOURS = 285.  # ceiling the user approved for the follow-up (ledger F-0012.7)


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


def fit_shards(needed: dict[str, int], limit: int) -> dict[str, int]:
    """Scale shard counts down to at most ``limit`` in total, with at least one shard per model.

    Args:
        needed: Shards each model would need to start all its unfinished rows.
        limit: Most shards the GPU ceiling allows.
    """
    require(limit >= len(needed), f'the GPU ceiling allows {limit} shards for {len(needed)} models')
    total = sum(needed.values())
    if total <= limit:
        return dict(needed)
    share = {model: count*limit/total for model, count in needed.items()}
    fitted = {model: min(needed[model], max(1, math.floor(share[model]))) for model in needed}
    # Hand out the shards the floors left over to the largest remainders.
    for model in sorted(share, key=lambda model: share[model]-math.floor(share[model]), reverse=True):
        if sum(fitted.values()) < limit and fitted[model] < needed[model]:
            fitted[model] += 1
    # The one-shard minimum can overshoot the limit; take it back from the largest models.
    while sum(fitted.values()) > limit:
        fitted[max(fitted, key=fitted.get)] -= 1
    return fitted


def parent_shard_jobs(parent: str) -> list[dict]:
    """Every production job of the parent with its native state, as {id, model, shard, state}."""
    jobs = [job for shards in load(REPORTS/parent/'JOBS.json')['production'].values() for job in shards]
    accounting = whole.native_accounting([{'id': job['id']} for job in jobs])
    return [{'id': job['id'], 'model': job['model'], 'shard': job['shard'],
             'state': accounting.get(job['id'], {}).get('state', 'UNKNOWN')} for job in jobs]


def shard_outcomes(jobs: list[dict], families: list[str], started: set[str]) -> dict:
    """One model's parent shard jobs by state, and how many ended unsuccessfully before touching a row.

    Args:
        jobs: The model's parent_shard_jobs.
        families: Sorted families of the parent manifest (a shard takes those with rank % count == index).
        started: Families with any saved state in the parent store.
    """
    outcomes = Counter(job['state'] for job in jobs)
    failed = [job for job in jobs if whole.terminal({'state': job['state']}) and job['state'] != 'COMPLETED']
    before_start = [job for job in failed
                    if not {f for i, f in enumerate(families) if i % job['shard'][1] == job['shard'][0]} & started]
    return {'jobs': len(jobs), 'states': dict(outcomes), 'failed': len(failed),
            'failed_before_start': len(before_start)}


def plan_followup(parent: str, max_gpu_hours: float = APPROVED_GPU_HOURS, shards: dict | None = None) -> dict:
    """Count the unfinished rows of a complement release and size the follow-up (read-only).

    Args:
        parent: Complement release (directory under report/) whose shard jobs have ended.
        max_gpu_hours: Most GPU-hours the follow-up may reserve; its shards are scaled down to fit.
        shards: Optional {model: shard count} that replaces the computed sizes.

    Returns:
        Per model: the parent's dose, rows by saved state, the families they cover, the parent's shard jobs
        by state (and how many failed before starting a row) and the shards needed and chosen (capacity is
        the parent's own rows per shard); the totals, ceiling and live parent jobs.
    """
    root, release = REPORTS/parent, verified(REPORTS/parent/'RELEASE.json')
    require('complement_sha256' in release, f'{parent} is not a complement release')
    jobs = parent_shard_jobs(parent)
    models = {}
    for model in release['models']:
        dose = release['sweep'][model]
        base = verified(whole.manifest_path(model, dose, root))
        store = whole.driver.Store(root/'stores'/model/dose/'units')
        rows, counts = whole.unfinished_rows(base, store)
        if not rows:
            continue
        capacity = base['executable']/release['shards'][model]
        families = len({row['family'] for row in rows})
        # A shard takes whole families, so it cannot be smaller than one family of rows.
        needed = min(families, max(1, math.ceil(len(rows)/capacity)))
        outcomes = shard_outcomes([job for job in jobs if job['model'] == model],
                                  sorted({row['family'] for row in base['records']}),
                                  whole.started_families(base, store))
        models[model] = {'dose': dose, 'unfinished': len(rows), 'families': families, 'rows_per_shard': capacity,
                         'shards_needed': needed, 'parent_shards': outcomes, **{state: counts[state] for state in
                         ('never_started', 'partial', 'complete', 'output', 'not_nominated')}}
    require(models, f'{parent} has no unfinished rows')
    limit = math.floor((max_gpu_hours-CEILING_MARGIN)/SHARD_GPU_HOURS)
    chosen = shards or fit_shards({model: value['shards_needed'] for model, value in models.items()}, limit)
    require(set(chosen) == set(models), 'shards must name exactly the models with unfinished rows')
    require(sum(chosen.values()) <= limit, f'{sum(chosen.values())} shards exceed the {max_gpu_hours} GPU-hour limit')
    uncovered = sum(max(0, value['unfinished']-chosen[model]*value['rows_per_shard']) for model, value in models.items())
    return {'parent': parent, 'models': models, 'shards': chosen, 'shard_walltime_hours': 4,
            'gpu_ceiling': SHARD_GPU_HOURS*sum(chosen.values())+CEILING_MARGIN, 'max_gpu_hours': max_gpu_hours,
            'rows_beyond_shard_capacity': round(uncovered),
            'live_parent_jobs': [job['id'] for job in jobs if not whole.terminal({'state': job['state']})]}


def format_plan(plan: dict) -> str:
    """The plan as a table: rows by state, the parent's shard outcomes, shards needed and chosen, ceiling."""
    lines = [f'follow-up of {plan["parent"]}',
             '  model     dose     never_started partial complete output | parent shards: failed '
             '(before start) of jobs | families rows/shard shards needed->chosen']
    for model, value in plan['models'].items():
        parent = value['parent_shards']
        lines.append(f'  {model:9s} {value["dose"]:8s} {value["never_started"]:5d} {value["partial"]:7d} '
                     f'{value["complete"]:8d} {value["output"]:6d} |   {parent["failed"]:3d} '
                     f'({parent["failed_before_start"]:3d}) of {parent["jobs"]:3d} | {value["families"]:4d} '
                     f'{value["rows_per_shard"]:6.1f} {value["shards_needed"]:3d} -> {plan["shards"][model]}')
    lines.append(f'shards {sum(plan["shards"].values())} x {plan["shard_walltime_hours"]} h: GPU ceiling '
                 f'{plan["gpu_ceiling"]:.0f} h of {plan["max_gpu_hours"]:.0f} h allowed; '
                 f'{plan["rows_beyond_shard_capacity"]} rows beyond shard capacity (stay unknown or partial)')
    lines.append(f'parent shard jobs not terminal: {len(plan["live_parent_jobs"])}')
    return '\n'.join(lines)


def chain_sources(chain: str, parent: str, name: str, models: list[str]) -> list[dict]:
    """The source list of release ``chain`` with ``name`` added to the complements that hold ``parent``."""
    sources = verified(REPORTS/chain/'SOURCES.json')['sources']
    for model in models:
        source = next((s for s in sources if s['model'] == model and str(REPORTS/parent) in s.get('complements', [])),
                      None)
        require(source, f'{chain} has no {model} source with {parent} among its complements')
        source['complements'].append(str(REPORTS/name))
    return sources


def parse_cutoff(text: str) -> float:
    """Epoch seconds of a Unix time or an ISO date-time (Europe/Rome when it names no zone)."""
    if text.isdigit():
        return float(text)
    moment = datetime.fromisoformat(text)
    return (moment if moment.tzinfo else moment.replace(tzinfo=ZoneInfo('Europe/Rome'))).timestamp()


def followup(name: str, local_tests: int, parent: str, cutoff: float, chain: str = CHAIN,
             max_gpu_hours: float = APPROVED_GPU_HOURS, shards: dict | None = None, link: Path = LINK) -> dict:
    """Write every receipt of a follow-up of a complement release and materialize its code snapshot.

    The follow-up runs the parent's never-started rows and continues its PARTIAL rows (CONTINUATION.json,
    whole.continuation_manifest). Its SOURCES.json is the source chain of ``chain`` with this release added to
    the parent's place in each model's complements, and its combine uses the efficiency link ``link``.

    Args:
        name: Release directory under report/ (must not exist).
        local_tests: Tests that passed locally on this commit (the allocated validation reruns all).
        parent: Complement release whose shard jobs have all ended.
        cutoff: Epoch seconds after which no GPU work starts.
        chain: Release whose SOURCES.json lists every release of the final combine.
        max_gpu_hours, shards: GPU-hour limit and optional shard counts (see plan_followup).
        link: Efficiency-link receipt for the combined analysis.
    """
    require(not subprocess.run(['git', 'status', '--porcelain', '--', *RUN_PATHS], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout, 'commit the code first')
    root, old = REPORTS/name, verified(REPORTS/parent/'RELEASE.json')
    require(not root.exists(), f'{root} exists')
    plan = plan_followup(parent, max_gpu_hours, shards)
    require(not plan['live_parent_jobs'], f'{parent} still has shard jobs that have not ended: {plan["live_parent_jobs"]}')
    require(cutoff > max(time.time(), old['cutoff']), 'the cutoff must be later than now and than the parent\'s')
    link_receipt = verified(link)
    require(link_receipt['models'] == list(MODELS), 'the link receipt does not cover the eight models')
    models, shards = list(plan['models']), plan['shards']
    sweep = {m: plan['models'][m]['dose'] for m in models}
    sources = chain_sources(chain, parent, name, models)
    gpu_ceiling, cpu_ceiling = plan['gpu_ceiling'], 200.

    root.mkdir(mode=0o700)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    publish(root/'PRIOR_USAGE.json', {'earlier_whole_usage_snapshot': whole_usage(), 'snapshot_time': time.time(),
                                      'budget_reset': False,
                                      'separate_budget': f'{gpu_ceiling} GPU-h and {cpu_ceiling} CPU core-h'})
    publish(root/'LOCAL_VALIDATION.json', {'passed': True, 'tests': local_tests, 'commit': commit,
                                           'scope': 'tests/study/test_whole*.py; the allocated validation runs '
                                                    'tests/study and tests/outcome_steering in full'})
    publish(root/'RESOURCE_PLAN.json', {'models': models, 'sweep': sweep, 'shards': shards, 'plan': plan,
                                        'new_gpu_hour_ceiling': gpu_ceiling, 'cpu_core_hour_ceiling': cpu_ceiling,
                                        'cutoff': float(cutoff), 'no_budget_reset': True})
    publish(root/'PREFLIGHT.json', {
        'host': os.uname().nodename, 'user': 'lmolfett', 'account': 'iscrc_miosr', 'time': time.time(),
        'parent_release_sha256': old['sha256'], 'link_sha256': link_receipt['sha256'],
        'balance': subprocess.run(['/cineca/bin/saldo', '-b'], capture_output=True, text=True).stdout})
    # CONTINUATION.json makes whole.prepare build each model's never-started and partial rows of the parent.
    publish(root/'CONTINUATION.json', {'models': {m: {'parent': str(REPORTS/parent)} for m in models}})
    publish(root/'SOURCES.json', {'sources': sources})
    (root/'README.md').write_text(
        f'# Whole-reasoning follow-up ({name})\n\n'
        f'Continues the complement release {parent}, whose shards could not all start or finish before its\n'
        'science cutoff. It runs the rows the parent never started and continues the rows it left PARTIAL, with\n'
        'the parent UIDs, seeds and actions; each saved partial is copied into this release\'s own store and\n'
        're-prefilled through the same hooks, and no parent store is written. Rows the parent finished stay\n'
        'with it. Its combined analysis joins every outcome to the parent\'s rows by UID, with the efficiency\n'
        'link of the eight-model receipt (driver/EFFICIENCY_LINK_V2.json): analysis-combined/EFFECTS_WHOLE.json.\n'
        'Rows (never started + partial) and shards: '
        + ', '.join(f'{m} {v["never_started"]}+{v["partial"]} in {shards[m]}' for m, v in plan['models'].items())
        + f'.\nGPU work stops at {datetime.fromtimestamp(cutoff, ZoneInfo("Europe/Rome")):%Y-%m-%d %H:%M} Rome.\n')

    archive, snapshot = materialize(root, commit)
    files = {str(p): file_sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    inputs = {k.replace(old['release_root'], str(snapshot)): v for k, v in old['inputs'].items()}
    # The continuation reads each parent manifest; the combine reads the link receipt.
    for model in models:
        path = whole.manifest_path(model, sweep[model], REPORTS/parent)
        inputs[str(path)] = file_sha(path)
    inputs[str(link)] = file_sha(link)
    for path, expected in inputs.items():
        require(file_sha(Path(path)) == expected, 'input mismatch: '+path)
    value = publish(root/'RELEASE.json', {
        'commit': commit, 'release_root': str(snapshot), 'models': models, 'sweep': sweep, 'shards': shards,
        'cutoff': float(cutoff), 'continuation_sha256': verified(root/'CONTINUATION.json')['sha256'],
        'source_archive': {'path': str(archive), 'sha256_file': file_sha(archive)},
        'code_files': files, 'inputs': inputs, 'classifier': old['classifier'],
        'nomination_reference': old['nomination_reference'],
        'preflight_sha256': verified(root/'PREFLIGHT.json')['sha256'],
        'local_validation_sha256': verified(root/'LOCAL_VALIDATION.json')['sha256'],
        'resource_plan_sha256': verified(root/'RESOURCE_PLAN.json')['sha256'],
        'prior_usage': verified(root/'PRIOR_USAGE.json'), 'new_gpu_hour_ceiling': gpu_ceiling,
        'cpu_core_hour_ceiling': cpu_ceiling, 'engineering_only': False,
        'template_release_sha256': old['template_release_sha256'], 'parent_release_sha256': old['sha256']})
    (root/'launch.env').write_text(
        f"MODELS={','.join(models)}\nSWEEP={','.join(f'{m}:{sweep[m]}' for m in models)}\n"
        f"SHARDS={','.join(f'{m}:{shards[m]}' for m in models)}\nGPU_CEILING={gpu_ceiling}\n"
        f"CPU_CEILING={cpu_ceiling}\nCUTOFF={int(cutoff)}\nCOMMIT={commit}\nLINK={link}\n")
    return {'release': name, 'commit': commit, 'sha256': value['sha256'], 'snapshot': str(snapshot),
            'shards': shards, 'gpu_ceiling': gpu_ceiling}


def parse_shards(text: str) -> dict[str, int]:
    """Shard counts from "model:count,model:count"."""
    return {model: int(count) for model, count in (item.split(':') for item in text.split(',') if item)}


def main() -> None:
    parser = argparse.ArgumentParser(prog='python -m moe_exp.study.whole_complement')
    parser.add_argument('--name', help='release to seal (not needed with --plan)')
    parser.add_argument('--local-tests', type=int)
    parser.add_argument('--continue-from', metavar='PARENT',
                        help='seal a follow-up of this complement release instead of a complement')
    parser.add_argument('--plan', action='store_true', help='with --continue-from: print the plan, seal nothing')
    parser.add_argument('--cutoff', help='follow-up: last moment GPU work may start (ISO time, Rome, or epoch)')
    parser.add_argument('--chain', default=CHAIN, help='follow-up: release whose SOURCES.json it extends')
    parser.add_argument('--link', type=Path, default=LINK, help='follow-up: efficiency-link receipt')
    parser.add_argument('--max-gpu-hours', type=float, default=APPROVED_GPU_HOURS)
    parser.add_argument('--shards', type=parse_shards, help='follow-up: model:count,... instead of the computed sizes')
    args = parser.parse_args()
    if args.continue_from and args.plan:
        print(format_plan(plan_followup(args.continue_from, args.max_gpu_hours, args.shards)))
        return
    require(args.name and args.local_tests is not None, '--name and --local-tests are required to seal')
    if not args.continue_from:
        print(json.dumps(release(args.name, args.local_tests)))
        return
    require(args.cutoff, '--cutoff is required for a follow-up')
    print(json.dumps(followup(args.name, args.local_tests, args.continue_from, parse_cutoff(args.cutoff),
                              args.chain, args.max_gpu_hours, args.shards, args.link)))


if __name__ == '__main__':
    main()
