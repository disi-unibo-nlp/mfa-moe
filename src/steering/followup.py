"""Size the follow-up of a complement release: what its shards left unfinished and how many shards to run now.

A complement release runs many 4-hour shards and can reach its cutoff before all of them start. Its follow-up
(`steering.release continuation`) runs the rows the complement never started and continues the ones it left
PARTIAL. This module reads the complement's manifests, stores and shard jobs (read-only, nothing is sealed or
submitted) and sizes the follow-up under a GPU-hour limit. A shard takes whole question families, so a model
never needs more shards than it has unfinished families.

    python -m steering.followup report/PARENT [--max-gpu-hours H] [--shards model:N,...]

It prints, per model, the parent's rows by saved state, the parent's shard jobs by outcome (and how many ended
before touching a row), and the shards needed and chosen. `steering.release continuation` uses the same plan.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import re
import subprocess

from steering import pulse
from steering.whole import manifest, started_families, unfinished_rows
from store import load, require, verified

SHARD_GPU_HOURS = 8.  # one 2-GPU shard for its 4-hour walltime
CEILING_MARGIN = 10.  # GPU-hours above the shards' walltimes, as in the complement release
APPROVED_GPU_HOURS = 285.  # GPU-hours the follow-up may reserve unless --max-gpu-hours says otherwise
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED')


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


def submitted_jobs(root: Path) -> dict[str, list[dict]]:
    """The jobs a release submitted, as {'run': [...], 'extract': [...]} with {id, model, shard} for each.

    Reads JOBS.txt (written by scripts/steering/launch.sh) or, in a release that holds JOBS.json instead,
    that file.
    """
    jobs = {'run': [], 'extract': []}
    if (root/'JOBS.txt').exists():
        for line in (root/'JOBS.txt').read_text().splitlines():
            _, job, command = line.split(' ', 2)
            stage = re.search(r'steering\.whole (run|extract) ', command)
            if stage:
                shard = re.search(r'--shard (\d+)/(\d+)', command)
                jobs[stage.group(1)].append({'id': job, 'model': re.search(r'--model (\w+)', command).group(1),
                                             'shard': [int(shard.group(1)), int(shard.group(2))] if shard else None})
    else:
        record = load(root/'JOBS.json')
        for shards in record['production'].values():
            jobs['run'] += [{'id': job['id'], 'model': job['model'], 'shard': job['shard']} for job in shards]
        jobs['extract'] += [{'id': job['id'], 'model': model, 'shard': None}
                            for model, job in record.get('extraction', {}).items()]
    return jobs


def job_states(ids: list[str]) -> dict[str, str]:
    """Slurm's state of each job (the first word of sacct's State; read-only). A job sacct no longer knows is UNKNOWN."""
    if not ids:
        return {}
    answer = subprocess.run(['sacct', '-nP', '-X', '-j', ','.join(ids), '--format=JobIDRaw,State'],
                            capture_output=True, text=True, check=True)
    states = dict(line.split('|')[:2] for line in answer.stdout.splitlines() if '|' in line)
    return {job: states.get(job, 'UNKNOWN').split()[0] for job in ids}


def parent_shard_jobs(root: Path) -> list[dict]:
    """Every shard job of the parent with its Slurm state, as {id, model, shard, state}."""
    jobs = submitted_jobs(root)['run']
    states = job_states([job['id'] for job in jobs])
    return [{**job, 'state': states[job['id']]} for job in jobs]


def shard_outcomes(jobs: list[dict], families: list[str], started: set[str]) -> dict:
    """One model's parent shard jobs by state, and how many ended unsuccessfully before touching a row.

    Args:
        jobs: The model's parent_shard_jobs.
        families: Sorted families of the parent manifest (a shard takes those with rank % count == index).
        started: Families with any saved state in the parent store.
    """
    outcomes = Counter(job['state'] for job in jobs)
    failed = [job for job in jobs if job['state'] in TERMINAL and job['state'] != 'COMPLETED']
    before_start = [job for job in failed
                    if not {f for i, f in enumerate(families) if i % job['shard'][1] == job['shard'][0]} & started]
    return {'jobs': len(jobs), 'states': dict(outcomes), 'failed': len(failed),
            'failed_before_start': len(before_start)}


def plan_followup(root: Path, max_gpu_hours: float = APPROVED_GPU_HOURS, shards: dict | None = None) -> dict:
    """Count the unfinished rows of a complement release and size the follow-up (read-only).

    Args:
        root: Complement release folder (report/NAME) whose shard jobs have ended.
        max_gpu_hours: Most GPU-hours the follow-up may reserve; its shards are scaled down to fit.
        shards: Optional {model: shard count} that replaces the computed sizes.

    Returns:
        Per model: the parent's dose, rows by saved state, the families they cover, the parent's shard jobs
        by state (and how many failed before starting a row) and the shards needed and chosen (capacity is
        the parent's own rows per shard); the totals, ceiling, the live parent jobs and the parent's
        extraction jobs (the follow-up's combine must wait for them).
    """
    release = verified(root/'RELEASE.json')
    require('complement_sha256' in release, f'{root.name} is not a complement release')
    jobs = parent_shard_jobs(root)
    models = {}
    for model in release['models']:
        dose = release['sweep'][model]
        base = manifest(root, model, dose)
        store = pulse.Store(root/'stores'/model/dose/'units')
        rows, counts = unfinished_rows(base, store)
        if not rows:
            continue
        capacity = base['executable']/release['shards'][model]
        families = len({row['family'] for row in rows})
        # A shard takes whole families, so it cannot be smaller than one family of rows.
        needed = min(families, max(1, math.ceil(len(rows)/capacity)))
        outcomes = shard_outcomes([job for job in jobs if job['model'] == model],
                                  sorted({row['family'] for row in base['records']}), started_families(base, store))
        models[model] = {'dose': dose, 'unfinished': len(rows), 'families': families, 'rows_per_shard': capacity,
                         'shards_needed': needed, 'parent_shards': outcomes, **{state: counts[state] for state in
                         ('never_started', 'partial', 'complete', 'output', 'not_nominated')}}
    require(models, f'{root.name} has no unfinished rows')
    limit = math.floor((max_gpu_hours-CEILING_MARGIN)/SHARD_GPU_HOURS)
    chosen = shards or fit_shards({model: value['shards_needed'] for model, value in models.items()}, limit)
    require(set(chosen) == set(models), 'shards must name exactly the models with unfinished rows')
    require(sum(chosen.values()) <= limit, f'{sum(chosen.values())} shards exceed the {max_gpu_hours} GPU-hour limit')
    uncovered = sum(max(0, value['unfinished']-chosen[model]*value['rows_per_shard']) for model, value in models.items())
    return {'parent': root.name, 'models': models, 'shards': chosen, 'shard_walltime_hours': 4,
            'gpu_ceiling': SHARD_GPU_HOURS*sum(chosen.values())+CEILING_MARGIN, 'max_gpu_hours': max_gpu_hours,
            'rows_beyond_shard_capacity': round(uncovered),
            'live_parent_jobs': [job['id'] for job in jobs if job['state'] not in TERMINAL],
            'parent_extraction_jobs': [job['id'] for job in submitted_jobs(root)['extract']]}


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
    lines.append('parent extraction jobs (AFTER="..." when sealing, so the combine waits for them): '
                 + ' '.join(plan['parent_extraction_jobs']))
    return '\n'.join(lines)


def parse_shards(text: str) -> dict[str, int]:
    """Shard counts from "model:count,model:count"."""
    return {model: int(count) for model, count in (item.split(':') for item in text.split(',') if item)}


def main() -> None:
    parser = argparse.ArgumentParser(prog='steering.followup', description=__doc__.split('\n\n')[0])
    parser.add_argument('parent', type=Path, help='the complement release folder, report/NAME')
    parser.add_argument('--max-gpu-hours', type=float, default=APPROVED_GPU_HOURS)
    parser.add_argument('--shards', type=parse_shards, help='model:count,... instead of the computed sizes')
    args = parser.parse_args()
    print(format_plan(plan_followup(args.parent.resolve(), args.max_gpu_hours, args.shards)))


if __name__ == '__main__':
    main()
