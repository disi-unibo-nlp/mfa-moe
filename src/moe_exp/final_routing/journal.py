"""Verified stage receipts shared by the CLI, workers and finite dispatcher."""
from pathlib import Path
import os
import time
from .common import digest, load, publish, require, verified


def stage_directory(root, card):
    return Path(root)/'stages'/digest(card)


def result(root, card):
    path = stage_directory(root, card)/'RESULT.json'
    return verified(path) if path.exists() else None


def dependency_results(root, card):
    rows = []
    for identity in card.get('required_cards', []):
        path = Path(root)/'stages'/identity/'RESULT.json'
        rows.append(verified(path) if path.exists() else
                    {'status': 'upstream_unavailable', 'missing_receipt': str(path)})
    if card.get('previous_wave') is not None:
        path=Path(root)/'waves'/('%02d'%card['previous_wave'])/'PLAN.json'
        if not path.exists():rows.append({'status':'upstream_unavailable','missing_receipt':str(path)})
        else:
            for identity in verified(path)['terminal_cards']:
                result_path=Path(root)/'stages'/identity/'RESULT.json'
                rows.append(verified(result_path) if result_path.exists() else
                            {'status':'upstream_unavailable','missing_receipt':str(result_path)})
    return rows


def run_stage(binding, card, handler, *, clock=time.time, allocation_deadline=None):
    """A failed prerequisite never calls the stage handler or loads a model."""
    folder = stage_directory(binding['root'], card)
    old = result(binding['root'], card)
    if old:
        return old
    if card.get('recovery_of'):
        original = Path(binding['root'])/'stages'/card['recovery_of']/'RESULT.json'
        if original.exists() and verified(original)['status'] == 'complete':
            return publish(folder/'RESULT.json', {'status':'complete', 'card_sha256':digest(card),
                'reused_original':card['recovery_of'], 'job_id':os.environ.get('SLURM_JOB_ID'), 'model_loaded':False})
    dependencies = dependency_results(binding['root'], card)
    unavailable = [r for r in dependencies if r.get('status') != 'complete']
    if unavailable and card['stage'] not in ('advance', 'guard', 'finalize'):
        return publish(folder/'RESULT.json', {'status': 'upstream_unavailable',
            'card_sha256': digest(card), 'dependencies': dependencies,
            'job_id': os.environ.get('SLURM_JOB_ID'), 'model_loaded': False})
    started = clock()
    publish(folder/'START.json', {'card_sha256': digest(card),
        'job_id': os.environ.get('SLURM_JOB_ID'), 'started_epoch': started})
    try:
        deadline=min(started+card['minutes']*60-60,allocation_deadline or float('inf'))
        detail = handler(binding, card, deadline)
        require(isinstance(detail, dict) and 'status' in detail, 'stage returned no disposition')
        value = {'status': detail['status'], 'detail': detail}
    except Exception as exc:
        publish(folder/'FAILURE.json', {'exception': type(exc).__name__,
            'message': str(exc)[:4000], 'card_sha256': digest(card)})
        value = {'status': 'execution_failed', 'error_type': type(exc).__name__,
                 'error': str(exc)[:4000]}
    return publish(folder/'RESULT.json', {**value, 'card_sha256': digest(card),
        'job_id': os.environ.get('SLURM_JOB_ID'), 'elapsed_seconds': clock()-started})
