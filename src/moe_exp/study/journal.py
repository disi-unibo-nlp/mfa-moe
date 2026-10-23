"""Verified stage receipts shared by the CLI, workers and finite dispatcher."""
from pathlib import Path
import os
import time
import traceback
from moe_exp.io import digest, load, publish, require, verified, control_root


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
    folder = stage_directory(control_root(binding), card)
    old = result(control_root(binding), card)
    if old:
        return old
    failure = folder/'FAILURE.json'
    if failure.exists():
        saved = verified(failure)
        return publish(folder/'RESULT.json', {'status': 'execution_failed',
            'card_sha256': digest(card), 'error_type': saved['exception'],
            'error': saved['message'], 'recovered_failure_sha256': saved['sha256']})
    if card.get('recovery_of'):
        original = control_root(binding)/'stages'/card['recovery_of']/'RESULT.json'
        if original.exists() and verified(original)['status'] == 'complete':
            return publish(folder/'RESULT.json', {'status':'complete', 'card_sha256':digest(card),
                'reused_original':card['recovery_of'], 'job_id':os.environ.get('SLURM_JOB_ID'), 'model_loaded':False})
    dependencies = dependency_results(control_root(binding), card)
    unavailable = [r for r in dependencies if r.get('status') != 'complete']
    if unavailable and card['stage'] not in ('advance', 'guard', 'finalize'):
        return publish(folder/'RESULT.json', {'status': 'upstream_unavailable',
            'card_sha256': digest(card), 'dependencies': dependencies,
            'job_id': os.environ.get('SLURM_JOB_ID'), 'model_loaded': False})
    started = clock()
    first_start = folder/'START.json'
    if first_start.exists():
        require(verified(first_start)['card_sha256'] == digest(card), 'stage start identity changed')
    else:
        publish(first_start, {'card_sha256': digest(card),
            'job_id': os.environ.get('SLURM_JOB_ID'), 'started_epoch': started})
    attempt = folder/'attempts'/('%s-%s' % (os.environ.get('SLURM_JOB_ID', 'local'),
                                           os.environ.get('SLURM_RESTART_COUNT', '0')))/'START.json'
    if not attempt.exists():
        publish(attempt, {'card_sha256': digest(card), 'started_epoch': started})
    try:
        deadline=min(started+card['minutes']*60-60,allocation_deadline or float('inf'))
        detail = handler(binding, card, deadline)
        require(isinstance(detail, dict) and 'status' in detail, 'stage returned no disposition')
        value = {'status': detail['status'], 'detail': detail}
    except Exception as exc:
        publish(folder/'FAILURE.json', {'exception': type(exc).__name__,
            'message': str(exc)[:4000], 'traceback': traceback.format_exc()[-16000:],
            'card_sha256': digest(card)})
        value = {'status': 'execution_failed', 'error_type': type(exc).__name__,
                 'error': str(exc)[:4000]}
    return publish(folder/'RESULT.json', {**value, 'card_sha256': digest(card),
        'job_id': os.environ.get('SLURM_JOB_ID'), 'elapsed_seconds': clock()-started})
