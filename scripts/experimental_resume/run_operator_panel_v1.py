"""Slurm-only panel generation, with adapter fixtures outside scientific outcomes."""
import argparse
import fcntl
import os
from pathlib import Path
import time

from qualify_utility_pair_v2 import bootstrap

if __name__ in ('__main__', '__mp_main__'):
    bootstrap()

import operator_panel_v1 as P


def qualify(backend, manifest, out):
    from qualify_utility_pair_v2 import ScriptedController, inspect_case
    from utility_controller_v2 import NativeDecodeStream
    from overnight_routing_runner_v1 import SOURCE
    from utility_controller_interface_v1 import SideRequest
    design = P.U.sealed(P.DOC / 'OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json')
    native = P.U.sealed(SOURCE)['rows'][0]; prompt = native['prompt_ids']
    P.require(native['family'] not in P.U.sealed(P.PLAN)['family_order'], 'engineering fixture overlaps utility')
    out.mkdir(exist_ok=True)
    cases = [('native', None)] + [(op + '-' + t, manifest['policies'][op][t])
            for t in P.TRANSITIONS for op in P.ARMS[1:]]
    cases += [(op + '-early-closure', manifest['policies'][op]['candidate_to_verify']) for op in P.ARMS[1:]]
    checks = []
    safe = backend.tokenizer.encode(' x', add_special_tokens=False)[0]
    # Deterministic fixtures cover a full 256-token pulse, closure, and replay
    # on both ranks without selecting a fixture by scientific outcomes.
    for index, (name, selected) in enumerate(cases):
        ids = [safe] * 384; ids[64 if name.endswith('early-closure') else 320] = P.THINK_END_ID
        transition = selected['actions'][0]['transition'] if selected else None
        a = {'uid': 'panel-engineering|' + P.U.digest([manifest['sha256'], name, os.environ['SLURM_JOB_ID']])[:32],
             'family': native['family'], 'question': native['canonical_question'], 'seed': 0,
             'arm': 'native' if selected is None else selected['arm'], 'prompt_tokens': len(prompt),
             'prompt_token_ids_sha256': P.U.digest(prompt)}
        ctl = ScriptedController(selected, transition, decode=NativeDecodeStream(backend.tokenizer))
        result = backend.generate_one(a, prompt, native['question'], max_tokens=384, controller=ctl,
            qualification=True, qualification_token_ids=ids, qualification_preempt_at=80)
        check = inspect_case(result, backend.telemetry, design)
        if result['controller']['state']['completion_token_ids'] != ids:
            check['failures'].append('forced adapter token sequence differs')
        if selected and not any(check['expected_action_rows'].values()):
            check['failures'].append('operator pulse missing')
        if not result['forced_preemption_receipts']:
            check['failures'].append('recompute missing')
        import numpy as np
        route = out / f'ROUTES_{index:03d}.npz'
        with route.open('xb') as stream:
            np.savez_compressed(stream, routed=result.pop('routed'))
        checks.append(P.save(out / f'CASE_{index:03d}.json', {'schema': 'panel-adapter-case-v1',
            'case': name, 'result': result, 'check': check, 'route_sha256': P.U.file_sha(route)}))
    side = []
    common = 'panel-engineering-side-' + os.environ['SLURM_JOB_ID']
    for physical in ('first', 'second'):
        request = SideRequest(common + '~' + physical, 'engineering-only',
            'candidate_to_verify', {'problem': 'Find x if x + 1 = 3.',
             'emitted_prefix': 'The proposed value is x = 2.', 'triggering_sentence': 'The proposed value is x = 2.'},
            P.U.digest([]), 0)
        side.extend(backend.screen(request))
    for reader in (0, 1):
        first = P.U.sealed(backend.output / f'PANEL_READER_SEED_{common}~first_{reader}.json')
        second = P.U.sealed(backend.output / f'PANEL_READER_SEED_{common}~second_{reader}.json')
        P.require(first['sampler_seed'] == second['sampler_seed'], 'GPU common reader-seed adapter differs')
    failures = [e for c in checks for e in c['check']['failures']]
    if not all(r.finish_reason == 'stop' and type(r.vote) is bool for r in side):
        failures.append('frozen side-reader interface failed')
    result = P.save(out / 'QUALIFICATION.json', {'schema': 'panel-adapter-qualification-v1',
        'manifest_sha256': manifest['sha256'], 'source_files': P.generation_sources(),
        'case_sha256s': [c['sha256'] for c in checks], 'fixture_family': native['family'],
        'side_votes': [r.vote for r in side], 'failures': failures,
        'status': 'PASS_ENGINEERING' if not failures else 'FAIL_ENGINEERING',
        'job_id': os.environ['SLURM_JOB_ID'], 'scope': 'Engineering fixtures only; excluded from utility outcomes.'})
    P.require(result['status'] == 'PASS_ENGINEERING', 'panel adapter did not qualify: ' + repr(failures))
    return result


def run(manifest_path, out, deadline, pilot_index, resume):
    from operator_panel_backend_v1 import PanelBackend
    import numpy as np
    manifest = P.U.sealed(manifest_path); P.validate_manifest(manifest)
    P.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID'), 'panel needs actual GPU Slurm step')
    rows = manifest['rows']
    if pilot_index is not None:
        P.require(pilot_index in (0, 1) and len(manifest['family_order']) == 2, 'invalid pilot index')
        rows = [r for r in rows if r['assignment']['family'] == manifest['family_order'][pilot_index]]
    else:
        rows = [r for r in rows if r['assignment']['family'] in manifest['shards'][int(os.environ['SLURM_ARRAY_TASK_ID'])]]
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'WRITER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('attempts', 'receipts', 'routes', 'allocations', 'engineering'):
        (out / name).mkdir(exist_ok=True)
    binding = P.save(out / 'BINDING.json', {'schema': 'panel-generation-binding-v1',
        'manifest_sha256': manifest['sha256'], 'assigned': [r['assignment'] for r in rows]})
    pending = []
    for row in rows:
        path = out / 'receipts' / (P.U.digest(row['assignment']['uid']) + '.json')
        if path.exists():
            old = P.U.sealed(path)
            P.require(old['assignment'] == row['assignment'] and old['binding_sha256'] == binding['sha256'],
                      'prior committed cell rebound')
        else:
            pending.append(row)
    if not pending:
        return
    job = os.environ['SLURM_JOB_ID']; allocation = out / 'allocations' / job
    P.require(not allocation.exists(), 'allocation replay requires new job ID')
    with PanelBackend(manifest['policies'], allocation, deadline_epoch=deadline) as backend:
        if pilot_index is not None:
            qualification = qualify(backend, manifest, out / 'engineering' / job)
        else:
            qualification = P.U.sealed(Path(manifest['adapter_qualification_path']))
            P.require(qualification['status'] == 'PASS_ENGINEERING' and
                      qualification['sha256'] == manifest['adapter_qualification_sha256'] and
                      qualification['source_files'] == manifest['generation_sources'], 'adapter qualification changed')
        for row in pending:
            if time.time() >= deadline - 60:
                break
            a = row['assignment']; key = P.U.digest(a['uid'])
            prior = sorted((out / 'attempts').glob(key + '-*.json'))
            P.require(not prior or resume, 'uncommitted attempt needs bounded recovery authorization')
            attempt = P.save(out / 'attempts' / f'{key}-{len(prior):03d}.json', {
                'schema': 'panel-generation-attempt-v1', 'assignment': a, 'job_id': job,
                'binding_sha256': binding['sha256'], 'allocation_path': str(allocation),
                'adapter_qualification_sha256': qualification['sha256'],
                'prior_attempt_sha256s': [P.U.sealed(p)['sha256'] for p in prior]})
            failure = False
            try:
                result = backend.generate_one(a, row['original_prompt_ids'], row['problem'])
                route = out / 'routes' / (key + '.npz')
                with route.open('xb') as stream:
                    np.savez_compressed(stream, routed=result.pop('routed'))
                receipt = {'status': 'COMMITTED_GENERATION', 'result': result,
                           'routed_array_sha256': P.U.file_sha(route), 'error': None}
            except Exception as error:
                receipt = {'status': 'GENERATION_ERROR', 'result': getattr(error, 'utility_partial', None),
                           'routed_array_sha256': None, 'error': repr(error)}
                failure = True
            P.save(out / 'receipts' / (key + '.json'), {'schema': 'panel-generation-receipt-v1',
                'assignment': a, 'binding_sha256': binding['sha256'], 'attempt_sha256': attempt['sha256'], **receipt})
            if failure:
                break
    present = [P.U.sealed(out / 'receipts' / (P.U.digest(r['assignment']['uid']) + '.json'))
               for r in rows if (out / 'receipts' / (P.U.digest(r['assignment']['uid']) + '.json')).exists()]
    value = P.save(out / ('SUMMARY-' + job + '.json'), {'schema': 'panel-generation-summary-v1',
        'binding_sha256': binding['sha256'], 'assigned': len(rows), 'committed': len(present),
        'errors': sum(r['status'] == 'GENERATION_ERROR' for r in present), 'missing': len(rows) - len(present),
        'receipt_sha256s': [r['sha256'] for r in present]})
    print('PANEL_GENERATION', value['sha256'], len(present), '/', len(rows), flush=True)
    # Completed driver can contain explicit committed errors/missing cells;
    # downstream always reconciles ITT and terminal accounting rather than exit alone.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--pilot-index', type=int)
    parser.add_argument('--resume-uncommitted', action='store_true')
    args = parser.parse_args()
    run(args.manifest, args.out, args.deadline_epoch, args.pilot_index, args.resume_uncommitted)


if __name__ == '__main__':
    main()
