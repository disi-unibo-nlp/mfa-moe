"""Four-GPU utility engine qualification, separate from semantic discovery.

Dynamic episodes are scripted at token 32 to test exact execution. These are
engineering fixtures, never semantic outcomes or utility population estimates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def bootstrap():
    import sys
    overlay = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/'
                   'steering-v1/addenda/ordered/9727c10299b71e7a/moe_exp_src')
    if 'moe_exp' not in sys.modules:
        from diagnose_mechanism_validation_v3 import pin_qualified_worker
        pin_qualified_worker(overlay)


if __name__ in ('__main__', '__mp_main__'):
    bootstrap()


class ScriptedController:
    def __init__(self, selected, transition, *, trigger=32, decode=None):
        self.selected, self.transition, self.trigger = selected, transition, trigger
        self.state = SimpleNamespace(completion_token_ids=(), finish=None)
        self.episode = None
        self.decode, self.decoded_prefix = decode, ''

    def observe_cumulative(self, emitted_ids, finish_reason=None):
        from utility_scout_v1 import digest
        ids = tuple(emitted_ids)
        if len(ids) != len(self.state.completion_token_ids) + 1 or ids[:-1] != self.state.completion_token_ids:
            raise ValueError('qualification observed hidden or reordered generated tokens')
        self.state.completion_token_ids = ids
        self.state.finish = finish_reason
        if self.decode is not None:
            current = self.decode(ids)
            if not current.startswith(self.decoded_prefix):
                raise ValueError('native streaming decoder replaced previously observed text')
            self.decoded_prefix = current
        if self.selected is None or finish_reason is not None or len(ids) != self.trigger:
            return None
        self.episode = {'transition': self.transition, 'selected_arm': self.selected['arm'],
            'trigger_request_id': 'SCRIPTED_ENGINEERING_FIXTURE', 'trigger_emitted_tokens': len(ids),
            'trigger_emitted_ids_sha256': digest(ids), 'absolute_slots': [len(ids) + x for x in self.selected['slots']],
            'relative_slots': self.selected['slots'], 'action_names': self.selected['action_names'],
            'pulse_width': 256, 'stop_at_reasoning_closure': True}
        return self.episode

    def audit(self):
        return {'state': {'completion_token_ids': list(self.state.completion_token_ids), 'finish': self.state.finish},
                'decoded_prefix': self.decoded_prefix,
                'episode': self.episode, 'qualification_only': True,
                'claim_limit': 'Scripted dynamic-pulse engineering fixture, no semantic acceptance claim.'}


def fixtures(design):
    by_transition = {}
    cases = []
    for transition, arms in design['arms_by_transition'].items():
        target = [arm for arm in arms if arm['role'] == 'target']
        lookup = {a['name']: a for a in design['actions']}
        all_actions = {name: lookup[name] for arm in target for name in arm['policies'][transition]}
        first = target[0]
        by_transition[transition] = {'arm': first['name'], 'slots': first['slots'], 'pulse_width': 256,
            'action_names': first['policies'][transition], 'actions': list(all_actions.values())}
        for arm in target:
            selected = {'arm': arm['name'], 'slots': arm['slots'], 'pulse_width': 256,
                        'action_names': arm['policies'][transition]}
            cases.append({'name': transition + '-' + arm['name'], 'transition': transition,
                          'selected': selected, 'preempt': arm['name'] in ('force', 'reweight', 'repeat')})
    cases = [{'name': 'native-before', 'transition': None, 'selected': None, 'preempt': False},
             *cases, {'name': 'native-after', 'transition': None, 'selected': None, 'preempt': False}]
    cases.append({'name': 'deterministic-closure-recompute', 'transition': 'candidate_to_verify',
                  'selected': {k: v for k, v in by_transition['candidate_to_verify'].items() if k != 'actions'},
                  'preempt': True, 'forced_closure': True})
    return by_transition, cases


def inspect_case(result, telemetry, design):
    import numpy as np
    from utility_routing_worker_v2 import THINK_END_ID
    uid = result['assignment']['uid']
    records = []
    for path in telemetry.glob('rank*.jsonl'):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    records = [r for r in records if r.get('uid') == uid]
    failures = []
    if {r['rank'] for r in records} != {0, 1}:
        failures.append('both TP rank telemetry absent')
    for row in records:
        if any(c['expert_identity_mismatches'] or c['weight_mismatches']
               for c in row.get('inactive_native_checks', {}).values()):
            failures.append('inactive rows changed native routing')
    episode = result['controller']['episode']
    ids = result['controller']['state']['completion_token_ids']
    closure = ids.index(THINK_END_ID) if THINK_END_ID in ids else len(ids)
    expected = {}
    lookup = {a['name']: a for a in design['actions']}
    if episode:
        for slot, name in zip(episode['absolute_slots'], episode['action_names'], strict=True):
            positions = np.arange(slot, min(slot + 256, len(ids), closure + 1), dtype=int)
            expected[name] = expected.get(name, 0) + len(positions)
            if lookup[name]['kind'] == 'force':
                for layer, experts in lookup[name]['experts']:
                    for expert in experts:
                        if len(positions) and not np.all(np.any(result['routed'][positions, layer] == expert, axis=1)):
                            failures.append('forced expert absent in active routed row')
        for rank in (0, 1):
            rank_rows = [r for r in records if r['rank'] == rank]
            actual = {name: sum(r.get('utility_action_rows', {}).get(name, 0) for r in rank_rows) for name in expected}
            if actual != expected:
                failures.append('pulse boundary/count differs on TP rank ' + str(rank))
    if result['forced_preemption_receipts']:
        if not all(r['reset_ok'] for r in result['forced_preemption_receipts']) or any(
                sum(r.get('recompute_rows', 0) for r in records if r['rank'] == rank) == 0 for rank in (0, 1)):
            failures.append('active-request recompute not observed on both ranks')
    return {'uid': uid, 'failures': failures, 'expected_action_rows': expected,
            'closure_observed': closure < len(ids), 'rows_after_closure': max(0, len(ids) - closure - 1),
            'ranks': sorted({r['rank'] for r in records}), 'telemetry': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--startup-import-check', action='store_true')
    parser.add_argument('--prepared-plan', type=Path)
    args = parser.parse_args()
    import utility_scout_v1 as utility
    import rate_overnight_semantics_v2 as storage
    import overnight_routing_runner_v1 as source
    from utility_pair_backend_v2 import FourGPUBackend
    from utility_controller_interface_v1 import SideRequest
    design_path = utility.REPO / 'report/experimental-resume-v1/OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json'
    design = utility.sealed(design_path)
    selected, cases = fixtures(design)
    if args.startup_import_check:
        print('PASS_UTILITY_V2_IMPORTS', len(cases), 'engineering cases; no model load', flush=True)
        return
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('utility qualification requires Slurm GPU step')
    if args.prepared_plan is None:
        raise ValueError('sealed bounded engineering plan required before model loading')
    args.out.mkdir(parents=True, exist_ok=True)
    native = utility.sealed(source.SOURCE)['rows'][0]
    prompt = native['prompt_ids']
    from utility_source_contract_v2 import code_files
    from moe_steer import engine
    import importlib.metadata
    import sys
    prepared = utility.sealed(args.prepared_plan)
    storage.require(prepared['schema'] == 'utility-pair-engineering-plan-v2' and
        prepared['code_files'] == code_files() and prepared['cases'] == cases and
        prepared['design_sha256'] == design['sha256'] and
        prepared['source_enrollment_sha256'] == utility.sealed(source.SOURCE)['sha256'] and
        prepared['original_prompt_sha256'] == utility.digest(prompt), 'engineering plan changed')
    binding = storage.save(args.out / 'BINDING.json', {'schema': 'utility-pair-qualification-binding-v2',
        'design_sha256': design['sha256'], 'source_enrollment_sha256': utility.sealed(source.SOURCE)['sha256'],
        'prepared_plan_sha256': prepared['sha256'],
        'fixture_original_prompt_sha256': utility.digest(prompt), 'fixture_family': native['family'],
        'code_files': code_files(), 'engine_fingerprint': engine.fingerprint(),
        'python_executable': sys.executable, 'vllm_version': importlib.metadata.version('vllm'),
        'maximum_tokens_per_engineering_case': 1024, 'cases': cases,
        'allocated_gpus': 4, 'job_id': os.environ['SLURM_JOB_ID'],
        'profile': {'generator_and_reader_TP': 2, 'in_process_engine_core': True,
                    'max_num_generator_seqs': 1, 'enforce_eager': True, 'async_scheduling': False}})
    checks = []
    with FourGPUBackend(selected, args.out / 'pair', deadline_epoch=args.deadline_epoch) as backend:
        # Exercise the unchanged real semantic-reader prompt/parser and latency
        # independently of scripted pulse timing. This is no detector validation.
        request = SideRequest('utility-qualification-start-pair', 'engineering-only', 'candidate_to_verify',
            {'problem': 'Find x if x + 1 = 3.', 'emitted_prefix': 'The proposed value is x = 2.',
             'triggering_sentence': 'The proposed value is x = 2.'}, utility.digest([]), 0)
        side = backend.screen(request)
        side_ok = all(r.finish_reason == 'stop' and type(r.vote) is bool for r in side)
        storage.save(args.out / 'SIDE_CHECK.json', {'schema': 'utility-side-smoke-v2',
            'binding_sha256': binding['sha256'], 'parsed_votes': [r.vote for r in side],
            'claim_limit': 'Frozen two-reader interface exercised, not semantic detector accuracy validation.'})
        for index, case in enumerate(cases):
            assignment = {'uid': 'utility-qualification-v2|' + utility.digest([binding['sha256'], case])[:24],
                'family': native['family'], 'question': native['canonical_question'], 'seed': 0,
                'arm': 'native' if case['selected'] is None else 'frozen_policy',
                'prompt_tokens': len(prompt), 'prompt_token_ids_sha256': utility.digest(prompt)}
            from utility_controller_v2 import NativeDecodeStream
            ctl = ScriptedController(case['selected'], case['transition'], decode=NativeDecodeStream(backend.tokenizer))
            storage.save(args.out / f'ASSIGNMENT_{index:03d}.json', {'schema': 'utility-qualification-assignment-v2',
                'binding_sha256': binding['sha256'], 'assignment': assignment, 'case': case})
            forced = None
            cap = 1024
            if case.get('forced_closure'):
                cap = 128
                safe_token = backend.tokenizer.encode(' x', add_special_tokens=False)[0]
                forced = [safe_token] * cap
                forced[64] = 248069
            result = backend.generate_one(assignment, prompt, native['question'], max_tokens=cap,
                controller=ctl, qualification=True, qualification_token_ids=forced,
                qualification_preempt_at=(80 if forced else 64) if case['preempt'] else None)
            npz = args.out / f'ROUTES_{index:03d}.npz'
            import numpy as np
            with npz.open('xb') as stream:
                np.savez_compressed(stream, routed=result['routed'])
            check = inspect_case(result, backend.telemetry, design)
            if forced is not None and result['controller']['state']['completion_token_ids'] != forced:
                check['failures'].append('deterministic engineering closure sequence differs')
            if case['selected'] is not None and (result['controller']['episode'] is None or
                                                not any(check['expected_action_rows'].values())):
                check['failures'].append('scripted online pulse did not execute')
            if case['preempt'] and not result['forced_preemption_receipts']:
                check['failures'].append('required forced preemption was not reached')
            record = storage.save(args.out / f'RESULT_{index:03d}.json', {
                'schema': 'utility-qualification-case-v2', 'binding_sha256': binding['sha256'],
                'case': case, 'result': {k: v for k, v in result.items() if k != 'routed'},
                'routed_array_sha256': utility.file_sha(npz), 'check': check})
            checks.append(record)
    failures = [message for c in checks for message in c['check']['failures']]
    if not side_ok:
        failures.append('side-reader pair failed natural-stop boolean parsing or transport')
    if not any(c['check']['rows_after_closure'] > 0 for c in checks if c['case']['selected']):
        failures.append('GPU reasoning-closure cancellation coverage missing')
    result = storage.save(args.out / 'QUALIFICATION.json', {'schema': 'utility-pair-engine-qualification-v2',
        'status': 'PASS_ENGINEERING' if not failures else 'HOLD_FAILED_OR_INCOMPLETE_ENGINEERING',
        'binding_sha256': binding['sha256'], 'case_sha256s': [c['sha256'] for c in checks],
        'failures': failures, 'cases': len(checks), 'semantic_controller_validation': 'EXISTING_DISCOVERY_RUBRIC_ONLY',
        'production_status': 'HOLD_POLICY_BINDING_16K_PRICE_PILOT_AND_COMPLETE_STAGE_PROPOSAL',
        'claim_limit': 'Scripted one-episode execution; native inactive rows, operators, preemption and closure are checked. No semantic control or original-prompt utility result.'})
    print(result['status'], result['sha256'], flush=True)


if __name__ == '__main__':
    main()
