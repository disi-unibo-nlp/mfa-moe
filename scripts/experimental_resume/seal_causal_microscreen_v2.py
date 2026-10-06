"""Convert the discovery proposal into strict, runner-bound micro-screen manifests.

The input is observational and exploratory. This command performs no inference.
It writes the executable manifests separately from their audit/provenance receipt.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
PREPARED = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_PREPARED_v0.2.json'
OVERLAY_RECEIPT = ROOT / 'steering-v1/runs/ordered-qualification-v1/PREPARED.cpu-recovery-v2.json'
QUALIFICATION = ROOT / 'steering-v1/runs/ordered-qualification-v1/results-cpu-recovery-v2/QUALIFICATION.json'
PREFIX_SCOUT = ROOT / 'steering-v1/runs/routing-control-v1/dense-discovery/CANDIDATE_PREFIX_SCOUT_v1.json'
EXPERT_SOURCE = ROOT / 'steering-v1/runs/routing-control-v1/dense-verify-source-v2/RESULT.json'
DRIVER = REPO / 'scripts/experimental_resume/run_boundary_micro_screen.py'
EXPECTED_DRIVER_SHA = '1c0c3f0768fd853bc7ce8654b257ccf3cc00c9bcd332ca6f55b93fa5a784816d'
OUT = REPO / 'report/experimental-resume-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if digest({k: v for k, v in value.items() if k != 'sha256'}) != value.get('sha256'):
        raise ValueError('JSON seal differs: ' + str(path))
    return value


def write_sealed(path, body):
    value = {**body, 'sha256': digest(body)}
    if path.exists():
        if sealed(path) != value:
            raise ValueError('existing sealed file differs: ' + str(path))
    else:
        path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
    return value


def make_schedule(rows):
    # Outcome-blind balanced sets within seed and nonrepetition within family.
    if len(rows) % 4:
        raise ValueError('row count must balance four random sets per seed')
    ordered = sorted(rows, key=lambda r: digest(['micro-random-assign-v2', r['family']]))
    schedule = {r['family']: [i % 4, (i + 1) % 4] for i, r in enumerate(ordered)}
    for seed in (0, 1):
        if Counter(pair[seed] for pair in schedule.values()) != Counter({i: len(rows) // 4 for i in range(4)}):
            raise ValueError('random sets are not balanced within seed')
    if any(pair[0] == pair[1] for pair in schedule.values()):
        raise ValueError('one family repeats a random set')
    return schedule


def price(rows, arms):
    contexts = [len(r['prompt_ids']) + len(r['prefix_ids']) for r in rows]
    return {'families': len(rows), 'arms': len(arms), 'seeds': 2,
            'assigned_requests': len(rows) * len(arms) * 2,
            'prefill_tokens_no_cache_reuse': sum(contexts) * len(arms) * 2,
            'maximum_decode_tokens': len(rows) * len(arms) * 2 * 256,
            'context_tokens_min': min(contexts), 'context_tokens_max': max(contexts),
            'maximum_context_tokens_including_decode': max(contexts) + 256,
            'context_tokens_sum': sum(contexts)}


def main(check_only=False):
    prepared = sealed(PREPARED)
    overlay = sealed(OVERLAY_RECEIPT)
    qualification = sealed(QUALIFICATION)
    scout = sealed(PREFIX_SCOUT)
    source = sealed(EXPERT_SOURCE)
    if prepared['sha256'] != '932c088cdc1bd96db5b7e1887bf220dada048abe5696c63c75dc17f112b9b4ef':
        raise ValueError('source proposal differs')
    if file_sha(DRIVER) != EXPECTED_DRIVER_SHA:
        raise ValueError('driver not frozen at reviewed candidate SHA')
    if (prepared['worker_code_digest'] != overlay['sha256'] or
        prepared['qualified_worker_receipt_sha256'] != qualification['sha256'] or
        prepared['prefix_scout_sha256'] != scout['sha256'] or
        prepared['expert_source_sha256'] != source['sha256'] or
        qualification['worker_code_digest'] != overlay['sha256'] or
        qualification['base_tree'] != prepared['base_tree_sha256'] or
        not qualification['pass']):
        raise ValueError('qualified worker overlay differs')
    overlay_dir = Path(overlay['overlay'])
    overlay_files = [overlay_dir / ('moe_exp/routing_control/' + name)
                     for name in ('design.py', 'ordered_vllm.py', 'worker_adapter.py')]
    code_files = {str(DRIVER): EXPECTED_DRIVER_SHA}
    for path in overlay_files:
        if overlay['files'].get(str(path)) != file_sha(path):
            raise ValueError('qualified worker file changed: ' + str(path))
        code_files[str(path)] = file_sha(path)

    rows = [{key: r[key] for key in ('uid', 'question', 'family', 'prompt_ids', 'prefix_ids')}
            for r in prepared['rows']]
    arms = [{**arm, 'role': 'native' if arm['role'] == 'native_duplicate_isolation'
             else 'random' if arm['role'] == 'matched_random' else arm['role']}
            for arm in prepared['arms']]
    actions = [{key: action[key] for key in ('name', 'transition', 'experts', 'bias')}
               for action in prepared['actions']]
    common = {'schema': 'routing-boundary-micro-screen-v1',
              'base_tree_sha256': prepared['base_tree_sha256'],
              'driver_sha256': EXPECTED_DRIVER_SHA,
              'code_files': code_files,
              'family_freeze_sha256': prepared['family_freeze_sha256'],
              'prefix_scout_sha256': scout['sha256'],
              'prepared_sha256': prepared['sha256'],
              'qualified_worker_sha256': qualification['sha256'],
              'actions': actions, 'arms': arms, 'seeds': [0, 1], 'max_tokens': 256}

    full_price = price(rows, arms)
    full_body = {**common, 'stage': 'full', 'rows': rows,
                 'expected_requests': full_price['assigned_requests'],
                 'expected_prefill_tokens': full_price['prefill_tokens_no_cache_reuse'],
                 'maximum_decode_tokens': full_price['maximum_decode_tokens'],
                 'maximum_context_tokens': full_price['maximum_context_tokens_including_decode'],
                 'random_set_by_family_seed': make_schedule(rows)}
    # Cover the observed context-length range while selecting on prefix length only.
    ordered = sorted(rows, key=lambda r: (len(r['prompt_ids']) + len(r['prefix_ids']), r['uid']))
    pilot_rows = [ordered[i] for i in (0, 3, 7, 11)]
    pilot_price = price(pilot_rows, arms)
    pilot_body = {**common, 'stage': 'pilot', 'rows': pilot_rows,
                  'expected_requests': pilot_price['assigned_requests'],
                  'expected_prefill_tokens': pilot_price['prefill_tokens_no_cache_reuse'],
                  'maximum_decode_tokens': pilot_price['maximum_decode_tokens'],
                  'maximum_context_tokens': pilot_price['maximum_context_tokens_including_decode'],
                  'random_set_by_family_seed': make_schedule(pilot_rows)}
    sys.path.insert(0, str(overlay_dir))
    spec = importlib.util.spec_from_file_location('micro_screen_runner_validation', DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_manifest(full_body, DRIVER)
    module.validate_manifest(pilot_body, DRIVER)
    if check_only:
        print(json.dumps({'status': 'VALIDATED_CHECK_ONLY_NO_WRITES',
                          'full_price': full_price, 'qualification_price': pilot_price,
                          'qual4_selected_uids': [r['uid'] for r in pilot_rows],
                          'random_assignment_seed0': dict(Counter(pair[0] for pair in full_body['random_set_by_family_seed'].values())),
                          'random_assignment_seed1': dict(Counter(pair[1] for pair in full_body['random_set_by_family_seed'].values()))}))
        return
    full = write_sealed(OUT / 'CAUSAL_MICROSCREEN_MANIFEST_v2.json', full_body)
    pilot = write_sealed(OUT / 'CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v2.json', pilot_body)

    provenance = [{key: record[key] for key in (
                    'uid', 'family', 'prompt_ids_sha256', 'prefix_ids_sha256',
                    'tokenizer_sha256', 'trace_sha256', 'generation_messages_sha256',
                    'prefix_text_sha256', 'attempt_id', 'sentence_index')}
                  for record in prepared['rows']]
    binding = {'schema': 'causal-micro-screen-manifest-preflight-v1',
               'status': 'PREPARED_HOLD_CONTEXT_MATCHED_GPU_QUALIFICATION_AND_STAGE_PRICE',
               'prepared_path': str(PREPARED), 'prepared_sha256': prepared['sha256'],
               'source_prefix_scout_sha256': prepared['prefix_scout_sha256'],
               'observational_expert_source_sha256': prepared['expert_source_sha256'],
               'qualified_worker_receipt_sha256': prepared['qualified_worker_receipt_sha256'],
               'worker_code_digest': prepared['worker_code_digest'],
               'qualified_worker_overlay_receipt_path': str(OVERLAY_RECEIPT),
               'runner_path': str(DRIVER), 'runner_sha256': EXPECTED_DRIVER_SHA,
               'manifest_builder_path': str(Path(__file__)),
               'manifest_builder_sha256': file_sha(__file__),
               'manifest_full_sha256': full['sha256'],
               'manifest_qualification_sha256': pilot['sha256'],
               'full_price': price(rows, arms), 'qualification_price': price(pilot_rows, arms),
               'qual4_selection_rule': 'sort by prompt+prefix token count then UID, use zero-based indices 0,3,7,11',
               'qual4_selected_uids': [r['uid'] for r in pilot_rows],
               'qual4_random_assignments': pilot_body['random_set_by_family_seed'],
               'provenance': provenance,
               'interpretation': 'experts are proposed by observational dense Verify labels; exact same-prefix interventions and blind semantic ratings are required before any causal claim'}
    receipt = write_sealed(OUT / 'CAUSAL_MICROSCREEN_BINDING_v2.json', binding)
    print(json.dumps({'full_manifest': str(OUT / 'CAUSAL_MICROSCREEN_MANIFEST_v2.json'),
                      'full_sha256': full['sha256'], 'full_price': binding['full_price'],
                      'qualification_manifest': str(OUT / 'CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v2.json'),
                      'qualification_sha256': pilot['sha256'],
                      'qualification_price': binding['qualification_price'],
                      'binding': str(OUT / 'CAUSAL_MICROSCREEN_BINDING_v2.json'),
                      'binding_sha256': receipt['sha256']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    main(check_only=parser.parse_args().check_only)
