"""Separate, sealed discovery experiment; no controller or panel mutations."""
from __future__ import annotations

import json
from pathlib import Path

import utility_scout_v1 as U
from utility_production_v1 import require, save

DOC = U.REPO / 'report/experimental-resume-v1/native-finalization-v1'
ROOT = U.STAGE / 'runs/routing-control-v1/native-finalization-v1'
BASE = U.STAGE / 'code/s1-9a61e32f48c04c24'
MODEL = U.REPO.parent / 'cache/hf/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989'
ARMS = ('original', 'finalization')
ENDPOINTS = ('operational_correct', 'reasoning_tokens', 'tokens')
CAP = 16384
INSTRUCTION = ('Once you have a complete candidate answer, perform one concise independent check '
    'against the original problem. If it passes, give the final answer immediately. '
    'Reconsider the solution only if that check reveals a concrete error.')
FILES = ('native_finalization_v1.py', 'native_finalization_worker_v1.py',
    'run_native_finalization_v1.py', 'run_native_finalization_v1.sbatch',
    'native_finalization_outcomes_v1.py', 'native_finalization_audit_v1.py',
    'native_finalization_operations_v1.py', 'native_finalization_cpu_v1.sbatch',
    'native_finalization_transport_v2.py', 'native_finalization_native_dispatch_v3.py',
    'native_finalization_cpu_v2.sbatch', 'run_native_finalization_v2.sbatch')


def sources():
    paths = [Path(__file__).parent / n for n in FILES]
    paths += [Path(U.__file__), Path(__file__).with_name('utility_production_v1.py'),
              Path(__file__).with_name('operator_panel_outcomes_v1.py')]
    paths += list((BASE / 'moe_steer').glob('*.py'))
    paths += [U.REPO / 'tests/experimental_resume/test_native_finalization_v1.py']
    qualification = U.sealed(U.REPO / 'report/experimental-resume-v1/UTILITY_PAIR_ENGINEERING_PLAN_v2.json')
    require(all(U.file_sha(Path(p)) == h for p, h in qualification['code_files'].items()),
            'qualified scientific sources changed')
    return {**qualification['code_files'], **{str(p): U.file_sha(p) for p in paths}}


def append_instruction(ids, tokenizer):
    """Append inside the final user message, retaining the exact chat suffix."""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    require(tokenizer.encode(text, add_special_tokens=False) == ids, 'original prompt round trip differs')
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n'
    require(text.endswith(suffix) and text.count('<|im_start|>user\n') == 1,
            'unsupported original chat template')
    modified = text[:-len(suffix)] + '\n\n' + INSTRUCTION + suffix
    result = tokenizer.encode(modified, add_special_tokens=False)
    require(tokenizer.decode(result, skip_special_tokens=False) == modified, 'modified prompt differs')
    return result, modified


def assignment_rows(families, representatives, prompts, binding):
    rows = []
    for index, family in enumerate(families):
        for seed in ((0, 1) if index % 2 == 0 else (1, 0)):
            order = ARMS if (index + seed) % 2 == 0 else ARMS[::-1]
            for position, arm in enumerate(order):
                question = representatives[family]
                ids = prompts[question][arm]
                a = {'family': family, 'question': question, 'seed': seed, 'arm': arm,
                     'execution_position': position, 'prompt_tokens': len(ids),
                     'prompt_token_ids_sha256': U.digest(ids), 'maximum_tokens': CAP,
                     'execution_binding_sha256': U.digest(binding)}
                a['uid'] = 'native-finalization-v1|' + U.digest(a)[:32]
                rows.append({'assignment': a, 'prompt_token_ids': ids})
    return rows


def prepare():
    from utility_outcomes_v3 import scoring_modules
    _, engine = scoring_modules()
    from transformers import AutoTokenizer
    from operator_panel_outcomes_v1 import boundary_contract
    from native_finalization_audit_v1 import architecture
    family = U.sealed(U.FAMILY); prompts = U.sealed(U.PROMPTS)
    require(U.file_sha(U.PROMPTS) == family['inputs']['prompts']['sha256'], 'frozen prompts changed')
    pools = family['new_parent_pools']['parent_pools']
    require(len(pools['discovery']) == 48 and len(set(pools['discovery'])) == 48,
            'discovery order differs')
    families = pools['discovery'][:8]
    require(not set(families) & set(pools['utility']), 'utility contamination')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    boundary = boundary_contract(tokenizer, prompts['tokenizer_sha256'])
    prompt_variants, modified = {}, {}
    reps = family['new_parent_pools']['representative_questions']
    for f in families:
        q = reps[f]; p = prompts['questions'][q]; ids = p['prompt_token_ids']
        require(U.digest(ids) == p['prompt_sha256'] and len(ids) == p['prompt_tokens'], 'prompt integrity')
        changed, text = append_instruction(ids, tokenizer)
        prompt_variants[q] = {'original': ids, 'finalization': changed}
        modified[q] = {'text': text, 'prompt_token_ids': changed, 'original_ids_sha256': U.digest(ids)}
    arch = architecture(json.loads((MODEL / 'config.json').read_text()))
    require(arch['routed_experts_per_token'] == 8, 'current model expert count differs')
    binding = {'sources': sources(), 'family_freeze_sha256': family['sha256'],
        'prompt_table_sha256': prompts['sha256'], 'prompt_table_file_sha256': U.file_sha(U.PROMPTS),
        'model': str(MODEL), 'model_files': {str(MODEL / n): U.file_sha(MODEL / n)
            for n in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json')},
        'tokenizer_sha256': prompts['tokenizer_sha256'], 'architecture': arch,
        'sampler': engine.card_sampling(CAP, 0), 'batch_size': 4, 'tensor_parallel_size': 2,
        'routing': 'native; architecture-defined expert count; no interventions', 'instruction': INSTRUCTION}
    value = save(DOC / 'MANIFEST.json', {'schema': 'native-finalization-manifest-v1',
        'binding': binding, 'family_order': families, 'seeds': [0, 1], 'arms': list(ARMS),
        'boundary': boundary, 'rows': assignment_rows(families, reps, prompt_variants, binding),
        'assignment_count': 32, 'maximum_tokens': CAP,
        'order_rule': 'Alternate seed blocks by family parity; arm order by (family_index+seed)%2.',
        'inference': {'endpoints': list(ENDPOINTS), 'replicates': 50000, 'seed': 20261005,
            'method': 'Equal-family mean of two paired seed differences; simultaneous fixed-SE max-deviation family bootstrap.',
            'claim_limit': 'Discovery only; no equivalence or accuracy-retention claim.'},
        'budget': {'generation': {'partition': 'boost_usr_prod', 'qos': 'normal', 'gpus': 2,
            'cpus': 16, 'memory_GiB': 120, 'wall_seconds': 28800},
            'cpu': {'partition': 'lrd_all_viz', 'qos': 'normal', 'cpus': 2,
                    'memory_GiB': 32, 'wall_seconds': 3600}},
        'offline_reference_policy': 'Gold remains in frozen question table; never passed to generator.',
        'continuation_policy': '256/1024-token saved continuations remain separate.',
        'automatic_downstream_experiments': False})
    save(DOC / 'MODIFIED_PROMPTS.json', {'schema': 'native-finalization-prompts-v1',
        'manifest_sha256': value['sha256'], 'instruction': INSTRUCTION, 'questions': modified})
    validate(value)
    return value


def validate(manifest):
    require(manifest['schema'] == 'native-finalization-manifest-v1' and
            manifest['binding']['sources'] == sources(), 'manifest/source binding changed')
    b = manifest['binding']
    require(all(U.file_sha(Path(p)) == h for p, h in b['model_files'].items()), 'model/tokenizer files changed')
    f = U.sealed(U.FAMILY); p = U.sealed(U.PROMPTS)
    require(b['family_freeze_sha256'] == f['sha256'] and b['prompt_table_sha256'] == p['sha256'] and
            b['prompt_table_file_sha256'] == U.file_sha(U.PROMPTS) and
            manifest['family_order'] == f['new_parent_pools']['parent_pools']['discovery'][:8],
            'source enrollment changed')
    rows = manifest['rows']; variants = {}
    for row in rows:
        a = row['assignment']; ids = row['prompt_token_ids']
        require(U.digest(ids) == a['prompt_token_ids_sha256'] and len(ids) == a['prompt_tokens'], 'prompt changed')
        variants.setdefault(a['question'], {})[a['arm']] = ids
        if a['arm'] == 'original':
            require(ids == p['questions'][a['question']]['prompt_token_ids'], 'original prompt changed')
    require(len(rows) == 32 and rows == assignment_rows(manifest['family_order'],
        f['new_parent_pools']['representative_questions'], variants, b), 'factorial/order changed')
    require(manifest['maximum_tokens'] == CAP and b['instruction'] == INSTRUCTION and
            b['tensor_parallel_size'] == 2 and b['batch_size'] == 4, 'experiment changed')


def receipt_path(root, assignment):
    return Path(root) / 'receipts' / (U.digest(assignment['uid']) + '.json')


def reconcile(manifest, root):
    """Only committed exact receipts are reused; uncertain attempts are never rerun."""
    records = []
    for row in manifest['rows']:
        a = row['assignment']; path = receipt_path(root, a)
        attempt = Path(root) / 'attempts' / (U.digest(a['uid']) + '.json')
        if path.exists():
            r = U.sealed(path)
            require(r['assignment'] == a and r['manifest_sha256'] == manifest['sha256'] and
                    r['status'] in ('COMMITTED_GENERATION', 'GENERATION_ERROR'), 'receipt rebound')
            t = U.sealed(attempt)
            require(r['attempt_sha256'] == t['sha256'] and t['assignment'] == a and
                    t['manifest_sha256'] == manifest['sha256'], 'attempt binding changed')
            records.append({'assignment': a, 'status': r['status'], 'receipt_path': str(path),
                            'receipt_sha256': r['sha256'], 'attempt_path': str(attempt)})
        else:
            if attempt.exists():
                t = U.sealed(attempt)
                require(t['assignment'] == a and t['manifest_sha256'] == manifest['sha256'], 'uncommitted attempt rebound')
            records.append({'assignment': a, 'status': 'MISSING', 'receipt_path': None,
                'receipt_sha256': None, 'attempt_path': str(attempt) if attempt.exists() else None})
    return {'schema': 'native-finalization-reconciled-v1', 'manifest_sha256': manifest['sha256'],
            'family_order': manifest['family_order'], 'records': records,
            'uncommitted_attempts': sum(r['status'] == 'MISSING' and r['attempt_path'] is not None for r in records)}
