"""Resolve reader pricing from qualified existing maximum-input observations."""
from pathlib import Path
from moe_exp.research_campaign_v3_r6.contracts import load, publish, require
from moe_exp.research_campaign_v3_r6.reader_contract import ReaderUnits


def check_binding(campaign):
    from moe_exp.research_campaign_v3_r6.contracts import check_files
    from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1.policy import require_readiness
    root = Path(campaign['root'])
    cpu_proof = require_readiness(campaign)
    lookup = load(root / 'LOOKUP_R1_BINDING.json')
    lookup_proof = load(root / 'LOOKUP_R1_PREFLIGHT.json')
    allocation = load(root / 'CPU_ALLOCATION_R1_BINDING.json')
    allocation_proof = load(root / 'CPU_ALLOCATION_R1_PREFLIGHT.json')
    binding = load(root / 'READER_PRICE_R1_BINDING.json')
    for previous in (lookup, allocation, binding):
        check_files(previous['files'])
        require(previous['campaign_sha256'] == campaign['sha256'], 'Amendment campaign differs')
    check_files(binding['evidence_files'])
    require(lookup_proof['passed'] and lookup_proof['binding_sha256'] == lookup['sha256']
            and lookup_proof['cpu_dispatch_proof_sha256'] == cpu_proof['sha256']
            and allocation['lookup_binding_sha256'] == lookup['sha256']
            and allocation_proof['passed'] and allocation_proof['binding_sha256'] == allocation['sha256']
            and allocation_proof['lookup_proof_sha256'] == lookup_proof['sha256']
            and binding['cpu_allocation_binding_sha256'] == allocation['sha256']
            and binding['allocation_proof_sha256'] == allocation_proof['sha256']
            and binding['lookup_binding_sha256'] == lookup['sha256'], 'Inherited scientific proofs differ')
    require(binding['new_scientific_draws'] == binding['additional_gpu_allocations'] == 0
            and binding['remote_writes'] is False and binding['manuscript_writes'] is False,
            'Existing-evidence pricing scope differs')
    return binding


def measured_price(campaign, rows, observations, coverage, validated_original):
    require(len(rows) == len(observations) == 64, 'Full measured coverage batch required')
    require(len({r['uid'] for r in rows}) == 64, 'Distinct coverage assignments required')
    original_max = max(v['input_tokens'] for v in validated_original if v['context_eligible'])
    measured = []
    a = campaign['annotator']
    for row, value in zip(rows, observations):
        require(value and value.get('assignment') == row and value['status'] == 'complete',
                'Qualified original coverage disposition required')
        require(value['configuration'] == a['config'] and value['reader_revision'] == a['revision']
                and value['native_context'] == a['native_context']
                and value['engine_changes'] == a['engine_changes'], 'Reader profile/runtime differs')
        require(value['seconds'] > 0 and value['generated_tokens'] > 0 and value['load_seconds'] >= 0,
                'Actual execution timing required')
        measured.append({k: value[k] for k in ('seconds', 'generated_tokens', 'input_tokens', 'status', 'load_seconds')})
        measured[-1].update(model=row['model'], uid=row['uid'], bytes=len(str(value).encode()))
    maximum = max(v['input_tokens'] for v in measured)
    require(coverage['maximum_input_covered'] and maximum >= coverage['maximum_required_input_tokens']
            and maximum >= original_max, 'Existing measurements do not cover required original inputs')
    upper = max(v['seconds'] / v['generated_tokens'] * a['config']['max_tokens'] for v in measured)
    return {'status': 'complete', 'records': measured,
            'seconds_per_context_upper': upper,
            'load_seconds': max(v['load_seconds'] for v in measured),
            'bytes_per_context_upper': max(v['bytes'] for v in measured) * 4,
            'table_sha256': coverage['original_table_sha256'],
            'measured_table_sha256': coverage['supplemental_table_sha256'],
            'coverage_sha256': coverage['sha256'], 'maximum_input_covered': True,
            'maximum_measured_input_tokens': maximum, 'maximum_original_input_tokens': original_max,
            'original_assignments_retained': True, 'original_price_only_assignments_unexecuted': 64,
            'pricing_evidence': 'actual qualified largest-input coverage; original unexecuted outcomes remain unknown',
            'new_scientific_draws': 0}


def reconcile(campaign):
    root = Path(campaign['root'])
    binding = check_binding(campaign)
    coverage = load(root / 'ANNOTATOR_PRICE_COVERAGE.json')
    table = load(root / 'ANNOTATOR_PRICE_COVERAGE_TASKS.json')
    original = load(root / 'ANNOTATOR_PRICE_TASKS.json')
    validated = load(root / 'READER_INPUT_VALIDATION.json')
    require(len(original['records']) == len(validated['records']) == 64 and
            {r['uid'] for r in original['records']} == {r['uid'] for r in validated['records']},
            'Original retained roster differs from actual tokenizer/input validation')
    require(coverage['original_table_sha256'] == original['sha256'] and
            coverage['supplemental_table_sha256'] == table['sha256'] and
            coverage['outcomes_used'] is False and table['outcomes_used'] is False,
            'Outcome-blind input-coverage tables differ')
    require(all(ReaderUnits(root / r['model'] / 'annotations').state(r['uid']) is None
                for r in original['records']), 'Original assignment was executed or ambiguously admitted')
    failure = load(root / 'stages/reader_price-all-00-00000/RESULT.json')
    require(failure['sha256'] == binding['resolved_failure_sha256'] and
            failure['status'] != 'complete' and failure['error'] == 'original annotator execution price unavailable',
            'Only the preserved final pricing assertion is resolved')
    observations = [ReaderUnits(root / r['model'] / 'annotations').state(r['uid']) for r in table['records']]
    price = publish(root / 'ANNOTATOR_PRICE.json', measured_price(campaign, table['records'], observations,
                                                               coverage, validated['records']))
    return publish(root / 'READER_PRICE_R1_RESOLUTION.json', {
        'complete': True, 'price_sha256': price['sha256'], 'resolved_failure_sha256': failure['sha256'],
        'binding_sha256': binding['sha256'], 'failed_job_ids': ['60564472', '60566412'],
        'additional_gpu_allocations': 0, 'scheduler_outcomes_unchanged': True,
        'original_failure_not_deleted': True, 'measured_coverage_assignments_reused': 64,
        'original_assignments_unexecuted': 64, 'new_scientific_draws': 0,
        'original_table_sha256': original['sha256'], 'measured_table_sha256': table['sha256'],
        'actual_reader_input_validation_sha256': validated['sha256'],
    })


def resolved_stage(value, resolved):
    if value and value.get('sha256') == resolved['resolved_failure_sha256']:
        return {**value, 'status': 'complete', 'failure_scope': None,
                'source_bound_resolution_sha256': resolved['sha256'],
                'historical_failed_allocation_preserved': True}
    return value


def remaining_block(campaign, stage, model, resolved):
    for path in (Path(campaign['root']) / 'failure_dispositions').glob('*.json'):
        receipt = load(path)
        if receipt['failure'].get('sha256') == resolved['resolved_failure_sha256']:
            continue
        if stage in receipt['dependent_stages'] and (receipt['scope'] == 'shared' or receipt['model'] == model):
            return receipt
    return None


def install_resolution(campaign):
    root = Path(campaign['root'])
    resolved = load(root / 'READER_PRICE_R1_RESOLUTION.json')
    binding = check_binding(campaign)
    require(resolved['complete'] and resolved['binding_sha256'] == binding['sha256']
            and resolved['resolved_failure_sha256'] == binding['resolved_failure_sha256']
            and load(root / 'ANNOTATOR_PRICE.json')['sha256'] == resolved['price_sha256'],
            'Verified existing reader price resolution required')
    from moe_exp.research_campaign_v3_r6 import controller, failure_domains
    native_state, native_blocked = controller._stage_state, failure_domains.blocked
    def state(directory, stage, model='all', shard=0, threshold=0):
        value = native_state(directory, stage, model, shard, threshold)
        if Path(directory).resolve() == root.resolve() and (stage, model, shard, threshold) == ('reader_price', 'all', 0, 0):
            return resolved_stage(value, resolved)
        return value
    def blocked(value, stage, model):
        if Path(value['root']).resolve() != root.resolve():
            return native_blocked(value, stage, model)
        return remaining_block(value, stage, model, resolved)
    controller._stage_state = state
    failure_domains.blocked = blocked
