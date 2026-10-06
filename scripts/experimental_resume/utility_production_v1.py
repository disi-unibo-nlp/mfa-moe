"""Versioned production enrollment, measured pricing and exact receipt contract.

This module never selects actions, changes the frozen controller or loads a
model. The requested allocation ceiling is distinct from a runtime forecast.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import utility_scout_v1 as U

REPO = U.REPO
DOC = REPO / 'report/experimental-resume-v1'
SCRIPTS = REPO / 'scripts/experimental_resume'
RUNS = U.STAGE / 'runs/routing-control-v1'
FILES = ('utility_production_v1.py', 'run_utility_production_v1.py',
         'dispatch_utility_production_v1.py', 'run_utility_production_v1.sbatch',
         'dispatch_utility_production_v1.sbatch', 'dispatch_overnight_readers_v1.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, body):
    """Immutable atomic JSON commit; callers hold a stage or shard writer lock."""
    path = Path(path)
    value = U.seal(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(U.sealed(path) == value, 'immutable artifact differs: ' + str(path))
        return value
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.' + str(time.time_ns()))
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=1)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    # A hard link makes the final commit no-clobber even if a caller mislocks.
    try:
        os.link(temporary, path)
    except FileExistsError:
        require(U.sealed(path) == value, 'concurrent artifact differs: ' + str(path))
    finally:
        temporary.unlink()
    return value


def source_files():
    return {str(SCRIPTS / name): U.file_sha(SCRIPTS / name) for name in FILES}


def validate_config(config):
    require(config['schema'] == 'utility-production-config-v1' and
            config['source_files'] == source_files() and
            config['gpus_per_shard'] == 4 and config['max_tokens'] == 16384 and
            type(config['maximum_families_per_shard']) is int and 1 <= config['maximum_families_per_shard'] <= 8 and
            type(config['maximum_concurrent_shards']) is int and 1 <= config['maximum_concurrent_shards'] <= 96 and
            3600 <= config['wall_seconds'] <= 86400 and config['shutdown_margin_seconds'] == 600 and
            1 <= config['stress_factor'] <= 10 and config['side_query_count_cap'] is None and
            config['automatic_retry_committed_errors'] is False and
            config['maximum_infrastructure_recovery_waves'] == 1 and
            0 < config['infrastructure_recovery_reserve_fraction'] <= 1 and
            math.isfinite(config['offline_grading_reserve_gpu_hours']) and
            config['offline_grading_reserve_gpu_hours'] >= config['offline_grading_envelope']['requested_allocation_GPU_hour_ceiling'] and
            (config['available_generation_billing_core_hours'] is None or
             (math.isfinite(config['available_generation_billing_core_hours']) and config['available_generation_billing_core_hours'] > 0)),
            'production configuration or reviewed source files changed')
    proposal = U.sealed(Path(config['pilot_proposal_path']))
    require(proposal['sha256'] == config['pilot_proposal_sha256'], 'pilot proposal changed')
    measurement = U.sealed(Path(config['offline_measurement_plan_path']))
    require(measurement['sha256'] == config['offline_measurement_plan_sha256'], 'offline measurement plan changed')
    envelope = U.sealed(Path(config['offline_grading_envelope_path']))
    require(envelope == config['offline_grading_envelope'] and envelope['schema'] == 'utility-j1-max384-envelope-v2' and
            envelope['maximum_items'] == 384 and envelope['maximum_prompt_tokens_per_item'] == 16384 and
            envelope['outcome_measurement_plan_sha256'] == measurement['sha256'], 'maximum offline J1 envelope changed')
    require(U.file_sha(Path(config['offline_measured_reference_source'])) == config['offline_measured_reference_sha256'],
            'measured J1 reference changed')
    scientific = U.sealed(DOC / 'UTILITY_PAIR_ENGINEERING_PLAN_v2.json')['code_files']
    require(scientific == config['scientific_code_files'] and
            all(U.file_sha(Path(path)) == digest for path, digest in scientific.items()),
            'qualified 208-file scientific closure changed')


def validate_receipt(receipt, assignment, binding_sha, route_path=None):
    require(receipt['schema'] in ('utility-price-pilot-receipt-v2', 'utility-production-generation-receipt-v1') and
            receipt['assignment'] == assignment and receipt['binding_sha256'] == binding_sha and
            receipt['status'] in ('COMMITTED_GENERATION', 'GENERATION_ERROR'),
            'receipt identity, binding or status differs')
    result = receipt.get('result')
    if receipt['status'] == 'GENERATION_ERROR':
        require(bool(receipt.get('error')) and receipt.get('routed_array_sha256') is None,
                'error receipt lacks explicit failure')
        return
    require(result and receipt.get('error') is None and result['assignment'] == assignment and
            result['qualification_only'] is False and result['maximum_tokens'] == 16384,
            'production receipt is not a clean original-prompt request')
    controller = result['controller']
    state = controller['state']
    ids, sources = state['completion_token_ids'], state['token_sources']
    require(state['uid'] == assignment['uid'] and
            state['prompt_token_ids_sha256'] == assignment['prompt_token_ids_sha256'] and
            len(ids) == len(sources) <= 16384 and all(type(x) is int and x >= 0 for x in ids) and
            all(source == 'emitted' for source in sources) and
            state['finish'] in ('stop', 'length') and
            (state['finish'] != 'length' or len(ids) == 16384) and
            (assignment['arm'] != 'native' or (controller['episode'] is None and not controller['queries'])),
            'receipt token history, native behavior or stop condition differs')
    if route_path is not None:
        require(Path(route_path).is_file() and U.file_sha(Path(route_path)) == receipt['routed_array_sha256'],
                'routed array bytes differ')


def load_pilot(config, attachment):
    """Validate all eight pilot cells; preserve original artifacts, never copy outcomes."""
    validate_config(config)
    attachment = Path(attachment)
    proposal = U.sealed(Path(config['pilot_proposal_path']))
    manifest = U.sealed(attachment / 'PILOT_MANIFEST.json')
    policy = U.sealed(attachment / 'SELECTED_POLICY.json')
    chain = U.sealed(attachment / 'PILOT_CHAIN.json')
    accounting = U.sealed(attachment / 'PILOT_ACCOUNTING.json')
    plan = U.sealed(Path(proposal['utility_plan_path']))
    U.validate_plan(plan)
    qualification = U.sealed(Path(proposal['qualification_result']))
    qual_binding = U.sealed(Path(proposal['qualification_result']).parent / 'BINDING.json')
    require(qualification['status'] == 'PASS_ENGINEERING' and
            qualification['binding_sha256'] == qual_binding['sha256'] and
            qual_binding['prepared_plan_sha256'] == proposal['engineering_plan_sha256'] and
            qual_binding['code_files'] == config['scientific_code_files'], 'production engine is not qualified')
    require(manifest['schema'] == 'utility-price-pilot-manifest-v2' and
            manifest['plan_sha256'] == plan['sha256'] and
            manifest['policy_sha256'] == policy['sha256'] == manifest['policy']['sha256'] and
            manifest['qualification_sha256'] == qualification['sha256'] and
            manifest['code_files'] == config['scientific_code_files'] and
            manifest['family_order'] == plan['family_order'][:2] and
            [row['assignment'] for row in manifest['rows']] == plan['assignments'][:8] and
            chain['proposal_sha256'] == accounting['proposal_sha256'] == proposal['sha256'] and
            chain['pilot_manifest_sha256'] == manifest['sha256'] and
            accounting['pilot_chain_sha256'] == chain['sha256'] and
            accounting['pilot_job'] == chain['pilot_job'] and accounting['allocated_gpus'] == 4,
            'pilot assignment, policy, source or allocation provenance changed')
    output = Path(chain['output'])
    binding = U.sealed(output / 'BINDING.json')
    require(binding['manifest_sha256'] == manifest['sha256'] and
            binding['policy_sha256'] == policy['sha256'], 'pilot output rebound')
    imports, receipts = [], []
    for row in manifest['rows']:
        assignment = row['assignment']
        key = U.digest(assignment['uid'])
        path = output / 'receipts' / (key + '.json')
        require(path.exists(), 'pilot incomplete: missing assignment ' + assignment['uid'])
        receipt = U.sealed(path)
        route = output / 'routes' / (key + '.npz')
        validate_receipt(receipt, assignment, binding['sha256'], route if receipt['status'] == 'COMMITTED_GENERATION' else None)
        attempts = [U.sealed(p) for p in sorted((output / 'attempts').glob(key + '-*.json'))]
        require(any(a['sha256'] == receipt['attempt_sha256'] for a in attempts) and
                all(a['assignment'] == assignment and a['binding_sha256'] == binding['sha256'] and
                    a['job_id'] == chain['pilot_job'] for a in attempts),
                'pilot attempt provenance differs')
        imports.append({'assignment': assignment, 'origin': 'imported_pilot', 'status': receipt['status'],
            'receipt_path': str(path), 'receipt_sha256': receipt['sha256'],
            'source_manifest_sha256': manifest['sha256'], 'source_binding_sha256': binding['sha256'],
            'routed_path': str(route) if receipt['status'] == 'COMMITTED_GENERATION' else None,
            'routed_array_sha256': receipt['routed_array_sha256'],
            'attempt_provenance': [{'path': str(p), 'sha256': U.sealed(p)['sha256']}
                                   for p in sorted((output / 'attempts').glob(key + '-*.json'))]})
        receipts.append(receipt)
    loads = [U.sealed(p) for p in sorted((output / 'allocations').glob('*/PAIR_LOAD.json'))]
    costs = [U.sealed(p) for p in sorted((output / 'allocations').glob('*/PAIR_COST.json'))]
    require(len(loads) <= 1 and len(costs) <= 1,
            'pilot has multiple allocations; all-allocation accounting is required before production pricing')
    return {'plan': plan, 'policy': policy, 'policy_path': str(attachment / 'SELECTED_POLICY.json'),
            'qualification': qualification, 'qualification_binding': qual_binding,
            'manifest': manifest, 'imports': imports, 'receipts': receipts, 'loads': loads, 'costs': costs,
            'accounting': accounting, 'chain': chain, 'attachment': str(attachment)}


def measured_profile(pilot):
    """A conservative empirical stress reference, never a mathematical bound."""
    reasons = []
    accounting = pilot['accounting']
    if accounting['status'] != 'COMPLETE_UNGRADED_RUNTIME_PILOT':
        reasons.append('pilot allocation did not complete all assigned cells successfully')
    if not pilot['loads'] or not pilot['costs']:
        reasons.append('joint model load or shutdown accounting is missing')
    rows, reader_rates = [], []
    for receipt in pilot['receipts']:
        if receipt['status'] != 'COMMITTED_GENERATION':
            reasons.append('pilot includes a committed generation error; no complete runtime measurement')
            continue
        result, assignment = receipt['result'], receipt['assignment']
        ctl = result['controller']
        tokens = len(ctl['state']['completion_token_ids'])
        queries = ctl['queries']
        side_seconds = sum(r['elapsed_seconds'] for q in queries for r in q['results'])
        total_seconds = result['generation_wall_seconds']
        require(all(math.isfinite(v) and v >= 0 for v in (side_seconds, total_seconds)) and
                side_seconds <= total_seconds + 1., 'nonfinite or inconsistent pilot timing')
        if not tokens:
            reasons.append('pilot produced an empty completion; generator rate is not measurable')
            continue
        for query in queries:
            require(len(query['results']) == 2, 'side query omitted a reader')
            for reader in query['results']:
                if reader['prompt_tokens'] > 0 and reader['generated_tokens'] > 0:
                    reader_rates.append({'seconds': reader['elapsed_seconds'],
                        'prompt_tokens': reader['prompt_tokens'], 'generated_tokens': reader['generated_tokens']})
        rows.append({'uid': assignment['uid'], 'family': assignment['family'], 'arm': assignment['arm'],
            'emitted_tokens': tokens, 'prompt_tokens': assignment['prompt_tokens'],
            'generator_and_controller_seconds': max(0., total_seconds - side_seconds),
            'total_request_seconds': total_seconds, 'side_seconds': side_seconds,
            'side_queries': len(queries), 'nonfire': ctl['episode'] is None,
            'side_prompt_tokens': ctl['side_prompt_tokens'], 'side_generated_tokens': ctl['side_generated_tokens']})
    policies = [row for row in rows if row['arm'] == 'frozen_policy']
    natives = [row for row in rows if row['arm'] == 'native']
    if len(policies) != 4 or len(natives) != 4:
        reasons.append('four policy and four native runtime measurements are required')
    if not reader_rates or not any(row['side_queries'] for row in policies):
        reasons.append('no measured production side-reader work; an additional frozen-policy runtime probe is needed')
    if reasons:
        return {'status': 'HOLD_INCOMPLETE_MEASURED_PROFILE', 'reasons': sorted(set(reasons)), 'rows': rows}
    generator_rate = max(row['generator_and_controller_seconds'] /
                         (row['emitted_tokens'] + row['prompt_tokens']) for row in rows)
    # Both factors are explicit stress extrapolations. They do not impose a query
    # limit or claim that unobserved families cannot be slower.
    queries_per_16k = max(math.ceil(row['side_queries'] * 16384 / row['emitted_tokens']) for row in policies)
    max_reader_seconds = max(row['seconds'] * 1024 / row['generated_tokens'] for row in reader_rates)
    max_prompt = max(row['prompt_tokens'] for row in reader_rates)
    context_multiplier = max(1., (49152 - 1024) / max_prompt)
    projected_pair_seconds = 2 * max_reader_seconds * context_multiplier
    load_seconds = max(row['load_wall_seconds'] for row in pilot['loads'])
    measured_request_seconds = sum(row['total_request_seconds'] for row in rows)
    allocated = accounting['elapsed_seconds']
    other_seconds = max(0., allocated - measured_request_seconds - sum(row['load_wall_seconds'] for row in pilot['loads']))
    return {'status': 'MEASURED_STRESS_REFERENCE', 'reasons': [], 'rows': rows,
        'generator_seconds_per_prompt_or_emitted_token': generator_rate,
        'side_queries_per_16k_policy_request_stress': queries_per_16k,
        'side_pair_seconds_stress': projected_pair_seconds,
        'side_reader_context_multiplier': context_multiplier,
        'maximum_observed_side_prompt_tokens': max_prompt,
        'maximum_output_normalized_reader_seconds': max_reader_seconds,
        'joint_cold_load_seconds': load_seconds, 'shutdown_and_allocation_overhead_seconds': other_seconds,
        'pilot_allocated_gpu_hours': accounting['actual_allocated_gpu_hours'],
        'pilot_accounting_sha256': accounting['sha256'],
        'pilot_receipt_sha256s': [r['sha256'] for r in pilot['receipts']],
        'scope': 'Max observed per-token generator time and query density, with reader output/context stress scaling. Empirical projection from two fixed families; not a worst-case bound or statistical upper confidence limit.'}


def price_stage(plan, imports, profile, config):
    U.validate_plan(plan)
    imported = {row['assignment']['uid'] for row in imports}
    expected = {row['uid'] for row in plan['assignments']}
    require(len(imported) == len(imports) and imported <= expected, 'foreign or duplicate pilot assignment')
    remaining = [row for row in plan['assignments'] if row['uid'] not in imported]
    offline_envelope = config['offline_grading_envelope']
    base = {'schema': 'utility-production-price-v1', 'plan_sha256': plan['sha256'],
            'config_sha256': config['sha256'], 'profile': profile,
            'remaining_assignments': len(remaining), 'imported_assignments': len(imported),
            'maximum_remaining_emitted_tokens': len(remaining) * 16384,
            'remaining_original_prefill_tokens': sum(row['prompt_tokens'] for row in remaining),
            'gpus_per_shard': 4, 'completion_guarantee': False,
            'grading_status': 'SEPARATE_OFFLINE_STAGE_NOT_LAUNCHED_OR_CHARGED_IN_THIS_GENERATION_PRICE',
            'offline_measurement': {'plan_sha256': config['offline_measurement_plan_sha256'],
                'maximum_strict_rejected_natural_stop_items': 384, 'votes_per_item': 3,
                'maximum_output_tokens_per_vote': 8192, 'maximum_judge_output_tokens': 384 * 3 * 8192,
                'maximum_transport_attempts_per_vote': 3,
                'maximum_all_attempt_judge_output_tokens': 384 * 3 * 8192 * 3,
                'reserved_gpu_hours': config['offline_grading_reserve_gpu_hours'],
                'maximum_envelope_path': config['offline_grading_envelope_path'],
                'maximum_envelope_sha256': offline_envelope['sha256'],
                'median_full_cap_all_attempts_projected_gpu_hours': offline_envelope['median_full_cap_all_attempts_projected_GPU_h'],
                'historical_item_rate_forecast_gpu_hours': offline_envelope['historical_item_rate_forecast_GPU_h'],
                'measured_reference': {'job_id': '59068750', 'items': 228, 'allocated_seconds': 1552,
                    'ready_seconds': 736, 'gpus': 2, 'per_item_stress_factor': 2,
                    'cold_load_count': 2, 'shutdown_seconds_each': 196,
                    'source_sha256': config['offline_measured_reference_sha256']},
                'cpu_preparation_and_analysis_requested_core_hours': 8,
                'runtime_forecast_status': 'BOUND_MAX384_J1_ENVELOPE_EXACT_PROMPT_PRICE_STILL_REQUIRED',
                'scope': 'Explicit nonzero reservation, not a measured completion guarantee. Frozen three-vote J1 launch requires its separate profile and exact item-token price; no generation allocation is charged as offline grading.'},
            'side_query_count_cap': None}
    if profile['status'] != 'MEASURED_STRESS_REFERENCE':
        return {**base, 'status': 'HOLD_COMPLETE_GENERATION_PRICE', 'reasons': profile['reasons'], 'shards': [],
                'revised_proposal': 'Complete or extend the deterministic runtime probe with the same frozen policy and canonical pending assignments; do not invent side-reader timing or change enrollment.'}
    stress = config['stress_factor']
    load = profile['joint_cold_load_seconds'] + profile['shutdown_and_allocation_overhead_seconds']
    by_family = {}
    for row in remaining:
        by_family.setdefault(row['family'], []).append(row)
    shards, current, current_cost = [], [], load
    oversized = []
    for family in plan['family_order']:
        rows = by_family.get(family, [])
        if not rows:
            continue
        seconds = sum(profile['generator_seconds_per_prompt_or_emitted_token'] *
                      (row['prompt_tokens'] + 16384) +
                      (profile['side_queries_per_16k_policy_request_stress'] *
                       profile['side_pair_seconds_stress'] if row['arm'] == 'frozen_policy' else 0)
                      for row in rows)
        if stress * (load + seconds) + config['shutdown_margin_seconds'] > config['wall_seconds']:
            oversized.append({'family': family, 'projected_seconds': stress * (load + seconds) + config['shutdown_margin_seconds']})
        if current and (len(current) >= config['maximum_families_per_shard'] or
                        stress * (current_cost + seconds) + config['shutdown_margin_seconds'] > config['wall_seconds']):
            shards.append({'families': current, 'stress_projected_seconds': stress * current_cost + config['shutdown_margin_seconds']})
            current, current_cost = [], load
        current.append(family)
        current_cost += seconds
    if current:
        shards.append({'families': current, 'stress_projected_seconds': stress * current_cost + config['shutdown_margin_seconds']})
    for index, shard in enumerate(shards):
        shard.update(index=index, assigned_uids=[row['uid'] for family in shard['families'] for row in by_family[family]],
                     wall_seconds=config['wall_seconds'], gpus=4)
    require(sorted(uid for shard in shards for uid in shard['assigned_uids']) == sorted(row['uid'] for row in remaining),
            'production shard partition skips or duplicates assignments')
    # All array tasks use the same rounded measured stress allowance. Avoid
    # reserving the full configurable maximum when complete shards price lower.
    requested_wall = min(config['wall_seconds'], max(3600, 300 * math.ceil(
        max((s['stress_projected_seconds'] for s in shards), default=0) / 300)))
    for shard in shards:
        shard['wall_seconds'] = requested_wall
    initial_ceiling = len(shards) * 4 * requested_wall / 3600
    recovery_shards = math.ceil(len(shards) * config['infrastructure_recovery_reserve_fraction'])
    recovery_ceiling = recovery_shards * 4 * requested_wall / 3600
    projected = sum(shard['stress_projected_seconds'] for shard in shards) * 4 / 3600
    reasons = ['at least one complete family exceeds the proposed walltime under the explicit stress projection'] if oversized else []
    billing_ceiling = (initial_ceiling + recovery_ceiling) * 8
    if config['available_generation_billing_core_hours'] is None:
        reasons.append('current generation billing-core-hour allowance is not recorded in the proposal')
    elif billing_ceiling > config['available_generation_billing_core_hours']:
        reasons.append('initial generation plus full reserved infrastructure wave exceeds the recorded billing allowance')
    return {**base, 'status': 'HOLD_COMPLETE_GENERATION_PRICE' if reasons else 'PASS_COMPLETE_GENERATION_PROJECTION_BOUNDED_ALLOCATION',
            'reasons': reasons,
            'shards': shards, 'oversized_families': oversized,
            'requested_array_wall_seconds': requested_wall,
            'maximum_configured_wall_seconds': config['wall_seconds'],
            'empirical_stress_projected_generation_gpu_hours': projected,
            'initial_allocation_gpu_hour_ceiling': initial_ceiling,
            'infrastructure_recovery_reserve_gpu_hour_ceiling': recovery_ceiling,
            'infrastructure_recovery_reserved_shards': recovery_shards,
            'infrastructure_recovery_reserve_fraction': config['infrastructure_recovery_reserve_fraction'],
            'total_generation_allocation_gpu_hour_ceiling': initial_ceiling + recovery_ceiling,
            'generation_and_reserved_measurement_gpu_hours': initial_ceiling + recovery_ceiling + config['offline_grading_reserve_gpu_hours'],
            'generation_billing_core_hour_ceiling': billing_ceiling,
            'billing_conversion': 'Four GPUs on one full node reserve 32 billing cores: 8 billing-core-hours per allocated GPU-hour. Offline grading billing is accounted separately by its exact job resources.',
            'available_generation_billing_core_hours': config['available_generation_billing_core_hours'],
            'production_cpu_dispatch_requested_core_hour_ceiling': 12,
            'previous_pilot_allocated_gpu_hours': profile['pilot_allocated_gpu_hours'],
            'revised_proposal': ('Reconcile the recorded account allowance and requested initial/recovery allocations; increase per-shard walltime within verified partition limits or collect runtime-only calibration under the same policy if necessary. Do not suppress side queries or split/drop factorial cells.' if reasons else None),
            'interpretation': 'All remaining fixed assignments and all shard loads, sequential reader waits and nonfires are priced; one infrastructure-recovery wave is reserved for ceil(fraction × initial shard count) shards. Greater missing work requires a concrete additional-work proposal before submission. Requested walltime is a spending ceiling; empirical stress projections do not guarantee completion. Committed errors are never rerun to improve outcomes.'}


def prepare(config, attachment, output):
    pilot = load_pilot(config, attachment)
    profile = measured_profile(pilot)
    price = save(Path(output) / 'PRICE.json', price_stage(pilot['plan'], pilot['imports'], profile, config))
    if price['status'] != 'PASS_COMPLETE_GENERATION_PROJECTION_BOUNDED_ALLOCATION':
        print(price['status'], price['reasons'], flush=True)
        return price, None
    prompts, questions = U.sealed(U.PROMPTS), U.sealed(U.QUESTIONS)
    plan = pilot['plan']
    require(prompts['sha256'] == plan['sources']['prompt_table_sha256'] and
            U.file_sha(U.PROMPTS) == plan['sources']['prompt_table_file_sha256'] and
            questions['sha256'] == pilot['manifest']['original_question_table_sha256'], 'original inputs changed')
    imported = {row['assignment']['uid'] for row in pilot['imports']}
    rows = []
    for assignment in plan['assignments']:
        if assignment['uid'] in imported:
            continue
        ids = prompts['questions'][assignment['question']]['prompt_token_ids']
        require(U.digest(ids) == assignment['prompt_token_ids_sha256'], 'original prompt mismatch')
        rows.append({'assignment': assignment, 'original_prompt_ids': ids,
                     'problem': questions['questions'][assignment['question']]['problem']})
    manifest = save(Path(output) / 'MANIFEST.json', {'schema': 'utility-production-manifest-v1',
        'config_sha256': config['sha256'], 'price_sha256': price['sha256'],
        'plan_path': str(DOC / 'UTILITY_SCOUT_PLAN_v1.json'), 'plan_sha256': plan['sha256'],
        'policy_path': pilot['policy_path'], 'policy_sha256': pilot['policy']['sha256'], 'policy': pilot['policy'],
        'qualification_sha256': pilot['qualification']['sha256'],
        'qualification_binding_sha256': pilot['qualification_binding']['sha256'],
        'scientific_code_files': config['scientific_code_files'], 'source_files': config['source_files'],
        'pilot_attachment': str(attachment), 'pilot_accounting_sha256': pilot['accounting']['sha256'],
        'imported_receipts': pilot['imports'], 'rows': rows, 'shards': price['shards'],
        'assigned': 384, 'maximum_tokens': 16384, 'gpus_per_shard': 4,
        'original_question_table_sha256': questions['sha256'],
        'controller_rule': 'Unchanged v2 first accepted episode; no re-entry and no side-query cap.',
        'infrastructure_recovery_rule': 'At most one separately priced wave for assignments without a committed receipt. All committed errors remain ITT errors; retain original and recovery attempt provenance.'})
    print('PREPARED_PRODUCTION', manifest['sha256'], 'shards', len(manifest['shards']), flush=True)
    return price, manifest


def validate_manifest(manifest):
    require(manifest['schema'] == 'utility-production-manifest-v1' and manifest['assigned'] == 384 and
            manifest['maximum_tokens'] == 16384 and manifest['gpus_per_shard'] == 4 and
            manifest['source_files'] == source_files() and
            manifest['scientific_code_files'] == U.sealed(DOC / 'UTILITY_PAIR_ENGINEERING_PLAN_v2.json')['code_files'] and
            all(U.file_sha(Path(p)) == sha for p, sha in manifest['scientific_code_files'].items()),
            'production manifest or frozen source closure changed')
    plan = U.sealed(Path(manifest['plan_path']))
    U.validate_plan(plan)
    require(plan['sha256'] == manifest['plan_sha256'] and U.sealed(Path(manifest['policy_path']))['sha256'] == manifest['policy_sha256'],
            'production plan or selected policy changed')
    imported = [r['assignment'] for r in manifest['imported_receipts']]
    remaining = [r['assignment'] for r in manifest['rows']]
    all_rows = {r['uid']: r for r in imported + remaining}
    require(len(all_rows) == len(imported) + len(remaining) == 384 and
            [all_rows.get(r['uid']) for r in plan['assignments']] == plan['assignments'] and
            sorted(uid for s in manifest['shards'] for uid in s['assigned_uids']) == sorted(r['uid'] for r in remaining),
            'production/import coverage differs from exact 384 assignments')
    return plan


def allocation_accounting(array_jobs, shards_by_wave):
    """Charge every allocated task, including failed loads and killed attempts."""
    identifiers = [row['array_job'] for row in array_jobs]
    require(all(str(value).isdigit() for value in identifiers), 'invalid production array job ID')
    response = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(identifiers),
        '--format=JobID%50,State%40,ExitCode,ElapsedRaw,AllocTRES'], check=True, capture_output=True, text=True)
    raw = [line.split('|') for line in response.stdout.splitlines() if line.strip()]
    records = []
    for submission in array_jobs:
        for shard in shards_by_wave[submission['wave']]:
            identifier = submission['array_job'] + '_' + str(shard)
            matches = [row for row in raw if row[0].strip() == identifier]
            require(len(matches) == 1, 'final array-task accounting missing or ambiguous: ' + identifier)
            row = matches[0]
            state = row[1].strip()
            require(state not in ('RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED', 'REQUEUED', 'RESIZING'),
                    'array task has not reached terminal state: ' + identifier)
            tres = dict(part.split('=', 1) for part in row[4].split(',') if '=' in part)
            gpus = int(tres.get('gres/gpu', 0))
            require(gpus in (0, 4) and (state != 'COMPLETED' or (gpus == 4 and row[2] == '0:0')),
                    'completed task has wrong GPU allocation or nonzero exit: ' + identifier)
            elapsed = int(row[3])
            require(elapsed >= 0, 'invalid allocation duration')
            records.append({'accounting_job_id': identifier, 'wave': submission['wave'], 'shard_index': shard,
                'state': state, 'exit_code': row[2], 'elapsed_seconds': elapsed, 'allocated_gpus': gpus,
                'allocated_gpu_hours': gpus * elapsed / 3600, 'raw': '|'.join(row)})
    return records


def reconcile(manifest_path, output, accounting, *, final):
    """Every canonical UID is represented, including missing and committed errors."""
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = U.sealed(manifest_path)
    plan = validate_manifest(manifest)
    imported = {row['assignment']['uid']: row for row in manifest['imported_receipts']}
    observed = dict(imported)
    accounted_jobs = {row['accounting_job_id'] for row in accounting}
    for row in imported.values():
        receipt = U.sealed(Path(row['receipt_path']))
        require(receipt['sha256'] == row['receipt_sha256'], 'imported pilot receipt changed')
        validate_receipt(receipt, row['assignment'], row['source_binding_sha256'], row['routed_path'])
    by_uid = {r['assignment']['uid']: r['assignment'] for r in manifest['rows']}
    for shard in manifest['shards']:
        directory = output / f'shard-{shard["index"]:03d}'
        binding_path = directory / 'BINDING.json'
        binding = U.sealed(binding_path) if binding_path.exists() else None
        if binding:
            require(binding['manifest_sha256'] == manifest['sha256'] and
                    binding['assigned_uids'] == shard['assigned_uids'], 'shard output was rebound')
        valid_names = {U.digest(uid) + '.json' for uid in shard['assigned_uids']}
        require(all(path.name in valid_names for path in (directory / 'receipts').glob('*.json')),
                'foreign production receipt found')
        for uid in shard['assigned_uids']:
            assignment = by_uid[uid]
            key = U.digest(uid)
            attempts = []
            for path in sorted((directory / 'attempts').glob(key + '-*.json')):
                attempt = U.sealed(path)
                require(binding and attempt['assignment'] == assignment and
                        attempt['binding_sha256'] == binding['sha256'] and
                        attempt['accounting_job_id'] in accounted_jobs,
                        'attempt identity differs or its allocation is unaccounted')
                attempts.append({'path': str(path), 'sha256': attempt['sha256'],
                                 'accounting_job_id': attempt['accounting_job_id']})
            path = directory / 'receipts' / (key + '.json')
            receipt = U.sealed(path) if path.exists() else None
            if receipt:
                require(binding is not None and any(a['sha256'] == receipt['attempt_sha256'] for a in attempts),
                        'receipt has no matching assignment attempt')
                validate_receipt(receipt, assignment, binding['sha256'], receipt['routed_path'])
            observed[uid] = {'assignment': assignment,
                'status': receipt['status'] if receipt else 'MISSING',
                'origin': 'production' if receipt else 'missing',
                'receipt_path': str(path) if receipt else None,
                'receipt_sha256': receipt['sha256'] if receipt else None,
                'source_manifest_sha256': manifest['sha256'],
                'source_binding_sha256': binding['sha256'] if binding else None,
                'routed_path': receipt['routed_path'] if receipt else None,
                'routed_array_sha256': receipt['routed_array_sha256'] if receipt else None,
                'attempt_provenance': attempts}
    records = [observed[assignment['uid']] for assignment in plan['assignments']]
    require(len(records) == 384 and all(row['assignment'] == assignment for row, assignment in zip(records, plan['assignments'])),
            'reconciliation omitted or rebound assigned UIDs')
    missing = sum(row['status'] == 'MISSING' for row in records)
    errors = sum(row['status'] == 'GENERATION_ERROR' for row in records)
    pilot_accounting = U.sealed(Path(manifest['pilot_attachment']) / 'PILOT_ACCOUNTING.json')
    require(pilot_accounting['sha256'] == manifest['pilot_accounting_sha256'], 'pilot cost provenance changed')
    body = {'schema': 'utility-production-reconciled-index-v1',
        'plan_path': manifest['plan_path'], 'plan_sha256': plan['sha256'],
        'policy_path': manifest['policy_path'], 'policy_sha256': manifest['policy_sha256'],
        'production_manifest_path': str(manifest_path), 'production_manifest_sha256': manifest['sha256'],
        'assigned': 384, 'records': records, 'missing': missing, 'generation_errors': errors,
        'imported_pilot_assignments': len(imported), 'final_recovery_wave_resolved': final,
        'status': 'INCOMPLETE_GENERATION' if missing else 'COMPLETE_GENERATION_WITH_ERRORS' if errors else 'COMPLETE_CLEAN_GENERATION',
        'allocation_accounting': accounting, 'pilot_accounting_sha256': pilot_accounting['sha256'],
        'pilot_allocated_gpu_hours': pilot_accounting['actual_allocated_gpu_hours'],
        'production_allocated_gpu_hours': sum(row['allocated_gpu_hours'] for row in accounting),
        'total_generation_allocated_gpu_hours': pilot_accounting['actual_allocated_gpu_hours'] +
                                               sum(row['allocated_gpu_hours'] for row in accounting),
        'cost_scope': 'Whole four-GPU allocation elapsed time, including both loads, side-reader waits, nonfires, errors, recovery attempts and shutdown. Do not add reader components again.',
        'grading_status': 'PENDING_OFFLINE_BLIND_STRICT_J1_AND_384_CELL_ITT',
        'outcome_rule': 'Preserve frozen utility convention: only natural stops gradeable; caps/errors operational-wrong; missing cells remain explicit in ITT.'}
    return save(output / ('RECONCILED_INDEX_FINAL.json' if final else 'RECONCILED_INDEX_INITIAL.json'), body)


def recovery_plan(manifest, index):
    require(index['production_manifest_sha256'] == manifest['sha256'] and
            index['final_recovery_wave_resolved'] is False, 'recovery must follow the initial exact reconciliation')
    pending = {row['assignment']['uid']: row for row in index['records'] if row['status'] == 'MISSING'}
    shards = {}
    for shard in manifest['shards']:
        uids = [uid for uid in shard['assigned_uids'] if uid in pending]
        if uids:
            shards[str(shard['index'])] = {'assigned_uids': uids, 'wall_seconds': shard['wall_seconds'], 'gpus': 4}
    require(all(len(row['attempt_provenance']) <= 1 for row in pending.values()),
            'initial missing assignment has unexpected prior retries')
    return {'schema': 'utility-production-recovery-v1', 'manifest_sha256': manifest['sha256'],
        'initial_index_sha256': index['sha256'], 'wave': 1, 'shards': shards,
        'prior_attempt_sha256s': {uid: [a['sha256'] for a in row['attempt_provenance']] for uid, row in pending.items()},
        'assigned': len(pending), 'allocation_gpu_hour_ceiling': sum(s['wall_seconds'] * 4 / 3600 for s in shards.values()),
        'rule': 'One infrastructure recovery for missing receipts only. Never rerun committed errors/caps/nonfires or select by grading outcomes. Retain every prior attempt and charge all allocations.'}


def config_main(args):
    proposal = U.sealed(args.pilot_proposal)
    scientific = U.sealed(DOC / 'UTILITY_PAIR_ENGINEERING_PLAN_v2.json')['code_files']
    measurement_path = DOC / 'UTILITY_OUTCOME_MEASUREMENT_PLAN_v3.json'
    measurement = U.sealed(measurement_path)
    offline_reference = SCRIPTS / 'price_m9.py'
    offline_envelope_path = DOC / 'UTILITY_J1_MAX384_ENVELOPE_v2.json'
    offline_envelope = U.sealed(offline_envelope_path)
    value = {'schema': 'utility-production-config-v1', 'pilot_proposal_path': str(args.pilot_proposal.resolve()),
        'pilot_proposal_sha256': proposal['sha256'], 'source_files': source_files(), 'scientific_code_files': scientific,
        'gpus_per_shard': 4, 'max_tokens': 16384, 'wall_seconds': args.wall_seconds,
        'maximum_families_per_shard': args.maximum_families_per_shard,
        'maximum_concurrent_shards': args.maximum_concurrent_shards, 'stress_factor': args.stress_factor,
        'shutdown_margin_seconds': 600, 'side_query_count_cap': None,
        'maximum_infrastructure_recovery_waves': 1, 'automatic_retry_committed_errors': False,
        'infrastructure_recovery_reserve_fraction': args.infrastructure_recovery_reserve_fraction,
        'offline_measurement_plan_path': str(measurement_path), 'offline_measurement_plan_sha256': measurement['sha256'],
        'offline_grading_reserve_gpu_hours': args.offline_grading_reserve_gpu_hours,
        'offline_grading_envelope_path': str(offline_envelope_path), 'offline_grading_envelope': offline_envelope,
        'available_generation_billing_core_hours': args.available_generation_billing_core_hours,
        'offline_measured_reference_source': str(offline_reference),
        'offline_measured_reference_sha256': U.file_sha(offline_reference),
        'dispatch_on_pass': args.dispatch_on_pass,
        'authorization': 'User explicitly authorized all needed GPU hours and long jobs. Pricing and qualification are technical gates, not a new permission request.',
        'production_scope': 'Generation only. Offline correctness grading and final utility inference have separate measured resource accounting.'}
    validate_config(U.seal(value))
    result = save(args.out, value)
    print(result['sha256'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('config')
    p.add_argument('--pilot-proposal', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--wall-seconds', type=int, default=28800)
    p.add_argument('--maximum-families-per-shard', type=int, default=2)
    p.add_argument('--maximum-concurrent-shards', type=int, default=32)
    p.add_argument('--stress-factor', type=float, default=2.)
    p.add_argument('--infrastructure-recovery-reserve-fraction', type=float, default=.25,
                   help='Aggregate reserve: ceil(fraction × initial shards), one wave; excess missing work holds for repricing')
    p.add_argument('--offline-grading-reserve-gpu-hours', type=float, default=216.,
                   help='Explicit nonzero reservation; exact frozen J1 runtime/price is a separate launch gate')
    p.add_argument('--available-generation-billing-core-hours', type=float,
                   help='Recorded current account allowance for this generation allocation and reserve; missing evidence holds dispatch')
    p.add_argument('--dispatch-on-pass', action='store_true')
    args = parser.parse_args()
    config_main(args)


if __name__ == '__main__':
    main()
