"""Measured component receipts; missing replay/grading never become zero prices."""
from __future__ import annotations

from pathlib import Path

from moe_exp.routing_control.study_v1 import require, save, sealed, file_sha, digest
from moe_exp.routing_control.study_v2 import CAP
from moe_exp.routing_control.study_v2 import verify


def measured_components(pilot_directory, output):
    pilot_directory, output = Path(pilot_directory), Path(output)
    pilot = sealed(pilot_directory / 'PILOT.json')
    output.mkdir(parents=True, exist_ok=False)
    measurements, missing = [], ['replay', 'grading']
    reader_path = pilot_directory / 'READER_COST.json'
    annotation = None
    if reader_path.exists():
        reader = sealed(reader_path)
        annotation = save(output / 'ANNOTATION_OBSERVED.json', {'schema': 'routing-prefix-annotation-cost-v2',
            'reader_cost_sha256': reader['sha256'], 'actual_input_tokens': reader['input_tokens'],
            'actual_output_tokens': reader['output_tokens'], 'load_seconds': reader['load_seconds'],
            'reader_seconds': reader['reader_seconds'], 'shutdown_seconds': reader['shutdown_seconds'],
            'partial_attempts': reader.get('partial_attempts', []), 'gpus': 2, 'reader_cap': 1024,
            'actual_readers': reader['actual_readers'], 'assigned_readers': reader['assigned_readers'],
            'status': 'OBSERVED_READER_WORK' if reader['actual_readers'] else 'LOAD_ONLY_NO_RATINGS',
            'scope': 'Observed six-decision prefix rubric expenditure, including incomplete attempts. Cannot price a different mechanism or J1 rubric.'})
    cells = pilot['pricing_cells']
    if pilot['status'] != 'MEASURED' or len(cells) != 4:
        return save(output / 'PRICE_STATUS.json', {'schema': 'routing-measured-components-v2',
            'pilot_sha256': pilot['sha256'], 'status': 'UNAVAILABLE_MEASURED_PRICE',
            'measurements': [], 'unavailable_roles': ['generation', 'controller', 'capture', 'runtime', 'recovery', 'replay', 'grading'],
            'annotation_observed_sha256': annotation['sha256'] if annotation else None,
            'reason': 'Semantic qualification or all four exact 16k pricing cells incomplete.',
            'scientific_dispatch_authorized': False})
    execution = sealed(pilot_directory / 'EXECUTION.json')
    runtime = sealed(pilot_directory / 'generator/RUNTIME.json')
    cost = sealed(pilot_directory / 'generator/COST.json')
    require(all(c['cap'] == CAP and c['status'] == 'MEASURED' and
                c['assignment']['execution_contract_sha256'] == execution['sha256'] for c in cells),
            'short or mismatched pricing contract')
    metrics = cost['metrics']
    receipts = [verify(sealed(Path(c['receipt_path']))['receipt']) for c in cells]
    # The bounded pricing path intentionally retains engineering-only receipts.
    # Bind their exact request UID to the parent qualified contract without
    # rewriting a receipt into a scientific execution/outcome.
    require(all(r['sha256'] == c['receipt_sha256'] and r['assignment_uid'] == c['assignment']['uid'] and
                r['continuation_cap'] == CAP and r['engineering_only'] is True and
                r['execution_contract_sha256'] is None and c['assignment']['uid'] == digest([
                    'routing-outcome-blind-price-v2', pilot['manifest_sha256'],
                    {k: v for k, v in c['assignment'].items() if k != 'uid'}])
                for r, c in zip(receipts, cells)), 'pricing generation receipt source mismatch')
    require(runtime['code_files'] == pilot['code_files'] and cost['gpus'] == 2,
            'pricing runtime or allocation source mismatch')
    require({(c['assignment']['arm'], c['assignment']['seed']) for c in cells} == {
        ('native', 0), ('native', 1), ('incumbent', 0), ('incumbent', 1)}, 'pricing cell enrollment changed')
    generated = sum(r['generated_tokens'] for r in receipts)
    initial = sum(len(c['assignment']['original_prompt_ids']) for c in cells)
    processed = sum(r['processed_input_tokens'] for r in receipts)
    # Subsequent decode forwards consume one token per emitted row after the
    # first. The remaining input count is exact chunked prefill.
    prefill = processed - (generated - 4)
    require(prefill >= initial and metrics['prefill_seconds'] > 0 and metrics['decode_seconds'] > 0 and generated > 4,
            'missing actual prefill/decode measurement')
    calls = sum(len(r['controller']['controller_calls']) for r in receipts)
    controller_seconds = sum(r['controller'].get('controller_seconds', 0.) for r in receipts)
    total_wall = sum(r['generation_wall_seconds'] for r in receipts)
    rpc = metrics['command_rpc_seconds'] + metrics['capture_rpc_seconds']
    host_remainder = max(0., total_wall - metrics['engine_step_seconds'] - rpc)
    base = {'execution_contract_sha256': execution['sha256'], 'measurement_artifact_sha256': pilot['sha256'],
        'continuation_cap': CAP, 'gpus': 2, 'cpus': 16, 'billing_cores': 16,
        'prefill_tps': 0., 'decode_tps': 0., 'load_seconds': 0., 'shutdown_seconds': 0., 'seconds_per_call': 0.,
        'status': 'MEASURED', 'runtime_sha256': runtime['sha256'],
        'measurement_scope': 'Outcome-blind engineering pricing requests bound to the qualified parent execution contract; not scientific outcomes.'}
    measured = {
        'generation': {**base, 'prefill_tps': prefill / metrics['prefill_seconds'],
            'decode_tps': (generated - 4) / metrics['decode_seconds'],
            'load_seconds': cost['load_seconds'], 'shutdown_seconds': metrics['shutdown_seconds'],
            'scope': 'Observed synchronous engine work; first prediction is included in prefill. Scientific cap is 16,384 even for naturally shorter stops.'},
        'controller': {**base, 'seconds_per_call': controller_seconds / calls if calls else 0.,
            'actual_calls': calls, 'actual_output_tokens': sum(r['controller']['controller_tokens'] for r in receipts),
            'scope': 'Cheap recognizer reported work; full decoder/controller/capture host remainder is reserved in capture to avoid underpricing.'},
        'capture': {**base, 'seconds_per_call': (rpc + host_remainder) / generated,
            'scope': 'Both command/capture RPCs plus all residual host work, including decoding, controller bookkeeping, byte hashing and binary flushes. Conservative upper bound; recognizer work can be counted twice.'},
        'runtime': {**base, 'seconds_per_call': max(0., metrics['allocated_driver_seconds'] - cost['load_seconds'] -
            metrics['shutdown_seconds'] - total_wall), 'scope': 'Observed generator lifetime overhead outside bound request loops.'}}
    for role, value in measured.items():
        receipt = save(output / (role.upper() + '.json'), {'schema': 'routing-component-measurement-v2', 'role': role, **value})
        measurements.append({'role': role, 'sha256': receipt['sha256'], 'path': str(output / (role.upper() + '.json'))})
    # A full assignment retry uses this exact 16k profile plus another complete
    # prefill/load. Recovery correctness is qualified separately; this price is
    # a conservative workload projection, not an observed death-recovery rate.
    recovery = save(output / 'RECOVERY_PROJECTION.json', {'schema': 'routing-recovery-price-projection-v2',
        'execution_contract_sha256': execution['sha256'], 'maximum_attempts_per_assignment': 2,
        'basis_generation_measurement_sha256': sealed(output / 'GENERATION.json')['sha256'],
        'scope': 'One full original-prefix assignment rerun; add load/shutdown, full prefill, CAP decode and capture/controller/runtime roles again.',
        'observed_full_16k_process_death_recovery': False})
    missing.append('observed_16k_death_recovery_price')
    return save(output / 'PRICE_STATUS.json', {'schema': 'routing-measured-components-v2', 'pilot_sha256': pilot['sha256'],
        'execution_contract_sha256': execution['sha256'], 'measurements': measurements,
        'annotation_observed_sha256': annotation['sha256'] if annotation else None,
        'recovery_projection_sha256': recovery['sha256'], 'unavailable_roles': missing,
        'capture_storage': {'observed_allocated_bytes': sum(r['capture_binding']['allocated_storage_bytes'] for r in receipts),
            'maximum_rows_per_assignment': CAP, 'ranks': 2, 'paths_per_rank': 2, 'layers': 40, 'top_k': 8,
            'ids_dtype': 'uint16', 'weights_dtype': 'float32', 'dose_dtype': 'float32',
            'prefix_storage': 'Original prompts and frozen prefix catalog retained in parent manifest; include their bytes for every retry.'},
        'status': 'UNAVAILABLE_COMPLETE_STAGE_PRICE', 'scientific_dispatch_authorized': False,
        'code_files': {str(Path(__file__)): file_sha(__file__)}})
