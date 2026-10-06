"""Freeze a bounded four-GPU engineering probe; this never submits a job."""
from pathlib import Path
import argparse


def main():
    from qualify_utility_pair_v2 import bootstrap, fixtures
    bootstrap()
    import utility_scout_v1 as utility
    import rate_overnight_semantics_v2 as storage
    import overnight_routing_runner_v1 as source
    from utility_source_contract_v2 import code_files
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    design = utility.sealed(utility.REPO / 'report/experimental-resume-v1/OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json')
    enrollment = utility.sealed(source.SOURCE)
    native = enrollment['rows'][0]
    _, cases = fixtures(design)
    reference = utility.sealed(source.REFERENCE)
    max_decode = 12 * 1024 + 128
    preempts = sum(c['preempt'] for c in cases)
    prefill = len(cases) * len(native['prompt_ids']) + preempts * (len(native['prompt_ids']) + 80)
    work = reference['repeat_factor'] * (max_decode / reference['serial_decode_stress_tokens_per_second'] +
        prefill / reference['serial_prefill_stress_tokens_per_second'] + 2048 / 8 + 2 * (49152 - 1024) / 1000)
    estimate = work + 2 * reference['cold_load_seconds'] + 2 * reference['shutdown_seconds'] + 900
    body = {'schema': 'utility-pair-engineering-plan-v2', 'status': 'READY_BOUNDED_ENGINEERING_PROBE',
        'design_sha256': design['sha256'], 'source_enrollment_sha256': enrollment['sha256'],
        'original_prompt_sha256': utility.digest(native['prompt_ids']), 'fixture_family': native['family'],
        'cases': cases, 'generation_cases': len(cases), 'maximum_generation_tokens': max_decode,
        'side_queries': 1, 'side_readers': 2, 'maximum_side_output_tokens': 2048,
        'maximum_side_context_plus_output_per_reader': 49152, 'maximum_engineering_prefill_tokens': prefill,
        'gpus': 4, 'wall_seconds': 7200, 'allocation_gpu_hour_ceiling': 8.0,
        'shutdown_deadline_margin_seconds': 300, 'runtime_reference_sha256': reference['sha256'],
        'conservative_runtime_projection_seconds': estimate,
        'conservative_projection_gpu_hours': 4 * estimate / 3600,
        'projection_scope': 'Existing serial 8 token/s generation stress, 1000 prefill token/s, all generation and two full reader caps, six live preemptions plus deterministic closure preemption, two cold loads, two shutdowns, 25% work reserve, 900s overhead. New four-GPU profile remains unmeasured; 2h is an allocation bound, not a guarantee every check finishes.',
        'timeout_rule': 'Preserve raw assignments and partial logs; qualification remains incomplete or failed. No utility production follows a missing or non-PASS result.',
        'production_status': 'HOLD_SELECTED_POLICY_16K_RUNTIME_PILOT_AND_COMPLETE_STAGE_PROPOSAL',
        'code_files': code_files()}
    storage.require(len(cases) == 13 and estimate < 7200, 'engineering probe no longer fits proposed allocation')
    value = storage.save(args.out, body)
    print(value['status'], value['sha256'], f'{estimate:.1f}s projected; 8 GPUh allocation ceiling', flush=True)


if __name__ == '__main__':
    main()
