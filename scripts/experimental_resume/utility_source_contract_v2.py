"""Explicit project source closure for utility qualification and price pilots."""
from pathlib import Path
import utility_scout_v1 as utility


def code_files():
    here = Path(__file__).parent
    names = ('utility_source_contract_v2.py', 'prepare_utility_engineering_v2.py',
        'run_utility_price_pilot_v2.py', 'qualify_utility_pair_v2.py',
        'utility_pair_backend_v2.py', 'utility_controller_v2.py', 'utility_routing_worker_v2.py',
        'utility_qualification_logits_v2.py', 'utility_controller_interface_v1.py', 'select_utility_policy_v2.py',
        'utility_scout_v1.py', 'overnight_routing_worker_v2.py', 'overnight_routing_runner_v1.py',
        'run_boundary_micro_screen.py', 'diagnose_mechanism_validation_v3.py',
        'rate_transition_v22_fullprefix_starts_v2.py', 'rate_overnight_semantics_v2.py',
        'mechanism_extension_reader_contract_v2.py')
    paths = {here / name for name in names}
    paths.add(here / 'qualify_utility_pair_v2.sbatch')
    roots = (utility.REPO / 'src/moe_exp/routing_control',
             utility.STAGE / 'code/s1-9a61e32f48c04c24/moe_steer',
             utility.STAGE / 'addenda/ordered/9727c10299b71e7a/moe_exp_src')
    for root in roots:
        paths.update(root.rglob('*.py'))
    paths.update((utility.REPO / 'report/experimental-resume-v1/TRANSITION_RUBRIC_v0.1.md',
                  utility.REPO / 'src/moe_exp/correlation_pipeline/model_profiles.py'))
    return {str(path.resolve()): utility.file_sha(path) for path in sorted(paths)}
