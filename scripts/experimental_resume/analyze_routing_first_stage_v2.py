"""Scheduling-provenance adapter; frozen v1 scientific analysis is unchanged."""
from __future__ import annotations

from pathlib import Path
import analyze_routing_first_stage_v1 as original

source = original.source
require = original.require
PLAN = original.DOC / 'ROUTING_FIRST_STAGE_PLAN_v2.json'
PRIOR = original.PLAN
AMENDMENT = original.DOC / 'SLURM_PARENT_ENVIRONMENT_RECOVERY_v1.json'
HELPER_SHA = '0c142175df5db52fdf00a85d0e8dd46099209a76c5161994b4279332d6c84a10'


def code_files():
    prior = source.sealed(PRIOR)
    files = dict(prior['code_files'])
    helper = str(Path(__file__).with_name('dispatch_overnight_readers_v1.py'))
    files[helper] = HELPER_SHA
    for name in ('analyze_routing_first_stage_v2.py', 'analyze_routing_first_stage_v2.sbatch',
                 'submit_routing_first_stage_v2.py', 'submit_routing_first_stage_v2.sbatch'):
        path = Path(__file__).with_name(name)
        files[str(path)] = source.file_sha(path)
    require(all(source.file_sha(path) == sha for path, sha in files.items()),
            'first-stage v2 source or preserved v1 source changed')
    return files


def validate_plan():
    plan, prior, amendment = source.sealed(PLAN), source.sealed(PRIOR), source.sealed(AMENDMENT)
    require(plan['schema'] == 'routing-first-stage-plan-v2' and
            plan['prior_plan_sha256'] == prior['sha256'] and
            plan['scheduling_amendment_sha256'] == amendment['sha256'] and
            plan['code_files'] == code_files() and
            plan['definitions_sha256'] == source.file_sha(original.DEFINITIONS) == prior['definitions_sha256'] and
            plan['bootstrap_replicates'] == original.BOOTSTRAPS == prior['bootstrap_replicates'] and
            plan['analysis_seed'] == original.SEED == prior['analysis_seed'],
            'first-stage v2 plan binding changed')
    return plan


def output_path(manifest, parent, plan):
    return Path(parent) / ('routing-first-stage-v2-' + manifest['sha256'][:16] + '-' + plan['sha256'][:12])


def main():
    validate_plan()
    original.validate_plan = validate_plan
    original.output_path = output_path
    original.main()


if __name__ == '__main__':
    main()
