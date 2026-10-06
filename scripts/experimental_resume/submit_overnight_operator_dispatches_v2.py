"""Attach reviewed one-shot execution dispatches to qualified stage preparation."""
import os
from pathlib import Path
import subprocess

import dispatch_overnight_readers_v1 as shared


def main():
    directory = shared.DOC / 'overnight-operator-dispatch-submissions-v2'
    directory.mkdir(exist_ok=True)
    scripts = shared.REPO / 'scripts/experimental_resume'
    wrapper = scripts / 'dispatch_overnight_stage_v2.sbatch'
    files = [wrapper, scripts / 'dispatch_overnight_stage_v2.py',
             scripts / 'prepare_overnight_operator_stage_v2.py']
    subprocess.run(['bash', '-n', str(wrapper)], check=True)
    records = []
    for stage, preparation in [('C', '59345112'), ('FRESH', '59345113')]:
        env = os.environ.copy()
        env.update(OVERNIGHT_STAGE=stage, OVERNIGHT_DISPATCH_PHASE='generation')
        binding = {'stage': stage, 'preparation_job': preparation,
                   'phase': 'generation',
                   'files': {str(p): shared.base.file_sha(p) for p in files}}
        job = shared.submit(directory, stage.lower() + '-generation-dispatch',
                            ['--dependency=afterok:' + preparation,
                             '--job-name=st-overnight-' + stage + '-launch-v2', str(wrapper)],
                            env, binding)
        records.append({'stage': stage, 'preparation_job': preparation, 'dispatcher_job': job})
    shared.save(directory / 'CHAIN.json', {'schema': 'overnight-operator-dispatch-submissions-v2',
                                         'records': records})


if __name__ == '__main__':
    main()
