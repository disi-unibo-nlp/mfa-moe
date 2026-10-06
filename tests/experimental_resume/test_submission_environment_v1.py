"""Regression for CPU dispatcher affinity contaminating a new GPU allocation."""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experimental_resume'))
import dispatch_overnight_readers_v1 as dispatch


def test_new_allocation_has_no_parent_placement():
    parent = {'SLURM_CPU_BIND': 'quiet,mask_cpu:0x3000000',
              'SLURM_CPU_BIND_LIST': '0x3000000', 'SLURM_STEP_ID': '0',
              'SLURM_JOB_ID': '123', 'SRUN_CPUS_PER_TASK': '2',
              'SBATCH_MEM_PER_NODE': '16384', 'PMI_RANK': '0',
              'PMIX_NAMESPACE': 'old-step', 'CUDA_VISIBLE_DEVICES': '1',
              'OVERNIGHT_STAGE': 'FRESH', 'CAP_RECOVERY_MODE': 'qualification',
              'PATH': '/verified/bin', 'VLLM_USE_FLASHINFER_SAMPLER': '0'}
    before = dict(parent)
    child = dispatch.submission_environment(parent)
    assert parent == before
    assert child == {key: parent[key] for key in
                     ('OVERNIGHT_STAGE', 'CAP_RECOVERY_MODE', 'PATH',
                      'VLLM_USE_FLASHINFER_SAMPLER')}


def test_test_only_and_real_submission_use_same_clean_environment(tmp_path, monkeypatch):
    calls = []
    parent = {'SLURM_CPU_BIND': 'mask_cpu:0x100', 'OVERNIGHT_STAGE': 'A'}
    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(stdout=('12345\n' if '--parsable' in arguments else
            'JobId=12345 UserId=lmolfett(133943)' if arguments[0] == 'scontrol' else ''))
    monkeypatch.setattr(dispatch.subprocess, 'run', run)
    assert dispatch.submit(tmp_path, 'test', ['--cpus-per-task=16', 'job.sbatch'], parent, {}) == '12345'
    submits = [options for args, options in calls if args[0] == 'sbatch']
    assert len(submits) == 2
    assert submits[0]['env'] == submits[1]['env'] == {'OVERNIGHT_STAGE': 'A'}
    assert parent['SLURM_CPU_BIND'] == 'mask_cpu:0x100'
    receipt = dispatch.base.sealed(tmp_path / 'test.json')
    assert receipt['removed_parent_environment_keys'] == ['SLURM_CPU_BIND']
    assert 'mask_cpu' not in (tmp_path / 'test.json').read_text()
