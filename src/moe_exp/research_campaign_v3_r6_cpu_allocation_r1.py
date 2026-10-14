"""Verify real CPU-only Booster steps without falsifying scheduler variables."""
import os
import re
import socket
import sys

from moe_exp.research_campaign_v3_r6.contracts import require


def guard_for(value, original, verify):
    checked = False
    def allocated(*, gpu=False):
        nonlocal checked
        require(not gpu or value['gpus'] > 0, 'GPU work exceeds the immutable CPU-only card')
        if gpu or value['gpus'] or value['partition'] != 'boost_usr_prod':
            return original(gpu=gpu)
        require(value['cpus'] == 4 and value['memory_gib'] == 24
                and value['stage'] not in ('controller', 'snapshot', 'validate'),
                'Unregistered CPU-only Booster shape')
        require(os.environ.get('SLURM_JOB_ID', '').isdigit()
                and os.environ.get('SLURM_STEP_ID') is not None,
                'scientific work requires a Slurm Python step')
        require(os.environ.get('USER') == 'lmolfett' and 'leonardo' in socket.getfqdn(),
                'wrong cluster/user')
        require(os.environ.get('SLURM_JOB_ACCOUNT', '').lower() == 'iscrc_miosr',
                'unexpected allocation account')
        require(os.environ.get('SLURM_JOB_PARTITION') == 'boost_usr_prod'
                and not socket.gethostname().startswith('login'), 'wrong CPU compute allocation')
        if not checked:
            proof = verify(os.environ['SLURM_JOB_ID'], value)
            require(not re.search(r'(?:^|,)gres/gpu=', proof['fields']['ReqTRES'])
                    and proof['resources']['billing_units'] <= 4,
                    'CPU-only worker requested GPU or expanded billing')
            checked = True
    return allocated


def verification_target(job_id, logical, physical):
    if physical.get('array_cards'):
        parent = os.environ.get('SLURM_ARRAY_JOB_ID', '')
        task = os.environ.get('SLURM_ARRAY_TASK_ID', '')
        require(parent.isdigit() and task.isdigit() and
                physical['array_cards'][int(task)] == logical,
                'Physical array allocation differs from the logical worker assignment')
        return parent + '_' + task, physical
    require(physical == logical, 'Physical worker card differs')
    return job_id, physical


def install(value, verify):
    from moe_exp.expert_memory_study import contracts
    original = contracts.allocated
    replacement = guard_for(value, original, verify)
    for name, module in list(sys.modules.items()):
        if name.startswith('moe_exp.') and getattr(module, 'allocated', None) is original:
            module.allocated = replacement
