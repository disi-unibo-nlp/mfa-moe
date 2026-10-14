"""Match sealed embedded failure payloads in the correct receipt namespace."""
from pathlib import Path
from moe_exp.research_campaign_v3_r6.contracts import load, require, seal


def failure_identity(value):
    identity = seal(value)['sha256']
    require('sha256' not in value or value['sha256'] == identity,
            'Embedded failure seal differs from its payload')
    return identity


def remaining_block(campaign, stage, model, resolved):
    for path in (Path(campaign['root']) / 'failure_dispositions').glob('*.json'):
        receipt = load(path)
        if failure_identity(receipt['failure']) == resolved['resolved_failure_sha256']:
            continue
        if stage in receipt['dependent_stages'] and (receipt['scope'] == 'shared' or receipt['model'] == model):
            return receipt
    return None


def check_binding(campaign):
    from moe_exp.research_campaign_v3_r6.contracts import check_files
    from moe_exp.research_campaign_v3_r6_reader_price_r1 import check_binding as parent_binding
    parent = parent_binding(campaign)
    root = Path(campaign['root'])
    binding = load(root / 'READER_PRICE_FAILURE_ID_R1_BINDING.json')
    check_files(binding['files']); check_files(binding['evidence_files'])
    require(binding['campaign_sha256'] == campaign['sha256'] and
            binding['reader_price_binding_sha256'] == parent['sha256'] and
            binding['resolution_sha256'] == load(root / 'READER_PRICE_R1_RESOLUTION.json')['sha256'],
            'Exact pricing ancestry and resolution required')
    require(binding['new_scientific_draws'] == binding['additional_gpu_allocations'] == 0 and
            binding['remote_writes'] is False and binding['manuscript_writes'] is False,
            'Failure identifier repair scope differs')
    proof = load(root / 'READER_PRICE_R1_PREFLIGHT.json')
    require(proof['passed'] and proof['binding_sha256'] == parent['sha256'], 'Parent pricing proof differs')
    return binding


def install():
    from moe_exp import research_campaign_v3_r6_reader_price_r1 as parent
    parent.remaining_block = remaining_block
