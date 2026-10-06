"""Freeze the 220 unused, confirm-disjoint families as an exploratory extension.

This enrollment was motivated by observed start-audit attrition in the registered
128-family pool. It does not amend that pool or rescue its feasibility gate.
No extension start rating or intervention outcome enters the family order.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
SOURCE = REPO / 'report/experimental-resume-v1/family-freeze.json'
OUT = REPO / 'report/experimental-resume-v1/MECHANISM_EXTENSION_220_FAMILY_FREEZE_v1.json'
POOLS = ('discovery', 'mechanism', 'utility')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed source seal: ' + str(path))
    return value


def derive(source):
    if source['schema'] != 'routing-control-family-freeze-v1':
        raise ValueError('wrong frozen family source')
    state = source['new_parent_pools']
    if state['eligible_family_count'] != 492:
        raise ValueError('eligible family population differs')
    all_families = set(state['families'])
    excluded = set(state['confirm_connected_excluded'])
    eligible = all_families - excluded
    if len(eligible) != 492 or len(excluded) != 994:
        raise ValueError('confirm-connected exclusion differs')
    ordered = sorted(eligible, key=lambda family: hashlib.sha256(
        ('routing-control-v1|' + family).encode()).hexdigest())
    allocated = [family for pool in POOLS for family in state['parent_pools'][pool]]
    if ([len(state['parent_pools'][pool]) for pool in POOLS] != [48, 128, 96] or
            len(allocated) != 272 or len(set(allocated)) != 272 or
            allocated != ordered[:272]):
        raise ValueError('registered parent pools or order differ')
    extension = ordered[272:]
    if len(extension) != 220 or set(extension) & set(allocated) or set(extension) & excluded:
        raise ValueError('extension is not the full unused eligible suffix')
    representatives = {family: state['representative_questions'][family]
                       for family in extension}
    if len(set(representatives.values())) != 220 or any(
            question not in state['families'][family]
            for family, question in representatives.items()):
        raise ValueError('canonical representative questions differ')
    return {'schema': 'mechanism-extension-220-family-freeze-v1',
            'source_family_freeze_sha256': source['sha256'],
            'parent_pool_counts': {pool: len(state['parent_pools'][pool]) for pool in POOLS},
            'eligible_parent_families': 492,
            'confirm_connected_excluded_families': 994,
            'registered_parent_families': 272,
            'extension_families': 220,
            'order_rule': 'unused suffix of ascending SHA256(routing-control-v1|family_id)',
            'families': extension,
            'representative_questions': representatives,
            'scope': ('Prospective exploratory extension motivated by observed start-audit '
                      'attrition in the registered 128-family pool. Frozen before extension '
                      'reader or intervention outcomes; not a replacement or rescue of the '
                      'registered mechanism validation.')}


def main():
    body = derive(sealed(SOURCE))
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('frozen extension family enrollment differs')
    else:
        OUT.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'out': str(OUT), 'sha256': value['sha256'],
                      'families': len(value['families']), 'scope': value['scope']}))


if __name__ == '__main__':
    main()
