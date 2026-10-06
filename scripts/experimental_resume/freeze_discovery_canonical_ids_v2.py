"""Bind each audited text prompt to one canonical world ID by exact token IDs."""
from __future__ import annotations

import json
from pathlib import Path

import run_boundary_micro_screen as base

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
SOURCE = REPORT / 'CAUSAL_DISCOVERY_FEASIBILITY_MICRO_AMENDMENT_v1.json'
OUT = REPORT / 'CAUSAL_DISCOVERY_FEASIBILITY_MICRO_AMENDMENT_v2.json'


def main():
    source = base.sealed(SOURCE)
    freeze = base.sealed(base.FAMILY_FREEZE)
    from moe_steer import manifests as M
    world = M.load_world()
    by_tokens = {}
    for key, info in world.infos.items():
        token_ids = tuple(info['prompt_token_ids'])
        if token_ids in by_tokens:
            raise ValueError('world has nonunique exact prompt token IDs')
        by_tokens[token_ids] = key
    rows = []
    for row in source['rows']:
        canonical = by_tokens.get(tuple(row['prompt_ids']))
        if canonical is None:
            raise ValueError('audited prompt has no exact canonical world match')
        if canonical not in freeze['new_parent_pools']['families'][row['family']]:
            raise ValueError('canonical prompt not in frozen duplicate family')
        if world.infos[canonical]['prompt_token_ids'] != row['prompt_ids']:
            raise ValueError('prompt IDs changed while mapping canonical key')
        rows.append({**row, 'canonical_question': canonical})
    body = {k: v for k, v in source.items() if k not in ('sha256', 'driver_sha256', 'rows')}
    body.update({'schema': 'routing-discovery-feasibility-micro-amendment-v2',
                 'source_v1_sha256': source['sha256'],
                 'driver_sha256': base.file_sha(__file__),
                 'canonical_mapping_rule': 'Exact full prompt_token_ids equality to exactly one immutable M.load_world().infos entry, then canonical ID membership in frozen duplicate family; original question text is preserved for blinded grading and never substituted into sampler IDs.',
                 'rows': rows})
    value = {**body, 'sha256': base.digest(body)}
    if OUT.exists():
        if base.sealed(OUT) != value:
            raise ValueError('existing canonical amendment differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'rows': len(rows), 'families': len({r['family'] for r in rows})}))


if __name__ == '__main__':
    main()
