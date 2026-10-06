"""Copy saved routing estimates into claim records without fitting new results."""
from __future__ import annotations

import json

import dispatch_overnight_readers_v1 as shared


def main():
    rows = []
    sources = {}
    for stage in ('A', 'B'):
        manifest = shared.base.sealed(shared.DOC / f'OVERNIGHT_DISCOVERY_{stage}_MANIFEST_v1.json')
        plan = shared.base.sealed(shared.DOC / 'ROUTING_FIRST_STAGE_PLAN_v2.json')
        directory = shared.RUNS / 'routing-first-stage' / (
            'routing-first-stage-v2-' + manifest['sha256'][:16] + '-' + plan['sha256'][:12])
        source = directory / 'RESULT.json'
        result = shared.base.sealed(source)
        if result['manifest_sha256'] != manifest['sha256'] or result['plan_sha256'] != plan['sha256']:
            raise ValueError('saved routing result source differs')
        sources[str(source)] = result['sha256']
        inference = result['inference']
        for contrast in inference['contrasts']:
            rows.append({'claim_id': f"routing-{stage}-{contrast['scope']}-{contrast['left']}-{contrast['right']}",
                'population': {'stage': stage, 'scope': contrast['scope'], 'families': contrast['families'],
                    'assigned_starts': contrast['assigned_starts'], 'paired_seed_cells': contrast['paired_seed_cells']},
                'intervention': {'left_arm': contrast['left'], 'right_arm': contrast['right'],
                    'manifest_sha256': manifest['sha256'], 'horizon': result['horizon']},
                'endpoint': inference['endpoint'], 'estimate': contrast['estimate'],
                'simultaneous_ci95': contrast['simultaneous_ci95'],
                'identification_bounds': contrast['identification_bounds'],
                'simultaneous_uncertainty_envelope95': contrast['simultaneous_uncertainty_envelope95'],
                'uncertainty_method': inference['method'], 'weighting': inference['weighting'],
                'multiplicity_family': f'routing-engagement-{stage}',
                'multiplicity_size': inference['multiplicity'], 'replicates': inference['replicates'],
                'status': 'ROUTING_ENGAGEMENT_ONLY_SEMANTIC_EFFECT_PENDING',
                'precision_status': contrast['precision_status'], 'source_result': str(source),
                'source_result_sha256': result['sha256'], 'claim_limit': result['claim_limit']})
    value = shared.save(shared.DOC / 'ROUTING_ENGAGEMENT_CLAIM_LEDGER_v1.json', {
        'schema': 'routing-engagement-claim-ledger-v1', 'source_results': sources, 'claims': rows,
        'interpretation': 'All saved A/B routing contrasts; no ranking or cross-stage pooled inference. Positive expert engagement alone does not establish semantic control, correctness or token benefit.'})
    print(json.dumps({'claims': len(rows), 'sha256': value['sha256']}))


if __name__ == '__main__':
    main()
