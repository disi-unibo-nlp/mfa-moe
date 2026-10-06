"""CPU qualification of frozen discovery prefix fixtures; labels enter scoring only.

Input JSON: {family_freeze_sha256, fixtures: [{family, problem, emitted_token_ids,
emitted_text, supported_candidate_gold: bool}]}. Fixtures must be discovery-family prefixes.
This measures candidate parsing only; semantic detector thresholds need a separate audit.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from moe_exp.routing_control.design import digest
from moe_exp.routing_control.prefix import StreamingAdapter


def qualify(fixtures, discovery_families):
    confusion = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    receipts = []
    for row in fixtures:
        if row['family'] not in discovery_families:
            raise ValueError('prefix qualification contains a nondiscovery family')
        # Label, correctness, future fields and family are never passed into the detector.
        decision = StreamingAdapter().observe({key: row[key] for key in
            ('problem', 'emitted_token_ids', 'emitted_text')})
        prediction = decision['candidate'] is not None
        label = row['supported_candidate_gold']
        if type(label) is not bool:
            raise ValueError('candidate qualification labels must be booleans')
        confusion['tp' if prediction and label else 'fp' if prediction else 'fn' if label else 'tn'] += 1
        receipts.append({'family': row['family'], 'prefix_tokens': len(row['emitted_token_ids']),
                         'closure': decision['closure'], 'trigger': decision['can_trigger'],
                         'candidate': asdict(decision['candidate']) if prediction else None})
    n = len(receipts)
    return {'confusion': confusion, 'n_prefixes': n,
            'coverage': (confusion['tp'] + confusion['fp']) / n if n else None,
            'trigger_frequency': sum(r['trigger'] for r in receipts) / n if n else None,
            'receipts': receipts, 'semantic_detector_status': 'NOT_QUALIFIED_BY_THIS_PARSER_CHECK'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--family-freeze', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.family_freeze.read_text())
    data = json.loads(args.fixtures.read_text())
    if frozen['sha256'] != digest({k: v for k, v in frozen.items() if k != 'sha256'}):
        raise ValueError('family freeze content changed')
    if data['family_freeze_sha256'] != frozen['sha256']:
        raise ValueError('prefix fixture family binding differs')
    result = qualify(data['fixtures'], frozen['new_parent_pools']['parent_pools']['discovery'])
    result['input_sha256'] = hashlib.sha256(args.fixtures.read_bytes()).hexdigest()
    result['family_freeze_sha256'] = frozen['sha256']
    result['sha256'] = digest(result)
    if args.out.exists() and json.loads(args.out.read_text()) != result:
        raise ValueError('changed qualification requires a versioned output')
    args.out.write_text(json.dumps(result, indent=1) + '\n')
