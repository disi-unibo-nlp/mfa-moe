"""Freeze registered G3 point-estimate selection after verified complete X2/NLL."""
import hashlib
import json
from pathlib import Path

S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')


def run():
    source = S/'runs/x2-resume-v1/analysis/estimates.json'
    value = json.loads(source.read_text())
    if value['status'] != 'COMPLETE' or value['requests'] != 6630 or len(value['cells']) != 42:
        raise ValueError('all X2 assigned branches and all 42 target cells must be complete')
    if set(value['native_NLL_validation']) != {'tf1', 'tf8'} or not all(v['pass'] for v in value['native_NLL_validation'].values()):
        raise ValueError('identical native parity must pass')
    rank = lambda c: (('L1', 'BAND', 'ALL').index(c['scope']), ('reweight', 'bias', 'force').index(c['operator']), c['magnitude'])
    selected = []
    for sign in (-1, 1):
        cells = [c for c in sorted(value['cells'], key=rank) if c['sign'] == sign and c['G3_pass']]
        chosen = cells[0] if cells else None
        if (chosen['policy'] if chosen else None) != value['selected_policy_per_sign'][str(sign)]:
            raise ValueError('G3 selection differs from the registered ranking')
        if chosen:
            if not all(c is True for c in chosen['checks'].values()) or not all(m['eligible'] for m in chosen['matches']):
                raise ValueError('selected G3 screen or measured-dose support failed')
            selected.append({k: chosen[k] for k in ('policy', 'scope', 'operator', 'sign', 'magnitude', 'G3_pass')})
            selected[-1]['random_magnitudes'] = {str(k): m['magnitude'] for k, m in enumerate(chosen['matches'])}
            selected[-1]['random_dose_ratios'] = {str(k): m['ratio'] for k, m in enumerate(chosen['matches'])}
    body = {'schema': 'G3-dose-selection-v1', 'native_nll_parity_pass': True,
        'manifest_sha256': value['manifest_sha256'], 'selected': selected,
        'source_results': str(source), 'source_results_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'population': '39 eligible of 48 dev-cal questions; two seeds; no replacement',
        'interpretation': 'diagnostic point-estimate dose selection only; no semantic steering or significance claim',
        'X3_launch': 'HOLD complete eligibility/pricing and new-study priority',
        'ranking': 'scope L1/BAND/ALL, operator reweight/bias/force, increasing magnitude, one per sign'}
    body['sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    dest = S/'runs/x2-resume-v1/analysis/G3_SELECTION.json'
    if dest.exists() and json.loads(dest.read_text()) != body:
        raise ValueError('a different G3 selection is already frozen')
    if not dest.exists():
        dest.write_text(json.dumps(body, indent=1)+'\n')
        dest.chmod(0o400)
    print(json.dumps({'path': str(dest), 'sha256': body['sha256'], 'selected': selected, 'X3_launch': 'HOLD'}))


if __name__ == '__main__':
    run()
