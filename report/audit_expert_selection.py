"""Audit saved identity policies; RD sensitivity retains our calibration weights.

This is not SteerMoE replication: it does not recover token-pooled contrasts.
Run: python3 report/audit_expert_selection.py
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'report'


def audit_policy(policy):
    pi = policy['baseline_accuracy']
    problems = len(policy['calibration_problems'])
    assert 0 < pi < 1 and problems > 0
    direction = 1 if policy.get('expert_polarity', 'positive') == 'positive' else -1
    result = {}
    for layer, info in policy['layers'].items():
        stats = []
        for expert in info['experts']:
            if not expert['frequency_mass']:
                continue
            f = expert['frequency_mass'] / problems
            acc = expert['accuracy']
            positive_rate = f * acc / pi
            negative_rate = f * (1 - acc) / (1 - pi)
            rd = positive_rate - negative_rate
            assert math.isclose(expert['lift'], acc - pi, abs_tol=1e-12)
            assert math.isclose(expert['lift'], pi * (1 - pi) * rd / f, abs_tol=1e-12)
            stats.append({**expert, 'mean_frequency': f,
                          'correct_conditional_rate': positive_rate,
                          'incorrect_conditional_rate': negative_rate,
                          'problem_balanced_rd': rd})
        eligible = [s for s in stats if s['problem_support'] >= policy['min_support']
                    and direction * s['lift'] > 0]
        lifted = sorted(eligible, key=lambda s: (-direction * s['lift'], -s['problem_support'], s['expert']))[:policy['max_experts']]
        rd_ranked = sorted(eligible, key=lambda s: (-direction * s['problem_balanced_rd'], -s['problem_support'], s['expert']))[:policy['max_experts']]
        saved = {i for i, s in enumerate(info['scores']) if s > 0}
        assert saved == {s['expert'] for s in lifted}
        peak = lifted[0]['lift'] if lifted else 1
        for s in lifted:
            assert math.isclose(info['scores'][s['expert']], s['lift'] / peak, abs_tol=1e-12)
        rd_set = {s['expert'] for s in rd_ranked}
        result[layer] = {'saved_selected': sorted(saved), 'rd_selected': sorted(rd_set),
                         'overlap': len(saved & rd_set), 'selected_count': len(saved),
                         'exceeds_dispatch_k': len(saved) > policy['top_k'], 'experts': stats}
    return result


def main():
    sources, policies, runs = {}, {}, []
    for path in sorted((ROOT / 'results/moe_identity_guiding').glob('**/sampling/full/guided/manifest.json')):
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if manifest['status'] != 'complete':
            continue
        policy = manifest['policy']
        source = str(path.relative_to(ROOT))
        sources[source] = hashlib.sha256(raw).hexdigest()
        # Steering method/strength do not enter expert selection; group identical selections/stats.
        signature = {k: policy[k] for k in ['model', 'baseline_accuracy', 'calibration_problems',
                     'layers', 'min_support', 'max_experts', 'top_k', 'estimator']}
        signature['expert_polarity'] = policy.get('expert_polarity', 'positive')
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        policies.setdefault(key, {'model': policy['model'], 'polarity': signature['expert_polarity'],
                                 'top_k': policy['top_k'], 'layers': audit_policy(policy)})
        runs.append({'manifest': source, 'selection_group': key,
                     'method': policy.get('guiding_method', 'fixed'), 'strength': manifest['strength']})
    assert policies
    for rel in ['src/moe_exp/moe_identity_guiding/calibration.py',
                'src/moe_exp/moe_identity_guiding/routing.py']:
        sources[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    output = {'scope': 'Completed full-sampling guided manifests; rank sensitivity on existing problem-balanced weights, not the paper token-pooled estimator',
              'source_sha256': sources, 'policies': policies, 'runs': runs}
    (REPORT / 'expert_selection_audit.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    text = [r'\begin{table}[ht]\centering\small',
            r'\caption{Selection sensitivity using saved calibration sufficient statistics. RD here uses the same problem-balanced trace weights, support filter, polarity, layer set, and per-layer budget as lift; it is not a reproduction of the paper\textquotesingle s token-pooled contrastive estimator. Overlap counts layer--expert identities.}',
            r'\label{tab:selection-audit}', r'\begin{tabular}{llrrl}\toprule',
            r'Model & Polarity & Layers & Overlap & Overlap per layer \\\midrule']
    for p in policies.values():
        name = 'GPT-OSS' if 'gpt-oss' in p['model'] else 'Qwen'
        layers = p['layers']
        common = sum(v['overlap'] for v in layers.values())
        total = sum(v['selected_count'] for v in layers.values())
        detail = ', '.join(f'{k}: {v["overlap"]}/{v["selected_count"]}' for k, v in layers.items())
        text.append(f'{name} & {p["polarity"]} & {len(layers)} & {common}/{total} & {detail}' + r' \\')
        print(name, p['polarity'], f'{common}/{total}', detail)
    text += [r'\bottomrule\end{tabular}\end{table}']
    (REPORT / 'expert_selection_audit_table.tex').write_text('\n'.join(text) + '\n')


if __name__ == '__main__':
    main()
