"""Rescore every saved, manifested identity/margin evaluation without inference.

Install report/rescoring_requirements.txt. Run with PYTHONPATH=src. Raw outputs,
manifests, policies and calibration labels remain immutable; outputs are sidecars.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from moe_exp.utils import extract_model_answer
from moe_exp.correlation_pipeline.offline_scoring import CONTRACT, equivalent
from moe_exp.moe_identity_guiding.bootstrap import paired_accuracy_bootstrap


def judge(pair):
    return equivalent(*pair)


def comparison(baseline, guided, partial=False):
    keys = sorted(guided); baseline = {k: baseline[k] for k in keys}
    assert baseline.keys() == guided.keys()
    for key in keys:
        for field in ['input', 'prompt_sha256', 'sampling_args']:
            assert baseline[key][field] == guided[key][field], (key, field)
    result = {'num_examples': len(keys), 'calibration_overlap': [], 'scoring_contract': CONTRACT}
    for name, group in [('baseline', baseline), ('guided', guided)]:
        result[name] = {'accuracy': sum(r['is_correct'] for r in group.values())/len(keys),
                        'correct': sum(r['is_correct'] for r in group.values()),
                        'mean_generated_tokens': sum(r['generated_token_count'] for r in group.values())/len(keys),
                        'truncated': sum(r['finish_reason'] == 'length' for r in group.values())}
    result['accuracy_delta'] = result['guided']['accuracy']-result['baseline']['accuracy']
    result['mean_generated_tokens_delta'] = result['guided']['mean_generated_tokens']-result['baseline']['mean_generated_tokens']
    result['wrong_to_right'] = sum(not baseline[k]['is_correct'] and guided[k]['is_correct'] for k in keys)
    result['right_to_wrong'] = sum(baseline[k]['is_correct'] and not guided[k]['is_correct'] for k in keys)
    if not partial:
        result['bootstrap'] = paired_accuracy_bootstrap(baseline, guided, workers=1)
    return result


def main():
    out = {'scoring_contract': CONTRACT,
           'method': 'Re-extract last nonempty balanced boxed answer, then existing answer-marker/last-number fallbacks. Compare normalized display text or pinned symbolic expressions; finite scalar numbers use relative tolerance 1e-6 with no absolute floor. Score only the extracted candidate. Parse/verify failures and unsupported embedded prose receive false unless display strings match; list them for review.',
           'scope': 'All manifested saved identity/margin evaluations, including 13 complete full stochastic, two greedy, one restricted Minerva and the interrupted Qwen subset. Original generation records, frozen policies and calibration/correlation labels are preserved.',
           'versions': {p: version(p) for p in ['math-verify','latex2sympy2_extended','antlr4-python3-runtime','sympy','mpmath']},
           'source_sha256': {}, 'conditions': {}, 'runs': {}, 'changes': []}
    for name in ['src/moe_exp/correlation_pipeline/offline_scoring.py','src/moe_exp/utils.py','src/moe_exp/moe_identity_guiding/bootstrap.py','report/rescoring_requirements.txt']:
        out['source_sha256'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    originals = {}; manifests = {}; candidates = set()
    for mp in sorted((ROOT/'results').glob('moe_*guiding/**/guided/manifest.json')):
        folder = mp.parent.parent; root = str(folder.relative_to(ROOT)); manifests[root] = []
        assert (folder/'baseline/manifest.json').exists()
        for condition in ['baseline','guided']:
            rel = root+'/'+condition; originals[rel] = {}
            path = folder/condition/'manifest.json'; raw = path.read_bytes()
            out['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
            m = json.loads(raw); manifests[root].append(m)
            assert m['condition'] == condition
            assert m['status'] == 'complete' or (condition == 'guided' and m['status'] == 'failed' and m['error'].startswith('KeyboardInterrupt'))
            digest = hashlib.sha256(); path = folder/condition/'generations.jsonl'
            with path.open('rb') as handle:
                for line in handle:
                    digest.update(line); row = json.loads(line); key = row['id']
                    assert key not in originals[rel] and type(row['is_correct']) is bool
                    answer = extract_model_answer(row['text']); gold = row['input']['gold_answer']; assert gold
                    candidates.add((answer,gold))
                    originals[rel][key] = {k:row[k] for k in ['input','is_correct','generated_token_count','finish_reason','model_answer','scoring_method']}
                    originals[rel][key].update(answer=answer, sampling_args=row.get('sampling_args'),
                        prompt_sha256=hashlib.sha256(json.dumps(row['prompt_token_ids']).encode()).hexdigest())
            out['source_sha256'][str(path.relative_to(ROOT))] = digest.hexdigest()
        b,g = manifests[root]
        for field in ['prompts_sha256','policy_sha256','engine_args','sampling_args','versions','scoring_contract']:
            assert b[field] == g[field], (root,field)
        assert not g['calibration_overlap']
    pairs = sorted(candidates); print(f'{sum(map(len, originals.values()))} records, {len(pairs)} distinct answer/gold pairs', flush=True)
    scored = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, (pair, decision) in enumerate(zip(pairs, pool.map(judge, pairs, chunksize=20)), 1):
            scored[pair] = decision
            if i % 1000 == 0: print(f'Scored {i}/{len(pairs)} distinct pairs', flush=True)
    for rel, rows in originals.items():
        out['conditions'][rel] = {}; counts = Counter()
        for key,row in rows.items():
            ok,method = scored[(row['answer'],row['input']['gold_answer'])]; old = row['is_correct']
            rec = {'is_correct':ok, 'model_answer':row['answer'], 'method':method,
                   'original_is_correct':old, 'original_model_answer':row['model_answer']}
            out['conditions'][rel][key] = rec
            if ok != old or row['answer'] != row['model_answer']:
                out['changes'].append({'condition':rel,'id':key,'gold_answer':row['input']['gold_answer'],**rec})
            row['is_correct'] = ok
            counts[method] += 1
    for root,(b,g) in manifests.items():
        a,z = (originals[root+'/'+c] for c in ['baseline','guided']); partial = g['status'] != 'complete'
        if not partial: assert a.keys() == z.keys()
        else: assert len(z) < len(a)
        c = comparison(a,z,partial)
        old_correct = [sum(out['conditions'][root+'/'+cond][k]['original_is_correct'] for k in z) for cond in ['baseline','guided']]
        out['runs'][root] = {'status':'partial' if partial else 'complete','comparison':c,
                             'original_correct':old_correct, 'original_delta':(old_correct[1]-old_correct[0])/len(z)}
        print(root, old_correct, '->', [c[k]['correct'] for k in ['baseline','guided']], 'delta', round(100*c['accuracy_delta'],3), flush=True)
    out['method_counts'] = dict(Counter(r['method'] for rows in out['conditions'].values() for r in rows.values()))
    out['judgment_changes'] = dict(Counter(f"{r['original_is_correct']}->{r['is_correct']}" for r in out['changes'] if r['original_is_correct'] != r['is_correct']))
    (ROOT/'report/guiding_rescoring.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Judgment changes',out['judgment_changes'],flush=True)


if __name__ == '__main__':
    main()
