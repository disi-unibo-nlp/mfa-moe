"""Report the interrupted Qwen paper intervention on exact completed-ID pairs.

Run: python3 report/analyze_partial_guiding.py
Does not modify experiment artifacts or mark the run complete.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/moe_identity_guiding/qwen_global/variants/positive_paper_eps_0.01/strength_1/sampling/full'
DATASETS = {'aime24': 'AIME24', 'aime25': 'AIME25', 'amc23': 'AMC23',
            'math500': 'MATH-500', 'minerva': 'Minerva', 'olympiad': 'Olympiad'}


def summarize(baseline, guided, ids):
    if not ids:
        return None
    result = {'attempts': len(ids), 'problems': len({(guided[k]['input']['dataset'], guided[k]['input']['source_problem_id']) for k in ids})}
    for name, group in [('baseline', baseline), ('guided', guided)]:
        tokens = [group[k]['generated_token_count'] for k in ids]
        result[name] = {'correct': sum(group[k]['is_correct'] for k in ids),
                        'accuracy': sum(group[k]['is_correct'] for k in ids) / len(ids),
                        'mean_tokens': statistics.mean(tokens), 'median_tokens': statistics.median(tokens),
                        'limit_hits': sum(group[k]['finish_reason'] == 'length' for k in ids),
                        'finish_reasons': dict(Counter(group[k]['finish_reason'] for k in ids))}
    result['accuracy_delta_pp'] = 100 * (result['guided']['accuracy'] - result['baseline']['accuracy'])
    result['mean_tokens_delta'] = result['guided']['mean_tokens'] - result['baseline']['mean_tokens']
    result['wrong_to_right'] = sum(not baseline[k]['is_correct'] and guided[k]['is_correct'] for k in ids)
    result['right_to_wrong'] = sum(baseline[k]['is_correct'] and not guided[k]['is_correct'] for k in ids)
    return result


def main():
    hashes = {}
    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return raw
    manifests = [json.loads(read(RUN / name / 'manifest.json')) for name in ('baseline', 'guided')]
    b, g = manifests
    assert b['status'] == 'complete' and g['status'] == 'failed'
    assert g['error'].startswith('KeyboardInterrupt')
    assert b['condition'] == 'baseline' and g['condition'] == 'guided'
    for key in ['prompts_sha256', 'policy_sha256', 'engine_args', 'sampling_args', 'versions',
                'scoring_contract', 'seed_strategy', 'token_scope', 'template_date']:
        assert b[key] == g[key], key
    assert not g['calibration_overlap']
    assert g['policy']['guiding_method'] == 'paper' and g['policy']['expert_polarity'] == 'positive'
    assert g['strength'] == 1 and g['policy']['top_k'] == 8
    assert all(sum(s > 0 for s in layer['scores']) == 8 for layer in g['policy']['layers'].values())
    assert all(r['condition'] == 'baseline' and not r['layers'] for r in b['routing_diagnostics'])
    assert g['routing_diagnostics']
    for r in g['routing_diagnostics']:
        assert r['condition'] == 'guided' and set(r['layers']) == set(g['policy']['layers'])
        assert all(v['token_evaluations'] > 0 for v in r['layers'].values())
    groups = []
    for condition in ('baseline', 'guided'):
        raw = read(RUN / condition / 'generations.jsonl')
        assert raw.endswith(b'\n'), 'Incomplete JSONL tail'
        group = {}
        for line in raw.splitlines():
            row = json.loads(line)
            assert row['id'] not in group and row['condition'] == condition
            assert type(row['is_correct']) is bool
            group[row['id']] = {k: row[k] for k in ['input', 'prompt_token_ids', 'sampling_args',
                               'is_correct', 'generated_token_count', 'finish_reason']}
        groups.append(group)
    rescoring = json.loads(read(ROOT / 'report/guiding_rescoring.json'))
    for condition, group in zip(('baseline', 'guided'), groups):
        decisions = rescoring['conditions'][str((RUN / condition).relative_to(ROOT))]
        assert group.keys() == decisions.keys()
        for key, row in group.items():
            assert row['is_correct'] == decisions[key]['original_is_correct']
            row['is_correct'] = decisions[key]['is_correct']
    baseline, guided = groups
    assert len(baseline) == b['completed_count'] == b['expected_count'] == g['expected_count']
    assert len(guided) == g['completed_count'] < len(baseline)
    assert set(guided) <= set(baseline)
    for k in guided:
        for field in ('input', 'prompt_token_ids', 'sampling_args'):
            assert guided[k][field] == baseline[k][field], (k, field)
    planned = Counter(r['input']['dataset'] for r in baseline.values())
    datasets = {}
    for d in DATASETS:
        ids = [k for k in guided if guided[k]['input']['dataset'] == d]
        datasets[d] = {'planned_attempts': planned[d], 'completed_attempts': len(ids),
                       'summary': summarize(baseline, guided, ids)}
    result = {'scoring_contract': rescoring['scoring_contract'], 'status': 'interrupted_partial', 'completed_attempts': len(guided),
              'planned_attempts': len(baseline), 'remaining_attempts': len(baseline) - len(guided),
              'overall': summarize(baseline, guided, list(guided)), 'datasets': datasets,
              'source_sha256': hashes,
              'limitations': 'Completed-ID subset only; scheduling, benchmark order and output length can affect completion. No full-run estimate or causal population claim; no intervals reported for this selected partial cohort.'}
    (ROOT / 'report/partial_guiding_statistics.json').write_text(json.dumps(result, indent=2) + '\n')
    r = result['overall']; bs, gs = r['baseline'], r['guided']
    text = [r'\subsection{Interrupted Qwen paper-style intervention: partial results}',
            r'\label{sec:partial-qwen-paper}',
            f'The Qwen positive-expert paper-style intervention at strength $+1$ was stopped by user request after {len(guided):,} of {len(baseline):,} planned attempts ({100*len(guided)/len(baseline):.1f}\\%). '
            f'The manifest records a KeyboardInterrupt; {len(baseline)-len(guided):,} attempts have no saved guided result. '
            'This policy targets eight experts at each of six layers, matching the model routing top-$k=8$; it is unaffected by the OSS target-count correction. '
            'The intervention uses epsilon 0.01, concurrency 25, and a 32,768-token completion budget. The saved outputs remain resumable; the run is not marked complete.',
            f'On the {len(guided):,} exact matched baseline/guided IDs ({r["problems"]} source problems), '
            f'baseline accuracy is {100*bs["accuracy"]:.2f}\\% ({bs["correct"]}/{len(guided)}), '
            f'and guided accuracy is {100*gs["accuracy"]:.2f}\\% ({gs["correct"]}/{len(guided)}), '
            f'a change of ${r["accuracy_delta_pp"]:+.2f}$ percentage points. '
            f'There are {r["wrong_to_right"]} wrong-to-right and {r["right_to_wrong"]} right-to-wrong flips. '
            f'Mean generated length changes from {bs["mean_tokens"]:,.0f} to {gs["mean_tokens"]:,.0f} tokens '
            f'(medians {bs["median_tokens"]:,.0f} and {gs["median_tokens"]:,.0f}); '
            f'token-limit hits change from {bs["limit_hits"]}/{len(guided)} to {gs["limit_hits"]}/{len(guided)}.',
            r'\begin{table}[ht]\centering\small',
            r'\caption{Partial Qwen paper-style intervention. Accuracy and token counts compare only the same completed IDs in each condition. B/G denotes baseline/guided. A dash indicates no completed guided attempts, not zero accuracy.}',
            r'\label{tab:partial-qwen-paper}',
            r'\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrr}\toprule',
            r'Benchmark & Completed/planned & Problems & Accuracy B/G (\%) & Change (pp) & Mean tokens B/G & Limit hits B/G \\\midrule']
    for d, name in DATASETS.items():
        x = datasets[d]; y = x['summary']
        if y:
            a, z = y['baseline'], y['guided']
            text.append(f'{name} & {x["completed_attempts"]}/{x["planned_attempts"]} & {y["problems"]} & '
                        f'{100*a["accuracy"]:.2f}/{100*z["accuracy"]:.2f} & ${y["accuracy_delta_pp"]:+.2f}$ & '
                        f'{a["mean_tokens"]:,.0f}/{z["mean_tokens"]:,.0f} & {a["limit_hits"]}/{z["limit_hits"]}' + r' \\')
        else:
            text.append(f'{name} & 0/{x["planned_attempts"]} & 0 & --- & --- & --- & ---' + r' \\')
    text += [r'\bottomrule\end{tabular}}\end{table}',
             r'These are descriptive partial results, excluded from the complete-run comparison and weighting tables. The completed subset is not a random sample: benchmark submission order, scheduling, and generation length can affect which requests have finished. Multiple attempts from one problem are also dependent. Consequently the pooled difference is not an estimate for the full 1,395-attempt evaluation, and no population confidence interval or significance claim is reported. Missing attempts are not counted as wrong or filled from the baseline. Increased token-limit incidence does not by itself establish repetitive looping.',
             r'All saved guided records have unique IDs, scored boolean outcomes, matching baseline inputs, prompt token IDs, and per-attempt sampling settings. Policy, engine, software, scoring, and template-date contracts also match; hook diagnostics confirm activity at all selected layers. Source hashes and benchmark denominators are saved in \path{report/partial_guiding_statistics.json}.']
    (ROOT / 'report/partial_guiding_results.tex').write_text('\n\n'.join(text) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
