"""Render complete guiding pairs, probe comparisons, and judge effort from saved outputs."""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report'
sources = {}

def read(path):
    path = ROOT / path
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

rescoring = read('report/guiding_rescoring.json')

def table(name, caption, label, columns, header, rows):
    text = '\\begin{table*}[ht]\n\\centering\\small\n'
    text += f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
    text += '\\resizebox{\\textwidth}{!}{%\n' if name in ('guiding_summary_table.tex', 'identity_comparison_table.tex', 'margin_comparison_table.tex', 'guiding_benchmark_changes.tex', 'guiding_weighting_table.tex', 'guiding_qwen_margin_breakdown.tex', 'guiding_token_table.tex') else ''
    text += '\\begin{tabular}{' + columns + '}\n\\toprule\n'
    text += header + ' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(row) + ' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}'
    text += '}' if name in ('guiding_summary_table.tex', 'identity_comparison_table.tex', 'margin_comparison_table.tex', 'guiding_benchmark_changes.tex', 'guiding_weighting_table.tex', 'guiding_qwen_margin_breakdown.tex', 'guiding_token_table.tex') else ''
    text += '\n\\end{table*}\n'
    (OUT / name).write_text(text)

models = [('Qwen', 'qwen3.5-35b-a3b-gptq-int4'), ('GPT-OSS', 'gpt-oss-20b'), ('Gemma', 'gemma-4-26b-a4b-it-nf4')]
probes = [read(f'results/probeTest/{directory}/probes/results.json') for _, directory in models]
rows = []
for target in probes[0]['config']['targets']:
    row = [target]
    for model in probes:
        r = model['best_by_target'][target]
        row += [str(r['layer_idx']), f"{r['test_f1']:.3f}", f"{r['test_auc']:.3f}"]
    rows.append(row)
table('probe_multimodel_table.tex', 'Accuracy-selected boundary probes: Qwen3.5 GPTQ, GPT-OSS-20B, and Gemma-4 NF4. Each model reports hidden-state index, F1, and AUROC; selection and evaluation share the sentence-level test split.', 'tab:probe-models', 'lrrrrrrrrr', ' & \\multicolumn{3}{c}{Qwen} & \\multicolumn{3}{c}{GPT-OSS} & \\multicolumn{3}{c}{Gemma} \\\\\nTarget & Index & F1 & AUROC & Index & F1 & AUROC & Index & F1 & AUROC', rows)
path = ROOT / 'results/gepaLLMAsJudge/thinking-low-medium-20260915/summary.csv'
sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
rows = [[r['level'].capitalize()] + [f"{float(r[k]):.4f}" for k in ('accuracy', 'balanced_accuracy', 'macro_f1')] for r in csv.DictReader(path.open())]
table('judge_effort_table.tex', 'Frozen Qwen3.8-27B NVFP4 judge on 407 reused validation units. Metrics are exact accuracy, balanced accuracy, and macro-F1.', 'tab:judge-effort', 'lrrr', 'Effort & Accuracy & Balanced accuracy & Macro-F1', rows)

def records(path):
    digest = hashlib.sha256()
    result = {}
    with (ROOT / path).open('rb') as handle:
        for line in handle:
            digest.update(line)
            r = json.loads(line)
            assert r['id'] not in result, 'Duplicate ID'
            assert type(r['is_correct']) is bool, 'Unscored answer'
            result[r['id']] = {k: r[k] for k in ('input', 'prompt_token_ids', 'sampling_args', 'is_correct', 'generated_token_count', 'finish_reason')}
    sources[path] = digest.hexdigest()
    assert sources[path] == rescoring['source_sha256'][path], path
    decisions = rescoring['conditions'][str(Path(path).parent)]
    assert result.keys() == decisions.keys()
    for key, row in result.items():
        assert row['is_correct'] == decisions[key]['original_is_correct']
        row['is_correct'] = decisions[key]['is_correct']
    return result

summaries = {}
rows, dataset_rows = [], []
for label, directory in [('Identity $+1$', 'moe_identity_guiding/qwen_global'), ('Identity $-1$', 'moe_identity_guiding/qwen_strength_-1'), ('Margin $1$', 'moe_margin_guiding/qwen_global')]:
    root = f'results/{directory}/sampling/full'
    manifests = [read(f'{root}/{condition}/manifest.json') for condition in ('baseline', 'guided')]
    for m, condition in zip(manifests, ('baseline', 'guided')):
        assert m['status'] == 'complete' and m['condition'] == condition
    for key in ('prompts_sha256', 'policy_sha256', 'engine_args', 'sampling_args', 'versions', 'scoring_contract'):
        assert manifests[0][key] == manifests[1][key], key
    baseline, guided = [records(f'{root}/{condition}/generations.jsonl') for condition in ('baseline', 'guided')]
    assert baseline and baseline.keys() == guided.keys()
    for key, a in baseline.items():
        b = guided[key]
        for field in ('input', 'prompt_token_ids', 'sampling_args'):
            assert a[field] == b[field], field
    def summarize(ids):
        n = len(ids)
        correct = [sum(group[k]['is_correct'] for k in ids) for group in (baseline, guided)]
        return dict(n=n, correct=correct, delta_pp=100*(correct[1]-correct[0])/n,
                    tokens=[sum(group[k]['generated_token_count'] for k in ids)/n for group in (baseline,guided)],
                    truncated=[sum(group[k]['finish_reason']=='length' for k in ids) for group in (baseline,guided)],
                    wrong_to_right=sum(not baseline[k]['is_correct'] and guided[k]['is_correct'] for k in ids),
                    right_to_wrong=sum(baseline[k]['is_correct'] and not guided[k]['is_correct'] for k in ids))
    comparison = read(f'{root}/comparison_bootstrap.json')
    for source, expected in comparison['source_sha256'].items():
        assert sources[source] == expected, f'Stale bootstrap comparison: {source}'
    bootstrap = rescoring['runs'][root]['comparison']['bootstrap']
    assert bootstrap['replicates'] == 5000 and bootstrap['seed'] == 42
    assert bootstrap['method'] == 'paired_dataset_stratified_problem_percentile'
    def interval(summary):
        bounds = summary['ci95']
        return '--' if bounds is None else f"[{100*bounds[0]:+.2f}, {100*bounds[1]:+.2f}]"
    total = summarize(list(baseline))
    assert total['n'] == 1395
    summaries[label] = {'total': total, 'datasets': {}, 'bootstrap': bootstrap}
    rows.append([label] + [f'{v/total["n"]:.4f}' for v in total['correct']] + [f'{total["delta_pp"]:+.2f}', interval(bootstrap['overall']), f'{total["wrong_to_right"]}/{total["right_to_wrong"]}', '/'.join(str(v) for v in total['truncated']), '/'.join(f'{v:.0f}' for v in total['tokens'])])
    for dataset in sorted({r['input']['dataset'] for r in baseline.values()}):
        r = summarize([k for k in baseline if baseline[k]['input']['dataset']==dataset])
        summaries[label]['datasets'][dataset] = r
        dataset_rows.append([label, dataset.replace('_', '\\_'), str(r['n']), '/'.join(map(str,r['correct'])), f'{r["delta_pp"]:+.2f}', interval(bootstrap['datasets'][dataset])])
table('guiding_summary_table.tex', 'Complete matched Qwen3.5 GPTQ stochastic evaluations, 1,395 attempts on 465 held-out problems per comparison. B/G denotes baseline/guided; changes are percentage points. Flips are wrong-to-right/right-to-wrong. Intervals are paired, dataset-stratified source-problem bootstrap percentiles (5,000 resamples, seed 42).', 'tab:guiding-summary', 'lrrrrrrr', 'Policy & B accuracy & G accuracy & Change & 95\\% CI & Flips & Limit hits B/G & Mean tokens B/G', rows)
table('guiding_dataset_table.tex', 'Qwen3.5 GPTQ guiding results by benchmark. Correct counts are baseline/guided; accuracy changes are percentage points. Intervals resample whole problems (5,000 resamples, seed 42); each AIME benchmark contains only nine evaluation problems.', 'tab:guiding-datasets', 'llrrrr', 'Policy & Dataset & Attempts & Correct B/G & Change & 95\\% CI', dataset_rows)

# Discover complete identity pairs, including polarity and paper variants.
sys.path.insert(0, str(ROOT / 'src'))
from moe_exp.moe_identity_guiding.run import compare

identity_summaries = {}
identity_rows = []
for folder in sorted((ROOT / 'results/moe_identity_guiding').glob('**/sampling/full')):
    paths = [folder / condition / 'manifest.json' for condition in ('baseline', 'guided')]
    if not all(p.exists() for p in paths):
        continue
    manifests = [json.loads(p.read_text()) for p in paths]
    if any(m['status'] != 'complete' for m in manifests):
        print(f'Skipping incomplete identity comparison: {folder.relative_to(ROOT)}')
        continue
    result = rescoring['runs'][str(folder.relative_to(ROOT))]['comparison']
    assert result['num_examples'] == 1395 and not result['calibration_overlap']
    assert result['bootstrap']['overall']['num_problems'] == 465
    for condition in ('baseline', 'guided'):
        for filename in ('manifest.json', 'generations.jsonl'):
            path = folder / condition / filename
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            sources[str(path.relative_to(ROOT))] = digest.hexdigest()
            assert digest.hexdigest() == rescoring['source_sha256'][str(path.relative_to(ROOT))]
    manifest = manifests[1]
    policy = manifest['policy']
    model = {'openai/gpt-oss-20b': 'GPT-OSS', 'Qwen/Qwen3.5-35B-A3B-GPTQ-Int4': 'Qwen'}[policy['model']]
    method = policy.get('guiding_method', 'fixed').capitalize()
    polarity = policy.get('expert_polarity', 'positive').capitalize()
    strength = f"{manifest['strength']:+g}"
    identity_summaries[str(folder.relative_to(ROOT))] = result
    identity_rows.append([model, method, polarity, strength,
                          f"{100*result['baseline']['accuracy']:.2f}",
                          f"{100*result['guided']['accuracy']:.2f}",
                          f"{100*result['accuracy_delta']:+.2f}",
                          interval(result['bootstrap']['overall'])])
table('identity_comparison_table.tex',
      'Complete identity-guiding stochastic comparisons. Accuracy is percent; changes and 95\\% intervals are percentage points relative to each matched baseline. Fixed denotes the lift-weighted additive bias; Paper denotes extreme-score steering with epsilon 0.01. Polarity identifies the selected expert set, independently of signed strength. Each comparison has 1,395 attempts on 465 problems; intervals use 5,000 paired, dataset-stratified source-problem bootstrap resamples (seed 42).',
      'tab:identity-comparison', 'llllrrrr',
      'Model & Method & Polarity & Strength & B accuracy & G accuracy & Change & 95\\% CI',
      identity_rows)

# Add the verified margin comparisons and secondary weighting analysis.
sensitivity = read('report/guiding_sensitivity.json')
for source, expected in sensitivity['source_sha256'].items():
    if source not in sources:
        digest = hashlib.sha256()
        with (ROOT / source).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        sources[source] = digest.hexdigest()
    assert sources[source] == expected, f'Stale sensitivity analysis: {source}'
margin_summaries, margin_rows = {}, []
benchmark_rows, weighting_rows = [], []
datasets = ['math500', 'aime24', 'aime25', 'olympiad', 'amc23', 'minerva']
dataset_names = ['MATH-500', 'AIME24', 'AIME25', 'Olympiad', 'AMC23', 'Minerva']
for root, result in sensitivity['runs'].items():
    label = result['label']
    benchmark_rows.append([label] + [f"{100*result['datasets'][d]['accuracy_delta']:+.2f}" for d in datasets])
    a, p = result['attempt_weighted'], result['problem_weighted']
    weighting_rows.append([label, f"{100*a['accuracy_delta']:+.2f}", interval(a),
                           f"{100*p['accuracy_delta']:+.2f}", interval(p)])
    if '/moe_margin_guiding/' in root:
        r = result['comparison']
        m = read(root + '/guided/manifest.json')
        model = 'GPT-OSS' if m['policy']['model'] == 'openai/gpt-oss-20b' else 'Qwen'
        margin_summaries[root] = r
        margin_rows.append([model, str(m['engine_args']['max_num_seqs']),
                            f"{100*r['baseline']['accuracy']:.2f}", f"{100*r['guided']['accuracy']:.2f}",
                            f"{100*r['accuracy_delta']:+.2f}", interval(r['bootstrap']['overall'])])
table('margin_comparison_table.tex',
      'Complete stochastic margin comparisons at strength 1, with 1,395 attempts on 465 problems each. B/G denotes baseline/guided accuracy in percent; changes and intervals are percentage points. Concurrency is the maximum active-sequence count. Each row uses its own matched baseline; this is not a controlled concurrency comparison.',
      'tab:margin-comparison', 'lrrrrr', 'Model & Concurrency & B & G & Change & 95\\% CI', margin_rows)
table('guiding_benchmark_changes.tex',
      'Benchmark-level accuracy changes in percentage points for all completed stochastic interventions. Positive/negative denotes expert polarity; c denotes concurrency. Entries are point estimates, not significance decisions. Benchmark counts (problems/attempts) are shown in the header. Full benchmark intervals and correct counts are saved in guiding\_sensitivity.json.',
      'tab:guiding-benchmark-changes', 'lrrrrrr',
      r'Condition & MATH-500 & AIME24 & AIME25 & Olympiad & AMC23 & Minerva \\ & 150/150 & 9/288 & 9/288 & 203/203 & 12/384 & 82/82', benchmark_rows)
table('guiding_weighting_table.tex',
      'Sensitivity to the evaluation unit: accuracy changes and 95\\% intervals in percentage points. Attempt weighting is the primary statistic; equal-problem weighting first averages correctness over each problem\'s attempts and then averages the 465 problems. Both use the same 5,000 paired, dataset-stratified problem resamples (seed 42). Equal problem weights are not equal benchmark weights. This is a secondary exploratory analysis without multiplicity adjustment.',
      'tab:guiding-weighting', 'lrrrr',
      r'Condition & Attempt change & 95\% CI & Problem change & 95\% CI', weighting_rows)
key = 'results/moe_margin_guiding/qwen_strength_1/sampling/full'
qwen_rows = []
for ds, label in zip(datasets, dataset_names):
    r = sensitivity['runs'][key]['datasets'][ds]
    qwen_rows.append([label, str(r['num_problems']), str(r['num_attempts']),
                      f"{r['baseline_correct']}/{r['guided_correct']}", f"{r['net_correct']:+d}",
                      f"{100*r['accuracy_delta']:+.2f}", interval(r)])
table('guiding_qwen_margin_breakdown.tex',
      'Benchmark contributions to Qwen margin guiding at concurrency 25. B/G denotes baseline/guided correct counts; net is guided minus baseline correct attempts. Changes and pointwise 95\\% intervals are percentage points, with 5,000 paired problem resamples (seed 42).',
      'tab:guiding-qwen-margin-breakdown', 'lrrrrrr',
      r'Benchmark & Problems & Attempts & Correct B/G & Net & Change & 95\% CI', qwen_rows)
token_rows = []
for run in sensitivity['runs'].values():
    comparison = run['comparison']
    baseline, guided = comparison['baseline'], comparison['guided']
    token_rows.append([run['label'], f"{baseline['mean_generated_tokens']:.0f}",
                       f"{guided['mean_generated_tokens']:.0f}",
                       str(baseline['truncated']), str(guided['truncated'])])
table('guiding_token_table.tex',
      'Generated-token means and token-limit hits for all 13 complete stochastic comparisons. Each row has 1,395 matched attempts; B/G denotes baseline/guided. Token statistics are descriptive.',
      'tab:guiding-tokens', 'lrrrr',
      'Condition & Mean tokens B & Mean tokens G & Limit hits B & Limit hits G', token_rows)
(OUT / 'current_results_sources.json').write_text(json.dumps({'sha256': sources, 'guiding': summaries, 'identity_guiding': identity_summaries, 'margin_guiding': margin_summaries, 'guiding_sensitivity': sensitivity, 'scoring_contract': rescoring['scoring_contract'], 'rescoring_source': 'report/guiding_rescoring.json'}, indent=2)+'\n')
print(f'Rendered probe, judge-effort, original guiding tables, and {len(identity_rows)} complete identity comparisons, plus margin and weighting tables.')
