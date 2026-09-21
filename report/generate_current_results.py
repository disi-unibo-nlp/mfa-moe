"""Render complete guiding pairs, probe comparisons, and judge effort from saved outputs."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report'
sources = {}

def read(path):
    path = ROOT / path
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def table(name, caption, label, columns, header, rows):
    text = '\\begin{table*}[ht]\n\\centering\\small\n'
    text += f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
    text += '\\begin{tabular}{' + columns + '}\n\\toprule\n'
    text += header + ' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(row) + ' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
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
    total = summarize(list(baseline))
    assert total['n'] == 1395
    summaries[label] = {'total': total, 'datasets': {}}
    rows.append([label] + [f'{v/total["n"]:.4f}' for v in total['correct']] + [f'{total["delta_pp"]:+.2f}', f'{total["wrong_to_right"]}/{total["right_to_wrong"]}', '/'.join(str(v) for v in total['truncated']), '/'.join(f'{v:.0f}' for v in total['tokens'])])
    for dataset in sorted({r['input']['dataset'] for r in baseline.values()}):
        r = summarize([k for k in baseline if baseline[k]['input']['dataset']==dataset])
        summaries[label]['datasets'][dataset] = r
        dataset_rows.append([label, dataset.replace('_', '\\_'), str(r['n']), '/'.join(map(str,r['correct'])), f'{r["delta_pp"]:+.2f}'])
table('guiding_summary_table.tex', 'Complete matched Qwen3.5 GPTQ stochastic evaluations, 1,395 attempts on 465 held-out problems per comparison. B/G denotes baseline/guided; changes are percentage points. Flips are wrong-to-right/right-to-wrong. These are descriptive point estimates without cluster confidence intervals.', 'tab:guiding-summary', 'lrrrrrr', 'Policy & B accuracy & G accuracy & Change & Flips & Limit hits B/G & Mean tokens B/G', rows)
table('guiding_dataset_table.tex', 'Qwen3.5 GPTQ guiding results by benchmark. Correct counts are baseline/guided; accuracy changes are percentage points. Repeated attempts from one source problem are dependent.', 'tab:guiding-datasets', 'llrrr', 'Policy & Dataset & Attempts & Correct B/G & Change', dataset_rows)
(OUT / 'current_results_sources.json').write_text(json.dumps({'sha256': sources, 'guiding': summaries}, indent=2)+'\n')
print('Rendered probe, judge-effort, and three complete matched guiding comparisons.')
