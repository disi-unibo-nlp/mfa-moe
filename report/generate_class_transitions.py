"""Render outcome-conditioned, directed transitions from saved sentence labels.

Run: MPLCONFIGDIR=/tmp/moe-report-mpl python3 report/generate_class_transitions.py
"""
import json

import numpy as np
import matplotlib.pyplot as plt

import generate_correlation_tables as api

MODELS = [
    ('qwen', 'Qwen3.5-35B-A3B', 'reasoning-vllm-v1',
     'unsloth--Qwen3.5-35B-A3B', 'Qwen--Qwen3.5-35B-A3B-GPTQ-Int4'),
    ('gemma', 'Gemma-4-26B-A4B', 'gemma-nvfp4-nf4/reasoning-vllm-v1',
     'google--gemma-4-26B-A4B-it', 'nvidia--Gemma-4-26B-A4B-NVFP4'),
    ('gptoss', 'GPT-OSS-20B', 'gpt-oss-20b/reasoning-vllm-v1',
     'openai--gpt-oss-20b', 'openai--gpt-oss-20b'),
]


def transition_counts(units, labels):
    """Preserve direction and self-transitions; never bridge unlabelled indices."""
    units = sorted(units, key=lambda u: u['index'])
    if len({u['index'] for u in units}) != len(units):
        raise ValueError('Duplicate sentence index')
    lookup = {label: i for i, label in enumerate(labels)}
    for unit in units:
        if unit['label'] not in lookup:
            raise ValueError('Unknown sentence label')
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for left, right in zip(units, units[1:]):
        if right['index'] == left['index'] + 1:
            counts[lookup[left['label']], lookup[right['label']]] += 1
    return counts


def summarize(matrices, size):
    counts = sum(matrices, start=np.zeros((size, size), dtype=int))
    n = len(matrices)
    return {'attempts': n, 'attempts_with_pairs': sum(bool(m.sum()) for m in matrices),
            'pairs': int(counts.sum()), 'counts': counts.tolist(),
            'mean_counts_per_attempt': (counts / n).tolist() if n else None}


def main():
    results = {}
    for slug, title, run, model, generation in MODELS:
        root = api.ROOT / 'results/correlation_pipeline' / run
        rows = api.read_csv(root / 'analysis' / model / 'trace_features.csv')
        outcomes = {(r['dataset'], r['problem_id']): r for r in rows}
        assert len(outcomes) == len(rows)
        groups = {(d, y): [] for d in api.DATASETS for y in (0, 1)}
        seen, problems = set(), set()
        empty = 0
        for d in api.DATASETS:
            path = root / 'annotations' / generation / d / 'annotations.jsonl'
            for line in api.fingerprint(path).splitlines():
                ann = json.loads(line)
                key = ann['dataset'], ann['problem_id']
                assert key[0] == d and key not in seen
                seen.add(key)
                row = outcomes[key]
                problem = d, row['source_problem_id']
                assert problem not in problems
                problems.add(problem)
                value = float(row['is_correct'])
                assert value in (0, 1)
                if not ann['units']:
                    empty += 1
                    continue
                groups[d, int(value)].append(transition_counts(ann['units'], api.CLASSES))
        pooled = {name: summarize([m for d in api.DATASETS for m in groups[d, y]], len(api.CLASSES))
                  for y, name in [(1, 'correct'), (0, 'incorrect')]}
        per_dataset = {d: {name: summarize(groups[d, y], len(api.CLASSES))
                           for y, name in [(1, 'correct'), (0, 'incorrect')]}
                       for d in api.DATASETS}
        results[slug] = {'model': title, 'empty_label_attempts_excluded': empty,
                         'pooled': pooled, 'by_dataset': per_dataset}
        print(title, {k: {f: v[f] for f in ['attempts', 'attempts_with_pairs', 'pairs']}
                      for k, v in pooled.items()}, flush=True)
    vmax = max(np.max(g['mean_counts_per_attempt']) for model in results.values()
               for g in model['pooled'].values() if g['attempts'])
    text = [r'\subsection{Directed class transitions by answer correctness}',
            r'For each model, we separate labelled attempts by whole-answer correctness and count directed transitions between originally adjacent labelled sentences. Rows are source classes and columns are destination classes. Self-transitions are retained; missing labels and sampling gaps are never bridged. Each cell is the total observed count divided by the number of labelled attempts in that outcome group: equivalently, the mean per-attempt count, including zero-pair attempts. Empty-label attempts are excluded. These are counts per attempt, not row-normalized probabilities or full-trace transition estimates.',
            r'The two panels per model pool the six benchmarks and all six panels share one colour scale. Benchmark-specific counts and denominators are saved in the accompanying statistics JSON. Outcome groups differ in benchmark mix, reasoning length, and label coverage; Qwen also retains a different annotation protocol from Gemma and GPT-OSS. Differences are descriptive associations, not evidence that a transition causes correctness or a controlled comparison between models.']
    for slug, result in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), layout='constrained')
        for ax, (name, group) in zip(axes, result['pooled'].items()):
            matrix = (np.array(group['mean_counts_per_attempt']) if group['attempts']
                      else np.full((len(api.CLASSES), len(api.CLASSES)), np.nan))
            im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=vmax, cmap='Blues')
            for i, j in np.ndindex(matrix.shape):
                ax.text(j, i, f'{matrix[i,j]:.2f}' if np.isfinite(matrix[i,j]) else '—',
                        ha='center', va='center', fontsize=8,
                        color='white' if matrix[i,j] > .55 * vmax else 'black')
            ax.set_xticks(range(len(api.CLASSES)), api.CLASSES, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(api.CLASSES)), api.CLASSES, fontsize=8)
            ax.set_xlabel('Next sentence class')
            ax.set_ylabel('Current sentence class')
            ax.set_title(f'{name.title()}: {group["attempts"]:,} attempts\n'
                         f'{group["pairs"]:,} pairs; {group["attempts_with_pairs"]:,} attempts with pairs', fontsize=10)
        fig.suptitle(result['model'] + ': observed directed class transitions')
        fig.colorbar(im, ax=axes, shrink=.8, label='Mean observed transitions per labelled attempt')
        filename = slug + '_class_transitions_by_correctness'
        api.save_figure(fig, filename)
        caption = (result['model'] + r': directed class transitions for correct and incorrect answers. '
                   r'Cells show mean observed counts per labelled attempt, including self-transitions on the diagonal. '
                   r'Panel headings give denominators and adjacent-pair coverage; zero-pair attempts remain in the mean. '
                   r'The shared colour scale is identical across all three models. Sampling and population differences limit interpretation.')
        text.append(r'\begin{figure}[p]\centering' + '\n' +
                    r'\includegraphics[width=\linewidth]{figures/' + filename + '.pdf}\n' +
                    r'\caption{' + caption + '}\n' + r'\label{fig:' + filename.replace('_', '-') + '}\n' +
                    r'\Description{Two square heatmaps show directed class transition counts, split by answer correctness.}' + '\n' +
                    r'\end{figure}')
    (api.REPORT / 'class_transition_results.tex').write_text('\n\n'.join(text) + '\n')
    (api.REPORT / 'class_transition_statistics.json').write_text(json.dumps(
        {'classes': api.CLASSES, 'normalization': 'mean observed counts per labelled attempt, including zero-pair attempts',
         'shared_color_max': float(vmax), 'models': results}, indent=2, allow_nan=False) + '\n')
    (api.REPORT / 'class_transition_sources.json').write_text(json.dumps(api.MANIFEST, indent=2) + '\n')


if __name__ == '__main__':
    main()
