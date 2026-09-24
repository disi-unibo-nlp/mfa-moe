"""Audit imported labels against replay checkpoints and render their report summary."""
import hashlib
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def render(api, figure):
    tables, audits, populations, class_views = [], {}, [], []
    for source, label, run, model, generation in [
        ('gemma', 'Gemma', 'gemma-nvfp4-nf4', 'google--gemma-4-26B-A4B-it', 'nvidia--Gemma-4-26B-A4B-NVFP4'),
        ('gpt', 'GPT-OSS', 'gpt-oss-20b', 'openai--gpt-oss-20b', 'openai--gpt-oss-20b'),
    ]:
        root = api.ROOT / 'results/correlation_pipeline' / run / 'reasoning-vllm-v1'
        imported = api.read(root / 'annotations/import_manifest.json')
        summary = api.read(root / 'annotations/source_summary.json')
        sampling = api.read(root / 'annotations/sampling_manifest.json')
        api.read(root / 'annotations/source_manifest.json')
        failures = api.read(root / 'annotations/failures.json')
        analysis = root / 'analysis' / model
        csvs, views = {}, {}
        for cls in api.CLASSES:
            folder = analysis / 'views-v1/class' / cls
            csvs[cls] = {(r['dataset'], r['problem_id']): r for r in api.read_csv(folder / 'trace_features.csv')}
            views[cls] = api.read(folder / 'correlations.json')
        annotations, dataset_counts, labels = {}, Counter(), Counter()
        for path in sorted((root / 'annotations' / generation).glob('*/annotations.jsonl')):
            payload = api.fingerprint(path)
            assert hashlib.sha256(payload).hexdigest() == imported['datasets'][path.parent.name]['sha256']
            for line in payload.splitlines():
                ann = json.loads(line)
                key = ann['dataset'], ann['problem_id']
                assert key not in annotations
                annotations[key] = ann
                dataset_counts[ann['dataset']] += len(ann['units'])
                labels.update(u['label'] for u in ann['units'])
        assert sum(labels.values()) == imported['labels'] and dict(labels) == summary['labels']
        assert len(annotations) == imported['traces'] == sampling['selected_traces']
        assert len(failures) == imported['unknown']
        assert sum(dataset_counts.values()) + len(failures) == sampling['selected_sentences'] == 100000
        rows = api.read_csv(analysis / 'trace_features.csv')
        problems = {(r['dataset'], r['source_problem_id']) for r in rows if (r['dataset'], r['problem_id']) in annotations}
        assert len(problems) == len(annotations)  # One labelled completion per problem.
        checked = 0
        # Check every replay, including traces that no longer carry old labels.
        for row in rows:
            key = row['dataset'], row['problem_id']
            ann = annotations.get(key)
            safe_id = row['problem_id'].replace('/', '_').replace('\\', '_')
            path = root / 'forward' / model / row['dataset'] / 'tensors' / f'{row["dataset"]}_{safe_id}_extraction.json'
            if not path.exists():
                # Empty reasoning can legitimately have no extraction checkpoint.
                assert ann is None and all(not float(csvs[c][key]['token_count'] or 0) for c in api.CLASSES)
                continue
            checkpoint = api.read(path)
            expected = hashlib.sha256(json.dumps(ann, sort_keys=True).encode()).hexdigest()
            assert checkpoint['config']['annotation_sha256'] == expected, path
            scopes = {v['name']: v for v in checkpoint['correlation_views']['scopes'] if v['view'] == 'class'}
            for cls in api.CLASSES:
                saved = csvs[cls][key]
                scope = scopes[cls]
                assert float(saved['token_count'] or 0) == scope['token_count'], (path, cls)
                for feature in api.FEATURES:
                    if feature not in scope['values']:
                        continue
                    expected_value = scope['values'][feature]
                    observed = float(saved[feature]) if saved.get(feature) else np.nan
                    assert (expected_value is None and not np.isfinite(observed)) or np.isclose(observed, expected_value, equal_nan=True), (path, cls, feature)
            checked += 1
        for dataset in api.DATASETS:
            n = sum(a['dataset'] == dataset for a in annotations.values())
            missing = sum(r['identity']['dataset'] == dataset for r in failures)
            assert dataset_counts[dataset] + missing == summary['datasets'][dataset]
            tables.append(f'{label} & {api.DATASETS[dataset]} & {n:,} & {dataset_counts[dataset]:,} & {missing}')
        audits[source] = {'labelled_traces': len(annotations), 'source_problems': len(problems),
                          'valid_labels': sum(labels.values()), 'unknown_labels': len(failures),
                          'replay_checkpoints_checked': checked, 'datasets': dict(dataset_counts),
                          'labels': dict(labels), 'partial_traces': sum(a['status'] == 'partial' for a in annotations.values())}
        populations.append((label, labels, dataset_counts))
        class_views.append((label, views))
        print(label, 'stratified labels and replay/CSV bindings verified', flush=True)
    text = [r'\subsection{Stratified annotation coverage}',
        r'The September 21 export replaces the earlier Gemma error-weighted sample and GPT-OSS corpus-prefix labels. Each model has 100,000 selected sentences, drawn with seed 42 from one completion per source problem, with quotas proportional to dataset $\times$ correctness $\times$ reasoning-length-quartile supply. The four 25,000-sentence parts are merged, not independent experimental replicates. GPT-OSS uses the corrected length-based sampling plan, which falls back to replay tokens when server reasoning-token counts are zero.',
        r'GPT-OSS contributes 100,000 valid labels on 1,510 problems; Gemma contributes 99,999 on 1,503 problems. One Gemma sentence has a missing judge response and is excluded; its trace remains partially labelled. Both models now have class-token coverage on all six benchmarks. Sampled class populations remain smaller than the full 4,647-attempt corpus and need not contain the same problems across models. Sampling eligibility uses the saved scored-trace population; subsequently retained invalid answers in the full-corpus accuracy analysis do not imply annotation eligibility.',
        api.table('Stratified annotation population. Problems each contribute one labelled completion; valid labels are sentence counts, not token counts. Missing labels are retained in the sampling record.', 'tab:stratified-labels', ['Model', 'Dataset', 'Problems', 'Valid labels', 'Missing'], tables, 'llrrr')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for ax, keys, index, title in [(axes[0], api.CLASSES, 1, 'Predicted class composition'), (axes[1], list(api.DATASETS), 2, 'Benchmark composition')]:
        x = np.arange(len(keys))
        for i, item in enumerate(populations):
            counts = item[index]
            ax.bar(x + (i-.5)*.36, [100*counts[k]/sum(counts.values()) for k in keys], .36,
                   label=item[0], color=['#b24b2b', '#54803b'][i])
        ax.set_xticks(x, keys if index == 1 else list(api.DATASETS.values()), rotation=35, ha='right')
        ax.set_ylabel('Share of valid sampled sentence labels (%)')
        ax.set_title(title)
        ax.legend()
    api.save_figure(fig, 'stratified_label_composition')
    text.append(figure('stratified_label_composition', r'Gemma and GPT-OSS stratified label composition. Denominators are 99,999 and 100,000 valid sampled sentences. Equal sample budgets do not imply equal class or benchmark proportions. These sentence shares are neither complete-corpus token shares nor adjusted model effects.'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    for ax, (label, views) in zip(axes, class_views):
        matrix = np.full((len(api.CLASSES), len(api.DATASETS)), np.nan)
        for i, cls in enumerate(api.CLASSES):
            lookup = api.lookup(views[cls])
            for j, dataset in enumerate(api.DATASETS):
                r = lookup.get((dataset, 'router_confidence_mean_layers'))
                if r is not None:
                    matrix[i, j] = r['point_biserial_r']
                ax.text(j, i, '—' if r is None else f'{matrix[i,j]:+.2f}\nn={r["n_traces"]}',
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(matrix[i,j]) > .65 else 'black')
        im = ax.imshow(np.ma.masked_invalid(matrix), vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(6), api.DATASETS.values(), rotation=40, ha='right')
        ax.set_yticks(range(7), api.CLASSES)
        ax.set_title(label + ': class-conditioned confidence')
    fig.colorbar(im, ax=axes, shrink=.75, label='Point-biserial r with correctness')
    api.save_figure(fig, 'stratified_class_confidence')
    text.append(figure('stratified_class_confidence', r'Class-conditioned router confidence versus whole-attempt correctness after stratified relabelling. Each cell reports the saved point-biserial coefficient and its finite-attempt count. Cohorts include only attempts with tokens assigned to that class; missing coefficients are undefined, not zero. Sparse AIME/AMC samples, near-ceiling outcomes, and model-specific problem selection limit comparisons. This descriptive screen does not adjust for length or difficulty or establish a causal class effect.'))
    text.append(r'The common annotation procedure improves benchmark coverage but does not match model cohorts. Full class/metric matrices are in the appended atlases. Qwen retains its earlier sampling and judge deployment, so the three-model class comparisons are not a controlled model comparison.')
    (api.REPORT / 'stratified_label_results.tex').write_text('\n'.join(text)+'\n')
    return audits
