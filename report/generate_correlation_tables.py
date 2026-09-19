"""Render tables and figures from the completed six-benchmark run; with descriptive sampled-label summaries.

Requires NumPy and Matplotlib. Run from any directory. All numerical inputs
are fingerprinted in correlation_tables_sources.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/correlation_pipeline/reasoning-vllm-v1'
SOURCE = RUN / 'analysis/unsloth--Qwen3.5-35B-A3B'
REPORT = ROOT / 'report'
DATASETS = {'math500': 'MATH-500', 'aime24': 'AIME24', 'aime25': 'AIME25', 'olympiad': 'Olympiad', 'amc23': 'AMC23', 'minerva': 'Minerva'}
CLASSES = ['Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor']
FEATURES = {
    'token_count': ('T', 'Token count'),
    'router_confidence_mean_layers': ('C', 'Router confidence'),
    'router_margin_mean_layers': ('M', 'Leading-expert margin'),
    'router_selected_mass_mean_layers': ('S', 'Selected probability mass'),
    'router_boundary_margin_mean_layers': ('B', 'Selection-boundary margin'),
    'router_topk_non_topk_gap_mean_layers': ('G', 'Selected/rest mean gap'),
    'hidden_norm_mean_layers': ('H', 'Hidden-state norm'),
    'hidden_step_distance_mean_layers': ('D', 'Hidden-state step distance'),
    'hidden_router_geometry_mean_layers': ('R', 'Hidden/router geometry'),
    'router_switch_rate_mean_layers': ('W', 'Top-1 switch rate'),
    'router_topk_overlap_mean_layers': ('O', 'Adjacent selected-set overlap'),
    'character_count': ('Ch', 'Character count'),
    'step_count': ('St', 'Step count'),
}
TARGETS = {'is_correct': 'Correctness', 'has_backtracking': 'Backtracking', 'has_contradiction': 'Contradiction', 'has_self_correction': 'Self-correction'}
MANIFEST = {}


def fingerprint(path):
    payload = path.read_bytes()
    MANIFEST[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return payload


def read(path):
    return json.loads(fingerprint(path))


def read_csv(path):
    fingerprint(path)
    with path.open() as handle:
        return list(csv.DictReader(handle))


def number(value):
    return '---' if value is None or not math.isfinite(value) else f'${value:.3f}$'


def cell(record, key='point_biserial_r'):
    return '---' if record is None else number(record[key]) + f' ({record["n_traces"]:,})'


def lookup(view, target='is_correct'):
    return {(r['scope'], r['feature']): r for r in view['binary_correlations'] if r['target'] == target}


def table(caption, label, headers, rows, spec=None):
    return '\n'.join([
        r'\begingroup\scriptsize\setlength{\tabcolsep}{4pt}',
        r'\begin{longtable}{@{}' + (spec or 'l' + 'r' * (len(headers) - 1)) + '@{}}',
        '\\caption{' + caption + '}\\label{' + label + r'}\\',
        r'\toprule', ' & '.join(headers) + r' \\', r'\midrule\endfirsthead',
        rf'\multicolumn{{{len(headers)}}}{{l}}{{\textit{{Continued from previous page}}}}\\',
        r'\toprule', ' & '.join(headers) + r' \\', r'\midrule\endhead',
        r'\bottomrule\endfoot',
        *[row if row == r'\midrule' else row + r' \\' for row in rows],
        r'\end{longtable}\endgroup', '',
    ])


def binary_rows(view, target='is_correct', features=None):
    records = lookup(view, target)
    if features is None:
        features = [f for f in FEATURES if any((d, f) in records for d in DATASETS)]
    return [FEATURES[f][1] + ' & ' + ' & '.join(cell(records.get((d, f))) for d in DATASETS) for f in features]


def target_pair_rows(view):
    """Tutor's layout: fix a target, then features in rows and datasets in columns."""
    records = view['cross_feature_correlations']['trace_level']
    pairs = {(r['scope'], frozenset((r['feature_x'], r['feature_y']))): r for r in records}
    features = [f for f in FEATURES if any(f in (r['feature_x'], r['feature_y']) for r in records)]
    if view.get('view') == 'position':
        assert 'token_count' not in features
    rows = []
    for target in features:
        rows.append(r'\multicolumn{7}{l}{\textbf{Target: ' + FEATURES[target][1] + '}}')
        for feature in features:
            if feature != target:
                rows.append(FEATURES[feature][1] + ' & ' + ' & '.join(
                    cell(pairs.get((d, frozenset((target, feature)))), 'spearman_rho') for d in DATASETS))
        rows.append(r'\midrule')
    return rows


def save_figure(fig, name):
    (REPORT / 'figures').mkdir(exist_ok=True)
    for suffix in ['pdf', 'png']:
        fig.savefig(REPORT / f'figures/{name}.{suffix}', dpi=180, bbox_inches='tight',
                    metadata={'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else {})
    plt.close(fig)


def figures(base, full, classes, positions):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True, layout='constrained')
    for ax, (name, view) in zip(axes, [('Whole continuation', base), ('Full reasoning', full)]):
        records = lookup(view)
        for j, (feature, color) in enumerate([('router_confidence_mean_layers', '#146a91'), ('router_margin_mean_layers', '#b24b2b')]):
            for i, d in enumerate(DATASETS):
                r = records.get((d, feature))
                if r:
                    y = i + (j - .5) * .22
                    if r['cluster_bootstrap_ci']:
                        ax.plot(r['cluster_bootstrap_ci'], [y, y], color=color, lw=1.5)
                    ax.plot(r['point_biserial_r'], y, 'o', color=color, label=FEATURES[feature][1] if i == 0 else None, ms=5)
                elif j == 0:
                    ax.text(0, i, 'undefined (all correct)', ha='center', va='center', fontsize=8, color='#666666')
        ax.axvline(0, color='#aaaaaa', lw=.8, zorder=0)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Correctness correlation r (95% bootstrap interval)')
        ax.set_xlim(-.8, .85)
        ax.set_yticks(range(len(DATASETS)), [f'{name} ({round(next(a["n_traces"] * (1 - a["accuracy_all"]) for a in base["generation_budget_audit"]["scopes"] if a["scope"] == d))} errors)' for d, name in DATASETS.items()])
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower left', fontsize=8)
    save_figure(fig, 'qwen_correctness_intervals')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout='constrained')
    for ax, feature in zip(axes, ['router_confidence_mean_layers', 'hidden_step_distance_mean_layers']):
        values = np.array([[lookup(v).get((d, feature), {}).get('point_biserial_r', np.nan) for d in DATASETS] for v in classes.values()])
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#eeeeee')
        plot = ax.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1, cmap=cmap, aspect='auto')
        for i, v in enumerate(classes.values()):
            for j, d in enumerate(DATASETS):
                r = lookup(v).get((d, feature))
                n = v['coverage']['datasets'][d]['traces_with_tokens']
                label = f'{r["point_biserial_r"]:+.2f}\nn={r["n_traces"]}' if r else f'—\nn={n}'
                ax.text(j, i, label, ha='center', va='center', fontsize=8,
                        color='white' if r and abs(r['point_biserial_r']) > .65 else '#222222')
        ax.set_xticks(range(len(DATASETS)), DATASETS.values(), rotation=25, ha='right')
        ax.set_yticks(range(7), CLASSES)
        ax.set_title(FEATURES[feature][1] + ' vs correctness', fontsize=11)
    fig.colorbar(plot, ax=axes, shrink=.8, label='Point-biserial r with correctness')
    save_figure(fig, 'qwen_class_correlations')

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True, layout='constrained')
    for ax, d in zip(axes.flat, DATASETS):
        for i, v in enumerate(positions.values()):
            r = lookup(v).get((d, 'router_confidence_mean_layers'))
            if r:
                ax.plot(i, r['point_biserial_r'], 'o', color='#146a91', ms=4)
                if r['cluster_bootstrap_ci']:
                    ax.vlines(i, *r['cluster_bootstrap_ci'], color='#146a91', lw=1)
        ax.axhline(0, color='#aaaaaa', lw=.8)
        ax.set_title(DATASETS[d]); ax.set_ylim(-1, 1)
        ax.set_xticks(range(11), [str(i) for i in range(10)] + ['Ov.'], fontsize=8)
    fig.supylabel('Router confidence vs correctness: r (95% problem-bootstrap CI)')
    fig.supxlabel(f'Absolute reasoning window (~{full["contract"]["position_reference"]["mean_reasoning_tokens"] / 10:,.1f} tokens per bin; Ov. = overflow)')
    save_figure(fig, 'qwen_position_coverage')

    # The call explicitly requests cross-metric relationships, not only accuracy.
    features = list(FEATURES)[:11]
    pairs = {(r['scope'], frozenset((r['feature_x'], r['feature_y']))): r
             for r in base['cross_feature_correlations']['trace_level']}
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), layout='constrained')
    for ax, d in zip(axes.flat, DATASETS):
        values = np.full((len(features), len(features)), np.nan)
        for i, left in enumerate(features):
            for j, right in enumerate(features):
                if j < i:
                    r = pairs.get((d, frozenset((left, right))))
                    if r:
                        values[i, j] = r['spearman_rho']
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('white')
        plot = ax.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1, cmap=cmap)
        codes = [FEATURES[f][0] for f in features]
        ax.set_xticks(range(len(features)), codes, fontsize=8)
        ax.set_yticks(range(len(features)), codes, fontsize=8)
        ax.set_title(f'{DATASETS[d]} (n={base["datasets"][d]})', fontsize=10)
    fig.supxlabel(' | '.join(code + ': ' + name for code, name in list(FEATURES.values())[:6]) + '\n' +
                  ' | '.join(code + ': ' + name for code, name in list(FEATURES.values())[6:11]), fontsize=8)
    fig.colorbar(plot, ax=list(axes.flat), shrink=.8, label='Spearman correlation between metrics')
    save_figure(fig, 'qwen_metric_pairs')


def repeated_outputs(base, full):
    datasets = ['aime24', 'aime25', 'amc23']
    rows = []
    records = {(r['dataset'], r['feature']): r for r in base['repeated_problem_analysis']['problem_level']
               if r['problem_feature'] == 'feature_mean'}
    for feature in list(FEATURES)[:11]:
        cells = []
        for d in datasets:
            r = records.get((d, feature))
            cells.append('---' if r is None else number(r['spearman_rho']) + f" ({r['n_problems']})")
        rows.append(FEATURES[feature][1] + ' & ' + ' & '.join(cells))
    text = table('Whole-continuation mean feature versus avg@32 correctness, Spearman $\\rho$ (source problems). All 32 attempts, including token-limit completions, enter each eligible group.',
                 'tab:qwen-repeated', ['Feature', *[DATASETS[d] for d in datasets]], rows)
    within = {(r['dataset'], r['feature']): r for r in base['repeated_problem_analysis']['within_problem']}
    rows = []
    for d in datasets:
        r = within[d, 'router_confidence_mean_layers']
        lo, hi = r['problem_bootstrap_ci']
        rows.append(f"{DATASETS[d]} & {r['n_mixed_outcome_problems']} & {r['mean_correct_minus_incorrect']:.5f} & [{lo:.5f}, {hi:.5f}]")
    text += table('Within-problem confidence differences, Qwen3.5: mean correct minus incorrect, equally weighted over mixed-outcome problems; saved 95\\% whole-problem bootstrap intervals.',
                  'tab:qwen-within', ['Dataset', 'Mixed problems', 'Difference', '95\\% interval'], rows)
    (REPORT / 'correlation_repeated_tables.tex').write_text(text)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True, layout='constrained')
    for ax, (name, view) in zip(axes, [('Whole continuation', base), ('Full reasoning', full)]):
        rec = {(r['dataset'], r['feature']): r for r in view['repeated_problem_analysis']['problem_level'] if r['problem_feature'] == 'feature_mean'}
        for j, (feature, color) in enumerate([('router_confidence_mean_layers', '#146a91'), ('token_count', '#b24b2b')]):
            for i, d in enumerate(datasets):
                r = rec[d, feature]; y = i + (j - .5) * .22
                if r['problem_bootstrap_ci']:
                    ax.plot(r['problem_bootstrap_ci'], [y, y], color=color)
                ax.plot(r['spearman_rho'], y, 'o', color=color, label=FEATURES[feature][1] if i == 0 else None)
        ax.axvline(0, color='#aaaaaa', lw=.8)
        ax.set_xlim(-1, 1); ax.set_title(name)
        ax.set_xlabel('Spearman correlation with avg@32 (95% interval)')
        ax.set_yticks(range(3), ['AIME24 (30 problems)', 'AIME25 (30 problems)', 'AMC23 (40 problems)'])
    axes[0].invert_yaxis(); axes[0].legend(fontsize=8, loc='upper center')
    save_figure(fig, 'qwen_repeated_intervals')


def main():
    base = read(SOURCE / 'correlations.json')
    fingerprint(ROOT / 'Call with Lorenzo Molfetta (2).vtt')
    audit = read(REPORT / 'correlation_run_audit.json')
    assert audit['analysis_sha256'] == MANIFEST[str((SOURCE / 'correlations.json').relative_to(ROOT))]
    forward = read(RUN / 'forward/unsloth--Qwen3.5-35B-A3B/summary.json')
    read(SOURCE / 'views-v1/summary.json')
    assert base['status'] == forward['status'] == base['reasoning_view_analysis']['status'] == 'complete'
    assert base['n_traces'] == 4647 and set(base['datasets']) == set(DATASETS)
    full = read(SOURCE / 'views-v1/full/reasoning/correlations.json')
    classes = {c: read(SOURCE / f'views-v1/class/{c}/correlations.json') for c in CLASSES}
    positions = {f'Bin {i}': read(SOURCE / f'views-v1/position/bin_{i:02d}/correlations.json') for i in range(10)}
    positions['Overflow'] = read(SOURCE / 'views-v1/position/overflow/correlations.json')
    frames = read_csv(SOURCE / 'trace_features.csv')
    for v in [full, *classes.values(), *positions.values()]:
        assert v['contract'] == full['contract'] and v['coverage']['traces'] == len(frames)
    overview = []
    for d in [*DATASETS, 'all']:
        a = next(r for r in base['generation_budget_audit']['scopes'] if r['scope'] == d)
        rows = [r for r in frames if d == 'all' or r['dataset'] == d]
        correct = sum(int(float(r['is_correct'])) for r in rows)
        overview.append(' & '.join([DATASETS.get(d, 'Pooled'), str(a['n_traces']), str(a['token_limit_hits']),
                                    str(len(rows)), str(correct), str(len(rows) - correct),
                                    f'{a["accuracy_all"]:.3f}', f'{correct / len(rows):.3f}']))
    generated = ['% Generated by generate_correlation_tables.py; numerical cells must not be edited.',
        table('Completed-run coverage and accuracy, Qwen3.5-35B-A3B. All supplied attempts are retained, including limit hits. AIME/AMC use 32 attempts per problem; other benchmarks use one. Unparseable answers with known gold are incorrect. Pooled accuracy is descriptive.',
              'tab:qwen-corr-data', ['Dataset', 'Input', 'Limit hits', 'Analyzed', 'Correct', 'Errors', 'Input acc.', 'Analyzed acc.'], overview)]
    generated.append(table('Whole-continuation aggregate features versus correctness: point-biserial $r$ (finite traces), Qwen3.5-35B-A3B. Token-limit completions are included. A dash denotes an undefined coefficient.',
                           'tab:qwen-corr-correct', ['Feature', *DATASETS.values()], binary_rows(base, features=list(FEATURES)[:11])))
    (REPORT / 'correlation_overview_tables.tex').write_text('\n'.join(generated))

    label_counts = Counter()
    retained_ids = {(r['dataset'], r['problem_id']) for r in frames}
    for d in DATASETS:
        path = RUN / f'annotations/Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/{d}/annotations.jsonl'
        for line in fingerprint(path).decode().splitlines():
            a = json.loads(line)
            if (d, a['problem_id']) in retained_ids:
                label_counts.update(u['label'] for u in a['units'])
    rows = []
    for c, v in classes.items():
        counts = [v['coverage']['datasets'][d]['traces_with_tokens'] for d in DATASETS]
        rows.append(c + ' & ' + f'{label_counts[c]:,}' + ' & ' + ' & '.join(map(str, counts)) + f' & {sum(counts)}')
    (REPORT / 'correlation_class_table.tex').write_text(table(
        'Class coverage in the completed run. Qwen3.8-27B labels on Qwen3.5-35B-A3B generations. Units are tagged sentences; dataset columns count traces with selected class tokens. Classes overlap across traces; missing classes are not zero-valued features.',
        'tab:qwen-class-coverage', ['Class', 'Units', *DATASETS.values(), 'Total traces'], rows))
    rows = []
    for d in [*DATASETS, 'all']:
        rs = [r for r in frames if d == 'all' or r['dataset'] == d]
        counts = [sum(int(float(r[t])) for r in rs) for t in list(TARGETS)[1:]]
        rows.append(DATASETS.get(d, 'Pooled') + f' & {len(rs)} & ' + ' & '.join(f'{n} ({100*n/len(rs):.1f}\\%)' for n in counts))
    (REPORT / 'correlation_event_table.tex').write_text(table(
        'Whole-attempt lexical-event prevalence among retained Qwen3.5 traces. Counts and percentages refer to attempts, not sentence classes; detectors are not independently validated episode annotations.',
        'tab:qwen-corr-events', ['Dataset', 'Traces', 'Backtracking', 'Contradiction', 'Self-correction'], rows))
    rows = []
    for d in DATASETS:
        rs = [r for r in base['expert_identity_analysis']['trace_point_biserial'] if r['scope'] == d and r['target'] == 'is_correct']
        if not rs:
            rows.append(DATASETS[d] + f' & Undefined (all correct) & {base["datasets"][d]} & 0 & --- & ---')
            continue
        r = max(rs, key=lambda x: abs(x['point_biserial_r']))
        feature = r['feature'].replace('expert_pair_top1_e', 'Top-1 ').replace('_top2_e', ' / top-2 ').replace('expert_any_topk_rate_e', 'Top-8 ID ').replace('expert_top1_rate_e', 'Top-1 ID ').removesuffix('_rate')
        rows.append(' & '.join([DATASETS[d], feature, str(r['n_traces']), str(r['n_negative']), number(r['point_biserial_r']), f'${r["benjamini_hochberg_q_value"]:.3g}$']))
    (REPORT / 'correlation_expert_table.tex').write_text(table(
        'Post-hoc largest absolute correctness coefficient per dataset in the expert-rate screen, Qwen3.5-35B-A3B. Point-biserial $r$ and BH $q$ within dataset/target; no expert bootstrap intervals. Numeric IDs pool six distinct layer-local modules. Repeated-attempt trace-level q-values are diagnostic, not cluster-adjusted inference.',
        'tab:qwen-corr-experts', ['Dataset', 'Pooled-ID rate', 'Traces', 'Errors', '$r$', '$q$'], rows, 'llrrrr'))

    # Reuse the reported correctness-selected candidates: do not select new
    # maxima independently for each continuous target.
    candidates = []
    for d in DATASETS:
        rs = [r for r in base['expert_identity_analysis']['trace_point_biserial']
              if r['scope'] == d and r['target'] == 'is_correct']
        if rs:
            r = max(rs, key=lambda x: abs(x['point_biserial_r']))
            candidates.append((d, r['feature']))
    rates = {(r['scope'], r['feature'], r['target']): r
             for r in base['expert_identity_analysis']['trace_spearman']}
    rows = []
    for dataset, candidate in candidates:
        rows.append(r'\multicolumn{7}{l}{\textbf{Candidate selected on ' + DATASETS[dataset] + r':} \texttt{' + candidate.replace('_', r'\_') + '}}')
        for target, (_, name) in FEATURES.items():
            rows.append(name + ' & ' + ' & '.join(cell(rates.get((d, candidate, target)), 'spearman_rho') for d in DATASETS))
        rows.append(r'\midrule')
    (REPORT / 'correlation_expert_metric_table.tex').write_text(table(
        'Whole-continuation expert-rate associations with continuous metrics, Qwen3.5-35B-A3B. Spearman $\\rho$ (finite traces). The six candidates are the correctness-selected maxima in Table~\\ref{tab:qwen-corr-experts}; this is a descriptive follow-up, not independent validation or layer-specific expert identification.',
        'tab:qwen-expert-metrics', ['Target metric', *DATASETS.values()], rows))

    parts = [r'''% Generated by generate_correlation_tables.py; do not edit numerical cells.
\section{Complete aggregate correlation tables}
\label{sec:complete-correlation-tables}
These tables use the Qwen3.5-35B-A3B run completed September 17, 2026
(GPTQ-Int4 generation, runtime-4-bit replay): all 4,647 attempts from 1,547
source problems, including 412 token-limit completions. AIME24, AIME25,
and AMC23 retain all 32 attempts per problem. All values are saved
coefficients, not newly fitted statistics. Cells give a coefficient and
its finite trace count. A dash denotes an undefined coefficient.

Whole-continuation metrics include every continuation token; full-reasoning
and position views use every reasoning token, independently of tagging.
Class views use available tagged sentence tokens on sample-zero attempts.
Position bins use the all-generation mean of 12,498.062 reasoning tokens,
with ten absolute windows and overflow. Empty views are pairwise missing;
class transitions require original adjacency within a sentence.

Each binary table contains all available prespecified aggregate metrics for
correctness and the three whole-attempt lexical flags. Feature-pair tables
use Spearman $\rho$ and the tutor's requested layout: each panel fixes a
target metric, with the remaining named metrics in rows and benchmarks in
columns. Each symmetric association appears under both target orientations;
these are duplicated presentations, not independent results. Self-correlations
are omitted. Metric codes used in the overview figure are defined below.
These tables make no significance claim. Pointwise intervals and
within-family BH adjustments remain in the fingerprinted JSON files;
neither adjusts comparisons selected across views. Exploratory layer-specific
coefficients and the full 640-feature expert screen remain in the source
artifacts. Repeated-problem and within-problem summaries appear in the main text.
''']
    parts.append(table('Aggregate metric key for Qwen3.5 correlation tables.', 'tab:qwen-metric-key', ['Code', 'Metric'], [code + ' & ' + name for code, name in FEATURES.values()], 'll'))
    views = [('Whole continuation', 'continuation', base), ('Full reasoning', 'reasoning', full)]
    views += [('Class: ' + c, 'class-' + c, v) for c, v in classes.items()]
    views += [('Position: ' + name, 'position-' + name.lower().replace(' ', '-'), v) for name, v in positions.items()]
    for name, slug, view in views:
        parts.append(r'\subsection{' + name + '}')
        rows = []
        for target, title in TARGETS.items():
            rows.append(r'\multicolumn{7}{l}{\textbf{' + title + '}}')
            rows.extend(binary_rows(view, target))
            rows.append(r'\midrule')
        parts.append(table(name + ': aggregate point-biserial correlations with whole-attempt targets, $r$ (finite traces). Qwen3.5-35B-A3B completed run.', 'tab:qwen-binary-' + slug, ['Feature', *DATASETS.values()], rows))
        parts.append(table(name + ': correlations organized by target metric, Spearman $\\rho$ (finite traces). Qwen3.5-35B-A3B completed run. Each panel repeats the same feature-row, benchmark-column layout for a different target.', 'tab:qwen-pairs-' + slug, ['Feature', *DATASETS.values()], target_pair_rows(view)))
    parts.append(r'\subsection{Expert rates versus continuous metrics}')
    parts.append(r'\input{correlation_expert_metric_table}')
    (REPORT / 'correlation_tables.tex').write_text('\n'.join(parts))
    repeated_outputs(base, full)
    figures(base, full, classes, positions)
    import sys
    import tutor_figures
    tutor_figures.generate(sys.modules[__name__], base, full, classes, positions, frames, views)
    (REPORT / 'correlation_tables_sources.json').write_text(json.dumps(MANIFEST, indent=2) + '\n')
    print(f'Generated {len(views)} views, coverage/correctness/class/event/expert and repeated-attempt tables, plot atlas and tutor-requested figures; {len(MANIFEST)} fingerprinted inputs.')


if __name__ == '__main__':
    main()
