"""Audit saved three-model results and render comparisons plus complete scope atlases.

Run: MPLCONFIGDIR=/tmp/moe-report-mpl python3 report/generate_multimodel_correlations.py
No inference or experiment outputs are modified.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import generate_correlation_tables as api
from audit_tagged_correlation_run import check_coefficients, check_repeated

MODELS = [
    ('qwen', 'Qwen3.5-35B-A3B', 'reasoning-vllm-v1', 'unsloth--Qwen3.5-35B-A3B'),
    ('gemma', 'Gemma-4-26B-A4B', 'gemma-nvfp4-nf4/reasoning-vllm-v1', 'google--gemma-4-26B-A4B-it'),
    ('gptoss', 'GPT-OSS-20B', 'gpt-oss-20b/reasoning-vllm-v1', 'openai--gpt-oss-20b'),
]


def figure(name, caption):
    return (r'\begin{figure}[p]\centering' + '\n' +
            r'\includegraphics[width=\linewidth]{figures/' + name + '.pdf}\n' +
            r'\caption{' + caption + '}\n' + r'\Description{' + caption + '}\n' +
            r'\end{figure}' + '\n')


def atlas_plot(label, name, view, filename):
    fs = [f for f in list(api.FEATURES)[:11] if not (view.get('view') == 'position' and f == 'token_count')]
    pairs = {(r['scope'], frozenset((r['feature_x'], r['feature_y']))): r
             for r in view['cross_feature_correlations']['trace_level']}
    correct = api.lookup(view)
    fig, axes = plt.subplots(2, 3, figsize=(11, 8), layout='constrained')
    for ax, d in zip(axes.flat, api.DATASETS):
        m = np.full((len(fs), len(fs)+1), np.nan)
        for i, f in enumerate(fs):
            for j, g in enumerate(fs[:i]):
                m[i, j] = pairs.get((d, frozenset((f, g))), {}).get('spearman_rho', np.nan)
            m[i, -1] = correct.get((d, f), {}).get('point_biserial_r', np.nan)
        im = ax.imshow(np.ma.masked_invalid(m), vmin=-1, vmax=1, cmap='RdBu_r')
        for i, j in np.ndindex(m.shape):
            if j < i or j == len(fs):
                ax.text(j, i, f'{m[i,j]:.2f}' if np.isfinite(m[i,j]) else '—',
                        ha='center', va='center', fontsize=5,
                        color='white' if abs(m[i,j]) > .65 else 'black')
        ax.set_xticks(range(len(fs)+1), [api.FEATURES[f][0] for f in fs]+['Acc'], fontsize=7)
        ax.set_yticks(range(len(fs)), [api.FEATURES[f][0] for f in fs], fontsize=7)
        n = view.get('coverage', {}).get('datasets', {}).get(d, {}).get('traces_with_tokens', view.get('datasets', {}).get(d, 0))
        ax.set_title(f'{api.DATASETS[d]} (token-bearing n={n})', fontsize=9)
    fig.suptitle(label + ': ' + name)
    fig.colorbar(im, ax=list(axes.flat), shrink=.8, label='Correlation coefficient')
    api.save_figure(fig, filename)


def supplementary_plots(loaded):
    """Compare repeated-problem estimates and display each model's expert IDs."""
    parts = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout='constrained')
    for ax, feature in zip(axes, ['router_confidence_mean_layers', 'token_count']):
        for j, (slug, label, base, full) in enumerate(loaded):
            records = {(r['dataset'], r['feature']): r for r in base['repeated_problem_analysis']['problem_level'] if r['problem_feature'] == 'feature_mean'}
            for i, d in enumerate(['aime24', 'aime25', 'amc23']):
                r = records.get((d, feature))
                if r is None:
                    continue
                y = i + (j-1)*.22
                color = ['#146a91', '#b24b2b', '#54803b'][j]
                if r['problem_bootstrap_ci']:
                    ax.plot(r['problem_bootstrap_ci'], [y,y], color=color)
                ax.plot(r['spearman_rho'], y, 'o', color=color, label=label if i == 0 else None)
        ax.axvline(0, color='gray', lw=.6)
        ax.set_xlim(-1, 1)
        ax.set_yticks(range(3), ['AIME24 (30)', 'AIME25 (30)', 'AMC23 (40)'])
        ax.set_title(api.FEATURES[feature][1])
        ax.set_xlabel('Spearman rho with avg@32 (95% bootstrap CI)')
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8)
    api.save_figure(fig, 'multimodel_repeated')
    parts.append(figure('multimodel_repeated', 'All models: whole-continuation problem means versus avg@32, with saved pointwise problem-bootstrap intervals. All 32 attempts enter each problem mean.'))
    for slug, label, base, full in loaded:
        expert = base['expert_identity_analysis']
        assert expert['status'] == 'complete'
        ids = sorted({r['expert_id'] for r in expert['feature_metadata'] if r['kind'] == 'any_topk'})
        assert ids
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), layout='constrained')
        for ax, target, key, coefficient in zip(axes, ['is_correct', 'token_count'], ['trace_point_biserial', 'trace_spearman'], ['point_biserial_r', 'spearman_rho']):
            records = {(r['scope'], r['feature']): r for r in expert[key] if r['target'] == target}
            m = np.array([[records.get((d, f'expert_any_topk_rate_e{e}'), {}).get(coefficient, np.nan) for e in ids] for d in api.DATASETS])
            im = ax.imshow(np.ma.masked_invalid(m), vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto', interpolation='nearest')
            ticks = np.unique(np.linspace(0, len(ids)-1, min(12, len(ids)), dtype=int))
            ax.set_xticks(ticks, [ids[i] for i in ticks])
            ax.set_yticks(range(6), api.DATASETS.values())
            ax.set_title(label + ': selected-expert frequency vs ' + ('correctness (r)' if target == 'is_correct' else 'token count (rho)'))
        axes[-1].set_xlabel('Model-local expert ID, pooled across selected layers')
        fig.colorbar(im, ax=axes, shrink=.8, label='Descriptive coefficient')
        api.save_figure(fig, slug+'_multimodel_experts')
        parts.append(figure(slug+'_multimodel_experts', label+': selected-pool expert frequency correlations with correctness and token count. IDs are model-local and pooled across selected layers; they do not identify matching experts across models. Missing coefficients are blank.'))
    (api.REPORT/'multimodel_supplementary_plots.tex').write_text('\n'.join(parts)+'\n')


def main():
    loaded, audit, overview, protocol, coverage = [], {}, [], [], []
    atlas = [r'\section{Gemma and GPT-OSS correlation plot atlas}',
             r'Lower triangles show metric--metric Spearman $\rho$; Acc shows point-biserial $r$ with correctness. Missing or undefined values are dashes, not zero. Token-bearing counts are upper bounds on pairwise finite counts. Each model has whole-continuation, full-reasoning, seven class, and eleven position scopes. Qwen plots appear in the preceding atlas. These are descriptive screens without significance selection.',
             api.table('Metric key for all models.', 'tab:multi-key', ['Code', 'Metric'], [c+' & '+n for c,n in list(api.FEATURES.values())[:11]], 'll')]
    numeric = []
    for slug, label, run, model in MODELS:
        root = api.ROOT/'results/correlation_pipeline'/run
        source = root/'analysis'/model
        b = api.read(source/'correlations.json')
        f = api.read(root/'forward'/model/'summary.json')
        summary = api.read(source/'views-v1/summary.json')
        assert b['status'] == f['status'] == summary['status'] == b['reasoning_view_analysis']['status'] == 'complete'
        rows = api.read_csv(source/'trace_features.csv')
        ids = {(r['dataset'], r['problem_id']) for r in rows}
        assert len(ids) == len(rows) == b['n_traces'] == 4647
        assert len({(r['dataset'],r['source_problem_id']) for r in rows}) == b['n_problem_clusters'] == 1547
        assert Counter(r['dataset'] for r in rows) == b['datasets']
        assert all(d['status'] == 'complete' and d['traces'] == b['datasets'][d['dataset']] for d in f['datasets'])
        assert sum(g['eligible_repeated_analysis_groups'] for g in b['repeated_problem_analysis']['group_completeness']) == 100
        views = [('Whole continuation', 'continuation', b, source)]
        views += [('Full reasoning', 'reasoning', api.read(source/'views-v1/full/reasoning/correlations.json'), source/'views-v1/full/reasoning')]
        for kind, names in [('class', api.CLASSES), ('position', [f'bin_{i:02d}' for i in range(10)]+['overflow'])]:
            for name in names:
                folder = source/'views-v1'/kind/name
                views.append((kind.title()+': '+name.replace('_',' '), kind+'-'+name, api.read(folder/'correlations.json'), folder))
        checks = {}
        for title, key, v, folder in views:
            assert v.get('status', 'complete') == 'complete'
            vr = rows if key == 'continuation' else api.read_csv(folder/'trace_features.csv')
            assert {(r['dataset'],r['problem_id']) for r in vr} == ids
            if key != 'continuation':
                assert v['contract'] == summary['contract'] and v['coverage']['traces'] == 4647
            binary = [r for r in v['binary_correlations'] if r['target'] == 'is_correct' and r['feature'] in api.FEATURES]
            pairs = [r for r in v['cross_feature_correlations']['trace_level'] if r['feature_x'] in api.FEATURES and r['feature_y'] in api.FEATURES]
            checks[key] = {'correctness': check_coefficients(vr, binary), 'metric_pairs': check_coefficients(vr, pairs, rank=True)}
            numeric.append(api.table(label+' / '+title+': correctness correlations $r$ (finite attempts).', 'tab:multi-'+slug+'-'+key, ['Metric', *api.DATASETS.values()], api.binary_rows(v)))
            if slug != 'qwen':
                filename = slug+'_atlas_'+key
                atlas_plot(label, title, v, filename)
                atlas += [figure(filename, label+' / '+title+'. Metric correlations and correctness; model-specific cohorts and tokenization.'), r'\clearpage']
        checks['repeated'] = check_repeated(rows, b['repeated_problem_analysis'])
        audit[slug] = {'status': 'complete', 'attempts': len(rows), 'problems':1547, 'checks':checks}
        for d in api.DATASETS:
            rr = [r for r in rows if r['dataset'] == d]
            correct = sum(int(float(r['is_correct'])) for r in rr)
            budget = next(a for a in b['generation_budget_audit']['scopes'] if a['scope'] == d)
            assert abs(correct/len(rr)-budget['accuracy_all']) < 1e-12
            overview.append(f'{label} & {api.DATASETS[d]} & {len(rr)} & {correct} & {correct/len(rr):.3f} & {budget["token_limit_hits"]}')
        protocol.append(label+' & '+f['quantization']+' & '+', '.join(map(str,f['layer_indices']))+f' & {summary["contract"]["position_reference"]["mean_reasoning_tokens"]:,.1f}')
        for c in api.CLASSES:
            v = next(v for title,key,v,folder in views if key == 'class-'+c)
            coverage.append(label+' & '+c+' & '+' & '.join(str(v['coverage']['datasets'][d]['traces_with_tokens']) for d in api.DATASETS))
        loaded.append((slug,label,b,views[1][2]))
        print(label, 'audited 20 scopes', flush=True)
    text = [r'\section{Three-model correlation results}', r'\label{sec:multimodel}',
        r'Forward replay and saved correlation analyses are complete for Qwen3.5-35B-A3B, Gemma-4-26B-A4B, and GPT-OSS-20B. Each includes 4,647 attempts from 1,547 source problems on the same six benchmarks, with 100 complete avg@32 groups. All token-limit completions are retained. Completion of class analysis means analysis of available labels, not full annotation coverage. GPT-OSS class tokens occur only in MATH-500 and part of AIME24; the other four benchmarks have no class estimates.',
        r'Qwen uses the local deterministic NVFP4 judge described above. Gemma and GPT-OSS use imported Qwen3.8-27B labels, with temperature 1, top-$p$ 0.95, top-$k$ 20, low reasoning effort and a 16,384-token judge budget. Consequently class comparisons also differ in annotation protocol and cohort. Generation uses Qwen GPTQ-Int4, Gemma NVFP4, and GPT-OSS checkpoints respectively; replay settings and selected decoder indices are below. Token counts and absolute position windows are model-specific. Correlations are marginal descriptive associations, not controlled model effects.',
        api.table('Replay configuration and mean reasoning length used for position windows.', 'tab:multi-protocol', ['Model','Replay','Decoder indices','Mean tokens'], protocol, 'llll'),
        api.table('All-attempt accuracy and termination by model. Invalid known-gold answers count as incorrect; repeated attempts are not independent problems.', 'tab:multi-population', ['Model','Dataset','Attempts','Correct','Accuracy','Limit hits'], overview, 'llrrrr'),
        api.table('Class token-bearing attempts; classes overlap. Zero coverage implies unavailable estimates.', 'tab:multi-coverage', ['Model','Class',*api.DATASETS.values()], coverage, 'llrrrrrr')]
    for scope, vi in [('Whole continuation',2),('Full reasoning',3)]:
        fig, axes = plt.subplots(1, 3, figsize=(13,5), sharey=True, layout='constrained')
        for ax, feature in zip(axes, ['router_confidence_mean_layers','router_margin_mean_layers','token_count']):
            for j, item in enumerate(loaded):
                records = api.lookup(item[vi])
                for i,d in enumerate(api.DATASETS):
                    r = records.get((d,feature))
                    if not r: continue
                    y=i+(j-1)*.22; color=['#146a91','#b24b2b','#54803b'][j]
                    if r['cluster_bootstrap_ci']: ax.plot(r['cluster_bootstrap_ci'], [y,y], color=color)
                    ax.plot(r['point_biserial_r'],y,'o',color=color,label=item[1] if i==0 else None,ms=4)
            ax.axvline(0,color='gray',lw=.6); ax.set_xlim(-1,1)
            ax.set_title(api.FEATURES[feature][1]); ax.set_yticks(range(6),api.DATASETS.values())
            ax.set_xlabel('Correctness r (95% problem-bootstrap CI)')
        axes[0].invert_yaxis(); axes[-1].legend(fontsize=7)
        fig.suptitle(scope+' — all three models')
        filename='multimodel_'+('continuation' if vi==2 else 'reasoning')
        api.save_figure(fig,filename)
        text.append(figure(filename, scope+': saved correctness correlations and pointwise source-problem bootstrap intervals. Overlap or separation of intervals is not a paired test of model differences.'))
    text.append('Across models, confidence--correctness associations do not have a uniform sign. Qwen is positive on all six benchmarks; Gemma is negative on AMC23 and Minerva, and GPT-OSS is negative on MATH-500, AIME25, AMC23, and Minerva. The pointwise intervals in the comparison figures show substantial uncertainty for several of these estimates. Token-count associations are negative except for Gemma on AMC23. These differences motivate model-specific validation rather than assuming that a Qwen-derived routing rule transfers to other models.')
    for slug,label,b,full in loaded:
        text += [r'\subsection{'+label+'}',api.table(label+': whole-continuation correctness $r$ (finite attempts).','tab:multi-summary-'+slug,['Metric',*api.DATASETS.values()],api.binary_rows(b,features=list(api.FEATURES)[:11]))]
        rec=api.lookup(b)
        vals=[rec.get((d,'router_confidence_mean_layers'),{}).get('point_biserial_r') for d in api.DATASETS]
        text.append('Confidence coefficients in benchmark order are '+', '.join(api.number(x) for x in vals)+'.')
        repeated={(r['dataset'],r['feature']):r for r in b['repeated_problem_analysis']['problem_level'] if r['problem_feature']=='feature_mean'}
        rr=[]
        for feature in list(api.FEATURES)[:11]:
            cells=[]
            for d in ['aime24','aime25','amc23']:
                r=repeated.get((d,feature));cells.append('---' if r is None else api.number(r['spearman_rho'])+f" ({r['n_problems']})")
            rr.append(api.FEATURES[feature][1]+' & '+' & '.join(cells))
        text.append(api.table(label+': problem mean versus avg@32, Spearman $\\rho$ (problems).','tab:multi-repeated-'+slug,['Metric','AIME24','AIME25','AMC23'],rr))
    text.append(r'Aggregate correctness and metric-pair coefficients were independently recomputed from saved CSVs in all 60 model/scope combinations; repeated-problem mean correlations and within-problem contrasts were also checked. This validates numerical consistency, not judge accuracy or replay fidelity. Audit counts and numerical errors are saved in \path{report/multimodel_correlation_audit.json}; input SHA-256 hashes are in \path{report/multimodel_correlation_sources.json}. The appended atlases include every Gemma and GPT-OSS scope, including empty class panels. Full correctness tables are available in \path{report/multimodel_correlation_tables.tex}.')
    supplementary_plots(loaded)
    text.append(r'\input{multimodel_supplementary_plots}')
    for name, parts in [('multimodel_correlation_results.tex',text),('multimodel_correlation_atlas.tex',atlas),('multimodel_correlation_tables.tex',numeric)]:
        (api.REPORT/name).write_text('\n'.join(parts)+'\n')
    (api.REPORT/'multimodel_correlation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (api.REPORT/'multimodel_correlation_sources.json').write_text(json.dumps(api.MANIFEST,indent=2)+'\n')


if __name__ == '__main__':
    main()
