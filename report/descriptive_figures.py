"""Descriptive routing and reasoning plots from saved features and sampled labels.

Called by generate_correlation_tables.py; never reconstructs missing labels or
router probabilities. Every input is registered by the parent reader.
"""
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def figure_tex(name, caption, label):
    return ('\\begin{figure}[p]\n\\centering\n'
            '\\includegraphics[width=\\linewidth]{figures/' + name + '.pdf}\n'
            '\\caption{' + caption + '}\n\\label{fig:' + label + '}\n'
            '\\Description{Six benchmark heatmaps compare aggregate metrics with each other and with correctness.}\n'
            '\\end{figure}\n')


def binned(rows, xkey, ykey, edges, bootstrap=False):
    """Fixed edges, with the rightmost edge inclusive; whole-problem bootstrap."""
    result = []
    rng = np.random.default_rng(42)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        rs = [r for r in rows if lo <= r[xkey] and (r[xkey] < hi or (i == len(edges)-2 and r[xkey] <= hi))]
        if not rs:
            continue
        y = np.array([r[ykey] for r in rs], dtype=float)
        interval = None
        if bootstrap:
            groups = {}
            for r in rs:
                groups.setdefault((r['dataset'], r['source_problem_id']), []).append(r[ykey])
            sums = np.array([sum(v) for v in groups.values()]); ns = np.array([len(v) for v in groups.values()])
            if len(ns) >= 2:
                idx = rng.integers(0, len(ns), (500, len(ns)))
                interval = np.quantile(sums[idx].sum(1)/ns[idx].sum(1), [.025, .975]).tolist()
        result.append({'lo': float(lo), 'hi': float(hi), 'x': float(np.mean([r[xkey] for r in rs])),
                       'mean': float(y.mean()), 'n': len(rs), 'ci': interval})
    assert sum(r['n'] for r in result) == len(rows)
    return result


def generate(api, base, full, classes, positions, frames, views):
    datasets, labels, features = api.DATASETS, api.CLASSES, api.FEATURES
    save, read, report = api.save_figure, api.read, api.REPORT
    numerical = {'seed': 42, 'bootstrap_replicates': 500}
    full_rows = api.read_csv(api.SOURCE / 'views-v1/full/reasoning/trace_features.csv')
    lengths = {(r['dataset'], r['problem_id']): float(r['token_count']) for r in full_rows}
    rows = []
    for d in datasets:
        vals = [float(r['token_count']) for r in full_rows if r['dataset'] == d]
        rows.append(f'{datasets[d]} & {len(vals)} & {np.mean(vals):,.1f} & {np.median(vals):,.0f} & {np.quantile(vals,.9):,.0f}')
    (report/'reasoning_length_table.tex').write_text(api.table(
        'Reasoning length in tokens, all attempts including limit hits. This replaces the position-coverage count plot; lengths are observed under the generation budget.',
        'tab:reasoning-length', ['Dataset','Attempts','Mean','Median','90th percentile'], rows))

    annotations = {}
    for d in datasets:
        path = api.RUN/f'annotations/Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/{d}/annotations.jsonl'
        for line in api.fingerprint(path).decode().splitlines():
            a = json.loads(line); annotations[a['trace_sha256']] = a
    frames_by_id = {(r['dataset'],r['problem_id']): r for r in frames}
    cumulative = np.zeros((7,11), dtype=int)
    transitions = {d: np.zeros((7,7), dtype=int) for d in datasets}
    sample_rows = []
    matched = set()
    for d in datasets:
        for path in sorted((api.RUN/f'forward/unsloth--Qwen3.5-35B-A3B/{d}/tensors').glob('*__sample_00_extraction.json')):
            payload = read(path)['correlation_views']
            a = annotations.get(payload['trace_sha256'])
            if a is None:
                continue
            matched.add(payload['trace_sha256'])
            original = frames_by_id[d,a['problem_id']]
            units = sorted(a['units'], key=lambda u:u['index'])
            if not units:
                continue
            counts = Counter(u['label'] for u in units)
            spans = {u['index']:u for u in payload['sentence_spans']}
            all_ranges = [r for s in spans.values() for r in s['token_ranges']]
            origin = min(r[0] for r in all_ranges)
            length = payload['reasoning_token_count']
            assert length == lengths[d,a['problem_id']]
            for u in units:
                token_ranges = spans[u['index']]['token_ranges']
                if not token_ranges:
                    raise ValueError('Labelled sentence has no token position')
                # Count a sentence when its last token has been reached.
                endpoint = (max(r[1] for r in token_ranges)-origin)/length
                assert 0 < endpoint <= 1
                cumulative[labels.index(u['label'])] += (np.linspace(0,1,11) >= endpoint)
            for left,right in zip(units,units[1:]):
                if right['index'] == left['index']+1:
                    transitions[d][labels.index(left['label']),labels.index(right['label'])] += 1
            r = {'dataset':d, 'source_problem_id':original['source_problem_id'],
                 'is_correct':float(original['is_correct']), 'token_count':float(original['token_count']),
                 'labelled_sentences':len(units)}
            r.update({c:counts[c]/len(units) for c in labels})
            sample_rows.append(r)
    assert matched == set(annotations)
    assert cumulative[:,-1].sum() == sum(r['labelled_sentences'] for r in sample_rows)
    assert np.all(np.diff(cumulative,axis=1)>=0) and np.all(cumulative[:,0]==0)
    numerical['sampled_attempts'] = len(sample_rows)
    summary_rows = []
    for d in datasets:
        rs = [r for r in sample_rows if r['dataset'] == d]
        summary_rows.append(datasets[d] + f' & {len(rs)} & ' + ' & '.join(f'{100*np.mean([r[c] for r in rs]):.1f}' for c in labels))
    (report/'class_share_table.tex').write_text(api.table(
        'Mean sampled sentence share (percent), with equal weight per labelled attempt. Empty-label attempts are excluded; absent classes in an observed sample have zero share. These are not full-corpus class proportions.',
        'tab:class-shares', ['Dataset','Attempts',*labels], summary_rows))
    numerical['cumulative_counts'] = cumulative.tolist()
    numerical['transition_counts'] = {d:m.tolist() for d,m in transitions.items()}
    colors = plt.get_cmap('tab10').colors
    fig,ax = plt.subplots(figsize=(10,4.4),layout='constrained')
    for c,ys,color in zip(labels,cumulative,colors):
        ax.plot(np.arange(11)*10,ys,label=f'{c} ({ys[-1]:,})',color=color,marker='.',ms=5)
    ax.set(xlabel='Within-attempt reasoning progress (% of reasoning tokens)',ylabel='Cumulative sampled sentence count',
           title='Predicted classes over reasoning progress — Qwen3.5')
    ax.legend(ncol=2,fontsize=8); ax.ticklabel_format(axis='y',style='plain')
    save(fig,'qwen_class_cumulative')
    fig,axes = plt.subplots(2,3,figsize=(11,7.5),layout='constrained')
    for ax,(d,m) in zip(axes.flat,transitions.items()):
        n=m.sum(1); percent=np.divide(100*m,n[:,None],out=np.full(m.shape,np.nan),where=n[:,None]>0)
        im=ax.imshow(np.ma.masked_invalid(percent),vmin=0,vmax=100,cmap='Blues')
        for i,j in np.ndindex(m.shape):
            ax.text(j,i,f'{percent[i,j]:.0f}' if n[i] else '—',ha='center',va='center',fontsize=7,color='white' if percent[i,j]>55 else 'black')
        ax.set_xticks(range(7),labels,rotation=55,ha='right',fontsize=7)
        ax.set_yticks(range(7),[f'{c} (n={n[i]})' for i,c in enumerate(labels)],fontsize=7)
        ax.set_title(datasets[d],fontsize=10)
    fig.supxlabel('Next sentence class (originally adjacent labelled pairs only)',fontsize=10)
    fig.colorbar(im,ax=list(axes.flat),label='Row-normalized transitions (%)',shrink=.8)
    save(fig,'qwen_class_transitions')

    # Share correlations use one sample-zero attempt per problem, including
    # zero shares for observed samples without a given class.
    matrices=[]
    for target in ['is_correct','token_count']:
        values=np.full((7,6),np.nan)
        for i,c in enumerate(labels):
            for j,d in enumerate(datasets):
                rs=[r for r in sample_rows if r['dataset']==d]
                x=np.array([r[c] for r in rs]); y=np.array([r[target] for r in rs])
                if np.ptp(x)>0 and np.ptp(y)>0:
                    values[i,j]=np.corrcoef(x,y)[0,1] if target=='is_correct' else spearmanr(x,y)[0]
        matrices.append(values)
    numerical['share_correlations']=[np.where(np.isfinite(m),m,None).tolist() for m in matrices]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,m,title in zip(axes,matrices,['Sampled class share vs correctness (r)','Sampled class share vs total token count (ρ)']):
        im=ax.imshow(np.ma.masked_invalid(m),vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
        for i,j in np.ndindex(m.shape): ax.text(j,i,f'{m[i,j]:+.2f}' if np.isfinite(m[i,j]) else '—',ha='center',va='center',fontsize=8)
        ax.set_xticks(range(6),[f'{datasets[d]}\n(n={sum(r["dataset"]==d for r in sample_rows)})' for d in datasets],rotation=30,ha='right',fontsize=8)
        ax.set_yticks(range(7),labels);ax.set_title(title,fontsize=10)
    fig.colorbar(im,ax=axes,shrink=.8,label='Descriptive coefficient')
    save(fig,'qwen_class_share_correlations')
    # Actual share ranges and outcomes, separated by benchmark.
    share_bins={}
    for target, suffix, ylabel in [('is_correct','accuracy','Accuracy'), ('token_count','tokens','Mean total tokens (thousands)')]:
        fig,axes=plt.subplots(4,2,figsize=(10,10),layout='constrained')
        for ax,c in zip(axes.flat,labels):
            for d,color in zip(datasets,colors):
                rs=[r for r in sample_rows if r['dataset']==d]
                bins=binned(rs,c,target,np.linspace(0,1,6))
                share_bins[f'{d}/{c}/{target}']=bins
                scale=1000 if target=='token_count' else 1
                ax.plot([100*b['x'] for b in bins],[b['mean']/scale for b in bins],'.-',color=color,label=datasets[d],lw=1)
                for b in bins: ax.annotate(str(b['n']),(100*b['x'],b['mean']/scale),fontsize=6,xytext=(0,3),textcoords='offset points',color=color)
            ax.set_title(c+' share vs '+('accuracy' if target=='is_correct' else 'token count'),fontsize=10)
            if target=='is_correct':ax.set_ylim(-.05,1.12)
            ax.set_xlabel('Sampled sentence share (%)',fontsize=9); ax.set_ylabel(ylabel,fontsize=9)
            ax.set_xlim(-2,102);ax.tick_params(labelsize=8)
        handles,names=axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].axis('off');axes.flat[-1].legend(handles,names,loc='center',ncol=2,fontsize=10)
        save(fig,'qwen_class_share_'+suffix)
    numerical['share_bins']=share_bins

    # Actual router values versus accuracy, with problem-respecting intervals.
    metrics=['router_confidence_mean_layers','router_boundary_margin_mean_layers','router_topk_non_topk_gap_mean_layers']
    fig,axes=plt.subplots(3,6,figsize=(15,8),layout='constrained')
    numerical['router_value_bins']={}
    for j,d in enumerate(datasets):
        rs=[dict(r) for r in frames if r['dataset']==d]
        for r in rs:
            for k in metrics+['is_correct']:r[k]=float(r[k])
        for i,k in enumerate(metrics):
            edges=np.unique(np.quantile([r[k] for r in rs],np.linspace(0,1,6)))
            bins=binned(rs,k,'is_correct',edges,bootstrap=True)
            numerical['router_value_bins'][f'{d}/{k}']=bins
            ax=axes[i,j]
            for b in bins:
                if b['ci']:ax.vlines(b['x'],*b['ci'],color=colors[j],lw=1)
                ax.hlines(b['mean'],b['lo'],b['hi'],color=colors[j],lw=1)
                ax.plot(b['x'],b['mean'],'o',color=colors[j],ms=3)
            counts = [b['n'] for b in bins]
            count_label = f'n/bin: {counts[0]}' if len(set(counts)) == 1 else 'n: ' + '/'.join(map(str, counts))
            ax.text(.02, .98, count_label, transform=ax.transAxes, va='top', fontsize=6)
            ax.set_ylim(-.04,1.15);ax.tick_params(labelsize=7);ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2),useMathText=True)
            ax.set_xlabel(features[k][0]+' value',fontsize=8)
            if i==0:ax.set_title(datasets[d])
            if j==0:ax.set_ylabel(features[k][1]+'\nAccuracy',fontsize=8)
    save(fig,'qwen_router_value_accuracy')

    # All selected-pool IDs, avoiding a new ranking by maximum correlation.
    fig,axes=plt.subplots(2,1,figsize=(13,5),layout='constrained')
    expert=base['expert_identity_analysis']
    for ax,target,key,coef,title in zip(axes,['is_correct','token_count'],['trace_point_biserial','trace_spearman'],['point_biserial_r','spearman_rho'],['Selected-pool expert frequency vs correctness (r)','Selected-pool expert frequency vs total token count (ρ)']):
        rec={(r['scope'],r['feature']):r for r in expert[key] if r['target']==target}
        m=np.array([[rec.get((d,f'expert_any_topk_rate_e{e}'),{}).get(coef,np.nan) for e in range(256)] for d in datasets])
        assert np.isfinite(m).any()
        im=ax.imshow(np.ma.masked_invalid(m),vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto',interpolation='nearest')
        ax.set_yticks(range(6),datasets.values());ax.set_title(title,fontsize=10)
    axes[-1].set_xlabel('Expert ID (pooled across six layers; all 256 IDs)')
    fig.colorbar(im,ax=axes,shrink=.8,label='Descriptive coefficient')
    save(fig,'qwen_expert_outcomes')

    # Replace repeated target-wise tables with one triangle and correctness
    # column per dataset and scope; lexical targets intentionally excluded.
    atlas=[r'\section{Aggregate correlation plot atlas}',r'\label{sec:complete-correlation-tables}',
           r'Each page fixes a token scope. Lower triangles show metric--metric Spearman $\rho$; the last column (Acc) shows point-biserial $r$ with whole-attempt correctness. A dash means undefined or unavailable, not zero. Panels are descriptive; no significance threshold is implied. Sample sizes and saved intervals remain in the fingerprinted source JSON. Legacy lexical targets are omitted. Detailed numeric tables remain in \texttt{correlation\_tables.tex} as a separate generated artifact.',
           api.table('Common metric notation; $k=8$ for this model.', 'tab:qwen-metric-key-atlas',['Code','Metric'],[c+' & '+n for c,n in features.values()],'ll')]
    for name,slug,view in views:
        fs=[f for f in list(features)[:11] if not (view.get('view')=='position' and f=='token_count')]
        pairs={(r['scope'],frozenset((r['feature_x'],r['feature_y']))):r for r in view['cross_feature_correlations']['trace_level']}
        correct=api.lookup(view)
        fig,axes=plt.subplots(2,3,figsize=(11,8),layout='constrained')
        for ax,d in zip(axes.flat,datasets):
            m=np.full((len(fs),len(fs)+1),np.nan)
            for i,f in enumerate(fs):
                for j,g in enumerate(fs):
                    if j<i:m[i,j]=pairs.get((d,frozenset((f,g))),{}).get('spearman_rho',np.nan)
                m[i,-1]=correct.get((d,f),{}).get('point_biserial_r',np.nan)
            im=ax.imshow(np.ma.masked_invalid(m),vmin=-1,vmax=1,cmap='RdBu_r')
            for i,j in np.ndindex(m.shape):
                if j<i or j==len(fs):ax.text(j,i,f'{m[i,j]:.1f}' if np.isfinite(m[i,j]) else '—',ha='center',va='center',fontsize=5.5,color='white' if abs(m[i,j])>.65 else 'black')
            ax.set_xticks(range(len(fs)+1),[features[f][0] for f in fs]+['Acc'],fontsize=7)
            ax.set_yticks(range(len(fs)),[features[f][0] for f in fs],fontsize=7)
            ax.set_title(datasets[d],fontsize=10)
        fig.suptitle(name+' — metric correlations and correctness',fontsize=12)
        fig.colorbar(im,ax=list(axes.flat),shrink=.8,label='Correlation coefficient')
        save(fig,'qwen_atlas_'+slug)
        atlas.append(figure_tex('qwen_atlas_'+slug,name+': metric relationships and correctness. Codes are defined in Table~\\ref{tab:qwen-metric-key-atlas}.','atlas-'+slug))
        atlas.append(r'\clearpage')
    (report/'correlation_plot_atlas.tex').write_text('\n'.join(atlas))
    (report/'descriptive_plot_statistics.json').write_text(json.dumps(numerical,indent=2,allow_nan=False)+'\n')
