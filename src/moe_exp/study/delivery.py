"""Measured main-paper artifacts, diagnostics, figures and replacement map."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import CLASSES, MODELS, Units, digest, file_sha, load, publish, require, verified


def reference(path,models):
    path=Path(path)
    return {'path':str(path),'sha256_file':file_sha(path),'models':list(models)}


def catalogue(binding):
    from .readouts import HISTORICAL
    from .export import ARTIFACTS
    root=Path(binding['root']);source=Path(binding['source_campaign'])
    parent=load(source/'CAMPAIGN.json')
    entries={k:[] for k,_,_ in ARTIFACTS}
    for model in MODELS:
        paths={'native_profiles':root/model/'QUALIFIED.json',
            'native_benchmarks':(root if model=='lfm25' else source)/model/'NATIVE_ACCURACY.json',
            'native_dynamics':(root if model=='lfm25' else source)/model/'DYNAMICS.json',
            'selection_coverage':root/model/'SELECTION_COVERAGE.json' if model=='lfm25' else Path(parent['saved_inputs'][model]['selection']),
            'hypothesis_cards':root/model/'COUNT_NOMINATIONS.json'}
        for name,path in paths.items():
            if path.exists():entries[name].append(reference(path,[model]))
        confirmation=root/model/'COUNT_CONFIRMATION.json'
        if confirmation.exists():entries['hypothesis_cards'].append(reference(confirmation,[model]))
    old_readout=HISTORICAL/'readout-results.v3.json'
    if old_readout.exists():
        value=load(old_readout)
        models=[m for m in MODELS[:-1] if all(value['models'][m][mde]['status']=='CLEAN' for mde in ('token_identity','judge_context'))]
        entries['readout_results'].append(reference(old_readout,models))
    if (root/'lfm25/READOUT_RESULTS.json').exists():
        entries['readout_results'].append(reference(root/'lfm25/READOUT_RESULTS.json',['lfm25']))
    for model in MODELS[:-1]:
        path=source/model/'SUPPLEMENTARY.json'
        if path.exists() and load(path).get('status')=='complete':entries['legacy_supporting'].append(reference(path,[model]))
    return entries


def diagnostics(binding,data):
    root,bulk=Path(binding['root']),Path(binding['config']['bulk_root'])
    reader=Units(bulk/'reader/annotations')
    records=[]
    for row in data['assignments']:
        if row['arm']=='native' or row['uid'] not in data['outcomes']:continue
        path=bulk/row['model']/'contexts'/(row['uid']+'.json')
        contexts=verified(path)
        labelled=[(s,(reader.state(s['annotation_uid']) or {}).get('label')) for s in contexts['sentences']]
        pulse=row['pulse_tokens'];target=row['class'];horizon=row['horizon']
        entered=any(label==target and s['token_start']<pulse for s,label in labelled)
        post=[(s,l) for s,l in labelled if pulse<=s['token_start']<horizon]
        observed=sum(s['n_tokens'] for s,l in post if l is not None)
        retained=sum(s['n_tokens'] for s,l in post if l==target)
        capture=bulk/row['model']/'executions'/row['uid']/'CAPTURE.json'
        stats=verified(capture)['stats'] if capture.exists() else None
        records.append({k:row[k] for k in ('uid','model','benchmark','family','fold','class','depth_third','dose','arm')} |
            {'entry_during_pulse_observed':entered,'postpulse_target_tokens':retained,'postpulse_labelled_tokens':observed,
             'observed_persistence':retained/observed if entered and observed else None,
             'capture_available':capture.exists(),'target_engagement':stats,
             'interpretation':'descriptive mediator; no re-selection or claim of a mediated causal effect'})
    return records


def figures(folder,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outputs=[]
    fig,ax=plt.subplots(figsize=(7,3.5))
    contrasts=data['primary']['contrasts']
    for i,r in enumerate(contrasts):
        ax.plot([r['lower_bound'],r['upper_bound']],[i,i],color='C0',lw=5)
        if r['ci_lower'] is not None:ax.plot([r['ci_lower'],r['ci_upper']],[i,i],color='black',lw=1)
    ax.set_yticks(range(4),[r['contrast'] for r in contrasts]);ax.axvline(0,color='.6',lw=.7)
    ax.set_xlabel('Targeted minus matched control; suppression oriented positively')
    ax.set_xlim(-1,1);fig.tight_layout()
    path=folder/'primary_effects.pdf';fig.savefig(path,metadata={'CreationDate':None,'ModDate':None});plt.close(fig);outputs.append(reference(path,MODELS))
    for model in MODELS:
        rows=[r for r in data['responses'] if r['model']==model and r.get('dose')=='central' and
              r['arm']=='target_activate' and r.get('endpoint')=='occupancy']
        if not rows:continue
        matrix=np.full((7,7),np.nan)
        for i,c in enumerate(CLASSES):
            for j,other in enumerate(CLASSES):
                cells=defaultdict(list)
                for row in rows:
                    if row['class']==c and row['observed_class']==other:
                        cells[row['benchmark'],row['family'],row['depth_third']].append(row['lower'])
                families=defaultdict(list)
                for (b,f,d),values in cells.items():families[b,f].append(np.mean(values))
                benchmarks=defaultdict(list)
                for (b,f),values in families.items():benchmarks[b].append(np.mean(values))
                if benchmarks:matrix[i,j]=np.mean([np.mean(v) for v in benchmarks.values()])
        fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis')
        ax.set_xticks(range(7),CLASSES,rotation=60,ha='right');ax.set_yticks(range(7),CLASSES)
        ax.set_title(model+': observed post-pulse occupancy lower bound');fig.colorbar(im,ax=ax);fig.tight_layout()
        path=folder/('response-'+model+'.pdf');fig.savefig(path,metadata={'CreationDate':None,'ModDate':None});plt.close(fig);outputs.append(reference(path,[model]))
    return outputs


def analyze(binding):
    from .export import export
    root=Path(binding['root']);mode=verified(root/'DESIGN.json')['mode']
    result=export(binding,mode)
    publish(root/'FINAL_EXPORT.json',{'status':'complete','output':result,
        'manifest_file_sha256':file_sha(Path(result['path'])/'paper_results_manifest.json')})
    return {'status':'complete' if result['required_results_complete'] else 'incomplete_results',**result}


def lfm_complete(binding):
    root=Path(binding['root']);bulk=Path(binding['config']['bulk_root'])
    paths=['NATIVE_TASKS.json','DISCOVERY_SELECTION.json','DISCOVERY_INPUTS.json','READOUT_RESULTS.json','ANNOTATION_TASKS.json']
    if not all((root/'lfm25'/name).exists() for name in paths):return False
    selection=verified(root/'lfm25/DISCOVERY_SELECTION.json')
    if selection['native_attempts']!=4647 or selection['discovery_traces']!=1547 or selection['sampled_sentences']!=100000:return False
    if selection['unknown_native']:return False
    reader=Units(bulk/'reader/annotations')
    if any((reader.state(r['uid']) or {}).get('status')!='complete'
           for r in verified(root/'lfm25/ANNOTATION_TASKS.json')['records']):return False
    inputs=verified(root/'lfm25/DISCOVERY_INPUTS.json')
    if file_sha(inputs['counts'])!=inputs['counts_sha256'] or file_sha(inputs['sentences'])!=inputs['sentences_sha256']:return False
    result=verified(root/'lfm25/READOUT_RESULTS.json')
    return result['status']=='complete' and result['lexical_nulls']==5 and all(
        v['status']=='CLEAN' for v in result['models']['lfm25'].values())


def legacy_complete(binding):
    path=Path(binding['root'])/'INHERITED_TASKS.json'
    if not path.exists():return False
    for row in verified(path)['records']:
        if row['stage']=='analysis_finalize':continue
        receipt=Path(row['receipt_path'])
        if not receipt.exists() or load(receipt).get('status')!='complete':return False
    return True
