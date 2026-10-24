"""Standalone measurement report; incomplete models and metrics remain explicit."""
import argparse, json, os
from collections import defaultdict
from pathlib import Path
from measurement_study.production_plan import ROOT, SOURCES, atomic, sha
from measurement_study.contract import DATASETS, LEGACY, TEMPORAL


def macro_stats(groups, bootstrap=1000):
    import numpy as np
    if set(groups)!=set(DATASETS):return {'status':'unavailable_incomplete_benchmarks','observed':sorted(groups)}
    arrays=[]
    for dataset in DATASETS:
        questions=defaultdict(list)
        for r in groups[dataset]:
            if r['is_correct'] is None or r['completion_tokens'] is None:
                return {'status':'unavailable_unscored_or_missing_tokens','dataset':dataset}
            questions[r['question_id']].append([float(r['is_correct']),float(r['completion_tokens']),float(r['capped'])])
        if not questions:return {'status':'unavailable_empty_benchmark','dataset':dataset}
        arrays.append(np.array([np.mean(rs,axis=0) for _,rs in sorted(questions.items())]))
    estimate=np.mean([x.mean(0) for x in arrays],axis=0);rng=np.random.default_rng(42);draws=[]
    for _ in range(bootstrap):draws.append(np.mean([x[rng.integers(0,len(x),len(x))].mean(0) for x in arrays],axis=0))
    ci=np.quantile(draws,[.025,.975],axis=0)
    return dict(status='complete',weighting='attempt means within questions, question means within benchmarks, equal six-benchmark means',questions={d:len(x) for d,x in zip(DATASETS,arrays)},bootstrap=bootstrap,
        metrics={name:dict(value=float(estimate[i]),ci95=ci[:,i].tolist()) for i,name in enumerate(('accuracy','generated_tokens','truncation_rate'))})


def observed_capture(source, storage=None):
    from measurement_study.production_plan import storage_root,digest
    storage=storage or storage_root(source);shards=[]
    for shard in range(4):
        manifest_path=storage/'manifests'/source/f'{shard}.json'
        result_path=storage/'capture'/source/str(shard)/'result.json'
        if not manifest_path.exists() or not result_path.exists():
            shards.append(dict(shard=shard,status='not_complete'));continue
        m=json.loads(manifest_path.read_text());r=json.loads(result_path.read_text())
        ids=[x['id'] for x in r.get('records',[])]
        valid=(r.get('status')=='complete_native_collection' and r.get('manifest_binding')==m['binding']
               and digest({k:v for k,v in m.items() if k!='binding'})==m['binding']
               and len(ids)==len(set(ids))==m['expected_attempts']
               and set(ids)=={x['id'] for x in m['attempts']}
               and sum(x['tokens'] for x in r['records'])==m['expected_tokens'])
        shards.append(dict(shard=shard,status='complete' if valid else 'invalid_coverage_or_binding',
                           manifest_sha256=sha(manifest_path),manifest_binding=m['binding'],result_sha256=sha(result_path),attempts=len(ids)))
    return dict(status='complete' if all(x['status']=='complete' for x in shards) else 'not_complete',shards=shards)



def required_recovery_status(target, audit_root=None):
    """Keep explicitly required historical model conditions in coverage tables."""
    from measurement_study.artifact_layout import result_dir
    source=target['id']
    storage=ROOT/'results/measurement-study/recovery-native-v1'
    states={'capture':observed_capture(source,storage)}
    for stage,folder in [('primary','production-analysis'),('supplement','production-supplement')]:
        p=result_dir(source,folder)/'result.json'
        state=json.loads(p.read_text()) if p.exists() else {'status':'not_complete'}
        if state.get('status')=='complete':
            captured=states['capture'];binding=state.get('input_binding',{})
            if (captured['status']!='complete' or
                binding.get('capture_result_sha256')!={str(x['shard']):x.get('result_sha256') for x in captured['shards']} or
                binding.get('manifest_bindings')!=[x.get('manifest_binding') for x in captured['shards']]):
                state['status']='invalid_capture_binding'
        states[stage]=state
    audit_root=audit_root or ROOT/'results/measurement-study/qwen35-recovery-audit-v1'
    p=audit_root/(source+'.json')
    audit=json.loads(p.read_text()) if p.exists() else {'status':'pending_audit'}
    return dict(required=True,model=target['model'],saved_generations=target.get('saved_generations'),
        labeling_status=target.get('labeling_status','partial'),
        labeled_trace_versions=target.get('labeled_trace_versions'),
        preparation={k:v for k,v in audit.items() if k in ('status','exact_token_versions','missing_token_versions','conflicting_token_versions','checkpoint_configs')},
        stages=states,cohorts={},storage=str(storage),
        historical_replay_note='Existing Unsloth 4-bit replay is preserved separately and does not establish native GPTQ/GGUF equivalence.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'report/measurement-study/production-results');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm CPU allocation required')
    a.output.mkdir(parents=True,exist_ok=True)
    results={};hits=[];plot=[];lines=['# Native all-layer measurement results','',
        'These are observational associations on the available labeled completions. They do not estimate intervention effects. Label cohorts remain separate.','']
    from measurement_study.artifact_layout import result_dir
    from measurement_study.gemma_cache_policy import VARIANT
    campaign=json.loads((Path(__file__).resolve().parents[1]/'campaign.json').read_text())
    locations=campaign.get('measurement_locations',{})
    entries=[(source,source,'') for source in SOURCES]+[('gemma-bf16-kv-sensitivity','gemma',VARIANT)]
    for label,source,variant in entries:
        location=locations.get(label,{})
        storage=Path(location['storage']) if location.get('storage') else None
        base=result_dir(source,'production-analysis',variant);extra=result_dir(source,'production-supplement',variant)
        states={'capture':observed_capture(source,storage)}
        for name,path in [('primary',base),('supplement',extra)]:
            marker=path/'result.json';states[name]=json.loads(marker.read_text()) if marker.exists() else {'status':'not_complete'}
            state=states[name]
            if state.get('status')=='complete':
                if variant and state.get('replay_variant')!=variant:
                    state['status']='invalid_variant_binding'
                binding=state.get('input_binding')
                if binding:
                    observed=states['capture']
                    expected={str(x['shard']):x.get('result_sha256') for x in observed['shards']}
                    if observed['status']!='complete' or binding.get('capture_result_sha256')!=expected or binding.get('manifest_bindings')!=[x.get('manifest_binding') for x in observed['shards']]:
                        state['status']='invalid_capture_binding'
        record={'stages':states,'cohorts':{},'replay_variant':variant,'storage':str(storage) if storage else None};results[label]=record
        if variant:
            record['precision_note']='NVFP4 checkpoint with BF16 KV cache sensitivity; historical generation checkpoint revision and cache dtype remain unverified.'
        lines+=['## '+label,'',f"Capture: {states['capture']['status']}; primary analysis: {states['primary']['status']}; class/temporal supplement: {states['supplement']['status']}.",'']
        if variant:lines += [record['precision_note'],'']
        if states['primary']['status']!='complete':continue
        for directory in sorted((base/'compact').iterdir()):
            if not directory.is_dir():continue
            groups={p.stem:[json.loads(l) for l in p.open()] for p in sorted(directory.glob('*.jsonl'))}
            macro=macro_stats(groups);record['cohorts'][directory.name]={'macro':macro}
            if macro['status']=='complete':
                x=macro['metrics'];lines.append(f"- `{directory.name}`: equally weighted accuracy {100*x['accuracy']['value']:.2f}% (question-bootstrap 95% interval {100*x['accuracy']['ci95'][0]:.2f}–{100*x['accuracy']['ci95'][1]:.2f}%), mean generated tokens {x['generated_tokens']['value']:.1f}, truncation rate {100*x['truncation_rate']['value']:.2f}%.")
            else:lines.append(f"- `{directory.name}`: six-benchmark aggregate unavailable ({macro['status']}).")
        coverage=extra/'coverage.json'
        if coverage.exists():
            coverage=json.loads(coverage.read_text());missing=[]
            for layer in coverage['layers']:
                for width in (64,256):
                    cells=coverage['temporal'].get(f'L{layer}/W{width}',{})
                    missing.extend(f'L{layer}/W{width}/{metric}' for metric in TEMPORAL if cells.get(metric,0)==0)
            record['temporal_coverage']={'layers':coverage['layers'],'unavailable_cells':missing,'complete':not missing}
            lines.append(f"\nTemporal coverage: {len(coverage['layers'])} actual routed layers; {len(missing)} layer/window/metric cells without observations.")
        for stage,path in [('primary',base),('supplement',extra)]:
            if states[stage]['status']!='complete':continue
            for file in sorted((path/'associations').glob('**/correlations.jsonl')):
                relative=file.relative_to(path/'associations');cohort=relative.parts[0];dataset=relative.parts[1]
                for line in file.open():
                    r=json.loads(line)
                    if r.get('adjustment')=='log_length_and_cap' and r.get('q_bh') is not None and r['q_bh']<.05:
                        hits.append(dict(source=label,cohort=cohort,dataset=dataset,stage=stage,**r))
        own=[h for h in hits if h['source']==label];own.sort(key=lambda r:(r['q_bh'],-abs(r.get('r') or 0),r['metric']))
        lines+=['','Length/cap-adjusted associations passing their prespecified family BH correction: '+str(len(own))+'.','']
        for h in own[:12]:lines.append(f"- {h['cohort']} / {h['dataset']}: `{h['metric']}`, r={h['r']:+.3f}, q={h['q_bh']:.3g}, questions={h['n']}.")
        plot.extend(own[:12]);lines.append('')
    required=campaign.get('required_recovery_sources',[])
    for target in required:
        label=target['id'];record=required_recovery_status(target);results[label]=record
        lines+=['## '+target['model'],'',
            'Required campaign model. Saved generations: '+str(record['saved_generations'])+'; labeling: '+record['labeling_status']+'.',
            'Preparation: '+record['preparation']['status']+'.',
            'Capture: '+record['stages']['capture']['status']+'; primary analysis: '+record['stages']['primary']['status']+'; class/temporal supplement: '+record['stages']['supplement']['status']+'.',
            record['historical_replay_note'],'']
    unavailable={} if required else {
        'older_qwen_gptq':'Original native checkpoint/backend not yet verified; six saved attempts also lack exact token IDs.',
        'older_qwen_gguf':'Original native checkpoint/backend not yet verified; all ten saved attempts lack exact token IDs.'}

    for source, record in results.items():
        missing = [stage for stage, state in record['stages'].items() if state.get('status') != 'complete']
        if missing: unavailable[source] = 'Incomplete observed stages: ' + ', '.join(missing)
    lines+=['## Remaining coverage','']+[f'- {s}: {reason}' for s,reason in unavailable.items()]
    lines+=['','Bootstrap replicates resample questions within each benchmark. Matched repeated attempts stay together. Reported extremes are selected exploratory findings; independent replication is required. Pooled motif/hazard tables are descriptive.','']
    atomic(a.output/'results.json',dict(models=results,unavailable_models=unavailable,full_requested_model_coverage=not unavailable,reporting_code_sha256=sha(Path(__file__)),reporting_bundle_sha256=sha(Path(__file__).resolve().parents[1]/'SHA256.json')))
    (a.output/'significant-adjusted.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in hits))
    (a.output/'report.md').write_text('\n'.join(lines))
    if plot:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(13,max(4,len(plot)*.25)))
        ys=list(range(len(plot)));ax.scatter([r['r'] for r in plot],ys,s=16)
        ax.axvline(0,color='gray',lw=.8);ax.set_yticks(ys,[f"{r['source']} | {r['dataset']} | {r['metric']}" for r in plot],fontsize=6);ax.invert_yaxis();ax.set_xlabel('Question-level partial correlation, adjusted for log length and cap')
        ax.set_title('Selected exploratory associations passing family BH correction')
        fig.tight_layout();fig.savefig(a.output/'associations.pdf');fig.savefig(a.output/'associations.png',dpi=160);plt.close(fig)
    print(json.dumps({'report':str(a.output/'report.md'),'adjusted_bh_hits':len(hits),'full_requested_model_coverage':False}),flush=True)
if __name__=='__main__':main()
