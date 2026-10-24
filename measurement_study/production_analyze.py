"""CPU analyses of complete captures; disjoint boundary calibration and all cohorts."""
import argparse, copy, gzip, hashlib, io, json, os, random, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from measurement_study.production_plan import ROOT,FAST,SOURCES,digest,sha,atomic

def splits(rows):
    result={};rng=random.Random(42)
    for dataset in sorted({r['dataset'] for r in rows}):
        qs=sorted({r['identity']['question_id'] for r in rows if r['dataset']==dataset});rng.shuffle(qs)
        n=len(qs);a=int(.6*n);b=a+int(.2*n)
        result.update({q:('calibration' if i<a else 'tuning' if i<b else 'test') for i,q in enumerate(qs)})
    return result

def outcome_fingerprint(binding):
    """Re-read the explicitly passed sidecar and its manifest before cache reuse.

    This is the only place outcomes enter an analysis: the binding was produced
    at analysis time by campaign_v3_plan.read_outcomes from the sidecar (or its
    pointer) the caller named, never from a capture manifest.
    """
    if not binding:return None
    result={}
    for field,hash_field in (("path","sha256"),("manifest_path","manifest_sha256")):
        if not binding.get(field) or not binding.get(hash_field):
            raise ValueError("Outcome sidecar needs both content and manifest bindings")
        actual=sha(binding[field])
        if actual!=binding.get(hash_field):raise ValueError("Outcome sidecar changed: "+field)
        result[field]=str(Path(binding[field]).resolve());result[hash_field]=actual
    if binding.get('pointer_path') is not None:
        if not binding.get('pointer_sha256') or sha(binding['pointer_path'])!=binding['pointer_sha256']:
            raise ValueError("Outcome pointer changed")
        result['pointer_path']=str(Path(binding['pointer_path']).resolve())
        result['pointer_sha256']=binding['pointer_sha256']
    result['corpus']=binding.get('corpus');result['model']=binding.get('model');result['rows']=binding.get('rows')
    return result


def capture_code_binding(manifests):
    """Capture provenance is recorded; it need not equal the analysis bundle.

    Artifact hashes and the capture gate already establish that the captures
    came from the code named here. Analysis runs from its own frozen bundle.
    """
    codes={m.get('code_binding') for m in manifests}
    if None in codes or len(codes)!=1:
        raise ValueError('Mixed or missing capture code bindings in one plan')
    return codes.pop()


def full_analysis_binding(ready_path,manifest_paths,manifests,outcome_binding,code_binding,cohort,label_binding):
    """Bind ready/shard manifests, capture results, the analysis-time sidecar and code."""
    input_binding=dict(ready_manifest_sha256=sha(ready_path),
        manifests={str(Path(path).resolve()):sha(path) for path in manifest_paths},
        capture_result_sha256={str(Path(m['capture_dir'])/'result.json'):sha(Path(m['capture_dir'])/'result.json') for m in manifests})
    return dict(schema_version=3,input_binding=input_binding,
        outcome_binding=outcome_fingerprint(outcome_binding),
        capture_code_binding=capture_code_binding(manifests),code_binding=code_binding,
        cohort=cohort,label_binding=label_binding)


def analysis_code_binding(metrics,bundle=None):
    if bundle is not None:
        from measurement_study.bundle import verify_bundle,verify_imports,verify_loaded_imports
        receipt=verify_bundle(Path(bundle))
        verify_loaded_imports(Path(bundle))
        imported=verify_imports(Path(bundle),['analyze_available_metrics','analyze_pilot_metrics'])
        return dict(bundle=receipt,imports=imported)
    # Development tests also bind the actual helper files; never ROOT/scripts.
    import analyze_pilot_metrics as pilot
    paths=[Path(__file__),Path(metrics.__file__),Path(pilot.__file__)]
    return {str(p.resolve()):sha(p) for p in paths}


def checked_complete(path,binding):
    if not path.exists():return None
    result=json.loads(path.read_text())
    if result.get('status')!='complete' or result.get('analysis_binding')!=binding:
        raise ValueError('Stale analysis completion; select a fresh output root')
    for name,expected in result.get('output_hashes',{}).items():
        if sha(path.parent/name)!=expected:raise ValueError('Completed analysis output changed')
    return result


def claim_output(out,binding):
    marker=out/'analysis-binding.json'
    if marker.exists():
        if json.loads(marker.read_text())!=binding:
            raise ValueError('Analysis inputs/code changed; select a fresh output root')
    elif out.exists() and any(out.iterdir()):
        raise ValueError('Unbound existing analysis output; select a fresh output root')
    else:
        out.mkdir(parents=True,exist_ok=True);atomic(marker,binding)


def analyze_group(args):
    if len(args)==2:path,out=args;options={}
    else:path,out,options=args
    from measurement_study.bundle import import_frozen_helper
    bundle=options.get('bundle')
    metrics=import_frozen_helper('analyze_available_metrics',Path(bundle) if bundle else None)
    for label_path,expected in (options.get('label_binding') or {}).items():
        if sha(label_path)!=expected:raise ValueError('Label attachment input changed before worker reuse')
    binding=dict(schema_version=3,input_sha256=sha(path),
        outcome_binding=outcome_fingerprint(options.get('outcome_binding')),
        code_binding=analysis_code_binding(metrics,bundle),cohort_binding=options.get('cohort_binding'),
        label_binding=options.get('label_binding'))
    out=Path(out)
    existing=checked_complete(out/'complete.json',binding)
    if existing is not None:return existing
    claim_output(out,binding)
    rows=[json.loads(l) for l in Path(path).open() if l.strip()]
    summary=metrics.finish_groups(rows,out)
    hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('complete.json','analysis-binding.json')}
    result=dict(status='complete',input_sha256=binding['input_sha256'],analysis_binding=binding,
                output_hashes=hashes,groups=summary)
    atomic(out/'complete.json',result);return result


def compact_variant(base,variant,metrics,path,*,outcome,outcome_binding,cohort,fold):
    """Derive routing first, then attach the separate corrected outcome identity."""
    row=copy.deepcopy(variant.get('source_trace',base));meta=row['metadata']
    meta['forward_provenance']=base['metadata']['forward_provenance']
    meta['correlation_features']=base['metadata']['correlation_features']
    meta['routing_dynamics']=copy.deepcopy(base['metadata']['routing_dynamics'])
    row['model_logs']['layer_indices']=base['model_logs']['layer_indices']
    annotation=variant.get('annotation')
    if annotation is not None:
        meta['reasoning_annotation']=annotation;meta['correlation_views']=variant.get('views')
        meta['routing_dynamics']['annotation_sha256']=metrics.digest(annotation)
    else:
        meta.pop('reasoning_annotation',None);meta.pop('correlation_views',None)
    compact=metrics.routing_features(row,Path(path))
    compact.update(split=fold,labeled_sentences=0,transitions=0,label_status='unavailable',
        class_fraction_denominator=0,coverage=None)
    if annotation is not None:
        labels,_=metrics.label_features(metrics.TraceRecord(**row),annotation)
        compact['features'].update(labels['features'])
        compact.update(labeled_sentences=labels['labeled_sentences'],transitions=labels['transitions'],
            label_status='observed' if labels['labeled_sentences'] else 'unknown',
            class_fraction_denominator=labels['class_fraction_denominator'],coverage=labels['coverage'])
    correct=outcome['adjudicated_correct']
    if correct is not None and type(correct) is not bool:raise ValueError('Outcome must be boolean or unknown')
    compact.update(is_correct=correct,outcomes=outcome,outcome_binding=outcome_binding,cohort=cohort)
    if variant.get('label_attachment_sha256'):
        compact.update(label_attachment_sha256=variant['label_attachment_sha256'],
            class_view_status=variant['class_view_status'])
    if 'termination' in outcome:
        compact.update(termination=outcome['termination'],capped=outcome['termination']=='length')
    return compact


def verify_capture_inputs(records):
    for record in records:
        for info in record['files']:
            if sha(info['path'])!=info['sha256']:raise ValueError('Corrupted capture input artifact')


def load_label_attachments(root,manifests,manifest_paths,records):
    """Validate immutable posthoc labels against their exact native parents."""
    if root is None:return {},None
    root=Path(root);byid={r['id']:r for r in records};attached={};fingerprints={}
    expected_code=sha(Path(__file__).with_name('attach_labels.py'))
    for manifest,manifest_path in zip(manifests,manifest_paths,strict=True):
        cohorts={c['id'] for row in manifest['attempts'] for c in row['cohorts']}
        for cohort in sorted(cohorts):
            summary_path=root/cohort/(str(manifest['shard'])+'.result.json')
            if not summary_path.exists():
                if cohort=='B':raise ValueError('Missing complete B label attachment summary')
                continue
            summary=json.loads(summary_path.read_text())
            if (summary.get('status')!='complete' or summary.get('manifest_binding')!=manifest['binding']
                    or summary.get('manifest_sha256')!=sha(manifest_path) or summary.get('cohort')!=cohort):
                raise ValueError('Label attachment summary manifest mismatch')
            expected={row['id'] for row in manifest['attempts'] if any(c['id']==cohort for c in row['cohorts'])}
            entries=summary['records']
            if len(entries)!=len(expected) or {entry['id'] for entry in entries}!=expected:
                raise ValueError('Incomplete/duplicate label attachment coverage')
            for info in summary['annotation_inputs']:
                if sha(info['path'])!=info['sha256']:raise ValueError('Label annotation input changed')
                fingerprints[str(Path(info['path']).resolve())]=info['sha256']
            for entry in entries:
                path=Path(entry['path'])
                if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Label attachment escaped root')
                if sha(path)!=entry['sha256']:raise ValueError('Label attachment content changed')
                value=json.loads(path.read_text());binding=value['binding']
                if (value.get('schema_version')!=1 or value.get('status')!='complete'
                        or value.get('id')!=entry['id'] or value.get('cohort')!=cohort
                        or value.get('binding_sha256')!=digest(binding)
                        or binding.get('manifest_binding')!=manifest['binding']
                        or binding.get('cohort')!=cohort or binding.get('code_sha256')!=expected_code
                        or binding.get('annotations')!=summary['annotation_inputs']
                        or binding.get('capture_files')!=byid[entry['id']]['files']):
                    raise ValueError('Label attachment provenance mismatch')
                verify_capture_inputs([byid[entry['id']]])
                # Only ID/class counts survive posthoc attachment. Full hidden/
                # router class-conditioned summaries cannot be reconstructed.
                attached[entry['id'],cohort]={k:value[k] for k in ('source_trace','annotation','expert_class')}
                attached[entry['id'],cohort].update(views=None,
                    class_view_status='unavailable_discarded_hidden_router_tensors',
                    label_attachment_sha256=entry['sha256'])
                fingerprints[str(path.resolve())]=entry['sha256']
            fingerprints[str(summary_path.resolve())]=sha(summary_path)
    if not attached:raise ValueError('No complete label attachments found at explicit labels root')
    return attached,fingerprints


def fresh_output_root(path):
    allowed=(ROOT/'results/measurement-study/campaign-v3').resolve()
    path=Path(path).resolve()
    if not path.is_relative_to(allowed) or path==allowed:
        raise ValueError('New analysis outputs must be below results/measurement-study/campaign-v3')
    return path


def load_complete_manifests(directory,source,cohort,storage):
    """The ready receipt defines the whole cohort; a surviving subset is invalid."""
    directory,storage=Path(directory),Path(storage)
    ready_path=directory/'ready.json'
    ready=json.loads(ready_path.read_text())
    from measurement_study.annotation_provenance import verify_annotation_inputs
    verify_annotation_inputs(ready.get('annotation_binding'))
    # One label-complete `--cohort all` capture set may serve both cohort
    # analyses; a cohort-specific plan must name that exact cohort.
    if (ready.get('status')!='ready' or ready.get('schema_version')!=3
            or ready.get('source')!=source or ready.get('cohort') not in (cohort,'all')
            or not isinstance(ready.get('spec'),dict) or not isinstance(ready.get('cohort_model'),str)
            or Path(ready.get('storage_root','')).resolve()!=storage.resolve()):
        raise ValueError('Expected complete ready v3 cohort matching source/storage/spec')
    shards=ready.get('shards',[])
    indices=[entry.get('index') for entry in shards]
    if (not indices or any(type(index) is not int or index<0 for index in indices)
            or len(set(indices))!=len(indices) or set(indices)!=set(range(len(indices)))):
        raise ValueError('Invalid/empty ready shard indices')
    paths={path.name:path for path in directory.glob('[0-9]*.json')}
    expected={str(index)+'.json' for index in indices}
    if set(paths)!=expected:raise ValueError('Missing or extra planned collection shard manifest')
    excluded={'status','attempts','tokens','verification_manifest','shards'}
    shared={key:value for key,value in ready.items() if key not in excluded}
    byindex={entry['index']:entry for entry in shards};manifests=[];ordered=[];ids=set()
    for index in sorted(indices):
        path=paths[str(index)+'.json'];manifest=json.loads(path.read_text());entry=byindex[index]
        if (manifest.get('shard')!=index or any(manifest.get(key)!=value for key,value in shared.items())
                or digest({k:v for k,v in manifest.items() if k!='binding'})!=manifest.get('binding')
                or manifest.get('binding')!=entry.get('binding')):
            raise ValueError('Collection shard differs from ready source/spec/binding')
        rows=manifest['attempts'];tokens=sum(row['completion_tokens'] for row in rows)
        if (len(rows)!=entry.get('attempts') or len(rows)!=manifest.get('expected_attempts')
                or tokens!=entry.get('tokens') or tokens!=manifest.get('expected_tokens')
                or manifest.get('requested_seconds')!=entry.get('requested_seconds')
                or Path(manifest['capture_dir']).resolve()!=(storage/'capture'/source/ready['cohort']/str(index)).resolve()):
            raise ValueError('Collection shard counts/tokens/capture path differ from ready receipt')
        for row in rows:
            if row['id'] in ids:raise ValueError('Duplicate attempt across ready collection shards')
            ids.add(row['id'])
        ordered.append(path);manifests.append(manifest)
    if (len(ids)!=ready.get('attempts')
            or sum(m['expected_tokens'] for m in manifests)!=ready.get('tokens')):
        raise ValueError('Collection coverage differs from complete ready totals')
    return ordered,manifests,ready_path


def main():
    global FAST
    from measurement_study.sentence_schema import install
    install()
    p=argparse.ArgumentParser();p.add_argument('source',choices=SOURCES);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--manifest-dir',type=Path,required=True);p.add_argument('--storage-root',type=Path,required=True)
    p.add_argument('--cohort',choices=('A','B','all'),required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--outcomes',type=Path,required=True,help='Explicit corrected-outcome sidecar or its pointer; joined at analysis time')
    p.add_argument('--outcome-manifest',type=Path)
    p.add_argument('--labels-root',type=Path,help='Immutable posthoc class attachments; original packets remain unchanged')
    p.add_argument('--bundle',type=Path,default=os.environ.get('MEASUREMENT_BUNDLE'))
    a=p.parse_args()
    if a.bundle is None:raise ValueError('An explicit frozen --bundle is required for v3 analysis')
    FAST=a.storage_root
    out=fresh_output_root(a.output_root)
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm CPU allocation required')
    import numpy as np
    import torch
    from measurement_study.bundle import import_frozen_helper,worker_init
    metrics=import_frozen_helper('analyze_available_metrics',a.bundle)
    manifest_paths,manifests,ready_path=load_complete_manifests(a.manifest_dir,a.source,a.cohort,FAST)
    ready=json.loads(ready_path.read_text());cohort_model=ready['cohort_model']
    from measurement_study.campaign_v3_plan import read_outcomes,outcome_key
    sidecar_rows,outcomes=read_outcomes(a.outcomes,a.outcome_manifest,corpus='card-v3',model=cohort_model)
    outcome_fingerprint(outcomes)
    sidecar_index={outcome_key(row):row for row in sidecar_rows}
    rows=[r for m in manifests for r in m['attempts']];byid={r['id']:r for r in rows}
    if len(rows)!=len(byid):raise ValueError('Overlapping collection shards')
    joined={}
    for row in rows:
        identity=row['identity']
        outcome=sidecar_index.get(('card-v3',cohort_model,identity['dataset'],identity['problem_id'],identity['sample_id']))
        if outcome is None:
            raise ValueError('Missing outcome sidecar row for a planned attempt')
        if (outcome['trace_sha256']!=row['trace_sha256']
                or outcome['completion_tokens']!=row['completion_tokens']
                or outcome['termination']!=identity['termination']):
            raise ValueError('Analysis-time outcome differs from the captured trace identity')
        joined[row['id']]=outcome
    del sidecar_rows,sidecar_index
    records=[]
    for m in manifests:
        capture=Path(m['capture_dir'])
        if not capture.resolve().is_relative_to(FAST.resolve()):raise ValueError('Capture path outside explicit storage root')
        result=json.loads((capture/'result.json').read_text())
        if (result['status']!='complete_native_collection' or result['manifest_binding']!=m['binding']
                or result['attempts']!=m['expected_attempts'] or result.get('tokens')!=m['expected_tokens']
                or result.get('source')!=a.source or result.get('shard')!=m['shard']
                or len(result['records'])!=m['expected_attempts']
                or sum(r['tokens'] for r in result['records'])!=m['expected_tokens']
                or {r['id'] for r in result['records']}!={r['id'] for r in m['attempts']}):
            raise ValueError('Incomplete capture shard')
        expected_rows={row['id']:row for row in m['attempts']}
        if any(r.get('manifest_binding')!=m['binding'] or len(r['files'])!=2
               or r['tokens']!=expected_rows[r['id']]['completion_tokens'] for r in result['records']):
            raise ValueError('Capture attempt binding/token coverage mismatch')
        records.extend(result['records'])
    if len(records)!=len(byid) or {r['id'] for r in records}!=set(byid):raise ValueError('Incomplete/duplicate planned attempt coverage')
    # Every native tensor is an accepted analysis input, including tuning/test
    # tensors not loaded for calibration. Check all bytes before claiming output.
    verify_capture_inputs(records)
    attachments,label_binding=load_label_attachments(a.labels_root,manifests,manifest_paths,records)
    replay_variants={m['spec'].get('replay_variant','') for m in manifests}
    if len(replay_variants)!=1:raise ValueError('Mixed replay variants')
    replay_variant=next(iter(replay_variants))
    code_binding=analysis_code_binding(metrics,a.bundle)
    full_binding=full_analysis_binding(ready_path,manifest_paths,manifests,outcomes,code_binding,a.cohort,label_binding)
    if checked_complete(out/'result.json',full_binding) is not None:return
    claim_output(out,full_binding)
    split=splits(rows);atomic(out/'question-split.json',dict(seed=42,rule='60/20/20 within benchmark, grouped by question, remainder to test',questions=split))
    layers=manifests[0]['spec']['layers'];cal=[[] for _ in layers]
    for record in records:
        if split[byid[record['id']]['identity']['question_id']]!='calibration':continue
        info=record['files'][0]
        if sha(info['path'])!=info['sha256']:raise ValueError('Corrupted tensor artifact')
        with gzip.open(info['path'],'rb') as f:t=torch.load(f,map_location='cpu',weights_only=True)
        ix=t['reasoning_tokens'].long()
        for i in range(len(layers)):cal[i].append(t['native_boundary_gaps'][i,ix].numpy().copy())
    thresholds={str(layer):float(np.quantile(np.concatenate(cal[i]),.1)) if cal[i] else None for i,layer in enumerate(layers)}
    del cal
    atomic(out/'weak-boundary-calibration.json',dict(question_ids=sorted(q for q,v in split.items() if v=='calibration'),thresholds=thresholds,deduplicated_tokens=True,scope='unique saved completion reasoning tokens; thresholds only applied to tuning/test questions'))
    grouped=defaultdict(list);expert=defaultdict(lambda:defaultdict(list));window_count=0
    for record in records:
        info=record['files'][1]
        if sha(info['path'])!=info['sha256']:raise ValueError('Corrupted metrics artifact')
        with gzip.open(info['path'],'rt') as f:packet=json.load(f)
        identity=byid[record['id']]['identity'];fold=split[identity['question_id']]
        base=packet['trace'];windows=base['metadata']['routing_dynamics']['windows'];window_count+=len(windows)
        weak=defaultdict(list)
        for w in windows:
            threshold=thresholds[str(w['layer'])]
            if fold!='calibration' and threshold is not None:
                w['weak_boundary_fraction']=float(np.mean(np.asarray(w['boundary_gaps'])<=threshold));w['weak_boundary_status']='held_out_question_calibration'
                weak[w['layer'],w['window']].append(w['weak_boundary_fraction'])
        variants=packet.get('cohorts') or {a.cohort:dict(source_trace=base)}
        for cohort,variant in variants.items():
            if a.cohort!='all' and cohort!=a.cohort:continue
            if (record['id'],cohort) in attachments:
                variant={**variant,**attachments[record['id'],cohort]}
            compact=compact_variant(base,variant,metrics,info['path'],
                outcome=joined[record['id']],outcome_binding=outcomes,cohort=cohort,fold=fold)
            membership=next((c for c in byid[record['id']]['cohorts'] if c['id']==cohort),None)
            if membership is None:raise ValueError('Capture cohort not present in input manifest')
            compact.update(attempt_id=record['id'],question_id=identity['question_id'],
                inclusion_probability=membership['inclusion_probability'],arm_inclusion_probability=membership.get('arm_inclusion_probability'),sampling_stratum=membership['stratum'])
            global_weak=defaultdict(list)
            for (layer,width),values in weak.items():
                mean=float(np.mean(values));compact['features'][f'new_layer/L{layer}/W{width}/weak_boundary_fraction']=mean;global_weak[width].append(mean)
            for width,values in global_weak.items():compact['features'][f'new_global/W{width}/weak_boundary_fraction']=float(np.mean(values))
            grouped[cohort,compact['dataset']].append(compact)
            # Keep question-level sufficient statistics for problem-balanced expert/class nPMI.
            x=variant.get('expert_class');n=x['tokens'] if x else 0
            if n:
                payload={'selection':np.asarray(x['counts'],float)/n,'classes':{c:{'p':v['tokens']/n,'joint':np.asarray(v['counts'],float)/n} for c,v in x['classes'].items()}}
                expert[cohort][compact['question_id']].append(payload)
        print(json.dumps({'phase':'compact','attempt':record['id']}),flush=True)
    expert_results={}
    for cohort,questions in expert.items():
        summaries=[]
        for q,attempts in questions.items():
            summaries.append(dict(selection=np.mean([x['selection'] for x in attempts],axis=0),classes={c:dict(p=np.mean([x['classes'][c]['p'] for x in attempts]),joint=np.mean([x['classes'][c]['joint'] for x in attempts],axis=0)) for c in attempts[0]['classes']}))
        px=np.mean([x['selection'] for x in summaries],axis=0);classes={}
        for c in summaries[0]['classes']:
            py=float(np.mean([x['classes'][c]['p'] for x in summaries]));joint=np.mean([x['classes'][c]['joint'] for x in summaries],axis=0)
            valid=(joint>0)&(joint<1)&(px>0)&(py>0)
            npmi=np.full_like(px,np.nan);npmi[valid]=np.log(joint[valid]/(px[valid]*py))/-np.log(joint[valid])
            rate=np.divide(joint,py,out=np.full_like(px,np.nan),where=py>0)
            other=np.divide(px-joint,1-py,out=np.full_like(px,np.nan),where=py<1)
            def safe(v):return [[float(x) if np.isfinite(x) else None for x in row] for row in v]
            classes[c]=dict(class_probability=py,npmi=safe(npmi),selection_rate_difference=safe(rate-other))
        expert_results[cohort]=dict(questions=len(summaries),layers=layers,weighting='equal attempts within question, equal questions',classes=classes)
    atomic(out/'expert-class-statistics.json',expert_results);del expert
    jobs=[]
    for (cohort,dataset),rs in grouped.items():
        path=out/'compact'/cohort/(dataset+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rs))
        jobs.append((str(path),str(out/'associations'/cohort/dataset),dict(bundle=str(a.bundle) if a.bundle else None,
            outcome_binding=outcomes,cohort_binding=[m['cohort_binding'] for m in manifests],label_binding=label_binding)))
    del grouped
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=worker_init if a.bundle else None,
                             initargs=(a.bundle,) if a.bundle else ()) as pool:
        for f in as_completed([pool.submit(analyze_group,x) for x in jobs]):results.append(f.result());atomic(out/'progress.json',dict(completed_groups=len(results),expected_groups=len(jobs)))
    output_hashes={str(path.relative_to(out)):sha(path) for path in out.rglob('*') if path.is_file() and path.name!='result.json'}
    atomic(out/'result.json',dict(output_hashes=output_hashes,status='complete',source=a.source,replay_variant=replay_variant,input_binding=full_binding['input_binding'],analysis_code_binding=full_binding['code_binding'],analysis_binding=full_binding,attempts=len(rows),windows=window_count,groups=results,calibration_only_thresholds=True,all_cohorts_separate=True,limitations=['Exploratory selected-sample associations; no causal inference','Weak-boundary associations use tuning/test questions only','Missing native models remain explicit in campaign coverage']))
if __name__=='__main__':main()
