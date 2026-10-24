"""Small immutable-cache regressions; no corpus work or GPU imports."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from measurement_study import production_analyze as a


def sidecar(tmp_path):
    path=tmp_path/'outcomes.parquet';path.write_bytes(b'outcome-v1')
    manifest=tmp_path/'outcomes.manifest.json';manifest.write_text('{"schema_version":3}')
    return dict(path=str(path),sha256=a.sha(path),manifest_path=str(manifest),manifest_sha256=a.sha(manifest))


@pytest.fixture
def worker(tmp_path,monkeypatch):
    from measurement_study import bundle
    calls=[]
    def finish(rows,out):
        calls.append(rows);(out/'summary.json').write_text(json.dumps(rows));return ['ok']
    metrics=SimpleNamespace(finish_groups=finish)
    monkeypatch.setattr(bundle,'import_frozen_helper',lambda *args:metrics)
    code={'version':'one'}
    monkeypatch.setattr(a,'analysis_code_binding',lambda *args:dict(code))
    path=tmp_path/'rows.jsonl';path.write_text('{"is_correct":null}\n')
    args=(str(path),str(tmp_path/'analysis'),dict(outcome_binding=sidecar(tmp_path),cohort_binding='cohort-one'))
    return args,calls,code


def test_identical_completed_analysis_is_reused(worker):
    args,calls,code=worker
    first=a.analyze_group(args);second=a.analyze_group(args)
    assert first==second and len(calls)==1


@pytest.mark.parametrize('change',['input','sidecar','sidecar_manifest','cohort','code','output'])
def test_changed_analysis_inputs_cannot_reuse_or_overwrite(worker,change):
    args,calls,code=worker
    a.analyze_group(args)
    saved=(Path(args[1])/'complete.json').read_bytes()
    if change=='input':Path(args[0]).write_text('{"is_correct":true}\n')
    elif change=='sidecar':Path(args[2]['outcome_binding']['path']).write_bytes(b'outcome-v2')
    elif change=='sidecar_manifest':Path(args[2]['outcome_binding']['manifest_path']).write_text('{}')
    elif change=='cohort':args[2]['cohort_binding']='cohort-two'
    elif change=='code':code['version']='two'
    else:(Path(args[1])/'summary.json').write_text('changed')
    with pytest.raises(ValueError):a.analyze_group(args)
    assert (Path(args[1])/'complete.json').read_bytes()==saved and len(calls)==1


def test_legacy_unbound_completion_rejected(worker):
    args,calls,code=worker
    out=Path(args[1]);out.mkdir();(out/'complete.json').write_text('{"status":"complete"}')
    with pytest.raises(ValueError,match='fresh output root'):a.analyze_group(args)
    assert not calls


def test_explicit_output_root_cannot_overwrite_old_results(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'ROOT',tmp_path)
    with pytest.raises(ValueError):a.fresh_output_root(tmp_path/'results/measurement-study/production-analysis/gpt')
    good=tmp_path/'results/measurement-study/campaign-v3/analysis/gpt'
    assert a.fresh_output_root(good)==good


def test_label_free_variant_uses_corrected_outcome_without_mutating_trace():
    base=dict(is_correct=True,metadata=dict(forward_provenance={'checkpoint':'native'},
        correlation_features={},routing_dynamics={}),model_logs=dict(layer_indices=[0]))
    seen=[]
    def routing(trace,path):
        seen.append(trace['is_correct']);return dict(features={'routing':1},is_correct=trace['is_correct'])
    metrics=SimpleNamespace(routing_features=routing)
    compact=a.compact_variant(base,{},metrics,'packet.json',outcome={'adjudicated_correct':None},
        outcome_binding={'sha256':'corrected'},cohort='B',fold='test')
    assert seen==[True] and base['is_correct'] is True
    assert compact['is_correct'] is None and compact['features']=={'routing':1}
    assert compact['label_status']=='unavailable' and compact['coverage'] is None
    assert compact['class_fraction_denominator']==0


def test_completed_parent_revalidates_actual_capture_files(tmp_path):
    packet=tmp_path/'packet.gz';packet.write_bytes(b'capture-one')
    records=[{'files':[{'path':str(packet),'sha256':a.sha(packet)}]}]
    a.verify_capture_inputs(records)
    packet.write_bytes(b'capture-two')
    with pytest.raises(ValueError,match='capture input'):a.verify_capture_inputs(records)


def attachment_fixture(tmp_path):
    manifest_path=tmp_path/'0.json'
    manifest=dict(shard=0,binding='manifest-one',attempts=[dict(id='attempt',cohorts=[{'id':'B'}])])
    manifest_path.write_text(json.dumps(manifest))
    native=tmp_path/'native.gz';native.write_bytes(b'native')
    files=[dict(path=str(native),sha256=a.sha(native))]
    record=dict(id='attempt',files=files)
    annotation=tmp_path/'annotation.json';annotation.write_text('{}')
    inputs=[dict(path=str(annotation),sha256=a.sha(annotation))]
    root=tmp_path/'labels';(root/'B').mkdir(parents=True)
    binding=dict(manifest_binding=manifest['binding'],capture_files=files,annotations=inputs,
        cohort='B',code_sha256=a.sha(Path(a.__file__).with_name('attach_labels.py')))
    attachment=dict(schema_version=1,status='complete',id='attempt',cohort='B',binding=binding,
        binding_sha256=a.digest(binding),source_trace={},annotation={},expert_class={'tokens':1})
    path=root/'B/attempt.labels.json';path.write_text(json.dumps(attachment))
    summary=dict(status='complete',manifest_sha256=a.sha(manifest_path),manifest_binding=manifest['binding'],
        cohort='B',annotation_inputs=inputs,records=[dict(id='attempt',path=str(path),sha256=a.sha(path))])
    (root/'B/0.result.json').write_text(json.dumps(summary))
    return root,manifest,manifest_path,record,annotation,path


def test_posthoc_labels_are_bound_to_native_parents_and_mark_missing_views(tmp_path):
    root,manifest,path,record,annotation,labels=attachment_fixture(tmp_path)
    attached,binding=a.load_label_attachments(root,[manifest],[path],[record])
    value=attached['attempt','B']
    assert value['expert_class']['tokens']==1 and value['views'] is None
    assert value['class_view_status']=='unavailable_discarded_hidden_router_tensors'
    assert binding[str(labels.resolve())]==a.sha(labels)


@pytest.mark.parametrize('change',['annotation','native','attachment','parent','missing'])
def test_changed_posthoc_label_inputs_are_rejected(tmp_path,change):
    root,manifest,path,record,annotation,labels=attachment_fixture(tmp_path)
    if change=='annotation':annotation.write_text('{"changed":true}')
    elif change=='native':Path(record['files'][0]['path']).write_bytes(b'changed')
    elif change=='attachment':labels.write_text('{}')
    elif change=='parent':manifest['binding']='different'
    else:(root/'B/0.result.json').unlink()
    with pytest.raises(ValueError):a.load_label_attachments(root,[manifest],[path],[record])


def test_worker_revalidates_label_attachment_hash_before_reuse(worker,tmp_path):
    args,calls,code=worker
    attachment=tmp_path/'extra-labels.json';attachment.write_text('{}')
    args[2]['label_binding']={str(attachment):a.sha(attachment)}
    a.analyze_group(args)
    attachment.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='Label attachment input changed'):a.analyze_group(args)
    assert len(calls)==1


def ready_fixture(tmp_path):
    root=tmp_path/'manifests';root.mkdir()
    storage=tmp_path/'storage'
    base=dict(schema_version=3,source='gpt',cohort='B',spec={'layers':[0],'checkpoint':'native'},
              storage_root=str(storage),cohort_model='gpt',code_binding='capture-code')
    manifests=[]
    for index in range(2):
        row=dict(id=str(index),completion_tokens=100+index)
        m=dict(base,shard=index,attempts=[row],expected_attempts=1,expected_tokens=row['completion_tokens'],
               requested_seconds=1800,capture_dir=str(storage/'capture/gpt/B'/str(index)))
        m['binding']=a.digest(m);manifests.append(m)
        directory=storage/'capture/gpt/B'/str(index);directory.mkdir(parents=True)
        (directory/'result.json').write_text('{}')
        (root/(str(index)+'.json')).write_text(json.dumps(m))
    ready=dict(base,status='ready',attempts=2,tokens=201,verification_manifest='verify.json',
        shards=[dict(index=m['shard'],attempts=m['expected_attempts'],tokens=m['expected_tokens'],
            requested_seconds=m['requested_seconds'],binding=m['binding']) for m in manifests])
    (root/'ready.json').write_text(json.dumps(ready))
    return root,storage,ready,manifests


def test_ready_receipt_requires_complete_planned_collection(tmp_path):
    root,storage,ready,manifests=ready_fixture(tmp_path)
    paths,loaded,receipt=a.load_complete_manifests(root,'gpt','B',storage)
    assert loaded==manifests and [p.name for p in paths]==['0.json','1.json']
    assert receipt==root/'ready.json'


def test_all_cohort_ready_receipt_serves_one_capture_set_for_both_cohorts(tmp_path):
    root,storage,ready,manifests=ready_fixture(tmp_path)
    ready['cohort']='all'
    for index,manifest in enumerate(manifests):
        manifest['cohort']='all'
        manifest['capture_dir']=str(storage/'capture/gpt/all'/str(index))
        manifest['binding']=a.digest({k:v for k,v in manifest.items() if k!='binding'})
        ready['shards'][index]['binding']=manifest['binding']
        (storage/'capture/gpt/all'/str(index)).mkdir(parents=True,exist_ok=True)
        (root/(str(index)+'.json')).write_text(json.dumps(manifest))
    (root/'ready.json').write_text(json.dumps(ready))
    paths,loaded,_=a.load_complete_manifests(root,'gpt','B',storage)
    assert len(loaded)==2
    paths,loaded,_=a.load_complete_manifests(root,'gpt','A',storage)
    assert len(loaded)==2


def test_analysis_bundle_need_not_equal_capture_code_but_capture_code_is_recorded(tmp_path):
    root,storage,ready,manifests=ready_fixture(tmp_path)
    outcome=sidecar(tmp_path)
    binding=a.full_analysis_binding(root/'ready.json',[root/'0.json',root/'1.json'],manifests,
                                    outcome,{'bundle':{'bundle_sha256':'analysis-bundle'}},'B',None)
    assert binding['capture_code_binding']=='capture-code'
    assert binding['code_binding']=={'bundle':{'bundle_sha256':'analysis-bundle'}}
    manifests[1]['code_binding']='different-capture-code'
    with pytest.raises(ValueError,match='Mixed or missing capture code'):
        a.full_analysis_binding(root/'ready.json',[root/'0.json',root/'1.json'],manifests,
                                outcome,{'bundle':{'bundle_sha256':'analysis-bundle'}},'B',None)


@pytest.mark.parametrize('change',['missing_shard','extra_shard','not_ready','source','spec','tokens','attempts','duplicate_index','extra_attempt'])
def test_ready_receipt_rejects_subset_or_inconsistent_analysis(tmp_path,change):
    root,storage,ready,manifests=ready_fixture(tmp_path)
    if change=='missing_shard':(root/'1.json').unlink()
    elif change=='extra_shard':(root/'2.json').write_text(json.dumps(manifests[0]))
    elif change=='not_ready':ready['status']='needs_measurement'
    elif change=='source':ready['source']='glm'
    elif change=='spec':ready['spec']={'layers':[1],'checkpoint':'other'}
    elif change=='tokens':ready['tokens']=200
    elif change=='attempts':ready['attempts']=1
    elif change=='duplicate_index':ready['shards'][1]['index']=0
    else:
        m=manifests[1];m['attempts'].append(dict(id='extra',completion_tokens=1))
        m['binding']=a.digest({k:v for k,v in m.items() if k!='binding'})
        ready['shards'][1]['binding']=m['binding']
        (root/'1.json').write_text(json.dumps(m))
    (root/'ready.json').write_text(json.dumps(ready))
    with pytest.raises(ValueError):a.load_complete_manifests(root,'gpt','B',storage)
