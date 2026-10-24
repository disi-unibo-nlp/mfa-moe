import hashlib,io,json,subprocess,urllib.error
from pathlib import Path
from unittest.mock import patch
import pytest
from measurement_study.gemma_download import fetch,verify,validate_index
from measurement_study import label_transport

def item(data,name='x'):
    return dict(name=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),url='https://example.test/x')

def test_download_interrupted_resume_and_verified_reuse(tmp_path):
    data=b'complete file';spec=item(data);partial=tmp_path/'x.partial';partial.write_bytes(data[:4])
    def transfer(args,check):
        assert '--continue-at' in args and '--proto' in args and '=https' in args
        assert partial.read_bytes()==data[:4]
        partial.write_bytes(data)
    assert fetch(spec,tmp_path,run=transfer)=='downloaded_verified'
    assert fetch(spec,tmp_path,run=lambda *a,**k:pytest.fail('redownload'))=='reused_verified'
    assert not partial.exists()

def test_download_corruption_and_interruption_are_not_published(tmp_path):
    spec=item(b'good');partial=tmp_path/'x.partial';partial.write_bytes(b'bad!')
    with pytest.raises(ValueError,match='hash'):fetch(spec,tmp_path)
    assert partial.read_bytes()==b'bad!' and not (tmp_path/'x').exists()
    partial.write_bytes(b'g')
    def interrupted(*a,**k):raise subprocess.CalledProcessError(28,['curl'])
    with pytest.raises(subprocess.CalledProcessError):fetch(spec,tmp_path,run=interrupted)
    assert partial.read_bytes()==b'g'
    (tmp_path/'x').write_bytes(b'bad!')
    with pytest.raises(ValueError,match='published'):fetch(spec,tmp_path)

def test_git_blob_and_missing_shard(tmp_path):
    data=b'abc';p=tmp_path/'config';p.write_bytes(data)
    spec=dict(bytes=3,sha256=None,git_blob_id=hashlib.sha1(b'blob 3\0abc').hexdigest())
    assert verify(p,spec);spec['git_blob_id']='0'*40;assert not verify(p,spec)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'w':'missing.safetensors'}}))
    with pytest.raises(ValueError):validate_index(tmp_path,[item(b'x','expected.safetensors')])

class Response(io.BytesIO):
    status=200

def test_transport_timeout_retry_bounded_and_payload_unchanged():
    response={'choices':[{'index':0}]};calls=[]
    def request(req,timeout):
        calls.append((req.full_url,req.data,timeout))
        if len(calls)==1:raise TimeoutError('read timeout')
        return Response(b'ok' if req.full_url.endswith('/health') else json.dumps(response).encode())
    with patch.object(label_transport.urllib.request,'urlopen',side_effect=request),patch.object(label_transport.time,'sleep') as sleep:
        assert label_transport.post_batch({'messages':[]},base_url='http://localhost/v1',api_key='test')==response
    assert calls[0][1]==calls[2][1] and calls[0][2]==calls[2][2]==1200
    sleep.assert_called_once_with(5)
    def always(req,timeout):
        if req.full_url.endswith('/health'):return Response(b'ok')
        raise TimeoutError('read timeout')
    with patch.object(label_transport.urllib.request,'urlopen',side_effect=always) as call,patch.object(label_transport.time,'sleep'):
        with pytest.raises(TimeoutError):label_transport.post_batch({},base_url='http://localhost/v1',api_key='test')
        assert call.call_count==5  # Three POSTs, two health checks.

def test_invalid_input_and_dead_server_are_not_retried():
    error=urllib.error.HTTPError('http://localhost',400,'invalid input',{},io.BytesIO())
    with patch.object(label_transport.urllib.request,'urlopen',side_effect=error) as call:
        with pytest.raises(urllib.error.HTTPError):label_transport.post_batch({},base_url='http://localhost',api_key='test')
        assert call.call_count==1
    with patch.object(label_transport.urllib.request,'urlopen',side_effect=[TimeoutError(),ConnectionRefusedError()]) as call,patch.object(label_transport.time,'sleep'):
        with pytest.raises(ConnectionRefusedError):label_transport.post_batch({},base_url='http://localhost',api_key='test')
        assert call.call_count==2

def test_native_bias_ties_dtypes_and_genuine_mismatch():
    import torch
    from measurement_study.native_selection import selection_scores,require_native_boundary
    for dtype in (torch.float32,torch.bfloat16):
        raw=torch.tensor([[0.,0.,-2.]],dtype=dtype);bias=torch.tensor([0.,.1,0.],dtype=dtype)
        scores=selection_scores(raw,bias,scoring='sigmoid',fused_grouped=True)
        assert scores.dtype==dtype and scores.argmax(-1).item()==1
        assert require_native_boundary(scores,torch.tensor([[1]])).item()>0
        with pytest.raises(ValueError,match='disagree'):require_native_boundary(scores,torch.tensor([[2]]))
    scores=torch.tensor([[.5,.5+5e-8,.1]])
    assert require_native_boundary(scores,torch.tensor([[0]])).item()==0


def test_legacy_annotation_keeps_original_offsets_and_rejects_corruption():
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline import spans
    from measurement_study.sentence_schema import infer_schema,install
    from measurement_study.production_plan import normalize_annotation
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    trace=TraceRecord(prompt='x',gold_answer='1',model_answer='1',dataset='math500',problem_id='1',sample_id=0,model_id='fixture',cot_text='<think>Let x$. First sentence. Second sentence. Third sentence. $y$.</think>',metadata={})
    legacy=spans.sentence_spans_v1(trace)
    current=spans.sentence_spans(trace)
    assert legacy!=current
    annotation=dict(schema_version=1,trace_sha256=spans.trace_digest(trace),units=[dict(u,label='Analyze') for u in legacy])
    install();assert infer_schema(trace,annotation)==1
    normalized=normalize_annotation(trace,annotation)
    assert normalized['units']==annotation['units']
    assert saved_layout(trace)['units']==legacy
    spans.validate_annotation(trace,normalized)
    broken={**annotation,'units':[dict(annotation['units'][0],text='corrupt')]}
    with pytest.raises(ValueError,match='offsets/text'):infer_schema(trace,broken)


def test_gemma_source_versions_are_separate_and_all_files_retained():
    from measurement_study.gemma_source_recovery import split_spec
    original=dict(path='/data/qwen38-mtp3-production/gemma/part-00/checkpoints/batch-000000.json')
    repaired=dict(path='/data/qwen38-mtp3-production-v2-affected/gemma/part-00/checkpoints/batch-000000.json')
    spec=dict(id='qwen38-mtp3-production-gemma',annotations=[original,repaired],traces=['unchanged'])
    versions=split_spec(spec)
    assert [s['annotations'] for s in versions]==[[original],[repaired]]
    assert len({s['id'] for s in versions})==2
    assert all(s['traces']==['unchanged'] for s in versions)
    assert spec['annotations']==[original,repaired]
    with pytest.raises(ValueError,match='Both original'):split_spec(dict(spec,annotations=[original]))
