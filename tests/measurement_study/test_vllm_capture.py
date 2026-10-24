import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from measurement_study.vllm_capture import frame_entries,NativeHook

def test_interleaved_prompt_padding_and_replay():
    requests={'a':{'spec':{'tokens':[1,2,3,4],'prompt_length':2}},'b':{'spec':{'tokens':[9,8,7],'prompt_length':1}}}
    result=frame_entries(requests,['b','a'],[0,2,4],[8,7,2,3,999],[1,2,1,2,999])
    assert result==[('b',0),('b',1),None,('a',0),None]
    assert frame_entries(requests,['a'],[0,2],[3,4],[2,3])==[('a',0),('a',1)]
    with pytest.raises(ValueError,match='token mismatch'):frame_entries(requests,['a'],[0,1],[5],[2])
    with pytest.raises(ValueError,match='Unregistered'):frame_entries(requests,['cancelled'],[0,1],[1],[0])

def make_hook(monolithic=False):
    import torch
    ids=torch.tensor([[2,0]],dtype=torch.int32);weights=torch.tensor([[.7,.3]])
    router=type('FusedTopKRouter',(),{})();router.select_experts=lambda *a,**kw:(weights,ids);router.scoring_func='softmax'
    shared=object();calls=[]
    quant=SimpleNamespace(is_monolithic=monolithic,fp8_backend=SimpleNamespace(name='MARLIN'))
    routed=SimpleNamespace(quant_method=quant,forward_modular=lambda **kw:calls.append(kw))
    runner=type('MoERunner',(),{})();runner.router=router;runner.routed_experts=routed
    records=[];owner={'frame':[('r',0)],'requests':{'r':{'spec':{'prompt_length':2},'layers':{3:{}}}},'store':lambda state,pos,values:records.append((pos,values))}
    return NativeHook(owner,3,SimpleNamespace(experts=runner)),ids,weights,shared,calls,records

def test_capture_uses_actual_dispatch_weights_and_preserves_shared_branch():
    import torch
    hook,ids,weights,shared,calls,records=make_hook()
    logits=torch.tensor([[.2,.1,.3]])
    w,i=hook.select(torch.zeros(1,2),logits)
    assert w is weights and i is ids
    hook.dispatch(topk_ids=i,topk_weights=w,shared_experts=shared)
    assert calls[0]['shared_experts'] is shared and calls[0]['topk_ids'] is ids
    assert torch.equal(records[0][1]['weights'],weights[0])
    assert not torch.allclose(records[0][1]['weights'],logits.softmax(-1)[0,ids[0].long()])
    assert hook.dispatches==1 and hook.pending is None

def test_rejects_bypassed_or_monolithic_dispatch():
    import torch
    with pytest.raises(RuntimeError,match='Unsupported'):make_hook(monolithic=True)
    hook,ids,weights,shared,calls,records=make_hook();hook.select(torch.zeros(1,2),torch.zeros(1,3))
    with pytest.raises(RuntimeError,match='did not reach'):hook.dispatch(topk_ids=ids.clone(),topk_weights=weights)


def test_only_verified_gemma_custom_routing_is_accepted():
    hook,ids,weights,shared,calls,records=make_hook()
    runner=hook.runner
    # Restore the original methods before attaching a replacement hook.
    runner.routed_experts.forward_modular=hook.dispatch_original
    custom=type('CustomRoutingRouter',(),{})()
    custom.select_experts=hook.select_original
    def fn(*args):pass
    custom.custom_routing_function=fn
    runner.router=custom
    block=type('Gemma4MoE',(),{})();block.experts=runner;block.per_expert_scale=object()
    with pytest.raises(RuntimeError,match='Unverified custom'):NativeHook(hook.owner,3,block)
    fn.__module__='vllm.model_executor.models.gemma4'
    fn.__qualname__='Gemma4MoE.__init__.<locals>.routing_function'
    captured=NativeHook(hook.owner,3,block)
    import torch
    w,i=captured.select(torch.zeros(1,2),torch.tensor([[.2,.1,.3]]))
    captured.dispatch(topk_ids=i,topk_weights=w)
    assert w is weights and torch.equal(records[0][1]['weights'],weights[0])


def test_dispatch_rejects_genuine_native_selector_disagreement():
    import torch
    hook,ids,weights,shared,calls,records=make_hook()
    # Native test selector returns [2,0], but experts [0,1] are unambiguously top-2.
    w,i=hook.select(torch.zeros(1,2),torch.tensor([[3.,2.,0.]]))
    with pytest.raises(ValueError,match='disagree'):
        hook.dispatch(topk_ids=i,topk_weights=w)
    assert calls == [] and records == []


@pytest.mark.parametrize('spec',[{'capture_version':2},{'corpus':'card-v3'}])
def test_v3_worker_requires_explicit_bundle_environment(spec,monkeypatch):
    from measurement_study.vllm_capture import MeasurementWorkerExtension
    monkeypatch.delenv('MEASUREMENT_BUNDLE',raising=False)
    with pytest.raises(RuntimeError,match='requires a frozen bundle'):
        MeasurementWorkerExtension().measurement_configure(spec)
