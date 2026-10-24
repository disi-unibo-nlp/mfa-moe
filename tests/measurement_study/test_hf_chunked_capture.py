from types import SimpleNamespace
from unittest.mock import patch
import torch
from measurement_study.hf_chunked_capture import capture_chunked, compare_native_captures


def test_kv_chunked_state_and_exact_prompt_boundary():
    calls=[]
    class Model:
        config=SimpleNamespace(model_type='gpt_oss')
        def parameters(self):return iter([torch.zeros(1)])
        @property
        def model(self):return self
        def __call__(self,**kw):
            start=int(kw['cache_position'][0]);end=int(kw['cache_position'][-1])+1
            assert kw['attention_mask'].shape==(1,end)
            assert kw['past_key_values']==(None if start==0 else start)
            calls.append((start,end))
            return SimpleNamespace(past_key_values=end)
    def stream(model,forward,kwargs,prompt,requested,hidden,details):
        forward(**kwargs)
        values=kwargs['input_ids'][0,prompt:].float()[None,:,None].expand(2,-1,2).clone()
        details.update(selected_experts=torch.zeros(values.shape[:-1]+(1,),dtype=torch.long),
                       expert_weights=torch.ones(values.shape[:-1]+(1,)),
                       semantics={'native_selection':True,'executed_weights':'native_expert_input_weights'})
        return values,values.clone()
    trace=SimpleNamespace(metadata={'token_replay':{}},cot_text='fixture')
    with patch('moe_exp.models.token_replay.validate_token_replay',return_value=(list(range(5)),list(range(5,13)))),patch('moe_exp.models.router_adapters.stream_router_signals',side_effect=stream):
        result=capture_chunked(Model(),None,trace,[0,2],4)
    assert calls==[(0,4),(4,8),(8,12),(12,13)]
    assert result['router'][0,:,0].tolist()==list(range(5,13))
    assert result['router'].shape==(2,8,2)
    assert result['audit']['native_dispatch']


def test_chunked_differential_rejects_real_expert_mismatch():
    import pytest
    a={'ids':torch.tensor([[[0,1]]]),'weights':torch.tensor([[[.7,.3]]]),'router':torch.tensor([[[1.,.5,0.]]]),'hidden':torch.ones(1,1,2)}
    assert compare_native_captures(a,{**a})['status']=='passed'
    with pytest.raises(ValueError,match='selected expert sets'):
        compare_native_captures(a,{**a,'ids':torch.tensor([[[0,2]]])})


def test_chunked_differential_rejects_nonfinite_reference_and_candidate():
    import pytest
    a={'ids':torch.tensor([[[0,1]]]),'weights':torch.tensor([[[.7,.3]]]),'router':torch.tensor([[[1.,.5,0.]]]),'hidden':torch.ones(1,1,2)}
    for field in ('router','hidden','weights'):
        for value in (float('nan'),float('inf')):
            corrupt={**a,field:torch.full_like(a[field],value)}
            for reference,candidate in ((corrupt,a),(a,corrupt),(corrupt,corrupt)):
                with pytest.raises(ValueError):
                    compare_native_captures(reference,candidate)


def test_forced_chunking_does_not_change_single_pass_reference():
    from measurement_study.production_capture import HFBackend
    backend=HFBackend.__new__(HFBackend)
    backend.spec={'generation_model':'openai/gpt-oss-20b','force_chunked':True,'layers':[0]}
    backend.model=object();backend.tokenizer=None
    trace=SimpleNamespace(prompt='test',cot_text='text',system_prompt=None,generation_messages=None,
                          metadata={'token_replay':{'prompt_token_ids':[1],'completion_token_ids':[2,3]}})
    def single(**kwargs):
        kwargs['routing_details'].update(semantics={'native_selection':True,'executed_weights':'native_expert_input_weights'},selected_experts=torch.zeros(1,2,1,dtype=torch.long),expert_weights=torch.ones(1,2,1))
        return torch.ones(1,2,2),torch.ones(1,2,2)
    with patch('moe_exp.models.inference.extract_logs_single_pass',side_effect=single) as direct,patch('measurement_study.hf_chunked_capture.capture_chunked',return_value='chunked') as chunk:
        assert backend.capture(trace,'actual')=='chunked'
        result=backend.capture_single(trace,'reference')
        assert result['audit']['backend']=='hf'
        assert direct.call_count==1 and chunk.call_count==1
