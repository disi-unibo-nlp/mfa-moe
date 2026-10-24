import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('measurement_contract',Path(__file__).resolve().parents[2]/'measurement_study/contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def test_hybrid_and_dense_layers_are_not_fabricated():
    assert c.routed_layers({'model_type':'nemotron_h','layers_block_type':['mamba','moe','attention','moe']})==[1,3]
    assert c.routed_layers({'model_type':'glm4_moe_lite','num_hidden_layers':4,'first_k_dense_replace':1})==[1,2,3]


def test_common_flags_include_temporal_and_hidden_metrics():
    args=c.extraction_options(dict(forward_model='m',generation_model='m',revision='hash',quantization='none'),'g','a','o',['math500'])
    assert '--all-router-layers' in args and '--dynamics' in args and '--router-only' not in args


def test_capture_rejects_all_layers_claim_without_dispatch_evidence():
    with pytest.raises(ValueError,match='native_dispatch_verified'):
        c.check_capture({'schema_version':1,'layers':[0,1]}, {'layers':[0,1]})


def test_incomplete_source_is_never_complete():
    with pytest.raises(ValueError):c.check_result({'status':'complete'}, {'layers':[0]})


def test_model_specific_native_topk_is_not_changed():
    assert c.routed_layers({'model_type':'qwen3_moe','num_hidden_layers':6,'decoder_sparse_step':2,'mlp_only_layers':[3]})==[1,5]
    assert len(c.TEMPORAL)==len(set(c.TEMPORAL))
