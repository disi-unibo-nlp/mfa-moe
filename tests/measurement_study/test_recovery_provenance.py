import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from measurement_study.production_plan import digest
from measurement_study.recovery_rebind import rebind_manifest, reuse_packet
from measurement_study.artifact_layout import require_result_variant, result_dir
from measurement_study.gemma_cache_policy import VARIANT


def test_rebind_preserves_semantics_and_rejects_changed_parent(tmp_path):
    spec={'checkpoint':'fixture','revision':'pinned','layers':[1],'precision':'nvfp4'}
    parent=dict(spec=spec,attempts=[dict(id='saved',completion_tokens=10)],expected_attempts=1,expected_tokens=10,code_binding='old',artifact_cap_bytes=20)
    parent['binding']=digest(parent)
    p=tmp_path/'parent.json';p.write_text(json.dumps(parent))
    result=rebind_manifest(parent,spec,'new',40,p)
    assert result['attempts']==parent['attempts'] and result['spec']==parent['spec']
    assert result['parent_manifest']['binding']==parent['binding']
    assert parent['code_binding']=='old' and parent['artifact_cap_bytes']==20
    with pytest.raises(ValueError,match='semantics'):
        rebind_manifest(parent,dict(spec,revision='different'),'new',40,p)
    changed=copy.deepcopy(parent);changed['attempts'][0]['completion_tokens']=9
    with pytest.raises(ValueError,match='digest'):
        rebind_manifest(changed,spec,'new',40,p)


def test_sensitivity_analysis_cannot_mix_with_primary():
    manifests=[{'spec':{'replay_variant':VARIANT}}]*4
    with patch.dict(os.environ,{'MEASUREMENT_RESULT_VARIANT':''}):
        with pytest.raises(ValueError,match='disagree'):
            require_result_variant('gemma',manifests)
    with patch.dict(os.environ,{'MEASUREMENT_RESULT_VARIANT':VARIANT}):
        assert require_result_variant('gemma',manifests)==VARIANT
        assert result_dir('gemma','production-analysis').name=='gemma-bf16-kv-sensitivity'
        with pytest.raises(ValueError,match='Unverified'):
            result_dir('nemotron','production-analysis')


def test_reused_artifact_keeps_tensor_and_rejects_corruption(tmp_path):
    tensor=tmp_path/'tensor.gz';tensor.write_bytes(gzip.compress(b'unchanged native tensor'))
    packet=tmp_path/'packet.gz';raw=json.dumps(dict(id='saved',manifest_binding='old')).encode();packet.write_bytes(gzip.compress(raw))
    def info(p,raw):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest())
    record=dict(id='saved',tokens=10,seconds=1,manifest_binding='old',files=[info(tensor,b'unchanged native tensor'),info(packet,raw)])
    manifest=dict(binding='new',attempts=[dict(id='saved',completion_tokens=10)])
    receipt=tmp_path/'accepted.json';receipt.write_text('{"status":"passed"}')
    out=tmp_path/'out';out.mkdir()
    result=reuse_packet(record,'old',manifest,out,receipt)
    assert result['files'][0]==record['files'][0]
    assert result['parent_capture']['files']==record['files']
    assert json.loads(gzip.decompress(packet.read_bytes()))['manifest_binding']=='old'
    packet.write_bytes(b'corruption')
    with pytest.raises(ValueError,match='changed'):
        reuse_packet(record,'old',manifest,out,receipt)
