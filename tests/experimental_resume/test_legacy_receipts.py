import json
from pathlib import Path
import pytest
from moe_exp.routing_control.receipts import ReceiptStore


def test_actual_legacy_request_uid_requires_explicit_binding(workdir):
    campaign=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
    uid=json.loads((campaign/'manifests/x1-v1.json').read_text())['requests'][0]['uid']
    assert len(uid)==32
    with pytest.raises(ValueError,match='explicitly bound'):
        ReceiptStore(workdir/'default',{'manifest':'test'},[uid])
    binding={'uid_format':'legacy-sha256-prefix32','manifest':'test'}
    with ReceiptStore(workdir/'legacy',binding,[uid]) as store:
        store.put({'uid':uid,'mean_native_surprisal':.25})
    with ReceiptStore(workdir/'legacy',binding,[uid]) as store:
        assert store.read()[uid]['mean_native_surprisal']==.25
        with pytest.raises(ValueError,match='replace'):
            store.put({'uid':uid,'mean_native_surprisal':.50})


def test_new_study_uid_format_remains_sha256(workdir):
    with pytest.raises(ValueError):
        ReceiptStore(workdir/'new',{},['0'*32])
    with ReceiptStore(workdir/'new',{},['0'*64]) as store:
        store.put({'uid':'0'*64,'status':'assigned failure'})
        assert set(store.read())=={'0'*64}
