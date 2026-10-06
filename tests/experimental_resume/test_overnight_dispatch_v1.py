import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/experimental_resume'))
import dispatch_overnight_readers_v1 as dispatch
from prepare_overnight_designs_v1 import design, expert_sets, fresh_spec


def test_singleton_controls_use_explicit_mapping():
    dictionary = dispatch.base.sealed(dispatch.DOC / 'CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json')
    assert [x[0][1][0] for x in expert_sets(dictionary, 'candidate_to_verify', 189)[1]] == [139,255,43,196]
    assert [x[0][1][0] for x in expert_sets(dictionary, 'candidate_to_verify', 9)[1]] == [120,133,5,24]
    assert len(fresh_spec(dictionary)['primary_contrasts']) == 28
    assert [len(design(dictionary, name)['arms']) for name in 'ABC'] == [8,6,8]


def test_submit_receipt_resumes_without_duplicate_sbatch(tmp_path, monkeypatch):
    calls = []
    def call(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=('12345\n' if '--parsable' in args else
                                       'JobId=12345 UserId=lmolfett(133943)' if args[0] == 'scontrol' else ''))
    monkeypatch.setattr(dispatch.subprocess, 'run', call)
    bind = {'manifest_sha256': 'test'}
    assert dispatch.submit(tmp_path, 'reader', ['reader.sbatch'], {}, bind) == '12345'
    count = len(calls)
    assert dispatch.submit(tmp_path, 'reader', ['reader.sbatch'], {}, bind) == '12345'
    assert len(calls) == count
    assert sum('--parsable' in x for x in calls) == 1
    with pytest.raises(ValueError, match='different inputs'):
        dispatch.submit(tmp_path, 'reader', ['other.sbatch'], {}, bind)


def test_unresolved_attempt_never_resubmits(tmp_path, monkeypatch):
    dispatch.save(tmp_path / 'reader.attempt.json', {'status': 'attempted'})
    monkeypatch.setattr(dispatch.subprocess, 'run', lambda *a, **k: pytest.fail('must not submit'))
    with pytest.raises(RuntimeError, match='unresolved prior'):
        dispatch.submit(tmp_path, 'reader', ['reader.sbatch'], {}, {})
