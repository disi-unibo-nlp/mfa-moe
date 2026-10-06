"""Exercise the post-parity measurement path without loading a GPU model."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np


def test_post_parity_prepared_prefix_measurement(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[2]/'scripts/experimental_resume/native_nll.py'
    spec = importlib.util.spec_from_file_location('measurement_driver_under_test', script)
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    fixture = NS(question='q')
    for batch in ('tf1', 'tf8'):
        (tmp_path/f'plugin.{batch}.json').write_text(json.dumps({'fixtures': [{'question': 'q', 'prompt_len': 3}]}))
        for name in ('plugin', 'free_A', 'free_B'):
            np.savez(tmp_path/f'{name}.{batch}.npz', l0=np.array([-1., -2.]))
    monkeypatch.setitem(sys.modules, 'vllm', NS(LLM=lambda **kwargs: object()))
    monkeypatch.setattr(d, 'actual_logprobs', lambda p, lp: np.array([-1., -2.]))
    monkeypatch.setattr(d, 'validate_parity', lambda *args: {'pass': True})
    calls = []
    def measurement(model, prompts, size):
        calls.append(prompts)
        yield from [None for _ in prompts]
    monkeypatch.setattr(d, 'measure', measurement)
    monkeypatch.setattr(d, 'pulse_surprisal', lambda p, lp, **kw:
        {'mean_nll': 1.25, 'n_tokens': kw['n_pulse_tokens']})
    parent = {'prefix_len': 2, 'trace_ref': {'dataset': 'd', 'problem_id': 'q'}}
    requests = [{'uid': 'a', 'question': 'q', 'parent': parent, 'seed_k': 0, 'policy_name': 'N'},
                {'uid': 'b', 'question': 'q', 'parent': parent, 'seed_k': 1, 'policy_name': 'E'}]
    records = {uid: {'completion_token_ids': ids, 'manifest_sha256': 'm', 'code_tree': 'c'}
               for uid, ids in [('a', [5, 6]), ('b', [])]}
    validated, saved = [], {}
    RS = NS(validate_result=lambda rec, req: validated.append(req['uid']))
    M = NS(load_manifest=lambda path: {'sha256': 'm', 'questions': {'q': {'prompt_token_ids': [1]}}},
           TraceStore=lambda **kw: NS(completion_ids=lambda *args: (_ for _ in ()).throw(AssertionError('CPU prepared prefix must be used'))))
    engine = NS(engine_kwargs=lambda **kw: {}, engine_env=lambda *args, **kw: {}, apply_env=lambda env: None)
    store = NS(read=lambda: dict(saved), put=lambda row: saved.update({row['uid']: row}))
    prefixes = {'questions': {'q': {'parent': parent, 'original_prompt_token_ids': [1], 'prefix_token_ids': [1, 2, 3]}}}
    args = NS(parity=tmp_path, out=tmp_path, manifest='unused')
    validation = d.run_measurement(args, engine, M, None, RS, {'code_tree': 'c'}, [fixture],
        [[1, 2, 3]], records, requests, store, tmp_path, prefixes)
    assert all(v['pass'] for v in validation.values())
    assert validated == ['a', 'b']
    assert calls[-1] == [[1, 2, 3, 5, 6]]
    assert saved['a']['mean_nll'] == 1.25
    assert saved['b']['status'] == 'NO_PULSE_TOKENS'
