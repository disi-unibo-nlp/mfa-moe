import importlib
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace


def items(n):
    from moe_exp.correlation_pipeline.annotation_batch import digest
    result = []
    for i in range(n):
        trace = SimpleNamespace(dataset='a' if i % 2 else 'b', problem_id='same', sample_id=i,
                                prompt='question', generation_messages=[], system_prompt=None,
                                cot_text='Sentence.', metadata={})
        sha = digest(dict(prompt=trace.prompt, messages=[], system_prompt=None, cot_text=trace.cot_text))
        identity = dict(dataset=trace.dataset, problem_id='same', sample_id=i, index=0,
                        start=0, end=9, trace_sha256=sha)
        unit = dict(index=0, start=0, end=9, text='Sentence.')
        inputs = dict(problem_statement='question', previous_sentence='<START OF RESPONSE>',
                      sentence='Sentence.', next_sentence='<END OF RESPONSE>')
        result.append((identity, trace, unit, inputs))
    return result


def test_genuine_cross_trace_batches_and_zero_request_resume(tmp_path):
    module = importlib.import_module('moe_exp.correlation_pipeline.annotation_batch')
    calls = []
    def classify(batch):
        calls.append(len(batch))
        return ['Analyze'] * len(batch)
    result = module.run_part(items(104), {'contract': 1}, output_dir=tmp_path / 'part', classify=classify, expected_count=104)
    assert calls == [64, 40]
    assert result['status'] == 'complete' and result['completed'] == 104
    calls.clear()
    assert module.run_part(items(104), {'contract': 1}, output_dir=tmp_path / 'part', classify=classify, expected_count=104) == result
    assert calls == []



UNKNOWN_OUTCOME = {
    'status': 'unknown',
    'raw_completion': 'truncated raw text',
    'failure': {
        'kind': 'non_stop',
        'error_type': 'ValueError',
        'message': "finish_reason='length'",
        'finish_reason': 'length',
    },
}


def test_unknown_outcome_is_persisted_and_resumes_without_requests(tmp_path):
    module = importlib.import_module('moe_exp.correlation_pipeline.annotation_batch')
    calls = []

    def classify(batch):
        calls.append(len(batch))
        return ['Analyze', UNKNOWN_OUTCOME]

    root = tmp_path / 'mixed'
    result = module.run_part(items(2), {'contract': 2}, output_dir=root, classify=classify, expected_count=2)

    assert calls == [2]
    assert result['status'] == 'complete'
    assert result['completed'] == 2
    assert result['unknown_count'] == 1

    records = json.loads((root / 'annotations.json').read_text(encoding='utf-8'))
    assert len(records) == 2
    assert set(records[0]) == {'identity', 'unit', 'inputs', 'label'}
    assert records[0]['label'] == 'Analyze'
    assert records[1]['status'] == 'unknown'
    assert records[1]['raw_completion'] == 'truncated raw text'
    assert records[1]['failure'] == UNKNOWN_OUTCOME['failure']
    assert set(records[1]) == {'identity', 'unit', 'inputs', 'status', 'raw_completion', 'failure'}

    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    assert summary == result
    assert summary['unknown_count'] == 1

    def never_called(batch):
        raise AssertionError(f'classifier must not run on resume, got {batch!r}')

    resumed = module.run_part(items(2), {'contract': 2}, output_dir=root, classify=never_called, expected_count=2)
    assert resumed == result


def test_all_valid_run_keeps_the_legacy_record_and_summary_shape(tmp_path):
    module = importlib.import_module('moe_exp.correlation_pipeline.annotation_batch')
    calls = []

    def classify(batch):
        calls.append(len(batch))
        return ['Verify'] * len(batch)

    root = tmp_path / 'clean'
    result = module.run_part(items(3), {'contract': 3}, output_dir=root, classify=classify, expected_count=3)

    assert calls == [3]
    records = json.loads((root / 'annotations.json').read_text(encoding='utf-8'))
    assert [record['label'] for record in records] == ['Verify', 'Verify', 'Verify']
    assert all(set(record) == {'identity', 'unit', 'inputs', 'label'} for record in records)

    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    assert summary == result
    assert 'unknown_count' not in result
    assert 'unknown_count' not in summary


if __name__ == '__main__':
    root = Path('/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/gpt-oss-20b-qwen38-b64-mtp3-100k-all-attempts-v2/runtime')
    root.mkdir(parents=True, exist_ok=True)
    suite = unittest.TestSuite()
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            def run(fn=fn):
                with tempfile.TemporaryDirectory(dir=root) as directory:
                    fn(Path(directory))
            suite.addTest(unittest.FunctionTestCase(run, description=name))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not outcome.wasSuccessful())
