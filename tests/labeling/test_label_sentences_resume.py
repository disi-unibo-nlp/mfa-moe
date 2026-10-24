import importlib
import json


def items(n):
    from labeling.spans import digest
    result = []
    for i in range(n):
        dataset = 'a' if i % 2 else 'b'
        sha = digest(dict(prompt='question', messages=[], system_prompt=None, cot_text='Sentence.'))
        identity = dict(dataset=dataset, problem_id='same', sample_id=i, index=0,
                        start=0, end=9, trace_sha256=sha)
        unit = dict(index=0, start=0, end=9, text='Sentence.')
        inputs = dict(problem_statement='question', previous_sentence='<START OF RESPONSE>',
                      sentence='Sentence.', next_sentence='<END OF RESPONSE>')
        result.append(dict(identity=identity, unit=unit, inputs=inputs))
    return result


def labels(label, n):
    return [{'label': label}] * n


def test_genuine_cross_trace_batches_and_zero_request_resume(tmp_path):
    module = importlib.import_module('labeling.label_sentences')
    calls = []
    def classify(batch):
        calls.append(len(batch))
        return labels('Analyze', len(batch))
    result = module.run_part(items(104), {'contract': 1}, output_dir=tmp_path / 'part', classify=classify)
    assert calls == [64, 40]
    assert result['status'] == 'complete' and result['completed'] == 104
    calls.clear()
    assert module.run_part(items(104), {'contract': 1}, output_dir=tmp_path / 'part', classify=classify) == result
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
    module = importlib.import_module('labeling.label_sentences')
    calls = []

    def classify(batch):
        calls.append(len(batch))
        return [{'label': 'Analyze'}, UNKNOWN_OUTCOME]

    root = tmp_path / 'mixed'
    result = module.run_part(items(2), {'contract': 2}, output_dir=root, classify=classify)

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

    resumed = module.run_part(items(2), {'contract': 2}, output_dir=root, classify=never_called)
    assert resumed == result


def test_all_valid_run_has_label_only_records_and_no_unknown_count(tmp_path):
    module = importlib.import_module('labeling.label_sentences')
    calls = []

    def classify(batch):
        calls.append(len(batch))
        return labels('Verify', len(batch))

    root = tmp_path / 'clean'
    result = module.run_part(items(3), {'contract': 3}, output_dir=root, classify=classify)

    assert calls == [3]
    records = json.loads((root / 'annotations.json').read_text(encoding='utf-8'))
    assert [record['label'] for record in records] == ['Verify', 'Verify', 'Verify']
    assert all(set(record) == {'identity', 'unit', 'inputs', 'label'} for record in records)

    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    assert summary == result
    assert 'unknown_count' not in result
    assert 'unknown_count' not in summary


def test_checkpoint_key_binds_traces_program_and_decoding_but_not_the_server(tmp_path):
    module = importlib.import_module('labeling.label_sentences')
    first = module.checkpoint_config({'traces': 'plan'}, b'program')
    assert 'base_url' not in first
    assert first == module.checkpoint_config({'traces': 'plan'}, b'program')
    assert module.checkpoint_config({'traces': 'plan'}, b'other program') != first
    assert module.checkpoint_config({'traces': 'other plan'}, b'program') != first
    # The keys and values of the checkpoint key (fixed: they are part of every checkpoint's binding).
    assert {key: first[key] for key in first if key != 'source_plan_sha256' and key != 'program_sha256'} == {
        'judge_model': 'Qwen/Qwen3.8-27B', 'batch_size': 64, 'max_tokens': 16384, 'temperature': 1.0, 'top_p': 0.95,
        'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0, 'reasoning_effort': 'low',
        'enable_thinking': True, 'preserve_thinking': False}

    # A job on another port resumes the checkpoints of the first one without a single request.
    root = tmp_path / 'part'
    result = module.run_part(items(3), first, output_dir=root, classify=lambda batch: labels('Analyze', len(batch)))

    def never_called(batch):
        raise AssertionError('a new port must not trigger a new request')

    assert module.run_part(items(3), first, output_dir=root, classify=never_called) == result


def test_checkpoints_written_with_a_server_address_still_resume(tmp_path):
    module = importlib.import_module('labeling.label_sentences')
    calls = []
    def classify(batch):
        calls.append(len(batch))
        return labels('Analyze', len(batch))
    with_server = {'contract': 1, 'base_url': 'http://127.0.0.1:61003/v1'}
    first = module.run_part(items(104), with_server, output_dir=tmp_path / 'part', classify=classify)
    (tmp_path / 'part/annotations.json').unlink()
    (tmp_path / 'part/checkpoints/batch-000001.json').unlink()
    calls.clear()
    resumed = module.run_part(items(104), {'contract': 1}, output_dir=tmp_path / 'part', classify=classify)
    assert calls == [40]
    assert resumed['annotations_sha256'] == first['annotations_sha256']
