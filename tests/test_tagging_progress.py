"""Progress must count durable labels and exclude reused work from throughput."""
import json
import logging

from moe_exp.correlation_pipeline import annotate
from moe_exp.schemas import TraceRecord


def test_progress_resume_and_limit(tmp_path, caplog):
    source = tmp_path / 'generation/test/math500'
    source.mkdir(parents=True)
    rows = [TraceRecord(dataset='math500', problem_id=f'p{i}', prompt='Question',
                        model_id='test', gold_answer='9', model_answer='9', cot_text='First. Second. Third.') for i in range(2)]
    (source / 'traces.jsonl').write_text(''.join(r.model_dump_json() + '\n' for r in rows))
    program = tmp_path / 'program.json'
    program.write_text(json.dumps({'classify': {'signature': {'instructions': 'Frozen'}}}))
    args = annotate.build_parser().parse_args([
        '--generation-dir', str(tmp_path / 'generation'), '--generation-model', 'test',
        '--output-dir', str(tmp_path / 'output'), '--datasets', 'math500',
        '--judge-program', str(program), '--judge-model', 'test', '--limit', '1',
    ])
    with caplog.at_level(logging.INFO, logger=annotate.__name__):
        annotate.annotate_all(args, predict=lambda **kw: 'Plan')
    assert '3/3 sentences (100.0%) | 0 remaining | 0 reused' in caplog.text
    assert 'traces 1/1' in caplog.text
    shard = next((tmp_path / 'output').rglob('shards/*.json'))
    saved = json.loads(shard.read_text())
    saved['units'] = saved['units'][::2]  # Sparse partial checkpoint.
    saved['status'] = 'partial'
    shard.write_text(json.dumps(saved))
    caplog.clear()
    calls = []
    with caplog.at_level(logging.INFO, logger=annotate.__name__):
        annotate.annotate_all(args, predict=lambda **kw: calls.append(kw) or 'Plan')
    assert len(calls) == 1
    assert '2/3 sentences (66.7%) | 1 remaining | 2 reused' in caplog.text
    assert '3/3 sentences (100.0%) | 0 remaining | 2 reused' in caplog.text


def test_progress_eta_excludes_reused_and_throttles(monkeypatch, caplog):
    now = [0.0]
    monkeypatch.setattr(annotate.time, 'monotonic', lambda: now[0])
    with caplog.at_level(logging.INFO, logger=annotate.__name__):
        progress = annotate.TaggingProgress(100, 50, 2)
        assert 'ETA estimating' in caplog.text
        caplog.clear()
        now[0] = 5
        progress.advance(5)
        assert not caplog.text
        now[0] = 10
        progress.advance(5)
        assert '40 remaining | 50 reused | 1.00 sentences/s | ETA 0.7 min' in caplog.text
