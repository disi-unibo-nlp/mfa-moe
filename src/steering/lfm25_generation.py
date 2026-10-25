"""The GPU stages of LFM2.5 that run in the steering engine: its native traces, its judge labels and its router replay.

LFM2.5 has no server script, so these stages use the engine of steering.pulsed_steering without hooks (native,
annotate) or the capture backend (replay). They run in this order, between the CPU stages of routing.lfm25_analyses:

  python -m steering.lfm25_generation native --shard I/N     GPU: generate the native population (unhooked engine)
  python -m steering.lfm25_generation annotate --shard I/N   GPU: the judge labels of the sampled sentences
  python -m steering.lfm25_generation replay --shard I/N     GPU: record the routers of the sampled traces (capture.lfm_replay)

Start them as Slurm jobs: sbatch scripts/slurm/run_module_gpu.sbatch steering.lfm25_generation native --shard 0/4.
The study folder is the one store.STUDY names (STUDY_DATA_ROOT, default report/final-routing-20261015-r1).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from generation.client import slurm_deadline
from steering.pulsed_steering import (ADMISSION_MARGIN_HOURS, Engine, Store, annotate_rows, bulk, code_record, log,
                                      parse_shard, plain_llm, segment_seed)
from store import STUDY, binding, load, publish, require, verified


def native_kwargs(b, model, max_num_seqs=128):
    """The model's study engine profile with CUDA graphs; no hooks, so no intervention state exists."""
    from models.engine import engine_kwargs
    from store import check_files
    p = b['models'][model]
    check_files(p['config_files'])
    kwargs = engine_kwargs(p, context=p['card']['max_model_len'])
    kwargs.update(max_num_seqs=max_num_seqs, enable_chunked_prefill=True, max_num_batched_tokens=8192,
                  enforce_eager=False, gpu_memory_utilization=.88)
    if model == 'lfm25':
        kwargs['kv_cache_memory_bytes'] = p['kv_cache_memory_bytes']
    return kwargs


def lfm_native(shard, deadline):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from labeling.lfm_inputs import population
    from steering.runtime import prompt_prefix, remaining
    from store import Units
    b = binding()
    rows = population(b)['records']
    rows = [r for i, r in enumerate(sorted(rows, key=lambda r: r['uid'])) if i % shard[1] == shard[0]]
    p = b['models']['lfm25']
    store = Store(bulk(b)/'lfm25/native-progress')
    units = Units(Path(b['config']['bulk_root'])/'lfm25/native')
    pending = [r for r in rows if units.state(r['uid']) is None]
    log({'event': 'start', 'stage': 'lfm-native', 'rows': len(rows), 'pending': len(pending)})
    if not pending:
        return {'status': 'complete'}
    llm, kwargs, load_seconds = plain_llm(native_kwargs(b, 'lfm25'))
    tokenizer = llm.get_tokenizer()
    sampler = {k: v for k, v in p['sampler'].items() if v is not None}

    def build(row, history, segment):
        prompt, _, _ = prompt_prefix(b, 'lfm25', row, tokenizer)
        total = remaining(p, prompt, [])
        left = total-len(history)
        if left <= 0:
            return None
        # Repetition over prompt|output is the same token set after re-admission; presence/frequency are zero.
        params = SamplingParams(**sampler, seed=segment_seed(row['seed'], segment), max_tokens=left,
                                skip_special_tokens=False, detokenize=False, output_kind=RequestOutputKind.DELTA)
        return {'prompt_ids': prompt+history, 'params': params, 'request': None, 'total': total}

    def finish(row, progress, reason, stop, capture):
        if progress is None:
            progress, reason = load(store.folder(row['uid'])/'PARTIAL.json'), 'length'
        require(units.intent(row, os.environ['SLURM_JOB_ID']), 'native unit claimed twice')
        raw = units.output(row, {'status': 'complete', 'token_ids': progress['tokens'],
            'text': tokenizer.decode(progress['tokens'], skip_special_tokens=False), 'finish_reason': reason,
            'stop_reason': stop, 'segments': progress['segments'], 'cap': progress['total_allowance']})
        units.commit(row, {'status': 'complete', 'raw_sha256': raw['sha256'], 'stochastic_redraw': False})

    require(sampler.get('presence_penalty', 0) == 0 and sampler.get('frequency_penalty', 0) == 0,
            'native resume requires history-free penalties')
    try:
        metrics = Engine(llm, store, deadline, hooked=False, max_inflight=kwargs['max_num_seqs']).run(pending, build, finish)
    finally:
        llm.llm_engine.engine_core.shutdown()
    receipt = {'stage': 'lfm-native', 'shard': list(shard), 'job': os.environ.get('SLURM_JOB_ID'),
               'load_seconds': load_seconds, 'metrics': metrics, 'code': code_record()}
    publish(bulk(b)/'lfm25/jobs'/('native-'+os.environ.get('SLURM_JOB_ID', 'local')+'.json'), receipt)
    log({'event': 'end', **receipt})
    return receipt


def lfm_annotate(shard, deadline):
    """Discovery judge labels of LFM2.5 (the 100,000 sampled sentences)."""
    b = binding()
    rows = verified(STUDY/'lfm25/ANNOTATION_TASKS.json')['records']
    return annotate_rows(rows, shard, deadline, units_root=Path(b['config']['bulk_root'])/'reader/annotations',
                         progress_root=bulk(b)/'reader-progress', jobs_root=bulk(b)/'lfm25/jobs',
                         stage='lfm-annotate')


def lfm_replay(shard, deadline):
    from capture.lfm_replay import replay
    b = binding()
    sources = verified(STUDY/'lfm25/SOURCES.json')['records']
    rows = [s for i, s in enumerate(sources) if i % shard[1] == shard[0]]
    rows = [s for s in rows if not (Path(b['config']['bulk_root'])/'lfm25/replay'/(s['attempt_id']+'.json')).exists()]
    log({'event': 'start', 'stage': 'lfm-replay', 'pending': len(rows)})
    if not rows:
        return {'status': 'complete'}
    done = []
    from steering.runtime import Backend
    backend = Backend(b, 'lfm25', 1, deadline, replay=True)
    try:
        from store import read_source
        for row in rows:
            if time.time()+600 > deadline:
                break
            done.append(backend.capture_trace(read_source(row['source_location']),
                                              Path(b['config']['bulk_root'])/'lfm25/replay'/(row['attempt_id']+'.npz')))
    finally:
        backend.close()
    log({'event': 'end', 'stage': 'lfm-replay', 'captured': len(done), 'remaining': len(rows)-len(done)})
    return {'captured': len(done)}


STAGES = {'native': lfm_native, 'annotate': lfm_annotate, 'replay': lfm_replay}


def main(argv=None):
    ap = argparse.ArgumentParser(prog='steering.lfm25_generation', description=__doc__.split('\n\n')[0])
    ap.add_argument('stage', choices=sorted(STAGES))
    ap.add_argument('--shard', default='0/1', type=parse_shard, help='I/N: this job takes every N-th row, starting at I')
    args = ap.parse_args(argv)
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'), 'GPU commands require a Slurm GPU allocation')
    STAGES[args.stage](args.shard, slurm_deadline(ADMISSION_MARGIN_HOURS))


if __name__ == '__main__':
    main()
