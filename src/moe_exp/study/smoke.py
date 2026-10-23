"""GPT engineering gate for the streaming driver (non-primary low/high-count rows only).

Checks: penalty-operator parity; hooked no-op and zero-pulse token equality; dispatched
inclusion/exclusion through the pulse; resume parity with and without pulse
re-application; continuous admission with mixed lengths on both ranks; drain and
resume through durable progress records; long-prefix admission; throughput.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from moe_exp.io import digest, publish, require
from .driver import (DRIVER, Engine, ROOT, Store, binding, bulk, contract_for, hooked_llm,
                     intervention_builder, log, code_hashes)


def penalty_parity():
    import torch
    from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality
    from vllm import SamplingParams
    from moe_exp.study.penalties import processor_class as original_class
    from .fast_penalties import processor_class as fast_class
    device = 'cuda'
    torch.manual_seed(0)
    vocab = 1000
    results = []
    for presence, repetition in ((1.5, 1.0), (0., 1.05), (1.5, 1.05), (0., 1.0)):
        a, f = original_class()(None, device, False), fast_class()(None, device, False)
        outputs = [[], [], []]
        rows = []
        for i in range(3):
            prefix = torch.randint(0, vocab, (50+i,)).tolist()
            prompt = torch.randint(0, vocab, (80,)).tolist()+prefix
            params = SamplingParams(seed=i, extra_args={'research_prefix_penalties': {
                'prefix': prefix, 'presence': presence, 'frequency': 0., 'repetition': repetition}})
            rows.append((i, params, prompt, outputs[i]))
        update = BatchUpdate(batch_size=3, removed=[], added=rows, moved=[])
        a.update_state(update)
        f.update_state(update)
        equal = True
        for step in range(6):
            logits = torch.randn(3, vocab, device=device)
            x, y = a.apply(logits.clone()), f.apply(logits.clone())
            equal &= bool(torch.equal(x, y))
            for out in outputs:
                out.append(int(torch.randint(0, vocab, ()).item()))
            if step == 2:
                swap = BatchUpdate(batch_size=3, removed=[], added=[], moved=[(0, 2, MoveDirectionality.SWAP)])
                a.update_state(swap)
                f.update_state(swap)
                outputs[0], outputs[2] = outputs[2], outputs[0]
            else:
                a.update_state(None)
                f.update_state(None)
        results.append({'presence': presence, 'repetition': repetition, 'exact': equal})
    return {'cases': results, 'passed': all(r['exact'] for r in results)}


def smoke_rows(b):
    from moe_exp.io import load
    rows = load(ROOT/'gpt/ASSIGNMENTS-full.json')['records']
    usable = [r for r in rows if r['dose'] in ('low', 'high') and r['source'] and r['action']
              and r['prefix_status'] == 'enrolled']
    usable.sort(key=lambda r: (r['source']['prefix_tokens'], r['uid']))

    def variant(r, name, **change):
        v = {**r, **change}
        v['capture_layers'] = [(v['action'] or r['action'])['layer']]
        v['uid'] = digest(['smoke-20261017', r['uid'], name])
        return v
    short = [r for r in usable if 200 <= r['source']['prefix_tokens'] <= 2000]
    high = next(r for r in short if r['dose'] == 'high' and r['arm'] == 'target_activate')
    deact = next(r for r in short if r['dose'] == 'high' and r['arm'] == 'target_deactivate'
                 and r['family'] != high['family'])
    longest = usable[-1]
    return usable, short, high, deact, longest, variant


def run_alone(engine_factory, rows, build, store):
    results = {}

    def finish(row, progress, reason, stop, capture):
        results[row['uid']] = {'tokens': list(progress['tokens']) if progress else None, 'reason': reason,
                               'capture': capture, 'logprobs': progress.get('logprobs', []) if progress else []}
    engine = engine_factory(store)
    for row in rows:
        engine.run([row], build, finish)
    return results


def smoke(deadline):
    b = binding()
    report = {'job': os.environ.get('SLURM_JOB_ID'), 'started': time.time(), 'checks': {}}
    checks = report['checks']
    root = bulk(b)/'smoke'/os.environ['SLURM_JOB_ID']
    out = DRIVER/'smoke'/(os.environ['SLURM_JOB_ID']+'.json')

    def save():
        report['finished'] = time.time()
        report['passed'] = all(c.get('passed', False) for c in checks.values())
        from .driver import checkpoint
        checkpoint(out, report)
        log({'event': 'smoke-progress', 'checks': {k: v.get('passed') for k, v in checks.items()}})

    try:
        checks['penalty_parity'] = penalty_parity()
        save()
        usable, short, high, deact, longest, variant = smoke_rows(b)
        native = variant(high, 'native', action=None, arm='native')
        native['capture_layers'] = [high['action']['layer']]
        zero = variant(high, 'zero-pulse', action={**high['action'], 'pulse_tokens': 0})
        nocap = variant(high, 'native-nocapture', action=None, arm='native')
        nocap['capture_layers'] = []
        act = variant(high, 'activate')
        dea = variant(deact, 'deactivate')
        rows = [native, zero, nocap, act, dea]
        mixed = [variant(r, 'mixed-'+str(i)) for i, r in enumerate(short[10:22])]
        drain_rows = [variant(r, 'drain-'+str(i)) for i, r in enumerate(short[40:46])]
        long_row = variant(longest, 'long')
        spread = usable[::max(1, len(usable)//160)][:160]
        throughput = [variant(r, 'throughput-'+str(i)) for i, r in enumerate(spread)]
        everything = rows+mixed+drain_rows+[long_row]+throughput
        llm, kwargs, load_seconds = hooked_llm(b, 'gpt', 64)
        report['load_seconds'] = load_seconds
        reports = llm.collective_rpc('fs_begin', kwargs={'contract': contract_for(b, 'gpt', everything)})
        require(len(reports) == 2, 'two ranks required')
        from .runtime import closure
        tok = llm.get_tokenizer()
        close = closure(tok, 'gpt')
        pulse = high['action']['pulse_tokens']
        K = pulse+48

        def factory(store, **kw):
            return Engine(llm, store, deadline, hooked=True, max_inflight=kw.get('max_inflight', 1),
                          drain_seconds=kw.get('drain_seconds', 0))

        # 1. no-op fidelity, zero pulse, dispatch contract (greedy, one request at a time)
        greedy = intervention_builder(b, 'gpt', tok, close, greedy=True, limit=K, logprobs=5)
        res = run_alone(factory, rows, greedy, Store(root/'fidelity'))
        n, z, c, a, d = (res[r['uid']] for r in rows)
        act_stats, dea_stats = a['capture']['stats'], d['capture']['stats']
        cap = a['capture']['capture'][str(high['action']['layer'])]
        targets = set(high['action']['experts'])
        dispatched = [set(ids) for ids, on in zip(cap['executed_ids'], cap['active']) if on]
        checks['noop_fidelity'] = {
            'native_equals_zero_pulse': n['tokens'] == z['tokens'],
            'native_equals_uncaptured': n['tokens'] == c['tokens'],
            'zero_pulse_inactive': not z['capture']['stats']['active_positions'],
            'activate_positions': act_stats['active_positions'], 'pulse': pulse,
            'activate_changed': len(act_stats['changed_positions']),
            'activate_contract': act_stats['target_contract'] and all(targets <= s for s in dispatched),
            'deactivate_contract': dea_stats['target_contract'] and bool(dea_stats['active_positions']),
            'noop_exact': all(x['capture']['stats']['noop_exact'] for x in (a, d)),
            'activation_differs_from_native': a['tokens'] != n['tokens']}
        f = checks['noop_fidelity']
        f['passed'] = bool(f['native_equals_zero_pulse'] and f['native_equals_uncaptured'] and f['zero_pulse_inactive']
                           and f['activate_positions'] and f['activate_contract'] and f['deactivate_contract']
                           and f['noop_exact'] and f['activate_changed'] > 0)
        save()

        # 2. resume parity: uninterrupted vs re-admitted at k (pulse re-applied) vs native replay of the pulse
        uninterrupted, reference = a['tokens'], a['logprobs']
        parity = {}
        for name, k, action in (('inside_pulse', pulse//2, high['action']), ('after_pulse', pulse+8, high['action']),
                                ('after_pulse_native_replay', pulse+8, None)):
            row = variant(high, 'resume-'+name, action=action, arm=high['arm'] if action else 'native')
            row['capture_layers'] = [high['action']['layer']]
            store = Store(root/'resume'/name)
            from .driver import checkpoint as write
            write(store.folder(row['uid'])/'PARTIAL.json', {'uid': row['uid'], 'assignment_sha256': digest(row),
                  'tokens': uninterrupted[:k], 'segments': [{'job': 'synthetic', 'history': 0}], 'total_allowance': K})
            got = run_alone(factory, [row], greedy, store)[row['uid']]
            tokens = got['tokens']
            first = next((i for i, (x, y) in enumerate(zip(tokens, uninterrupted)) if x != y), None)
            deltas = []
            for j, lp in enumerate(got['logprobs']):
                position = k+j
                if position >= len(reference) or (first is not None and position >= first):
                    break
                token = str(uninterrupted[position])
                if token in lp and token in reference[position]:
                    deltas.append(abs(lp[token]-reference[position][token]))
            parity[name] = {'resume_at': k, 'equal': tokens == uninterrupted, 'first_divergence': first,
                            'max_abs_logprob_delta': max(deltas) if deltas else None,
                            'boundary_logprob_delta': deltas[0] if deltas else None,
                            'passes': got['capture']['passes'] if got['capture'] else None}
        ok = lambda p: p['first_divergence'] is None or p['first_divergence'] >= p['resume_at']+8
        parity['passed'] = bool(ok(parity['inside_pulse']) and ok(parity['after_pulse']))
        checks['resume_parity'] = parity
        save()

        # 3. continuous admission, mixed lengths, attribution on both ranks
        sampled = intervention_builder(b, 'gpt', tok, close)
        finished = {}

        def keep(row, progress, reason, stop, capture):
            finished[row['uid']] = {'n': len(progress['tokens']), 'reason': reason, 'capture': capture}
        limits = {}
        for i, r in enumerate(mixed):
            limits[r['uid']] = 24+40*i

        def bounded(row, history, segment):
            spec = sampled(row, history, segment)
            if spec is None:
                return None
            spec['params'].max_tokens = min(spec['params'].max_tokens, limits[row['uid']]-len(history))
            return spec
        Engine(llm, Store(root/'mixed'), deadline, hooked=True, max_inflight=4, drain_seconds=0).run(mixed, bounded, keep)
        checks['continuous_admission'] = {
            'finished': len(finished), 'expected': len(mixed),
            'lengths_match': all(finished[r['uid']]['n'] <= limits[r['uid']] for r in mixed if r['uid'] in finished),
            'captures': sum(1 for v in finished.values() if v['capture'] and v['capture']['capture'])}
        checks['continuous_admission']['passed'] = bool(len(finished) == len(mixed) and
                                                        checks['continuous_admission']['lengths_match'])
        save()

        # 4. drain mid-flight, then resume through durable PARTIAL records (production finisher)
        from .driver import intervention_finisher
        store = Store(root/'drain')
        finisher = intervention_finisher(store, b, 'gpt')
        Engine(llm, store, time.time()+60, hooked=True, max_inflight=8, drain_seconds=0).run(
            drain_rows, intervention_builder(b, 'gpt', tok, close, limit=8000), finisher)
        first = {r['uid']: store.state(r['uid']) for r in drain_rows}
        partial = {u: len(s['tokens']) for u, s in first.items() if s and s['status'] == 'partial'}
        Engine(llm, store, min(deadline, time.time()+900), hooked=True, max_inflight=8, drain_seconds=0).run(
            drain_rows, intervention_builder(b, 'gpt', tok, close, limit=8000), finisher)
        final = {r['uid']: store.state(r['uid']) for r in drain_rows}
        resumed = [final[u] for u in partial if final[u] and final[u]['status'] != 'partial']
        checks['drain_resume'] = {'drained': len(partial), 'terminal': sum(1 for s in final.values() if s and s['status'] != 'partial'),
                                  'statuses': sorted({s['status'] for s in final.values() if s}),
                                  'resumed_segments': [s['segments'] for s in resumed],
                                  'monotone': all(final[u]['generated_tokens'] >= partial[u] for u in partial
                                                  if final[u] and final[u]['status'] != 'partial')}
        checks['drain_resume']['passed'] = bool(partial and checks['drain_resume']['terminal'] == len(drain_rows) and
                                                checks['drain_resume']['statuses'] == ['complete'] and
                                                checks['drain_resume']['monotone'] and
                                                all(n >= 2 for n in checks['drain_resume']['resumed_segments']))
        save()

        # 5. long prefix admission
        got = run_alone(factory, [long_row], intervention_builder(b, 'gpt', tok, close, limit=64),
                        Store(root/'long'))[long_row['uid']]
        checks['long_prefix'] = {'prefix_tokens': long_row['source']['prefix_tokens'], 'generated': len(got['tokens']),
                                 'passed': bool(got['tokens'] and got['capture'])}
        save()

        # 6. throughput at production concurrency, full allowances, then drain
        budget = max(300., min(1200., deadline-time.time()-600))
        e3 = Engine(llm, Store(root/'throughput'), min(deadline-300, time.time()+budget), hooked=True, max_inflight=64, drain_seconds=0)
        metrics = e3.run(throughput, sampled, lambda *a: None)
        rate = metrics['generated_tokens']/max(1., metrics['wall_seconds'])
        checks['throughput'] = {**metrics, 'tokens_per_second': rate, 'passed': rate > 0}
        report['native_validation'] = llm.collective_rpc('fs_validation')[0]
        report['memory'] = llm.collective_rpc('fr_metrics')
        llm.collective_rpc('fs_end')
        llm.llm_engine.engine_core.shutdown()
    except BaseException as error:
        import traceback
        report['error'] = traceback.format_exc()
        save()
        raise
    report['code'] = code_hashes()
    save()
    return report
