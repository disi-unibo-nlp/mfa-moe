"""Discovery-only native fidelity, request-isolation and measured routing prices."""
from pathlib import Path
import time
from .common import CLASSES, Units, digest, load, publish, require, control_root, engineering_namespace
from moe_exp.final_routing.design import seed_for, strengths


def fixtures(binding, model):
    from moe_exp.research_campaign_v3_r6.inputs import read_source
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.schemas import TraceRecord
    root, source = Path(binding['root']), Path(binding['source_campaign'])
    target = root/model/'QUALIFICATION_FIXTURES.json'
    if target.exists():
        return load(target)['records']
    source_model = 'gpt' if model == 'lfm25' else model
    sources = load(source/source_model/'SOURCES.json')['records']
    allocation = load(root/'ALLOCATION.json')['records']
    excluded = {r['family'] for r in allocation}
    candidates = [r for r in sources if r['family'] not in excluded]
    candidates.sort(key=lambda r: (r['completion_tokens'], r['attempt_id']))
    chosen = []
    from .common import BENCHMARKS
    for benchmark in BENCHMARKS:
        part=[r for r in candidates if r['benchmark']==benchmark]
        require(part,'pricing benchmark lacks a non-primary discovery family')
        chosen.append(min(part,key=lambda r:digest([binding['config']['seed'],'pricing',model,r['attempt_id']])))
    remaining=[r for r in candidates if r['family'] not in {s['family'] for s in chosen}]
    for part in (remaining[:len(remaining)//2],remaining[len(remaining)//2:]):
        chosen.append(min(part,key=lambda r:digest([binding['config']['seed'],'pricing-length',model,r['attempt_id']])))
    require(len(chosen) == 8, 'pricing requires eight non-causal discovery families')
    rows = []
    folds = load(source/'FOLDS.json')['folds']
    for record in chosen:
        trace = read_source(record['source_location'])
        if model == 'lfm25':
            example = {k: trace.get(k) for k in ('dataset', 'problem_id', 'source_problem_id', 'sample_id',
                       'prompt', 'system_prompt', 'generation_messages', 'gold_answer')}
            example['generation_messages'] = trace.get('generation_messages') or trace.get('metadata', {}).get('generation_messages')
            rows.append({'model': model, 'native_example': example, 'source': None,
                         'benchmark': record['benchmark'], 'family': record['family'], 'fold': folds[record['family']]})
        else:
            layout = saved_layout(TraceRecord(**trace))
            available = [i for i, v in enumerate(layout['unit_tokens']) if v and max(v)+1 < record['completion_tokens']]
            require(available, 'pricing source lacks a complete reasoning prefix')
            i = available[len(available)//3]
            rows.append({'model': model, 'benchmark': record['benchmark'], 'family': record['family'],
                         'fold': folds[record['family']], 'source': {'attempt_id': record['attempt_id'],
                          'question': record['question'], 'prefix_tokens': max(layout['unit_tokens'][i])+1,
                          'sentence_index': i}})
    publish(target, {'records': rows, 'role': 'engineering-and-pricing-only',
                     'primary_families_excluded': True, 'effects_not_used_for_selection': True})
    return rows


def probe_rows(binding, model, *, capacity, full_answers=False):
    profile = binding['models'][model]
    base = fixtures(binding, model)
    path = Path(binding['root'])/model/'COUNT_NOMINATIONS.json'
    nominations = load(path)['records'] if path.exists() else []
    rows = []
    for i, fixture in enumerate(base):
        depth = i % 3
        candidates = [n for n in nominations if n['fold'] == fixture['fold'] and
                      n['depth_third'] == depth and n['status'] == 'nominated']
        n = max(candidates, key=lambda n: (n['association'], n['class'])) if candidates else None
        if model != 'lfm25':
            require(n is not None, 'no discovery-supported pricing intervention')
        layers = profile['layers']
        # LFM hardware checks precede its semantic discovery; they are not paper targets.
        layer = n['layer'] if n else layers[len(layers)*(2*depth+1)//6]
        pulse, horizon = (n['pulse_tokens'], n['horizon']) if n else (16, 512)
        variants = [(arm, 1) for arm in ('native', 'target_activate', 'target_deactivate', 'random_activate', 'random_deactivate')] if full_answers else [('native', 1)]+[(arm, d) for d in range(3) for arm in ('target_activate', 'target_deactivate')]
        for arm, dose_index in variants:
            action = None
            if arm != 'native':
                ids = (n['control_sets' if arm.startswith('random') else 'expert_sets'][dose_index] if n else
                       list(range(4, 4+strengths(profile['native_k'])[dose_index])) if arm.startswith('random') else
                       list(range(strengths(profile['native_k'])[dose_index])))
                action = {'layer': layer, 'experts': ids, 'direction': 'activate' if arm.endswith('_activate') else 'deactivate',
                          'start': 0, 'pulse_tokens': pulse}
            r = {**fixture, 'action': action, 'arm': arm, 'horizon': horizon, 'pulse_tokens': pulse,
                 'seed': seed_for(binding['config']['seed'], 'engineering', model, fixture['family']),
                 'profile_sha256': profile['profile_sha256'], 'capacity': capacity, 'dose_index': dose_index,
                 'engineering_role': 'pricing' if full_answers else 'batch-qualification', 'capture_layers': [layer],
                 'engineering_source_version': binding['source_commit']}
            r['uid'] = digest(r)
            rows.append(r)
    return rows


def qualify(binding, model, deadline, attempt=0):
    from .runtime import Backend, recover_outputs, prompt_prefix
    config = binding['config']
    root, bulk = Path(binding['root']), Path(config['bulk_root'])
    target = root/model/'QUALIFIED.json'
    if target.exists():
        return load(target)
    if model == 'lfm25':
        from .common import verified
        require(verified(root/'LFM_CHECKPOINT_VERIFIED.json')['status'] == 'complete', 'LFM validation absent')
    measured, profiles = [], []
    for capacity in (1, 2, 4, 8):
        path = control_root(binding)/model/('BATCH-'+str(capacity)+'.json')
        if path.exists():
            profile = load(path)
            profiles.append(profile)
            if profile['status'] != 'complete':
                break
            continue
        require(time.time()+600 < deadline, 'insufficient remaining qualification/load reserve')
        backend = None
        try:
            backend = Backend(binding, model, capacity, deadline)
            rows = probe_rows(binding, model, capacity=capacity)
            units = Units(bulk/model/engineering_namespace(binding, 'qualification'))
            recover_outputs(units, rows)
            outputs = []
            # Interleave different prefixes/actions and reverse batches to exercise
            # UID attribution independently of fixture and submission order.
            rows = sorted(rows, key=lambda r: digest(['engineering-order', r['uid']]))
            for offset in range(0, len(rows), capacity):
                batch=rows[offset:offset+capacity]
                outputs += backend.run(batch, units, engineering_tokens=max(64,max(r['pulse_tokens'] for r in batch)+16))
            # Max-context prefill at candidate concurrency; no semantic result is used.
            stress = []
            for j, base in enumerate(rows[:capacity]):
                prompt, prefix, _ = prompt_prefix(binding, model, base, backend.tokenizer)
                total = binding['models'][model]['card']['max_model_len']-16
                cycle = prompt+prefix
                expanded = (cycle*((total+len(cycle)-1)//len(cycle)))[:total]
                r = {**base, 'engineering_input': {'prompt': expanded[:-1], 'prefix': expanded[-1:]},
                     'engineering_role': 'full-native-context-concurrency', 'horizon': 16}
                r['uid'] = digest({k: v for k, v in r.items() if k != 'uid'})
                stress.append(r)
            backend.run(stress, units, engineering_tokens=16)
            require(all((units.state(r['uid']) or {}).get('status') == 'complete' for r in rows+stress),
                    'qualification has ambiguous/unfinished assigned fixture')
            # Read committed timings as well as fresh returns on a bounded recovery.
            raw = [load(units.folder(r['uid'])/'OUTPUT.json') for r in rows]
            captures = [load(units.folder(r['uid'])/'CAPTURE.json') for r in rows]
            require(all(c['stats']['noop_exact'] and c['stats']['target_contract'] and
                        c['stats']['finite_weights'] and c['native_validation'] for c in captures),
                    'native fidelity, target engagement or finite-output check failed')
            boundaries={}
            for row, capture in zip(rows, captures):
                for record in capture['capture'].values():
                    require(all(not flag or position < row['pulse_tokens']
                                for position, flag in zip(record['positions'], record['active'])),
                            'intervention escaped pulse boundary')
                    if row.get('action'):
                        key=(len(row['action']['experts']),row['action']['direction'])
                        evidence=boundaries.setdefault(key,{'start':False,'end':False})
                        evidence['start']|=any(p==0 and flag for p,flag in zip(record['positions'],record['active']))
                        evidence['end']|=any(p==row['pulse_tokens'] and not flag for p,flag in zip(record['positions'],record['active']))
            require(len(boundaries)==6 and all(r['start'] and r['end'] for r in boundaries.values()),
                    'not every strength and direction has actual pulse-start/end coverage')
            abort = backend.abort_probe(rows[0])
            seconds = sum(max(v['seconds'] for v in raw[o:o+capacity]) for o in range(0, len(raw), capacity))
            profile = publish(path, {'status': 'complete', 'capacity': capacity,
                'seconds_per_token': seconds/max(1, sum(len(v['token_ids']) for v in raw)),
                'load_seconds': backend.load_seconds, 'full_native_context': True,
                'request_isolation': True, 'native_noop_fidelity': True, 'abort_probe': abort,
                'mixed_prefixes_and_chunked_prefill': True, 'fixture_uids': [r['uid'] for r in rows+stress],
                'memory':backend.metrics(),
                'pulse_boundary_evidence':[{'count':k[0],'direction':k[1],**v} for k,v in boundaries.items()],
                'full_context_prefill_seconds_upper':max(load(units.folder(r['uid'])/'OUTPUT.json')['seconds'] for r in stress)})
            profiles.append(profile)
        except Exception as exc:
            failure_path = path if capacity > 1 else control_root(binding)/model/('BATCH-1-FAILURE-'+str(attempt)+'.json')
            profile = publish(failure_path, {'status': 'unqualified', 'capacity': capacity,
                                     'error_type': type(exc).__name__, 'error': str(exc)[:2000]})
            profiles.append(profile)
            if capacity == 1:
                raise
            break
        finally:
            if backend is not None:
                backend.close()
    passed = [p for p in profiles if p['status'] == 'complete']
    require(passed and passed[0]['capacity'] == 1, 'batch-one reference did not qualify')
    selected = min(passed, key=lambda p: (p['seconds_per_token'], p['capacity']))
    backend = Backend(binding, model, selected['capacity'], deadline)
    try:
        rows = probe_rows(binding, model, capacity=selected['capacity'], full_answers=True)
        publish(root/model/'PRICING_ASSIGNMENTS.json', {'records': rows, 'primary_outcomes_used': False})
        units = Units(bulk/model/engineering_namespace(binding, 'pricing'))
        recover_outputs(units, rows)
        for offset in range(0, len(rows), selected['capacity']):
            if all(units.state(r['uid']) is not None for r in rows[offset:offset+selected['capacity']]):
                continue
            require(time.time()+max(600, 2*selected['load_seconds']) < deadline, 'pricing batch reserve exhausted')
            backend.run(rows[offset:offset+selected['capacity']], units)
        raw = [load(units.folder(r['uid'])/'OUTPUT.json') for r in rows]
        require(all((units.state(r['uid']) or {}).get('status') == 'complete' for r in rows), 'pricing lacks terminal draws')
        batches = [raw[o:o+selected['capacity']] for o in range(0, len(raw), selected['capacity'])]
        max_per_token = max(max(v['seconds'] for v in batch)/max(1, sum(len(v['token_ids']) for v in batch)) for batch in batches)
        output_bytes = sum(p.stat().st_size for r in rows for p in units.folder(r['uid']).iterdir() if p.is_file())
        from .contexts import build
        contexts = build(binding, model, rows, namespace=engineering_namespace(binding, 'pricing'))
        result = {'status': 'complete', 'source_commit': binding['source_commit'], 'capacity': selected['capacity'],
            'batch_profiles': profiles, 'pricing_assignments': len(rows),
            'seconds_per_token_upper': max_per_token, 'load_seconds': backend.load_seconds,
            'context_records': [{'uid': r['uid'], 'contexts': len(r['sentences']), 'family': r['family'],
                                  'benchmark': r['benchmark'], 'fold': r['fold']} for r in contexts['records']],
            'native_context': binding['models'][model]['card']['max_model_len'],
            'maximum_batch_seconds': max(v['seconds'] for v in raw),
            'maximum_execution_seconds': max(v['seconds'] for v in raw),
            'maximum_generated_tokens':max(len(v['token_ids']) for v in raw),
            'full_context_prefill_seconds_upper':selected['full_context_prefill_seconds_upper'],
            'capture_bytes_per_position_upper':max((units.folder(r['uid'])/'CAPTURE.json').stat().st_size /
                max(1,max((len(v['positions']) for v in load(units.folder(r['uid'])/'CAPTURE.json')['capture'].values()),default=0)) for r in rows),
            'output_bytes': output_bytes, 'output_bytes_per_token': output_bytes/max(1, sum(len(v['token_ids']) for v in raw)),
            'pricing_output_tokens': sum(len(v['token_ids']) for v in raw), 'memory':backend.metrics(),
            'profile_sha256':binding['models'][model]['profile_sha256'],
            'remaining_allowance_preserved': True, 'selection_frozen_before_primary_outcomes': True}
    finally:
        backend.close()
    if model == 'lfm25':
        from .lfm import trace_record
        replay_backend = Backend(binding, model, 1, deadline, replay=True)
        try:
            observations = []
            native_lengths=[]
            for row in rows:
                if row['arm'] != 'native':
                    continue
                raw = load(units.folder(row['uid'])/'OUTPUT.json')
                trace = trace_record(binding,row,raw,replay_backend.tokenizer)
                from moe_exp.schemas import TraceRecord
                from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
                native_lengths.extend(len(p) for p in saved_layout(TraceRecord(**trace))['unit_tokens'] if p)
                observations.append(replay_backend.capture_trace(trace,bulk/model/'replay-pricing'/(row['uid']+'.npz')))
            # Exercise the complete replay profile at the native context cap,
            # including all 38 layers and actual host capture memory.
            prompt,_,_=prompt_prefix(binding,model,rows[0],replay_backend.tokenizer)
            cap=binding['models'][model]['card']['max_model_len']-len(prompt)-1
            cycle=replay_backend.tokenizer.encode('We compute the result carefully. ',add_special_tokens=False)
            expanded=(cycle*((cap+len(cycle)-1)//len(cycle)))[:cap]
            synthetic={'metadata':{'token_replay':{'prompt_token_ids':prompt,'completion_token_ids':expanded}},
                       'engineering_only':True}
            full=replay_backend.capture_trace(synthetic,bulk/model/'replay-pricing'/'full-native-context.npz')
            observations.append(full)
            result['replay_price'] = {'seconds_per_input_token_upper':max(r['seconds']/r['input_tokens'] for r in observations),
                'bytes_per_completion_token_upper':max(r['bytes']/max(1,r['completion_tokens']) for r in observations),
                'load_seconds':replay_backend.load_seconds,'observations':observations,'memory':replay_backend.metrics(),
                'profile':'teacher-forced-native-input-token','full_native_context_receipt':full['sha256']}
            import numpy as np
            require(native_lengths,'LFM native pricing lacks sentence lengths')
            result['discovery_forecast_horizon']=16*max(1,round(float(np.quantile(native_lengths,.75))))
            result['discovery_forecast_role']='provisional pricing-family length envelope; recheck complete native discovery before primary'
        finally:
            replay_backend.close()
    return publish(target,result)
