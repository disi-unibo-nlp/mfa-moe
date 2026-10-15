"""LFM native population, original stratified sampling, and native analyses."""
from collections import defaultdict
from pathlib import Path
import hashlib
import time
from types import SimpleNamespace
from .common import BENCHMARKS, CLASSES, Units, canonical, digest, file_sha, load, publish, require, verified


def population(binding):
    from moe_exp.research_campaign_v3_r6.inputs import read_source
    from .design import seed_for
    root = Path(binding['root'])
    destination = root/'lfm25/NATIVE_TASKS.json'
    if destination.exists():
        return verified(destination)
    source = Path(binding['source_campaign'])
    original = load(source/'gpt/NATIVE_ESTIMANDS.json')['records']
    folds = load(source/'FOLDS.json')
    rows = []
    for old in original:
        trace = read_source(old['original_source'])
        example = {k: trace.get(k) for k in ('dataset', 'problem_id', 'source_problem_id', 'sample_id', 'prompt', 'system_prompt', 'gold_answer')}
        example['generation_messages'] = trace.get('generation_messages') or trace.get('metadata', {}).get('generation_messages')
        r = {'model': 'lfm25', 'benchmark': old['benchmark'], 'family': old['family'],
             'fold': folds['folds'][old['family']], 'native_example': example, 'source': None,
             'arm': 'native', 'action': None, 'pulse_tokens': 0, 'horizon': 512,
             'profile_sha256': binding['models']['lfm25']['profile_sha256'],
             'seed': seed_for(binding['config']['seed'], 'lfm-native', old['attempt_identity'][:3]),
             'population_source': old['original_source'], 'original_attempt_identity': old['attempt_identity'][:3]}
        r['uid'] = digest(r)
        rows.append(r)
    require(len(rows) == 4647 and len({r['uid'] for r in rows}) == 4647, 'native population differs')
    return publish(destination, {'records': rows, 'population': 4647, 'source_sha256': file_sha(source/'gpt/NATIVE_ESTIMANDS.json'),
                                'generation_profile': binding['models']['lfm25']['profile_sha256']})


def trace_record(binding, row, raw, tokenizer):
    from .runtime import prompt_prefix
    from .contexts import parts
    from moe_exp.models.token_replay import make_token_replay
    from moe_exp.correlation_pipeline.scoring import score_completion_detailed
    from moe_exp.schemas import TraceRecord
    prompt, prefix, example = prompt_prefix(binding, 'lfm25', row, tokenizer)
    require(not prefix, 'native population contains a continuation prefix')
    text, replay = make_token_replay(tokenizer, prompt, raw['token_ids'])
    reasoning, answer = parts(tokenizer, binding['models']['lfm25'], text)
    score = score_completion_detailed({'gold_answer': example['gold_answer']}, answer_type='math', model_text=answer or '')
    p = binding['models']['lfm25']
    metadata = {'token_replay': replay, 'reasoning_content': reasoning, 'assistant_content': answer,
        'finish_reason': raw['finish_reason'], 'generation_messages': example.get('generation_messages'),
        'generation_config': {**p['card'], 'seed': row['seed']}, 'final_routing_uid': row['uid'],
        'raw_sha256': raw['sha256'], 'score_detail': score}
    value = {k: v for k, v in example.items() if k not in ('generation_messages', 'metadata')}
    value.update(model_id=p['checkpoint'], cot_text=text, model_answer=score.get('model_answer') or '',
                 is_correct=score.get('is_correct') if answer is not None else None, metadata=metadata)
    return TraceRecord(**value).model_dump()


def materialize(binding):
    """Commit exact native lines and run the unchanged 100k stratified sampler."""
    from transformers import AutoTokenizer
    from moe_exp.correlation_pipeline.generate import _model_slug
    from moe_exp.correlation_pipeline.sample_stratified import _select
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.spans import trace_digest
    from moe_exp.research_campaign_v3_r6.annotation import sentence_contexts, context_id
    from moe_exp.schemas import TraceRecord
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    target = root/'lfm25/DISCOVERY_SELECTION.json'
    if target.exists():
        return verified(target)
    tasks = verified(root/'lfm25/NATIVE_TASKS.json')['records']
    units = Units(bulk/'lfm25/native')
    tokenizer = AutoTokenizer.from_pretrained(binding['models']['lfm25']['local_path'], local_files_only=True)
    grouped = defaultdict(list)
    unavailable = []
    for row in tasks:
        p = units.folder(row['uid'])/'OUTPUT.json'
        if p.exists():
            grouped[row['benchmark']].append((row, trace_record(binding, row, verified(p), tokenizer)))
        else:
            unavailable.append({'uid': row['uid'], 'state': units.state(row['uid'])})
    generation = bulk/'lfm25/generation'
    model = binding['models']['lfm25']['checkpoint']
    locations, traces, identities = {}, {}, {}
    for benchmark in BENCHMARKS:
        folder = generation/_model_slug(model)/benchmark
        folder.mkdir(parents=True, exist_ok=True)
        path, pending = folder/'traces.jsonl', folder/'traces.pending'
        with pending.open('wb') as stream:
            for row, value in sorted(grouped[benchmark], key=lambda x: x[0]['uid']):
                data = canonical(value)+b'\n'
                key = (benchmark, value['problem_id'], value['sample_id'])
                locations[key] = {'path': str(path), 'byte_offset': stream.tell(), 'line_bytes': len(data),
                                  'line_sha256': hashlib.sha256(data).hexdigest()}
                traces[key], identities[key] = value, row
                stream.write(data)
        if path.exists():
            require(file_sha(path) == file_sha(pending), 'native collection changed after materialization')
            pending.unlink()
        else:
            pending.replace(path)
        publish(folder/'manifest.json', {'status': 'complete', 'records': len(grouped[benchmark]),
            'file_sha256': file_sha(path), 'population_terminal': True, 'unknowns_preserved': True})
    native=[]
    by_uid={row['uid']:value for pairs in grouped.values() for row,value in pairs}
    from moe_exp.research_campaign_v3_r6.scoring import unknown_bounds
    from moe_exp.research_campaign_v3_r6.statistics import bootstrap_panel
    for row in tasks:
        value=by_uid.get(row['uid']);correct=value.get('is_correct') if value else None
        finish=value['metadata']['finish_reason'] if value else 'unknown'
        answer=bool(value and value['model_answer'])
        lo,hi=unknown_bounds(correct,finish,answer)
        native.append({'benchmark':row['benchmark'],'family':row['family'],'uid':row['uid'],
            'question':row['benchmark']+'|'+(row['native_example']['source_problem_id'] or row['native_example']['problem_id']),
            'sample_id':row['native_example']['sample_id'],'strict':correct,'unknown_lower':lo,'unknown_upper':hi,
            'tokens':len(value['metadata']['token_replay']['completion_token_ids']) if value else None,
            'termination':finish,'original_attempt_identity':row['original_attempt_identity']})
    publish(root/'lfm25/NATIVE_ESTIMANDS.json',{'records':native,'fixed_denominator':4647,'unknowns_retained':True})
    publish(root/'lfm25/NATIVE_ACCURACY.json',bootstrap_panel(native,['unknown_lower','unknown_upper','tokens'],
            panel='lfm25:native',resample_key='native-family',draws=10000))
    args = SimpleNamespace(generation_dir=generation, generation_model=model, datasets=list(BENCHMARKS),
                           total=100000, parts=4, part_size=25000, seed=42)
    manifest, selected, details = _select(args)
    publish(root/'lfm25/SAMPLING_MANIFEST.json', manifest)
    parent = load(Path(binding['source_campaign'])/'CAMPAIGN.json')
    sources, annotation, layouts = [], {}, []
    for choice in selected.values():
        info = choice.info
        key = (info.dataset, info.problem_id, info.sample_id)
        value, row = traces[key], identities[key]
        trace = TraceRecord(**value)
        layout = saved_layout(trace)
        ranks = {p:i for i,p in enumerate(layout['reasoning_tokens'])}
        sha = trace_digest(trace)
        question = info.dataset+'|'+(info.source_problem_id or info.problem_id)
        source = {'attempt_id': digest([*key, sha]), 'question': question, 'benchmark': info.dataset,
            'family': row['family'], 'fold': row['fold'], 'source_location': locations[key],
            'trace_sha256': sha, 'completion_tokens': len(value['metadata']['token_replay']['completion_token_ids']),
            'sample_id': info.sample_id, 'problem_id': info.problem_id, 'source_problem_id': info.source_problem_id,
            'native_uid': row['uid'], 'selected_sentences': list(choice.combined), 'stratum': choice.cell}
        sources.append(source)
        contexts = sentence_contexts(trace, value['metadata']['token_replay'],
            {'source': {'prefix_tokens': 0}, 'window': source['completion_tokens']},
            {'token_ids': value['metadata']['token_replay']['completion_token_ids'], 'finish_reason': value['metadata']['finish_reason']})
        by_index = {r['sentence_index']: r for r in contexts}
        for index in choice.combined:
            require(index in by_index, 'sampled native sentence lacks exact context')
            c = by_index[index]
            uid = context_id(c['inputs'], parent['annotator'])
            annotation[uid] = {'uid': uid, 'inputs': c['inputs'], 'model': 'lfm25'}
            owned = layout['unit_tokens'][index]
            layouts.append({'attempt_id': source['attempt_id'], 'question': question, 'benchmark': info.dataset,
                'family': row['family'], 'fold': row['fold'], 'sentence_index': index, 'annotation_uid': uid,
                'token_indices': owned, 'token_ranks':[ranks[p] for p in owned],
                'segment': c['segment'], 'text': layout['units'][index]['text'],
                'n_reasoning': len(layout['reasoning_tokens']),
                'progress':min((ranks[p] for p in owned),default=0)/max(1,len(layout['reasoning_tokens']))})
    expected_questions = set(load(Path(binding['source_campaign'])/'FOLDS.json')['question_families'])
    require(len(sources) == 1547 and {s['question'] for s in sources} == expected_questions,
            'designated discovery population incomplete; retain native unknowns without replacement')
    require(len(layouts)==100000,'original 100000-sentence sampling obligation is incomplete')
    sources.sort(key=lambda r: (r['benchmark'], r['family'], r['question']))
    publish(root/'lfm25/SOURCES.json', {'records': sources, 'profile_sha256': binding['models']['lfm25']['profile_sha256']})
    publish(bulk/'lfm25/SAMPLED_SENTENCES.json', {'records': layouts})
    publish(root/'lfm25/ANNOTATION_TASKS.json', {'records': sorted(annotation.values(), key=lambda r: r['uid']), 'sampled_sentences': 100000})
    return publish(target, {'status': 'complete', 'native_attempts': len(tasks), 'unknown_native': unavailable,
                            'discovery_traces': len(sources), 'sampling_manifest_sha256': digest(manifest),
                            'sampled_sentences': len(layouts), 'unique_contexts': len(annotation)})


def replay(binding, rows, deadline):
    from .runtime import Backend
    from moe_exp.research_campaign_v3_r6.inputs import read_source
    bulk = Path(binding['config']['bulk_root'])
    backend = Backend(binding, 'lfm25', 1, deadline, replay=True)
    try:
        receipts = []
        for row in rows:
            require(time.time()+60 < deadline, 'replay allocation reserve')
            receipts.append(backend.capture_trace(read_source(row['source_location']), bulk/'lfm25/replay'/(row['attempt_id']+'.npz')))
        return {'status': 'complete', 'receipts': receipts}
    finally:
        backend.close()


def derive(binding):
    import numpy as np
    import pandas as pd
    from moe_exp.schemas import TraceRecord
    from moe_exp.research_campaign_v3_r6.inputs import read_source
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.research_campaign_v3_r6.design import prefix_roster
    from moe_exp.research_campaign_v3_r6.statistics import trajectories
    from moe_exp.expert_memory_study.selection import analyze, validate_counts
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    sources = verified(root/'lfm25/SOURCES.json')['records']
    sampled = verified(bulk/'lfm25/SAMPLED_SENTENCES.json')['records']
    by_attempt = defaultdict(list)
    for r in sampled:
        by_attempt[r['attempt_id']].append(r)
    labels = Units(bulk/'reader/annotations')
    counts = np.zeros((len(sources), 9, 38, 64), np.uint64)
    tokens = np.zeros((len(sources), 9), np.uint64)
    sentences = []
    routing_rows=[]
    for index, source in enumerate(sources):
        trace = TraceRecord(**read_source(source['source_location']))
        layout = saved_layout(trace)
        path = bulk/'lfm25/replay'/(source['attempt_id']+'.npz')
        receipt = verified(path.with_suffix('.json'))
        require(file_sha(path) == receipt['file_sha256'], 'replay changed before analysis')
        with np.load(path, allow_pickle=False) as z:
            ids,weights = z['ids'],z['weights']
        require(ids.shape == (38, receipt['captured_completion_tokens'], 4) and
                all(p<ids.shape[1] for p in layout['reasoning_tokens']), 'LFM replay geometry/reasoning coverage differs')
        ownership = defaultdict(list)
        ownership[7] = layout['reasoning_tokens']
        positions=np.asarray(layout['reasoning_tokens'],int)
        adjacent=np.diff(positions)==1
        native_row={'benchmark':source['benchmark'],'family':source['family'],'attempt_id':source['attempt_id']}
        for li,layer in enumerate(range(2,40)):
            chosen=ids[li,positions]
            dominant=chosen[np.arange(len(chosen)),weights[li,positions].argmax(-1)]
            frequency=np.bincount(dominant,minlength=64)/max(1,len(dominant))
            nonzero=frequency[frequency>0]
            native_row['entropy:'+str(layer)]=float(-(nonzero*np.log2(nonzero)).sum()/6)
            native_row['dominant_switch:'+str(layer)]=float((dominant[1:][adjacent]!=dominant[:-1][adjacent]).mean()) if adjacent.any() else None
            if adjacent.any():
                left,right=chosen[:-1][adjacent],chosen[1:][adjacent]
                intersection=(left[:,:,None]==right[:,None,:]).sum((1,2))
                native_row['topk_turnover:'+str(layer)]=float((1-intersection/(8-intersection)).mean())
            else:native_row['topk_turnover:'+str(layer)]=None
        routing_rows.append(native_row)
        for r in by_attempt[source['attempt_id']]:
            label = (labels.state(r['annotation_uid']) or {}).get('label')
            if label not in CLASSES or not r['token_indices']:
                continue
            c = CLASSES.index(label)
            ownership[c].extend(r['token_indices'])
            ownership[8].extend(r['token_indices'])
            sentences.append({**r, 'dataset': r['benchmark'], 'class': label, 'cls': c,
                'n_tokens': len(r['token_indices']), 'tok_start': min(r['token_ranks']), 'tok_end': max(r['token_ranks'])+1,
                'rel_pos': r['progress'], 'prefix_tokens': max(r['token_indices'])+1,
                'source_class': label, 'compatible': max(r['token_indices'])+1 < source['completion_tokens']})
        for c, positions in ownership.items():
            require(len(positions) == len(set(positions)), 'overlapping native sentence ownership')
            tokens[index,c] = len(positions)
            for layer in range(38):
                counts[index,c,layer] = np.bincount(ids[layer,positions].ravel().astype(int), minlength=64)
    validate_counts(counts, tokens, 4)
    folder = bulk/'lfm25/derived'
    folder.mkdir(parents=True, exist_ok=True)
    count_path, sentence_path = folder/'class_counts.npz', folder/'sentences.parquet'
    with count_path.open('wb') as stream:
        np.savez_compressed(stream, counts=counts, tokens=tokens, attempt_id=[s['attempt_id'] for s in sources], classes=[*CLASSES,'ALL_REASONING','ALL_LABELLED'])
    pd.DataFrame(sentences).to_parquet(sentence_path, index=False)
    folds = load(Path(binding['source_campaign'])/'FOLDS.json')['folds']
    causal = load(Path(binding['source_campaign'])/'COHORTS.json')['causal']
    prefixes = prefix_roster(sentences, causal, folds, 'lfm25')
    require(len(prefixes) == 129, 'LFM prefix population changed')
    publish(root/'lfm25/PREFIXES.json', {'records': prefixes, 'folds_sha256': digest(folds)})
    coverage = analyze(counts[:,:7], [s['family'] for s in sources], keep=48, groups=1,
        layers=list(range(2,40)), model='lfm25', classes=CLASSES, folds=folds)
    publish(root/'lfm25/SELECTION_COVERAGE.json', coverage)
    from moe_exp.research_campaign_v3_r6.statistics import bootstrap_panel
    routing_columns=[name+':'+str(layer) for name in ('entropy','dominant_switch','topk_turnover') for layer in range(2,40)]
    routing_summary=bootstrap_panel(routing_rows,routing_columns,panel='lfm25:native-routing-dynamics',draws=10000,
        resample_key='native-family')
    publish(root/'lfm25/DYNAMICS.json', {**trajectories(sentences),'routing_inference':routing_summary,
        'routing_observations':routing_rows,'native_input_token_alignment':True,
        'adjacent_reasoning_only':True,'unknown_sentence_gaps_preserved':True})
    publish(root/'lfm25/NATIVE_PROFILE.json', {'layers': list(range(2,40)), 'experts': 64, 'native_k':4,
        'reasoning_tokens': tokens[:,7].sum(), 'labelled_tokens':tokens[:,8].sum(),
        'class_tokens':tokens[:,:7].sum(0), 'source_receipts':len(sources)})
    return publish(root/'lfm25/DISCOVERY_INPUTS.json', {'status': 'complete', 'sources': sources,
        'counts': str(count_path), 'counts_sha256': file_sha(count_path),
        'sentences': str(sentence_path), 'sentences_sha256':file_sha(sentence_path)})
