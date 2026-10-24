"""The LFM2.5 side of the labelling: its native task list, its 100k discovery sample and that
sample's judge inputs.

`population` lists the native attempts. `materialize` commits the finished native generations as
trace files, runs the same stratified sampler as the other seven models
(`labeling.sample_stratified.build_plan`) and writes the sampled sentences with their judge tasks.
`repair_annotation_inputs` rebuilds the judge tasks of a build made with other inputs so that they carry the
source-parity inputs. `materialize` runs as `python -m routing.lfm materialize`, `repair_annotation_inputs` as
`python -m steering.pulse lfm-repair-contexts`, and `population` feeds `python -m steering.pulse native`.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
from labeling.batch_predictor import judge_inputs as sentence_inputs, question_of
from store import BENCHMARKS, Units, canonical, digest, file_sha, load, publish, require, verified


def population(binding):
    from store import read_source
    from steering.allocation import seed_for
    root = Path(binding['root'])
    destination = root/'lfm25/NATIVE_TASKS.json'
    if destination.exists():
        return verified(destination)
    source = Path(binding['source_campaign'])
    gpt_native = load(source/'gpt/NATIVE_ESTIMANDS.json')['records']
    folds = load(source/'FOLDS.json')
    rows = []
    for native in gpt_native:
        trace = read_source(native['original_source'])
        example = {k: trace.get(k) for k in ('dataset', 'problem_id', 'source_problem_id', 'sample_id', 'prompt', 'system_prompt', 'gold_answer')}
        example['generation_messages'] = trace.get('generation_messages') or trace.get('metadata', {}).get('generation_messages')
        r = {'model': 'lfm25', 'benchmark': native['benchmark'], 'family': native['family'],
             'fold': folds['folds'][native['family']], 'native_example': example, 'source': None,
             'arm': 'native', 'action': None, 'pulse_tokens': 0, 'horizon': 512,
             'profile_sha256': binding['models']['lfm25']['profile_sha256'],
             'seed': seed_for(binding['config']['seed'], 'lfm-native', native['attempt_identity'][:3]),
             'population_source': native['original_source'], 'original_attempt_identity': native['attempt_identity'][:3]}
        r['uid'] = digest(r)
        rows.append(r)
    require(len(rows) == 4647 and len({r['uid'] for r in rows}) == 4647, 'native population differs')
    return publish(destination, {'records': rows, 'population': 4647, 'source_sha256': file_sha(source/'gpt/NATIVE_ESTIMANDS.json'),
                                'generation_profile': binding['models']['lfm25']['profile_sha256']})


def trace_record(binding, row, raw, tokenizer):
    from steering.runtime import prompt_prefix
    from labeling.contexts import parts
    from models.token_replay import make_token_replay
    from generation.scoring import score_completion_detailed
    from schemas import TraceRecord
    prompt, prefix, example = prompt_prefix(binding, 'lfm25', row, tokenizer)
    require(not prefix, 'native population contains a continuation prefix')
    text, replay = make_token_replay(tokenizer, prompt, raw['token_ids'])
    reasoning, answer = parts(tokenizer, binding['models']['lfm25'], text)
    score = score_completion_detailed({'gold_answer': example['gold_answer']}, answer_type='math', model_text=answer or '')
    p = binding['models']['lfm25']
    metadata = {'token_replay': replay, 'reasoning_content': reasoning, 'assistant_content': answer,
        'finish_reason': raw['finish_reason'], 'generation_messages': example.get('generation_messages'),
        'generation_config': {**p['card'], 'seed': row['seed']}, 'final_routing_uid': row['uid'],
        'raw_sha256': raw['sha256'], 'score_detail': score, 'answer_present': answer is not None}
    value = {k: v for k, v in example.items() if k not in ('generation_messages', 'metadata')}
    value.update(model_id=p['checkpoint'], cot_text=text, model_answer=score.get('model_answer') or '',
                 # Sampling parity with the seven source models: a trace without a submitted answer (cap or
                 # unclosed reasoning) is strictly incorrect and eligible, as in their sampling. Native
                 # estimands below keep it as an unknown bound.
                 is_correct=score.get('is_correct') if answer is not None else False, metadata=metadata)
    return TraceRecord(**value).model_dump()


def judge_inputs(value, units, index):
    """Judge inputs exactly as the seven source models' labelling built them: the user message(s) of
    the generation prompt (falling back to the bare prompt), one neighbouring unit on each side and
    the edge sentinels."""
    messages = value.get('generation_messages') or value['metadata'].get('generation_messages')
    return sentence_inputs(question_of(messages, value['prompt']), units, index)


def repair_annotation_inputs(binding):
    """Rebuild the LFM judge tasks of the frozen 100k sample with source-parity inputs (the question, one
    neighbouring unit on each side and the edge sentinels). Same sentences; new context identities. Tasks that
    already say `judge_inputs: source-parity-v1` are returned as they are; otherwise the replaced files stay
    beside the new ones as `*.problem-statement-v0.json`."""
    from store import read_source
    from labeling.contexts import context_id
    from capture.dynamics.classes import saved_layout
    from schemas import TraceRecord
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    parent = load(Path(binding['source_campaign'])/'CAMPAIGN.json')
    tasks_path, sampled_path = root/'lfm25/ANNOTATION_TASKS.json', bulk/'lfm25/SAMPLED_SENTENCES.json'
    old_tasks, old_sampled = verified(tasks_path), verified(sampled_path)
    if old_tasks.get('judge_inputs') == 'source-parity-v1':
        return old_tasks
    sources = {s['attempt_id']: s for s in verified(root/'lfm25/SOURCES.json')['records']}
    grouped = defaultdict(list)
    for r in old_sampled['records']:
        grouped[r['attempt_id']].append(r)
    annotation, layouts = {}, []
    for attempt, rows in grouped.items():
        value = read_source(sources[attempt]['source_location'])
        units = saved_layout(TraceRecord(**value))['units']
        for r in rows:
            require(units[r['sentence_index']]['text'] == r['text'], 'sampled sentence identity changed')
            inputs = judge_inputs(value, units, r['sentence_index'])
            uid = context_id(inputs, parent['annotator'])
            annotation[uid] = {'uid': uid, 'inputs': inputs, 'model': 'lfm25'}
            layouts.append({**r, 'annotation_uid': uid})
    require(len(layouts) == len(old_sampled['records']) == 100000, 'sample size changed')
    for path in (tasks_path, sampled_path):
        path.rename(path.with_name(path.stem+'.problem-statement-v0.json'))
    publish(sampled_path, {'records': layouts, 'judge_inputs': 'source-parity-v1'})
    return publish(tasks_path, {'records': sorted(annotation.values(), key=lambda r: r['uid']),
                                'sampled_sentences': 100000, 'judge_inputs': 'source-parity-v1',
                                'replaces_sha256': old_tasks['sha256']})


def materialize(binding):
    """Commit exact native lines and run the 100k stratified sampler."""
    from transformers import AutoTokenizer
    from generation.generate import _model_slug
    from labeling.sample_stratified import build_plan
    from capture.dynamics.classes import saved_layout
    from labeling.spans import trace_digest
    from labeling.contexts import sentence_contexts, context_id
    from schemas import TraceRecord
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
    from stats import bootstrap_panel, unknown_bounds
    for row in tasks:
        value=by_uid.get(row['uid'])
        correct=value.get('is_correct') if value and value['metadata'].get('answer_present') else None
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
    manifest, selected = build_plan(generation, model, list(BENCHMARKS))
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
            inputs = judge_inputs(value, layout['units'], index)
            uid = context_id(inputs, parent['annotator'])
            annotation[uid] = {'uid': uid, 'inputs': inputs, 'model': 'lfm25'}
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
    require(len(layouts)==100000,'100000-sentence sampling obligation is incomplete')
    sources.sort(key=lambda r: (r['benchmark'], r['family'], r['question']))
    publish(root/'lfm25/SOURCES.json', {'records': sources, 'profile_sha256': binding['models']['lfm25']['profile_sha256']})
    publish(bulk/'lfm25/SAMPLED_SENTENCES.json', {'records': layouts})
    publish(root/'lfm25/ANNOTATION_TASKS.json', {'records': sorted(annotation.values(), key=lambda r: r['uid']), 'sampled_sentences': 100000})
    return publish(target, {'status': 'complete', 'native_attempts': len(tasks), 'unknown_native': unavailable,
                            'discovery_traces': len(sources), 'sampling_manifest_sha256': digest(manifest),
                            'sampled_sentences': len(layouts), 'unique_contexts': len(annotation)})
