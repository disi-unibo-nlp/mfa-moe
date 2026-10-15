"""Exact sentence ownership and unchanged reader contexts for new continuations."""
from pathlib import Path
from types import SimpleNamespace
from .common import CLASSES, Units, digest, load, publish, require


def parts(tokenizer, profile, text):
    if profile['key'] == 'lfm25':
        reasoning, marker, final = text.partition('</think>')
        return reasoning.removeprefix('<think>'), final if marker else None
    from moe_exp.research_campaign_v3_r6.annotation import parser_parts
    return parser_parts(tokenizer, profile, text)


def build(binding, model, rows, *, namespace='executions'):
    from transformers import AutoTokenizer
    from moe_exp.models.token_replay import make_token_replay
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.research_campaign_v3_r6.annotation import sentence_contexts, context_id
    from moe_exp.correlation_pipeline.scoring import score_completion_detailed
    from moe_exp.research_campaign_v3_r6.scoring import unknown_bounds
    from .runtime import prompt_prefix, closure
    root = Path(binding['root'])
    bulk = Path(binding['config']['bulk_root'])
    profile = binding['models'][model]
    tokenizer = AutoTokenizer.from_pretrained(profile['local_path'], local_files_only=True)
    units = Units(bulk/model/namespace)
    task_rows, context_rows = {}, []
    parent = load(Path(binding['source_campaign'])/'CAMPAIGN.json')
    for row in rows:
        output = units.folder(row['uid'])/'OUTPUT.json'
        target = bulk/model/'contexts'/(row['uid']+'.json')
        if target.exists():
            existing = load(target)
            for t in existing['tasks']:
                task_rows[t['uid']] = t
            context_rows.append(existing)
            continue
        if not output.exists():
            continue
        raw = load(output)
        prompt, prefix, original = prompt_prefix(binding, model, row, tokenizer)
        tokens = prefix+raw['token_ids']
        text, replay = make_token_replay(tokenizer, prompt, tokens)
        reasoning, final = parts(tokenizer, profile, text)
        trace = SimpleNamespace(**{**original, 'cot_text': text, 'metadata': {
            **original.get('metadata', {}), 'token_replay': replay, 'reasoning_content': reasoning,
            'assistant_content': final, 'finish_reason': raw['finish_reason']}})
        horizon = row.get('horizon') or 512
        legacy_row = {'source': {'prefix_tokens': len(prefix)}, 'window': horizon}
        sentences = sentence_contexts(trace, replay, legacy_row, raw)
        tasks = []
        for sentence in sentences:
            inputs = sentence.pop('inputs')
            uid = context_id(inputs, parent['annotator'])
            task = {'uid': uid, 'inputs': inputs, 'model': model}
            tasks.append(task)
            task_rows[uid] = task
            sentence['annotation_uid'] = uid
        layout = saved_layout(trace)
        reasoning_positions = set(layout['reasoning_tokens'] or [])
        nonreasoning = [p-len(prefix) for p in range(len(prefix), len(tokens)) if p not in reasoning_positions]
        reasoning_end = None
        for seq in closure(tokenizer, model):
            for i in range(max(0, len(prefix)-len(seq)), len(tokens)-len(seq)+1):
                if tokens[i:i+len(seq)] == seq:
                    value = max(0, i-len(prefix))
                    reasoning_end = min(reasoning_end, value) if reasoning_end is not None else value
                    break
        # A truncated thought is not a submitted final answer. All supported
        # profiles have an explicit reasoning channel/closure.
        detail = score_completion_detailed({'gold_answer': original.get('gold_answer')},
                                           answer_type='math', model_text=final or '')
        correct = detail.get('is_correct')
        strict = bool(correct) if correct is not None else None
        lo, hi = unknown_bounds(strict, raw['finish_reason'], bool(detail.get('model_answer')))
        record = publish(target, {'uid': row['uid'], 'model': model, 'benchmark': row['benchmark'],
            'family': row['family'], 'fold': row['fold'], 'sentences': sentences, 'tasks': tasks,
            'reasoning_end': reasoning_end, 'nonreasoning': nonreasoning,
            'generated_tokens': len(raw['token_ids']), 'prefix_tokens': len(prefix),
            'reasoning_tokens': sum(p >= len(prefix) for p in reasoning_positions),
            'finish_reason': raw['finish_reason'], 'strict_correct': strict, 'accuracy_bounds': [lo, hi],
            'score_detail': detail, 'raw_sha256': raw['sha256']})
        context_rows.append(record)
    return {'records': context_rows, 'tasks': sorted(task_rows.values(), key=lambda r: r['uid'])}


def measured(binding, row, record):
    from moe_exp.final_routing.endpoints import occupancy, first_sentence
    bulk = Path(binding['config']['bulk_root'])
    annotations = Units(bulk/'reader'/'annotations')
    labels, sentences = {}, []
    for s in record['sentences']:
        state = annotations.state(s['annotation_uid'])
        label = (state or {}).get('label')
        sentences.append({'start': s['token_start'], 'label': label, 'complete': s['complete']})
        for p in range(s['token_start'], s['token_end']):
            labels[p] = label
    natural = record['finish_reason'] == 'stop'
    return {'uid': row['uid'], 'status': 'complete',
            'occupancy': occupancy(labels, row['pulse_tokens'], row['horizon'],
                 reasoning_end=record['reasoning_end'], natural_eos=record['generated_tokens'] if natural else None,
                 nonreasoning=record['nonreasoning']),
            'first': first_sentence(sentences, natural_eos=natural or record['reasoning_end'] is not None,
                                    truncated=not natural and record['reasoning_end'] is None),
            'accuracy_bounds': record['accuracy_bounds'], 'strict_correct': record['strict_correct'],
            'new_tokens': record['generated_tokens'], 'reasoning_tokens': record['reasoning_tokens'],
            'total_tokens': record['prefix_tokens']+record['generated_tokens'],
            'finish_reason': record['finish_reason'], 'reasoning_end': record['reasoning_end']}
