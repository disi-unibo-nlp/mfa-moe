"""CPU outcome extraction for driver continuations: sentences, closure, scoring, token counts.

Wraps `contexts.build` (exact sentence ownership, judge-equivalent inputs within the
16·Q75 window plus the first complete sentence, strict scoring with unknown bounds)
over the driver's durable store. Writes one parquet of sentences and one of
continuation outcomes per model shard under <bulk>/driver/<model>/outcomes/.
"""
from __future__ import annotations
import argparse
import json
import time
from store import require
from steering.pulse import DRIVER, STORES, Store, binding, bulk, design_rows, log

START, END = '<START OF RESPONSE>', '<END OF RESPONSE>'


def continuation_sentences(trace, replay, prefix, window, finish_reason):
    """`research_campaign_v3_r6.annotation.sentence_contexts`, except that a reasoning unit straddling the
    prefix boundary (the continuation does not open a new sentence) is kept, clipped to its continuation
    tokens and flagged, instead of aborting: it counts toward occupancy but is never the first sentence."""
    from capture.dynamics.classes import saved_layout
    from labeling.spans import reasoning_ranges
    layout = saved_layout(trace)
    units, ranges = layout['units'], reasoning_ranges(trace)
    generated = []
    for i, u in enumerate(units):
        owned = layout['unit_tokens'][i] or []
        if not owned or max(owned) < prefix:
            continue
        complete = i < len(units)-1 or finish_reason == 'stop' or u['text'].rstrip().endswith(('.', '!', '?'))
        generated.append((i, u, max(min(owned), prefix), max(owned)+1, complete, min(owned) < prefix))
    first = next((i for i, u, a, b, complete, straddles in generated
                  if complete and not straddles and 0 <= a-prefix < window), None)
    selected = []
    for i, u, start, end, complete, straddles in generated:
        if start-prefix >= window and i != first:
            continue
        selected.append({'sentence_index': i, 'segment': next(j for j, (a, b) in enumerate(ranges) if a <= u['start'] < b),
                         'token_start': start-prefix, 'token_end': end-prefix, 'n_tokens': end-start,
                         'complete': complete, 'first_complete': i == first, 'straddles_prefix': straddles,
                         'inputs': {'problem_statement': trace.prompt,
                                    'previous_sentence': units[i-1]['text'] if i else '',
                                    'sentence': u['text'],
                                    'next_sentence': units[i+1]['text'] if i+1 < len(units) else ''}})
    return selected, layout


def record(b, model, row, raw, tokenizer, close, *, frozen_input=None):
    """One continuation: exact text, sentence windows, closure, strict score (contexts.build semantics)."""
    from types import SimpleNamespace
    from models.token_replay import make_token_replay
    from generation.scoring import score_completion_detailed
    from stats import unknown_bounds
    from labeling.contexts import parts
    from steering.runtime import prompt_prefix
    profile = b['models'][model]
    prompt, prefix, original = frozen_input or prompt_prefix(b, model, row, tokenizer)
    tokens = prefix+raw['token_ids']
    text, replay = make_token_replay(tokenizer, prompt, tokens)
    reasoning, final = parts(tokenizer, profile, text)
    fallback = bool(reasoning) and reasoning not in text
    if fallback:  # parser text not verbatim in the decoded sequence: use the explicit <think> span instead
        reasoning = None
    trace = SimpleNamespace(**{**original, 'cot_text': text, 'metadata': {
        **original.get('metadata', {}), 'token_replay': replay, 'reasoning_content': reasoning,
        'assistant_content': final, 'finish_reason': raw['finish_reason']}})
    sentences, layout = continuation_sentences(trace, replay, len(prefix), row['horizon'], raw['finish_reason'])
    reasoning_positions = set(layout['reasoning_tokens'] or [])
    nonreasoning = [p-len(prefix) for p in range(len(prefix), len(tokens)) if p not in reasoning_positions]
    reasoning_end = None
    for seq in close:
        for i in range(max(0, len(prefix)-len(seq)), len(tokens)-len(seq)+1):
            if tokens[i:i+len(seq)] == seq:
                value = max(0, i-len(prefix))
                reasoning_end = value if reasoning_end is None else min(reasoning_end, value)
                break
    if row.get('whole') and reasoning_end is not None:
        # A direct final-channel answer has zero reasoning, even though the
        # legacy layout fallback permits annotating final-only model outputs.
        reasoning_positions = {p for p in reasoning_positions if p < len(prefix)+reasoning_end}
        nonreasoning = [p-len(prefix) for p in range(len(prefix), len(tokens)) if p not in reasoning_positions]
        sentences = [s for s in sentences if s['token_start'] < reasoning_end]
    detail = score_completion_detailed({'gold_answer': original.get('gold_answer')}, answer_type='math',
                                       model_text=final or '')
    strict = bool(detail['is_correct']) if final is not None and detail.get('is_correct') is not None else (
        False if final is None else None)
    lo, hi = unknown_bounds(strict if final is not None else None, raw['finish_reason'], bool(detail.get('model_answer')))
    return {'sentences': sentences, 'reasoning_end': reasoning_end, 'nonreasoning': nonreasoning,
            'generated_tokens': len(raw['token_ids']), 'prefix_tokens': len(prefix),
            'reasoning_tokens': sum(p >= len(prefix) for p in reasoning_positions),
            'finish_reason': raw['finish_reason'], 'answer_present': bool(final and final.strip()), 'reasoning_fallback': fallback,
            'strict_correct': strict, 'accuracy_bounds': [lo, hi]}


def extract(model, shard, design='main'):
    import pandas as pd
    from transformers import AutoTokenizer
    from store import load
    from steering.runtime import closure
    b = binding()
    rows = design_rows(model, design, shard)
    store = Store(bulk(b)/model/STORES[design])
    tokenizer = AutoTokenizer.from_pretrained(b['models'][model]['local_path'], local_files_only=True)
    close = closure(tokenizer, model)
    tick = time.time()
    sentences, outcomes, failures = [], [], []
    for row in rows:
        result = store.state(row['uid'])
        if (result or {}).get('status') != 'complete':
            continue
        try:
            rec = record(b, model, row, load(store.folder(row['uid'])/'OUTPUT.json'), tokenizer, close)
        except Exception as error:  # per-continuation failure is recorded, never silently dropped
            failures.append({'uid': row['uid'], 'error': f'{type(error).__name__}: {error}'})
            continue
        outcomes.append({'uid': row['uid'], 'model': model, 'arm': row['arm'], 'class': row['class'],
            'family': row['family'], 'fold': row['fold'], 'benchmark': row['benchmark'],
            'layer': ((row.get('actions') or [row.get('action')])[0] or {}).get('layer'),
            'layer_regime': row.get('layer_regime'), 'operator': ((row.get('actions') or [row.get('action')])[0] or {}).get('operator', 'hard_select'),
            'pulse_tokens': row['pulse_tokens'], 'horizon': row['horizon'],
            'generated_tokens': rec['generated_tokens'], 'prefix_tokens': rec['prefix_tokens'],
            'reasoning_tokens': rec['reasoning_tokens'], 'reasoning_end': rec['reasoning_end'],
            'finish_reason': rec['finish_reason'], 'answer_present': rec['answer_present'],
            'reasoning_fallback': rec['reasoning_fallback'],
            'strict_correct': rec['strict_correct'], 'accuracy_lower': rec['accuracy_bounds'][0],
            'accuracy_upper': rec['accuracy_bounds'][1], 'segments': result.get('segments'),
            'straddles_prefix': any(s['straddles_prefix'] for s in rec['sentences']),
            'nonreasoning': json.dumps(rec['nonreasoning'])})
        for s in rec['sentences']:
            sentences.append({'uid': row['uid'], 'model': model, 'arm': row['arm'], 'class': row['class'],
                'family': row['family'], 'fold': row['fold'], 'sentence_index': s['sentence_index'],
                'token_start': s['token_start'], 'token_end': s['token_end'], 'complete': s['complete'],
                'first_complete': s['first_complete'], 'straddles_prefix': s['straddles_prefix'],
                'sentence_text': s['inputs']['sentence'], 'prev_sentence': s['inputs']['previous_sentence'] or START,
                'next_sentence': s['inputs']['next_sentence'] or END})
    folder = bulk(b)/model/'outcomes'
    folder.mkdir(parents=True, exist_ok=True)
    tag = ('' if design == 'main' else design+'-')+f'{shard[0]}-of-{shard[1]}'
    pd.DataFrame(outcomes).to_parquet(folder/f'outcomes-{tag}.parquet', index=False)
    pd.DataFrame(sentences).to_parquet(folder/f'sentences-{tag}.parquet', index=False)
    (folder/f'failures-{tag}.json').write_text(json.dumps(failures))
    value = {'model': model, 'design': design, 'shard': list(shard), 'rows': len(rows), 'extracted': len(outcomes),
             'failures': len(failures), 'sentences': len(sentences),
             'straddling': sum(o['straddles_prefix'] for o in outcomes), 'seconds': time.time()-tick}
    log({'event': 'outcomes', **value})
    return value


def main(argv=None):
    """Extract the saved continuations of one model shard."""
    from steering.pulse import parse_shard
    ap = argparse.ArgumentParser(prog='steering.pulse_outcomes')
    ap.add_argument('--model', required=True)
    ap.add_argument('--shard', default='0/1', type=parse_shard)
    ap.add_argument('--design', choices=sorted(STORES), default='main')
    args = ap.parse_args(argv)
    require(args.model in [p.name for p in DRIVER.iterdir() if p.is_dir()], 'unknown model manifest')
    extract(args.model, args.shard, args.design)


if __name__ == '__main__':
    main()
