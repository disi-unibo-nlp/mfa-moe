"""LFM native population, original stratified sampling, and native analyses."""
from collections import defaultdict
from pathlib import Path
from moe_exp.io import CLASSES, Units, canonical, digest, file_sha, load, publish, require, verified


def derive(binding):
    import numpy as np
    import pandas as pd
    from moe_exp.schemas import TraceRecord
    from moe_exp.io import read_source
    from moe_exp.capture.dynamics.classes import saved_layout
    from moe_exp.routing.selection import prefix_roster
    from moe_exp.routing.dynamics import trajectories
    from moe_exp.routing.selection import analyze, validate_counts
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    sources = verified(root/'lfm25/SOURCES.json')['records']
    sampled = verified(bulk/'lfm25/SAMPLED_SENTENCES.json')['records']
    by_attempt = defaultdict(list)
    for r in sampled:
        by_attempt[r['attempt_id']].append(r)
    labels = Units(bulk/'reader/annotations')
    label_counts = {'complete': 0, 'unknown': 0}
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
        missing = receipt.get('unavailable_terminal_reasoning_tokens', 0)
        reasoning_tokens = [p for p in layout['reasoning_tokens'] if p < ids.shape[1]]
        require(ids.shape == (38, receipt['captured_completion_tokens'], 4) and
                len(layout['reasoning_tokens'])-len(reasoning_tokens) == missing, 'LFM replay geometry/reasoning coverage differs')
        ownership = defaultdict(list)
        ownership[7] = reasoning_tokens
        positions=np.asarray(reasoning_tokens,int)
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
            state = labels.state(r['annotation_uid'])
            require(state is not None and state['status'] in ('complete', 'unknown'),
                    'LFM discovery label missing: derive requires every sampled sentence to be terminal')
            label_counts[state['status']] += 1
            label = state.get('label')
            keep = [i for i, p in enumerate(r['token_indices']) if p < ids.shape[1]]
            require(len(r['token_indices'])-len(keep) <= missing, 'only the capped terminal token may be unavailable')
            if label not in CLASSES or not keep:
                continue
            r = {**r, 'token_indices': [r['token_indices'][i] for i in keep], 'token_ranks': [r['token_ranks'][i] for i in keep]}
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
    from moe_exp.stats import bootstrap_panel
    routing_columns=[name+':'+str(layer) for name in ('entropy','dominant_switch','topk_turnover') for layer in range(2,40)]
    routing_summary=bootstrap_panel(routing_rows,routing_columns,panel='lfm25:native-routing-dynamics',draws=10000,
        resample_key='native-family')
    publish(root/'lfm25/DYNAMICS.json', {**trajectories(sentences),'routing_inference':routing_summary,
        'routing_observations':routing_rows,'native_input_token_alignment':True,
        'adjacent_reasoning_only':True,'unknown_sentence_gaps_preserved':True})
    publish(root/'lfm25/NATIVE_PROFILE.json', {'layers': list(range(2,40)), 'experts': 64, 'native_k':4,
        'reasoning_tokens': tokens[:,7].sum(), 'labelled_tokens':tokens[:,8].sum(),
        'class_tokens':tokens[:,:7].sum(0), 'source_receipts':len(sources)})
    require(sum(label_counts.values()) == len(sampled), 'sampled sentences without a terminal label')
    return publish(root/'lfm25/DISCOVERY_INPUTS.json', {'status': 'complete', 'sources': sources,
        'label_states': label_counts, 'judge_inputs': verified(root/'lfm25/ANNOTATION_TASKS.json').get('judge_inputs'),
        'counts': str(count_path), 'counts_sha256': file_sha(count_path),
        'sentences': str(sentence_path), 'sentences_sha256':file_sha(sentence_path)})


def dynamics_summary(binding):
    """Native-dynamics panel for LFM, built as research_campaign_v3_r6.derive builds it for the other seven models:
    per-attempt occupancy, dwell, re-entry and next-state rates from the published trajectories, then the
    10,000-draw family bootstrap with simultaneous intervals."""
    import numpy as np
    import pandas as pd
    from moe_exp.stats import bootstrap_panel
    from moe_exp.routing.dynamics import trajectories
    root = Path(binding['root'])
    inputs = verified(root/'lfm25/DISCOVERY_INPUTS.json')
    require(file_sha(Path(inputs['sentences'])) == inputs['sentences_sha256'], 'LFM derived sentences changed')
    sentences = pd.read_parquet(inputs['sentences']).to_dict('records')
    dynamics = verified(root/'lfm25/DYNAMICS.json')
    recomputed = trajectories(sentences)
    require(all(canonical(dynamics[k]) == canonical(recomputed[k]) for k in ('dwell', 'transitions', 'reentry')),
            'published LFM trajectories differ from the current sentences')
    source_lookup = {s['attempt_id']: s for s in inputs['sources']}
    by_attempt = defaultdict(list)
    for s in sentences:
        by_attempt[s['attempt_id']].append(s)
    dwell_by_attempt, transitions_by_attempt, reentry_by_attempt = defaultdict(list), defaultdict(list), defaultdict(list)
    for d in dynamics['dwell']:
        dwell_by_attempt[d['attempt_id']].append(d)
    for d in dynamics['transitions']:
        transitions_by_attempt[d['attempt_id']].append(d)
    for d in dynamics['reentry']:
        reentry_by_attempt[d['attempt_id']].append(d)
    rows = []
    for aid, sent in by_attempt.items():
        source = source_lookup[aid]
        n = sum(s['n_tokens'] for s in sent)
        row = {'benchmark': source['benchmark'], 'family': source['family'], 'attempt_id': aid,
               'observed_labelled_tokens': n,
               'label_coverage': n/source['completion_tokens'] if source['completion_tokens'] else None}
        for c in CLASSES:
            row[c+':occupancy'] = sum(s['n_tokens'] for s in sent if s['class'] == c)/n if n else None
            dwell = [d for d in dwell_by_attempt[aid] if d['class'] == c]
            uncensored = [d['tokens'] for d in dwell if not d['left_censored'] and not d['right_censored']]
            row[c+':observed_dwell_tokens'] = float(np.mean([d['tokens'] for d in dwell])) if dwell else None
            row[c+':uncensored_dwell_tokens'] = float(np.mean(uncensored)) if uncensored else None
            trans = [d for d in transitions_by_attempt[aid] if d['source'] == c]
            row[c+':reentry_per_observed_boundary'] = (sum(d['class'] == c for d in reentry_by_attempt[aid])/len(trans)
                                                       if trans else None)
            for next_class in CLASSES:
                row[c+':next:'+next_class] = sum(d['destination'] == next_class for d in trans)/len(trans) if trans else None
        rows.append(row)
    pd.DataFrame(rows).to_parquet(Path(binding['config']['bulk_root'])/'lfm25/derived/native_dynamics_estimands.parquet',
                                  index=False)
    columns = sorted({k for r in rows for k in r if k not in ('benchmark', 'family', 'attempt_id')})
    return publish(root/'lfm25/NATIVE_DYNAMICS_SUMMARY.json',
                   bootstrap_panel(rows, columns, panel='native-dynamics:lfm25'))
