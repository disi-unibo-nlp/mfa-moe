"""LFM2.5 analyses that need no GPU, once its traces are replayed and its sentences labelled.

    python -m routing.lfm materialize   sampled sentences, judge inputs and annotation tasks
    python -m routing.lfm derive        class counts, class trajectories, expert-selection coverage, nominations
                                        and the main-design assignments (the replay and labels must exist)
    python -m routing.lfm readout       present-state readout of routing, five folds and five nulls (about 1.5 h, 4 CPUs)
    python -m routing.lfm dynamics      class occupancy, dwell and next-class panel (needs derive)

The study folder is STUDY_DATA_ROOT (default report/final-routing-20261015-r1). `scripts/routing/lfm.sh` runs
derive, then readout and dynamics. The replay itself is the GPU stage `python -m steering.pulse lfm-replay`.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from routing.dynamics import trajectories
from routing.selection import analyze, prefix_roster, validate_counts
import store
from stats import bootstrap_panel
from store import CLASSES, Units, digest, file_sha, load, publish, read_source, require, verified

LAYERS, EXPERTS, K = range(2, 40), 64, 4


def routing_row(source, ids, weights, positions):
    """Per-layer routing dynamics of one attempt over its adjacent reasoning tokens (entropy, switches, turnover)."""
    adjacent = np.diff(positions) == 1
    row = {'benchmark': source['benchmark'], 'family': source['family'], 'attempt_id': source['attempt_id']}
    for li, layer in enumerate(LAYERS):
        chosen = ids[li, positions]
        dominant = chosen[np.arange(len(chosen)), weights[li, positions].argmax(-1)]
        frequency = np.bincount(dominant, minlength=EXPERTS)/max(1, len(dominant))
        nonzero = frequency[frequency > 0]
        row[f'entropy:{layer}'] = float(-(nonzero*np.log2(nonzero)).sum()/6)
        row[f'dominant_switch:{layer}'] = (float((dominant[1:][adjacent] != dominant[:-1][adjacent]).mean())
                                           if adjacent.any() else None)
        if adjacent.any():
            left, right = chosen[:-1][adjacent], chosen[1:][adjacent]
            intersection = (left[:, :, None] == right[:, None, :]).sum((1, 2))
            row[f'topk_turnover:{layer}'] = float((1-intersection/(2*K-intersection)).mean())
        else:
            row[f'topk_turnover:{layer}'] = None
    return row


def discover(binding):
    """Class counts, sentence table, trajectories, selection coverage and prefixes from the replay and the labels.

    The LFM2.5 counterpart of the saved-data analysis of the seven campaign models. Writes DISCOVERY_INPUTS.json
    last, so its presence means every other output exists.
    """
    import pandas as pd
    from schemas import TraceRecord
    from capture.dynamics.classes import saved_layout
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    sources = verified(root/'lfm25/SOURCES.json')['records']
    sampled = verified(bulk/'lfm25/SAMPLED_SENTENCES.json')['records']
    by_attempt = defaultdict(list)
    for r in sampled:
        by_attempt[r['attempt_id']].append(r)
    labels = Units(bulk/'reader/annotations')
    label_counts = {'complete': 0, 'unknown': 0}
    counts = np.zeros((len(sources), 9, len(LAYERS), EXPERTS), np.uint64)
    tokens = np.zeros((len(sources), 9), np.uint64)
    sentences, routing_rows = [], []
    for index, source in enumerate(sources):
        layout = saved_layout(TraceRecord(**read_source(source['source_location'])))
        path = bulk/'lfm25/replay'/(source['attempt_id']+'.npz')
        receipt = verified(path.with_suffix('.json'))
        require(file_sha(path) == receipt['file_sha256'], 'replay changed before analysis')
        with np.load(path, allow_pickle=False) as z:
            ids, weights = z['ids'], z['weights']
        missing = receipt.get('unavailable_terminal_reasoning_tokens', 0)
        reasoning_tokens = [p for p in layout['reasoning_tokens'] if p < ids.shape[1]]
        require(ids.shape == (len(LAYERS), receipt['captured_completion_tokens'], K) and
                len(layout['reasoning_tokens'])-len(reasoning_tokens) == missing,
                'LFM replay geometry/reasoning coverage differs')
        routing_rows.append(routing_row(source, ids, weights, np.asarray(reasoning_tokens, int)))
        ownership = defaultdict(list)  # class 7 = all reasoning tokens, 8 = all labelled tokens
        ownership[7] = reasoning_tokens
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
            tokens[index, c] = len(positions)
            for layer in range(len(LAYERS)):
                counts[index, c, layer] = np.bincount(ids[layer, positions].ravel().astype(int), minlength=EXPERTS)
    validate_counts(counts, tokens, K)
    folder = bulk/'lfm25/derived'
    folder.mkdir(parents=True, exist_ok=True)
    count_path, sentence_path = folder/'class_counts.npz', folder/'sentences.parquet'
    with count_path.open('wb') as stream:
        np.savez_compressed(stream, counts=counts, tokens=tokens, attempt_id=[s['attempt_id'] for s in sources],
                            classes=[*CLASSES, 'ALL_REASONING', 'ALL_LABELLED'])
    pd.DataFrame(sentences).to_parquet(sentence_path, index=False)
    folds = load(Path(binding['source_campaign'])/'FOLDS.json')['folds']
    causal = load(Path(binding['source_campaign'])/'COHORTS.json')['causal']
    prefixes = prefix_roster(sentences, causal, folds, 'lfm25')
    require(len(prefixes) == 129, 'LFM prefix population changed')
    publish(root/'lfm25/PREFIXES.json', {'records': prefixes, 'folds_sha256': digest(folds)})
    publish(root/'lfm25/SELECTION_COVERAGE.json',
            analyze(counts[:, :7], [s['family'] for s in sources], keep=48, groups=1,
                    layers=list(LAYERS), model='lfm25', classes=CLASSES, folds=folds))
    routing_columns = [f'{name}:{layer}' for name in ('entropy', 'dominant_switch', 'topk_turnover') for layer in LAYERS]
    routing_summary = bootstrap_panel(routing_rows, routing_columns, panel='lfm25:native-routing-dynamics',
                                      draws=10000, resample_key='native-family')
    publish(root/'lfm25/DYNAMICS.json', {**trajectories(sentences), 'routing_inference': routing_summary,
        'routing_observations': routing_rows, 'native_input_token_alignment': True,
        'adjacent_reasoning_only': True, 'unknown_sentence_gaps_preserved': True})
    publish(root/'lfm25/NATIVE_PROFILE.json', {'layers': list(LAYERS), 'experts': EXPERTS, 'native_k': K,
        'reasoning_tokens': tokens[:, 7].sum(), 'labelled_tokens': tokens[:, 8].sum(),
        'class_tokens': tokens[:, :7].sum(0), 'source_receipts': len(sources)})
    require(sum(label_counts.values()) == len(sampled), 'sampled sentences without a terminal label')
    return publish(root/'lfm25/DISCOVERY_INPUTS.json', {'status': 'complete', 'sources': sources,
        'label_states': label_counts, 'judge_inputs': verified(root/'lfm25/ANNOTATION_TASKS.json').get('judge_inputs'),
        'counts': str(count_path), 'counts_sha256': file_sha(count_path),
        'sentences': str(sentence_path), 'sentences_sha256': file_sha(sentence_path)})


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m routing.lfm', description='LFM2.5 analyses that need no GPU.')
    parser.add_argument('stage', choices=('materialize', 'derive', 'readout', 'dynamics'))
    stage = parser.parse_args(argv).stage
    study = store.binding()
    if stage == 'materialize':
        from labeling.lfm_inputs import materialize
        result = materialize(study)
    elif stage == 'derive':
        from steering.nominations import derive_model
        from steering.pulse import plan
        if not (store.STUDY/'lfm25/DISCOVERY_INPUTS.json').exists():
            discover(study)
        derive_model(study, 'lfm25')
        result = {'plan': plan(['lfm25'])}
    elif stage == 'readout':
        from routing.lfm_readout import lfm
        value = lfm(study)
        result = {'sha256': value['sha256'], **{mode: {k: r[k] for k in ('routing', 'routing_excess_over_lexical_null')}
                                                 for mode, r in value['models']['lfm25'].items()}}
    else:
        from routing.dynamics import dynamics_summary
        result = {'sha256': dynamics_summary(study)['sha256']}
    print(json.dumps({'stage': stage, **result}, default=str), flush=True)


if __name__ == '__main__':
    main()
