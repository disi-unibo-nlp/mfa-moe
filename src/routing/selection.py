"""Expert selection from saved routing counts: held-out coverage, inclusion statistics and nominations.

Vendored verbatim: `expert_memory_study.selection` (outcome-free family folds, ranking, held-out coverage),
`research_campaign_v3_r6.design` (inclusion statistics, nominations, prefix roster) and
`research_campaign_v3_r6.inputs.counts_for`.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from receipts import CLASSES, SEED, digest, require, seal


def family_folds(families, n=5):
    ordered = sorted(set(families), key=lambda f: (digest([SEED, "fold", f]), f))
    require(len(ordered) >= n, "insufficient independent families")
    return {f: i % n for i, f in enumerate(ordered)}


def validate_counts(counts, tokens, k):
    counts, tokens = np.asarray(counts), np.asarray(tokens)
    require(counts.ndim == 4 and tokens.shape == counts.shape[:2], "archive shape differs")
    require(np.issubdtype(counts.dtype, np.integer) and np.issubdtype(tokens.dtype, np.integer),
            "counts/tokens must be integer")
    require(bool((counts >= 0).all() and (tokens >= 0).all()), "negative count")
    require(bool((counts.sum(-1, dtype=np.uint64) == tokens[:, :, None] * k).all()),
            "selected slots differ from tokens times native k")


def scores(counts, families, train):
    """Counts are [attempt,class,layer,expert]; duplicates have total weight one."""
    counts = np.asarray(counts)
    families = np.asarray(families)
    train = np.asarray(train, dtype=bool)
    require(len(families) == len(counts) == len(train) and train.any(), "empty/misaligned training fold")
    grouped = np.stack([counts[train & (families == f)].mean(0, dtype=np.float64)
                        for f in sorted(set(families[train]))])
    mass = grouped.sum(-1)
    shares = np.divide(grouped, mass[..., None], out=np.zeros_like(grouped), where=mass[..., None] > 0)
    support = (mass > 0).sum(0)
    require(bool((support > 0).all()), "class/layer unsupported in training fold")
    balanced = (shares.sum(0) / support[..., None]).mean(0)
    pooled = grouped.sum((0, 1))
    frequency = pooled / pooled.sum(-1, keepdims=True)
    return balanced, frequency, support


def keep_set(score, keep, groups=1, *, model=None, layers=None, random=False):
    score = np.asarray(score)
    require(score.ndim == 2 and np.isfinite(score).all(), "invalid expert scores")
    L, E = score.shape
    require(E % groups == keep % groups == 0 and 0 < keep <= E, "nonintegral group quota")
    layers = list(range(L)) if layers is None else list(layers)
    require(len(layers) == L and len(set(layers)) == L, "layer map differs")
    result = []
    for i, layer in enumerate(layers):
        ids = []
        for group in range(groups):
            candidates = list(range(group * (E // groups), (group + 1) * (E // groups)))
            if random:
                require(model is not None, "random set needs model binding")
                candidates.sort(key=lambda e: (digest([SEED, model, layer, group, e]), e))
            else:
                candidates.sort(key=lambda e: (-score[i, e], e))
            ids += candidates[:keep // groups]
        result.append(sorted(ids))
    return np.asarray(result, dtype=np.int64)


def coverage(counts, families, mask):
    """Mean within-family class slot coverage, with explicit class support."""
    counts = np.asarray(counts, dtype=np.float64)
    grouped = np.stack([counts[np.asarray(families) == f].mean(0) for f in sorted(set(families))])
    total = grouped.sum(-1)
    kept = np.take_along_axis(grouped, np.asarray(mask)[None, None], axis=-1).sum(-1)
    support = (total > 0).sum(0)
    shares = np.divide(kept, total, out=np.zeros_like(kept), where=total > 0)
    result = np.divide(shares.sum(0), support, out=np.zeros_like(total[0]), where=support > 0)
    class_coverage = [float(row[sup > 0].mean()) if (sup > 0).any() else None
                      for row, sup in zip(result, support)]
    return {"coverage": [[float(x) if s else None for x, s in zip(row, sup)]
                         for row, sup in zip(result, support)],
            "family_support": support.tolist(),
            "class_mean_layer_coverage": class_coverage,
            "minimum_class_coverage": min((x for x in class_coverage if x is not None), default=None),
            "minimum_class_layer_coverage": float(result[support > 0].min()) if (support > 0).any() else None,
            "minimum_complete": bool((support > 0).all())}


def overlap(a, b):
    return [len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)]


def analyze(counts, families, *, keep, groups, layers, model, classes, folds=None):
    folds = folds or family_folds(families)
    require(set(families) <= set(folds), "family missing from common folds")
    random = keep_set(np.zeros(counts.shape[2:]), keep, groups, model=model, layers=layers, random=True)
    rows, fitted = [], []
    for fold in range(5):
        test = np.array([folds[f] == fold for f in families])
        require(test.any() and (~test).any(), "empty fold")
        b, f, support = scores(counts, families, ~test)
        masks = {"state_balanced": keep_set(b, keep, groups),
                 "labelled_frequency": keep_set(f, keep, groups), "random": random}
        fitted.append(masks)
        rows.append({"fold": fold, "train_families": len(set(np.asarray(families)[~test])),
                     "test_families": len(set(np.asarray(families)[test])), "training_support": support.tolist(),
                     "masks": {a: m.tolist() for a, m in masks.items()},
                     "held_out": {a: coverage(counts[test], np.asarray(families)[test], m) for a, m in masks.items()},
                     "balanced_frequency_jaccard": overlap(masks["state_balanced"], masks["labelled_frequency"])})
    b, f, support = scores(counts, families, np.ones(len(families), dtype=bool))
    final = {"state_balanced": keep_set(b, keep, groups), "labelled_frequency": keep_set(f, keep, groups), "random": random}
    stability = {a: [{"folds": [i, j], "layer_jaccard": overlap(fitted[i][a], fitted[j][a])}
                     for i in range(5) for j in range(i + 1, 5)] for a in final}
    return seal({"schema": "expert-selection-v1", "model": model, "classes": list(classes),
                 "layers": list(map(int, layers)), "groups": groups, "experts": counts.shape[-1],
                 "keep": keep, "folds_sha256": digest(folds), "folds": rows,
                 "stability": stability, "final_masks": {a: m.tolist() for a, m in final.items()},
                 "full_training_support": support.tolist(), "outcomes_used": False})


def family_weights(rows):
    """One unit per family occurrence, equal benchmarks, arbitrary repeat counts."""
    cells = Counter((r["benchmark"], r["family"]) for r in rows)
    families = defaultdict(set)
    for b, f in cells:
        families[b].add(f)
    return np.array([1 / (len(families) * len(families[r["benchmark"]]) *
                          cells[r["benchmark"], r["family"]]) for r in rows])


def prefix_roster(sentences, selected, folds, model):
    """No replacement when the assigned progress third has no usable prefix."""
    wanted = {(r["benchmark"], r["family"]) for r in selected}
    by_cell = defaultdict(list)
    for s in sentences:
        if (s["benchmark"], s["family"]) in wanted:
            by_cell[s["benchmark"], s["family"]].append(s)
    cells = sorted(wanted, key=lambda c: (c[0], folds[c[1]], digest([SEED, "prefix", *c])))
    block = Counter()
    result = []
    for b, f in cells:
        key = (b, folds[f])
        third = block[key] % 3
        block[key] += 1
        compatible = [s for s in by_cell[b, f] if s.get("compatible",True) and min(2, int(s["progress"] * 3)) == third]
        chosen = min(compatible, key=lambda s: (digest([SEED, "boundary", model,
                     s["attempt_id"], s["sentence_index"]]), s["sentence_index"]), default=None)
        result.append({"model": model, "benchmark": b, "family": f, "fold": folds[f],
                       "progress_third": third, "prefix": chosen,
                       "status": "enrolled" if chosen else "upstream_unavailable"})
    # Balanced over benchmark x fold x assigned progress position, before any outcomes.
    blocks = defaultdict(list)
    for r in result:
        blocks[r["benchmark"], r["fold"], r["progress_third"]].append(r)
    for key, members in blocks.items():
        members.sort(key=lambda r: digest([SEED, "sweep", *key, r["family"]]))
        for i, r in enumerate(members):
            r["configuration_indices"] = [0, 1 + i % 8]
    return result


def weighted_quantile(values, weights, quantiles):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return [None] * len(quantiles)
    order = np.argsort(v[valid], kind="stable")
    v, w = v[valid][order], w[valid][order]
    c = np.cumsum(w) / w.sum()
    return [float(v[min(len(v) - 1, np.searchsorted(c, q))]) for q in quantiles]


def inclusion_statistics(counts, tokens, rows, train):
    """Per-family per-benchmark token inclusion; unsupported cells are NaN."""
    counts, tokens = np.asarray(counts), np.asarray(tokens)
    require(counts.shape[:2] == tokens.shape and len(rows) == len(counts), "count alignment")
    C, L, E = counts.shape[1:]
    by_cell = defaultdict(list)
    for i, r in enumerate(rows):
        if train[i]:
            by_cell[r["benchmark"], r["family"]].append(i)
    by_benchmark = defaultdict(list)
    for (b, _), ids in by_cell.items():
        n = tokens[ids].sum(0, dtype=np.float64)
        c = counts[ids].sum(0, dtype=np.float64)
        v = np.divide(c, n[:, None, None], out=np.full((C, L, E), np.nan),
                      where=n[:, None, None] > 0)
        by_benchmark[b].append(v)
    require(by_benchmark, "no discovery families")
    support = np.zeros((C, L, E), int)
    totals, weights = np.zeros((C, L, E)), np.zeros((C, L, E))
    for values in by_benchmark.values():
        a = np.stack(values)
        valid = np.isfinite(a)
        sup = valid.sum(0)
        mean = np.divide(np.nansum(a, axis=0), sup, out=np.zeros((C, L, E)), where=sup > 0)
        support += sup
        totals += mean * (sup > 0)
        weights += sup > 0
    rate = np.divide(totals, weights, out=np.full((C, L, E), np.nan), where=weights > 0)
    return rate, support


def enriched_set(excess, fraction):
    order = sorted((i for i, v in enumerate(excess) if np.isfinite(v) and v > 0),
                   key=lambda e: (-excess[e], e))
    if not order:
        return []
    threshold = sum(excess[e] for e in order) * fraction
    total, selected = 0., []
    for e in order:
        selected.append(e)
        total += excess[e]
        if total >= threshold:
            break
    return sorted(selected)


def nominate(excess, stability, layers, k):
    require(len(excess) == len(layers) == len(stability), "layer nomination alignment")
    result = []
    for indices in np.array_split(np.arange(len(layers)), 3):
        candidates = []
        for i in indices:
            ids = enriched_set(excess[i], .5)
            if ids:
                allocation = sum(excess[i, e] for e in ids) / k
                candidates.append((-allocation, -float(stability[i]), int(layers[i]), int(i)))
        best = min(candidates, default=None)
        result.append(None if best is None else best[3])
    return result


def random_control(ids, frequency, identity):
    """Match size and frequency-rank quintile counts, with recorded priorities."""
    E = len(frequency)
    order = sorted(range(E), key=lambda e: (float(frequency[e]), e))
    strata = {e: min(4, i * 5 // E) for i, e in enumerate(order)}
    demand = Counter(strata[e] for e in ids)
    chosen = []
    for s, count in sorted(demand.items()):
        candidates = [e for e in range(E) if strata[e] == s]
        candidates.sort(key=lambda e: (digest([SEED, "control", identity, e]), e))
        chosen += candidates[:count]
    return sorted(chosen)


def counts_for(campaign, model, rows):
    import pandas as pd
    path = campaign["saved_inputs"][model]["counts"]
    with np.load(path, allow_pickle=False) as archive:
        ids, counts, tokens = (archive[k] for k in ("attempt_id", "counts", "tokens"))
        require(list(archive["classes"]) == list(CLASSES) + ["ALL_REASONING", "ALL_LABELLED"], "class order mismatch")
    index = pd.Index(ids).get_indexer([r["attempt_id"] for r in rows])
    require((index >= 0).all() and len(index) == len(ids), "archive row join failed")
    counts, tokens = counts[index], tokens[index]
    validate_counts(counts, tokens, campaign["models"][model]["native_k"])
    return counts, tokens


# The LFM2.5 counterpart of the saved-data derive (former study/lfm.py).
def derive(binding):
    from collections import defaultdict
    from pathlib import Path
    from store import CLASSES, Units, digest, file_sha, load, publish, require, verified  # study primitives, not the campaign layer above
    import numpy as np
    import pandas as pd
    from schemas import TraceRecord
    from store import read_source
    from capture.dynamics.classes import saved_layout
    from routing.dynamics import trajectories
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
    from stats import bootstrap_panel
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
