"""Which experts each captured attempt uses while it is in each reasoning class (CPU job).

For one model of a capture release (capture.py):

1. every reasoning sentence of every attempt gets a class from the lexical readout v1 (TF-IDF,
   the classifier that labels every intervention continuation in this study; out-of-fold by family);
2. every reasoning token inherits the class of the sentence that owns it;
3. per attempt, the native top-k ids of the tokens of each class are counted per (layer, expert).

Output (FAST, beside the release's stores): ``derived/<model>/USAGE_CLASS.npz`` with

- ``counts``  (attempts, 9, layers, experts): top-k selections on tokens of class c (0-6), on all
  reasoning tokens (7) and on labelled tokens only (8);
- ``tokens``  (attempts, 9): the matching token counts;
- ``weights`` (attempts, layers, experts): native gate weight summed over all reasoning tokens;

and ``USAGE_CLASS.json`` with one record per attempt (question, attempt, outcome, length, answer).
An attempt without a finished, extracted result keeps zero counts and ``usable: false``.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np

from store import CLASSES, binding, file_sha, publish, require, verified

ALL_REASONING, ALL_LABELLED = 7, 8


def derived_folder(release: Path, model: str) -> Path:
    """Where a capture release keeps its derived arrays: next to its stores (FAST)."""
    return Path(os.path.realpath(release/'stores')).parent/'derived'/model


def labelled_sentences(release: Path, model: str):
    """The extracted reasoning sentences of one model with their lexical class (lexical_label)."""
    import pandas as pd
    from routing.lexical import LEXICAL, predict
    sentences = pd.read_parquet(release/model/'extraction'/'sentences.parquet')
    if sentences.empty:
        return sentences.assign(lexical_label=pd.Series(dtype=str))
    frozen = verified(release/'RELEASE.json')['classifier']
    return pd.concat([predict(sentences.iloc[start:start+10000], LEXICAL/'readout-v1', frozen=frozen)
                      for start in range(0, len(sentences), 10000)], ignore_index=True)


def token_classes(length: int, spans: list[tuple[int, int, int]]) -> np.ndarray:
    """Class index of every captured reasoning token (-1 for tokens no labelled sentence owns).

    Args:
        length: Captured reasoning tokens of the attempt (rows of its AUDIT arrays).
        spans: (token_start, token_end, class index) of its labelled sentences, generated-token positions.
    """
    owner = np.full(length, -1, dtype=np.int8)
    for start, end, label in spans:
        owner[max(0, start):min(end, length)] = label
    return owner


def attempt_counts(task: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """counts (9, layers, experts), tokens (9,) and weights (layers, experts) of one attempt."""
    path, layers, experts, spans = task
    with np.load(path) as saved:
        ids = [saved[f'layer_{layer}_native_ids'] for layer in layers]
        gates = [saved[f'layer_{layer}_native_weights'].astype(np.float32) for layer in layers]
    length = len(ids[0])
    require(all(len(x) == length for x in ids), f'layers captured different token counts: {path}')
    owner = token_classes(length, spans)
    masks = [owner == c for c in range(7)] + [np.ones(length, bool), owner >= 0]
    counts = np.zeros((9, len(layers), experts), dtype=np.float32)
    weights = np.zeros((len(layers), experts), dtype=np.float32)
    for i, (layer_ids, layer_gates) in enumerate(zip(ids, gates)):
        for c, mask in enumerate(masks):
            counts[c, i] = np.bincount(layer_ids[mask].ravel(), minlength=experts)
        np.add.at(weights[i], layer_ids.ravel().astype(np.int64), layer_gates.ravel())
    return counts, np.array([m.sum() for m in masks], dtype=np.int64), weights


def usage(release: Path, model: str, workers: int = 8) -> dict:
    """Write USAGE_CLASS.npz/.json for one model of a capture release."""
    import pandas as pd
    from steering.whole_reasoning import manifest_path
    value = verified(manifest_path(model, 'main', release))
    require(value.get('design') == 'outcome-capture', 'not a capture manifest')
    profile_layers = value['records'][0]['capture_layers']
    experts = int(binding()['models'][model]['experts'])
    outcomes = pd.read_parquet(release/model/'extraction'/'outcomes.parquet')
    by_uid = {r['uid']: r for r in outcomes.to_dict('records')}
    sentences = labelled_sentences(release, model)
    index = {c: i for i, c in enumerate(CLASSES)}
    spans = {uid: [(int(s.token_start), int(s.token_end), index[s.lexical_label]) for s in group.itertuples()]
             for uid, group in sentences.groupby('uid')} if len(sentences) else {}
    units = release/'stores'/model/'main'/'units'
    records, tasks = [], []
    for row in value['records']:
        outcome = by_uid.get(row['uid'])
        audit = units/row['uid']/'AUDIT.npz'
        usable = outcome is not None and audit.exists()
        finish = outcome['finish_reason'] if outcome is not None else 'missing'
        records.append({
            'uid': row['uid'], 'question': row['question'], 'attempt': row['attempt'], 'family': row['family'],
            'benchmark': row['benchmark'], 'fold': row['fold'], 'usable': usable, 'finish_reason': finish,
            # A capped attempt has no known outcome: it never enters a right/wrong comparison.
            'strict': None if not usable or finish != 'stop' or pd.isna(outcome['strict_correct'])
                      else bool(outcome['strict_correct']),
            'reasoning_tokens': int(outcome['reasoning_tokens']) if usable else None,
            'model_answer': outcome.get('model_answer') if usable else None,
            'audit': str(audit) if usable else None, 'audit_sha256': file_sha(audit) if usable else None})
        if usable:
            tasks.append((len(records)-1, (str(audit), profile_layers, experts, spans.get(row['uid'], []))))
    require(tasks, f'no usable capture for {model}')
    counts = np.zeros((len(records), 9, len(profile_layers), experts), dtype=np.float32)
    tokens = np.zeros((len(records), 9), dtype=np.int64)
    weights = np.zeros((len(records), len(profile_layers), experts), dtype=np.float32)
    with ProcessPoolExecutor(workers) as pool:
        for (i, _), (c, t, w) in zip(tasks, pool.map(attempt_counts, [task for _, task in tasks], chunksize=4)):
            counts[i], tokens[i], weights[i] = c, t, w
    for i, record in enumerate(records):
        record['class_tokens'] = tokens[i].tolist()
    folder = derived_folder(release, model)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    np.savez(folder/'USAGE_CLASS.npz', counts=counts, tokens=tokens, weights=weights)
    sentences.to_parquet(folder/'sentences_labelled.parquet', index=False)
    return publish(folder/'USAGE_CLASS.json', {
        'model': model, 'release': str(release), 'manifest_sha256': value['sha256'], 'layers': profile_layers,
        'experts': experts, 'classes': list(CLASSES)+['ALL_REASONING', 'ALL_LABELLED'],
        'arrays_sha256': file_sha(folder/'USAGE_CLASS.npz'), 'code_sha256': file_sha(__file__),
        'sentences_sha256': file_sha(folder/'sentences_labelled.parquet'), 'attempts': records})
