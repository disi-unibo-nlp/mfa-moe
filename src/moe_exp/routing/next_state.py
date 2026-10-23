"""Next reasoning state from the current sentence's routing, for all eight models.

For every adjacent pair of judge-labelled sentences (s, s+1) in a model's native traces, three nested
classifiers predict the reasoning state of s+1 from sentence s:

  markov   the current state only: the transition matrix of Fig. 6, refitted on each training fold;
  text     the current state, the token bag of s, its absolute-position bin and its log length;
  routing  text plus the routing of every token of s (per-layer top-1 expert fractions as PCA scores,
           and the path surprise of the present-state readout).

The routing gain is the held-out cross-entropy of text minus routing, in nats per pair with questions
weighted equally. It is computed by the present-state readout machinery unchanged (readout_fit,
readout_analysis): the same family folds, inner penalty search, five token-permutation nulls and
family-cluster intervals, with the next state as the target. Every sentence of a pair counts,
whatever its length, and relative progress is left out because it needs the completed trace length.
The population is the 522 questions of the present-state readout, for all eight models.

    scripts/study/cpu.sh moe_exp.routing.next_state run --model gpt      (one CPU job per model)
    scripts/study/cpu.sh moe_exp.routing.next_state collect              (after all eight)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from moe_exp.io import CLASSES, MODELS, digest, file_sha, load, publish, require, verified
from moe_exp.routing import readout_analysis, readout_fit as sr
from moe_exp.routing.dp import common as cm, s3_readout as historical
from moe_exp.routing.lfm_readout import HISTORICAL
from moe_exp.steering import pulse as driver

# The historical present-state readout of the seven campaign-v3 models: its frozen population and the
# token caches its loader reads (moe_exp/routing/readout.py binds the same files).
FROZEN = HISTORICAL/'FROZEN.readout.v3.json'
TOKEN_CACHE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/depth-paths/results/s3/_tok')
LAYERS_EXPERTS_LFM = (38, 64)
CODE = ('next_state.py', 'readout_fit.py', 'readout_analysis.py')


def campaign_model(model: str) -> sr.ModelData:
    """Labelled sentences and token routing of one campaign-v3 model, as the historical readout loads them.

    The historical loader reads cohort A; the readout kept its dev and tune questions (522) and mapped
    each kept attempt to its token cache by its original row.
    """
    split = cm.load_split()
    attempts = cm.load_attempts(model, 'A')
    keep = np.array([split[q] in ('dev', 'tune') for q in attempts['question']])
    table, original_rows = attempts.loc[keep].reset_index(drop=True), np.flatnonzero(keep)
    original_load = cm.load_attempts
    cm.load_attempts = lambda m, c: table if (m, c) == (model, 'A') else original_load(m, c)
    historical.tok_path = lambda m, i: TOKEN_CACHE/m/f'{int(original_rows[i]):05d}.npz'
    try:
        md = historical.load_model(model)
    finally:
        cm.load_attempts = original_load
    require(set(md.groups) == set(load(FROZEN)['ids'][model]), f'{model}: readout population differs')
    return sr.ModelData(md.model, md.L, md.E, md.sent, md.y, md.groups, md.w, md.base, md.bag, md.top1,
                        md.tok_sent, md.tokid, md.tok_abs, md.adj, md.n_before_filter)


def lfm_model(questions: set) -> sr.ModelData:
    """LFM2.5's labelled discovery sentences and their replayed routing, as readouts.lfm loads them."""
    binding = driver.binding()
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    inputs = verified(root/'lfm25/DISCOVERY_INPUTS.json')
    require(file_sha(Path(inputs['sentences'])) == inputs['sentences_sha256'], 'LFM derived sentences changed')
    sent = pd.read_parquet(inputs['sentences'])
    sent = sent[sent.question.isin(questions)].sort_values(['attempt_id', 'tok_start']).reset_index(drop=True)
    require(set(sent.question) == questions, 'LFM lacks readout questions')
    top, owner, token_ids, token_bins, adjacency = [], [], [], [], []
    for attempt, group in sent.groupby('attempt_id', sort=True):
        path = bulk/'lfm25/replay'/(attempt+'.npz')
        require(file_sha(path) == verified(path.with_suffix('.json'))['file_sha256'], 'LFM replay changed')
        with np.load(path, allow_pickle=False) as z:
            ids, weights, tokens = z['ids'], z['weights'], z['token_ids']
        dominant = np.take_along_axis(ids, weights.argmax(-1)[..., None], axis=-1)[..., 0]
        for r in group.itertuples():
            positions = np.asarray(r.token_indices, int)
            top.append(dominant[:, positions])
            owner.extend([r.Index]*len(positions))
            token_ids.extend(tokens[positions])
            token_bins.extend(sr.abs_bin(np.asarray(r.token_ranks)))
            adjacency.extend([False]+(np.diff(positions) == 1).tolist())
    owner, token_ids = np.asarray(owner, int), np.asarray(token_ids, np.int64)
    require(np.array_equal(np.bincount(owner, minlength=len(sent)), sent.n_tokens), 'LFM token ownership differs')
    bag = sparse.coo_matrix((np.ones(len(token_ids)), (owner, sr.bucket_of(token_ids))),
                            shape=(len(sent), sr.N_BUCKETS)).tocsr()
    sent['abs_bin'] = sr.abs_bin(sent.tok_start.to_numpy())
    base = np.column_stack([(sent.abs_bin.to_numpy() == b).astype(float) for b in range(sr.N_ABS)]
                           + [np.log(sent.n_tokens)])
    groups = sent.question.to_numpy()
    layers, experts = LAYERS_EXPERTS_LFM
    return sr.ModelData('lfm25', layers, experts, sent, sent.cls.to_numpy(int), groups, sr.question_weights(groups),
                        base, bag, np.concatenate(top, axis=1), owner, token_ids, np.asarray(token_bins),
                        np.asarray(adjacency), len(sent))


def adjacent_pairs(md: sr.ModelData, question_family: dict, seed_index: int) -> sr.ModelData:
    """One row per labelled sentence whose successor in the trace is labelled too; the target is the successor.

    Features stay those of the source sentence: its token bag and routing, its absolute-position bin, its
    log length and its own state (one-hot). The relative-progress columns of the readout are dropped.
    """
    sent = md.sent.reset_index(drop=True)
    same_attempt = sent.attempt_id.to_numpy()[1:] == sent.attempt_id.to_numpy()[:-1]
    consecutive = sent.sentence_index.to_numpy()[1:] == sent.sentence_index.to_numpy()[:-1]+1
    source = np.flatnonzero(same_attempt & consecutive)
    successor = source+1
    new_row = np.full(md.n, -1)
    new_row[source] = np.arange(len(source))
    tokens = new_row[md.tok_sent] >= 0
    position = md.base[:, :sr.N_ABS]
    log_length = md.base[:, -1:]
    current = np.eye(len(CLASSES))[md.y]
    base = np.column_stack([position, log_length, current])[source]
    pairs = sent.iloc[source].reset_index(drop=True).assign(next_cls=md.y[successor])
    groups = md.groups[source]
    return sr.ModelData(md.model, md.L, md.E, pairs, md.y[successor], groups, sr.question_weights(groups), base,
                        md.bag[source], md.top1[:, tokens], new_row[md.tok_sent[tokens]], md.tokid[tokens],
                        md.tok_abs[tokens], md.adj[tokens], md.n, seed_index=seed_index,
                        question_family=question_family)


def markov_losses(pairs: sr.ModelData) -> np.ndarray:
    """Held-out cross-entropy of the current-state transition matrix (add-one smoothed, question-weighted)."""
    current = pairs.base[:, -len(CLASSES):].argmax(1)
    folds = sr.outer_folds(pairs.groups, pairs.question_family)
    losses = np.empty(pairs.n)
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        counts = np.ones((len(CLASSES), len(CLASSES)))
        np.add.at(counts, (current[train], pairs.y[train]), pairs.w[train])
        log_transition = np.log(counts/counts.sum(1, keepdims=True))
        losses[test] = -log_transition[current[test], pairs.y[test]]
    return losses


def run(model: str) -> dict:
    """Fit and publish one model's next-state result.

    Args:
        model: Study model key.

    Returns:
        The published report/final-routing-20261015-r1/<model>/NEXT_STATE.json receipt.
    """
    binding = driver.binding()
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    target = root/model/'NEXT_STATE.json'
    require(not target.exists(), f'{target} exists')
    question_family = load(Path(binding['source_campaign'])/'FOLDS.json')['question_families']
    questions = set(load(FROZEN)['ids']['gpt'])
    md = lfm_model(questions) if model == 'lfm25' else campaign_model(model)
    pairs = adjacent_pairs(md, question_family, MODELS.index(model))
    provenance = {'code': {name: file_sha(Path(__file__).with_name(name)) for name in CODE},
                  'frozen_readout_sha256': file_sha(FROZEN), 'binding_sha256': binding['sha256']}
    family = {'sha256': digest(['next-state', model, provenance])}
    family_ids = np.asarray([question_family[q] for q in pairs.groups])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
        result = readout_analysis.analyse(sr, pairs, None, 'token_identity', workers, family, family_ids,
                                          bulk/model/'next-state')
    with np.load(bulk/model/'next-state/readout'/model/'token_identity/losses.npz') as z:
        text, routing = z['base'], z['aug']
    markov = markov_losses(pairs)
    weight = pairs.w/pairs.w.sum()
    return publish(target, {
        'model': model, 'status': result['status'], 'questions': len(questions), 'pairs': pairs.n,
        'labelled_sentences': md.n, 'source_tokens': int(len(pairs.tokid)),
        'cross_entropy': {'markov': float(weight @ markov), 'text': float(weight @ text),
                          'routing': float(weight @ routing)},
        'text_gain_over_markov': -sr.question_bootstrap(text-markov, pairs.groups, pairs.w, 5000)['gain'],
        'routing_gain': result['routing'], 'routing_excess_over_null': result['routing_excess_over_lexical_null'],
        'readout_analysis': result, 'provenance': provenance,
        'definition': 'gain = held-out cross-entropy of the smaller model minus the larger one (nats per adjacent '
                      'pair, questions weighted equally); routing_gain.P is routing minus text, so negative '
                      'values favour routing'})


def collect() -> dict:
    """Join the eight per-model receipts into driver/NEXT_STATE.json for the paper exporter."""
    root = Path(driver.binding()['root'])
    receipts = {m: verified(root/m/'NEXT_STATE.json') for m in MODELS}
    return publish(root/'driver/NEXT_STATE.json', {'models': list(MODELS), 'results': receipts,
                                                   'receipt_sha256': {m: r['sha256'] for m, r in receipts.items()}})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog='python -m moe_exp.routing.next_state')
    parser.add_argument('command', choices=('run', 'collect'))
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args(argv)
    if args.command == 'run':
        require(args.model, 'run needs --model')
        print(run(args.model)['sha256'], flush=True)
    else:
        print(collect()['sha256'], flush=True)


if __name__ == '__main__':
    main()
