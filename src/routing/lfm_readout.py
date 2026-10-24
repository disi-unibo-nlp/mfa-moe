"""Present-state readout of LFM2.5: does routing tell the class of a sentence beyond its text?

It runs the same fits as the seven-model readout (readout_fit, readout_analysis) on LFM's replayed routing and
its 522 readout questions, twice: token identity only, and with the judge's previous/current/next sentence text.
The population is the one frozen in PUBLISHED_RUN/FROZEN.readout.v3.json (the seven-model run's inputs).

    python -m routing.lfm25_analyses readout --out DIR      (about 1.5 h on 4 CPUs); result: DIR/READOUT_RESULTS.json

DIR must be new. The receipt published in the study folder (lfm25/READOUT_RESULTS.json, which the paper reads) is
never rewritten: a run must have the same inputs as the ones it records, and only its code hashes may differ.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from routing import readout_analysis, readout_fit as sr
from routing.readout_analysis import PUBLISHED_RUN
from routing.readout_text import hash_counts
from store import check_files, digest, file_sha, load, publish, require, verified


def provenance_of(binding):
    """What the readout is a function of: the code, the derived sentences, the sampled units and the folds."""
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    inputs = verified(root/'lfm25/DISCOVERY_INPUTS.json')
    require(file_sha(Path(inputs['sentences'])) == inputs['sentences_sha256'], 'LFM derived sentences changed')
    return {'code': {name: file_sha(Path(__file__).with_name(name)) for name in
                     ('lfm_readout.py', 'readout_fit.py', 'readout_analysis.py', 'readout_text.py')},
            'inputs_sha256': inputs['sha256'], 'sentences_sha256': inputs['sentences_sha256'],
            'sampled_sentences_sha256': file_sha(bulk/'lfm25/SAMPLED_SENTENCES.json'),
            'folds_sha256': file_sha(Path(binding['source_campaign'])/'FOLDS.json'), 'seed_index': 7}


def check_published(binding, provenance):
    """A receipt already published in the study folder was fitted on these inputs (its code hashes may differ)."""
    published = Path(binding['root'])/'lfm25/READOUT_RESULTS.json'
    if published.exists():
        inputs = lambda value: {k: v for k, v in value.items() if k != 'code'}
        require(inputs(verified(published)['provenance']) == inputs(provenance),
                'the inputs differ from the published LFM readout receipt')


def lfm(binding, out):
    """Fit the LFM2.5 readout into the new folder `out` and return its sealed READOUT_RESULTS.json."""
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    out = Path(out)
    target = out/'READOUT_RESULTS.json'
    require(not target.exists(), f'{out} already holds LFM readout results; use a new --out')
    inputs = verified(root/'lfm25/DISCOVERY_INPUTS.json')
    folds_path = Path(binding['source_campaign'])/'FOLDS.json'
    sampled_path = bulk/'lfm25/SAMPLED_SENTENCES.json'
    # The checkpoints are keyed by code as well as data, so a changed fit never reuses earlier checkpoints.
    provenance = provenance_of(binding)
    check_published(binding, provenance)
    check_files(binding['readout_files'])
    frozen = load(PUBLISHED_RUN/'FROZEN.readout.v3.json')
    questions = set(frozen['ids']['gpt'])
    require(len(questions) == 522, 'readout population differs from the published run')
    sent = pd.read_parquet(inputs['sentences'])
    sent = sent[sent.question.isin(questions)].copy().sort_values(['attempt_id','tok_start']).reset_index(drop=True)
    require(set(sent.question) == questions, 'LFM common-population readout has unavailable questions')
    price=verified(root/'CPU_PRICE.json')
    require(len(sent)<=price['limits']['sentences'] and sent.n_tokens.sum()<=price['limits']['tokens'],
            'actual LFM readout exceeds the measured CPU profile; no unpriced analysis')
    row_of = {s:i for i,s in enumerate(sorted(sent.attempt_id.unique()))}
    sent['row'] = sent.attempt_id.map(row_of)
    top, tsent, tokid, tabs, adjacency = [], [], [], [], []
    # Context slots read every sampled unit of the attempt, unknown labels included, as
    # `dynamics_routing.d3a_pairs.load_unit_texts` does; only sentences without any sampled unit stay '<missing>'.
    attempts = set(sent.attempt_id)
    text_lookup = {(r['attempt_id'], r['sentence_index']): r['text'] for r in verified(sampled_path)['records']
                   if r['attempt_id'] in attempts}
    require(all(text_lookup.get((r.attempt_id, r.sentence_index)) == r.text for r in sent.itertuples()),
            'readout sentence text differs from the sampled unit')
    texts = []
    for attempt, group in sent.groupby('attempt_id', sort=True):
        path = bulk/'lfm25/replay'/(attempt+'.npz')
        proof = verified(path.with_suffix('.json'))
        require(file_sha(path) == proof['file_sha256'], 'readout replay changed')
        with np.load(path, allow_pickle=False) as z:
            ids, weights, tokens = z['ids'], z['weights'], z['token_ids']
        dominant = np.take_along_axis(ids, weights.argmax(-1)[...,None], axis=-1)[...,0]
        for r in group.itertuples():
            positions = np.asarray(r.token_indices, int)
            top.append(dominant[:,positions]); tsent.extend([r.Index]*len(positions)); tokid.extend(tokens[positions])
            tabs.extend(sr.abs_bin(np.asarray(r.token_ranks))); adjacency.extend([False]+(np.diff(positions)==1).tolist())
    for r in sent.itertuples():
        texts.append('\n'.join(name+' '+text_lookup.get((r.attempt_id,r.sentence_index+d),'<missing>')
                               for name,d in [('previous',-1),('current',0),('future',1)]))
    tsent, tokid = np.asarray(tsent,int), np.asarray(tokid,np.int64)
    require(np.array_equal(np.bincount(tsent,minlength=len(sent)),sent.n_tokens), 'readout token ownership differs')
    bag = sparse.coo_matrix((np.ones(len(tokid)),(tsent,sr.bucket_of(tokid))),shape=(len(sent),sr.N_BUCKETS)).tocsr()
    ab = sr.abs_bin(sent.tok_start.to_numpy())
    sent['abs_bin'] = ab
    rb = np.minimum((sent.rel_pos.to_numpy()*sr.N_REL).astype(int),sr.N_REL-1)
    base = np.column_stack([(ab==b).astype(float) for b in range(sr.N_ABS)] +
                           [(rb==b).astype(float) for b in range(sr.N_REL)] + [np.log(sent.n_tokens)])
    groups = sent.question.to_numpy()
    md = sr.ModelData('lfm25',38,64,sent,sent.cls.to_numpy(int),groups,sr.question_weights(groups),base,bag,
        np.concatenate(top,axis=1),tsent,tokid,np.asarray(tabs),np.asarray(adjacency),len(sent), seed_index=7)
    folds = load(folds_path)
    qf = folds['question_families']
    md.question_family = qf
    family = {'sha256':digest([frozen['sha256'],binding['sha256'],provenance])}
    counts = hash_counts(texts)
    results = {}
    from threadpoolctl import threadpool_limits
    for mode in ('token_identity','judge_context'):
        with threadpool_limits(limits=1):
            results[mode] = readout_analysis.analyse(sr,md,counts,mode,4,family,np.asarray([qf[q] for q in groups]), out)
    require(all(v['status']=='CLEAN' for v in results.values()),'LFM readout or lexical-null validation failed')
    return publish(target, {'status':'complete','models':{'lfm25':results},'historical_binding':frozen['sha256'],
        'questions':522,'families':len({qf[q] for q in questions}),'lexical_nulls':5,'provenance':provenance,
        'fold_local_transforms':True,'five_outer_three_inner_folds':True})
