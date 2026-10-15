"""Bind and reuse the historical five-fold present-state readout implementation."""
from pathlib import Path
import sys
from .common import CLASSES, check_files, digest, file_sha, load, publish, require, verified

HISTORICAL = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context')


def binding_files():
    frozen = load(HISTORICAL/'FROZEN.readout.v3.json')
    return {**frozen['code'], str(HISTORICAL/'FROZEN.readout.v3.json'):file_sha(HISTORICAL/'FROZEN.readout.v3.json'),
            str(HISTORICAL/'readout-results.v3.json'):file_sha(HISTORICAL/'readout-results.v3.json')}


def lfm(binding):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    target = root/'lfm25/READOUT_RESULTS.json'
    if target.exists():
        return verified(target)
    check_files(binding['readout_files'])
    sys.path.insert(0, str(HISTORICAL/'code'))
    import r3d_readout as historical
    from dp import s3_readout as sr
    from dynrt.d3a_text import hash_counts
    frozen = load(HISTORICAL/'FROZEN.readout.v3.json')
    questions = set(frozen['ids']['gpt'])
    require(len(questions) == 522, 'original readout population differs')
    inputs = verified(root/'lfm25/DISCOVERY_INPUTS.json')
    sent = pd.read_parquet(inputs['sentences'])
    sent = sent[sent.question.isin(questions)].copy().sort_values(['attempt_id','tok_start']).reset_index(drop=True)
    require(set(sent.question) == questions, 'LFM common-population readout has unavailable questions')
    price=verified(root/'CPU_PRICE.json')
    require(len(sent)<=price['limits']['sentences'] and sent.n_tokens.sum()<=price['limits']['tokens'],
            'actual LFM readout exceeds the measured CPU profile; no unpriced analysis')
    sources = {s['attempt_id']:s for s in inputs['sources']}
    row_of = {s:i for i,s in enumerate(sorted(sent.attempt_id.unique()))}
    sent['row'] = sent.attempt_id.map(row_of)
    top, tsent, tokid, tabs, adjacency = [], [], [], [], []
    text_lookup = {(r.attempt_id,r.sentence_index):r.text for r in sent.itertuples()}
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
        np.concatenate(top,axis=1),tsent,tokid,np.asarray(tabs),np.asarray(adjacency),len(sent))
    folds = load(Path(binding['source_campaign'])/'FOLDS.json')
    qf = folds['question_families']
    sr.cv.question_folds = lambda qs,n,seed: historical.family_folds(qs,n,seed,qf)
    sr.FOLD_SEED = historical.SEED
    historical.OUT = bulk/'lfm25/readout'
    family = {'sha256':digest([frozen['sha256'],inputs['sha256'],binding['sha256']])}
    counts = hash_counts(texts)
    results = {}
    from threadpoolctl import threadpool_limits
    for mode in ('token_identity','judge_context'):
        with threadpool_limits(limits=1):
            results[mode] = historical.analyse(sr,md,counts,mode,4,family,np.asarray([qf[q] for q in groups]))
    require(all(v['status']=='CLEAN' for v in results.values()),'LFM readout or lexical-null validation failed')
    return publish(target, {'status':'complete','models':{'lfm25':results},'historical_binding':frozen['sha256'],
        'questions':522,'families':len({qf[q] for q in questions}),'lexical_nulls':5,
        'fold_local_transforms':True,'five_outer_three_inner_folds':True})
