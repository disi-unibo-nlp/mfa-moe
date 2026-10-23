"""Bind and reuse the historical five-fold present-state readout implementation."""
from pathlib import Path
import sys
from store import CLASSES, check_files, digest, file_sha, load, publish, require, verified

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
    inputs = verified(root/'lfm25/DISCOVERY_INPUTS.json')
    require(file_sha(Path(inputs['sentences'])) == inputs['sentences_sha256'], 'LFM derived sentences changed')
    folds_path = Path(binding['source_campaign'])/'FOLDS.json'
    sampled_path = bulk/'lfm25/SAMPLED_SENTENCES.json'
    # Checkpoints and the published result are keyed by code as well as data, so a fix never reuses old fits.
    provenance = {'code': {name: file_sha(Path(__file__).with_name(name)) for name in
                           ('lfm_readout.py', 'readout_fit.py', 'readout_analysis.py', 'readout_text.py')},
                  'inputs_sha256': inputs['sha256'], 'sentences_sha256': inputs['sentences_sha256'],
                  'sampled_sentences_sha256': file_sha(sampled_path), 'folds_sha256': file_sha(folds_path),
                  'seed_index': 7}
    if target.exists():
        value = verified(target)
        require(value.get('provenance') == provenance, 'published LFM readout was produced by other code or inputs')
        return value
    check_files(binding['readout_files'])
    from routing import readout_analysis as historical, readout_fit as sr
    from routing.readout_text import hash_counts
    frozen = load(HISTORICAL/'FROZEN.readout.v3.json')
    questions = set(frozen['ids']['gpt'])
    require(len(questions) == 522, 'original readout population differs')
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
    # Context slots read every sampled unit of the attempt, unknown labels included, as the historical
    # load_unit_texts does; only sentences without any sampled unit stay '<missing>'.
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
    output_root = bulk/'lfm25/readout'
    family = {'sha256':digest([frozen['sha256'],binding['sha256'],provenance])}
    counts = hash_counts(texts)
    results = {}
    from threadpoolctl import threadpool_limits
    for mode in ('token_identity','judge_context'):
        with threadpool_limits(limits=1):
            results[mode] = historical.analyse(sr,md,counts,mode,4,family,np.asarray([qf[q] for q in groups]), output_root)
    require(all(v['status']=='CLEAN' for v in results.values()),'LFM readout or lexical-null validation failed')
    return publish(target, {'status':'complete','models':{'lfm25':results},'historical_binding':frozen['sha256'],
        'questions':522,'families':len({qf[q] for q in questions}),'lexical_nulls':5,'provenance':provenance,
        'fold_local_transforms':True,'five_outer_three_inner_folds':True})
