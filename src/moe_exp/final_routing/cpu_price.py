"""Allocated CPU price at the declared LFM readout envelope, not paper data."""
from pathlib import Path
import resource
import sys
import time
from .common import digest, load, publish, require, verified
from .readouts import HISTORICAL


def measure(binding,deadline):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    target=Path(binding['root'])/'CPU_PRICE.json'
    if target.exists():return verified(target)
    sys.path.insert(0,str(HISTORICAL/'code'))
    import r3d_readout as historical
    from dp import s3_readout as sr
    from dynrt.d3a_text import hash_counts
    started=time.monotonic()
    n,tokens_per_sentence=100000,20
    rng=np.random.default_rng(20261015)
    questions=load(HISTORICAL/'FROZEN.readout.v3.json')['ids']['gpt']
    qf=load(Path(binding['source_campaign'])/'FOLDS.json')['question_families']
    groups=np.asarray([questions[i%len(questions)] for i in range(n)])
    cls=rng.integers(0,7,n)
    token_sentence=np.repeat(np.arange(n),tokens_per_sentence)
    tok=rng.integers(0,65536,len(token_sentence))
    ranks=rng.integers(0,131072,len(token_sentence))
    top=rng.integers(0,64,(38,len(token_sentence)),dtype=np.uint8)
    sent=pd.DataFrame({'attempt_id':groups,'question':groups,'sentence_index':np.arange(n),
        'cls':cls,'n_tokens':tokens_per_sentence,'n_reasoning':131072,
        'tok_start':np.arange(n)%131072,'tok_end':np.arange(n)%131072+tokens_per_sentence,
        'rel_pos':rng.random(n)})
    ab=sr.abs_bin(sent.tok_start.to_numpy());rb=(sent.rel_pos.to_numpy()*sr.N_REL).astype(int)
    base=np.column_stack([(ab==b).astype(float) for b in range(sr.N_ABS)]+
         [(rb==b).astype(float) for b in range(sr.N_REL)]+[np.log(sent.n_tokens)])
    bag=sparse.coo_matrix((np.ones(len(tok)),(token_sentence,sr.bucket_of(tok))),shape=(n,sr.N_BUCKETS)).tocsr()
    adj=np.ones(len(tok),bool);adj[::tokens_per_sentence]=False
    md=sr.ModelData('lfm25-engineering',38,64,sent,cls,groups,sr.question_weights(groups),base,bag,
        top,token_sentence,tok,sr.abs_bin(ranks),adj,n)
    texts=['previous token'+str(i%4096)+'\ncurrent class'+str(cls[i])+' word'+str(i%17389)+
           '\nfuture token'+str((i+1)%4096) for i in range(n)]
    counts=hash_counts(texts)
    sr.cv.question_folds=lambda qs,k,seed:historical.family_folds(qs,k,seed,qf)
    sr.FOLD_SEED=historical.SEED
    historical.OUT=Path(binding['config']['bulk_root'])/'cpu-pricing'
    proof={'sha256':digest([binding['sha256'],'cpu-envelope',n,len(tok),38,64])}
    results={}
    from threadpoolctl import threadpool_limits
    for mode in ('token_identity','judge_context'):
        require(time.time()+60<deadline,'CPU pricing deadline; fit checkpoints remain recoverable')
        with threadpool_limits(limits=1):
            results[mode]=historical.analyse(sr,md,counts,mode,4,proof,np.asarray([qf[q] for q in groups]))
    # Complete timing includes retained checkpoints on recovery; never price a
    # resumed fit by just its cheap final readback.
    from .resources import accounting
    usage=accounting(binding['root'])
    price_jobs=[]
    for p in (Path(binding['root'])/'cards').glob('*.json'):
        c=load(p)
        if c['stage']!='cpu_price':continue
        a=Path(binding['root'])/'submissions'/digest({k:v for k,v in c.items() if k!='sha256'})/'ACCEPTED.json'
        if a.exists():
            jid=verified(a)['job_id']
            if jid in usage:price_jobs.append({'job_id':jid,**usage[jid]})
    fold_seconds=max(time.monotonic()-started,sum(v['seconds'] for v in price_jobs))
    return publish(target,{'status':'complete','scientific_results':False,'engineering_fixture':True,
        'profile':'historical-r3d-five-fold-three-inner-two-baselines-five-nulls',
        'limits':{'sentences':n,'tokens':len(tok),'layers':38,'experts':64,'cpu':4,'memory_gib':24},
        'observed_seconds':time.monotonic()-started,'reserved_fit_seconds':fold_seconds,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'binding_sha256':proof['sha256'],'checkpoint_root':str(historical.OUT),'modes':results,
        'allocated_attempts':price_jobs})
