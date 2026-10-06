"""Seal fixed exploratory nested-CV diagnostic stage after registered primary."""
import hashlib
import json
from pathlib import Path

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
OUT = R / 'report/experimental-resume-v1/R3E_POSTPRIMARY_DIAGNOSTIC_PRICE_v1.json'


def d(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def sealed(p):
    v=json.loads(Path(p).read_text())
    if v.get('sha256')!=d({k:x for k,x in v.items() if k!='sha256'}):
        raise ValueError('bad seal '+str(p))
    return v


def main():
    primary=sealed(S/'runs/resume-v1/r3e-cv-clean-v1/full.json')
    prior=sealed(R/'report/experimental-resume-v1/R3E_CV_CLEAN_PRICE_v1.json')
    if primary['freeze_sha256']!=prior['freeze_seal'] or primary['primary']['gain_nats_per_attempt']>=0:
        raise ValueError('trigger/result differs from fixed diagnostic plan')
    inputs=dict(prior['input_sha256'])
    for p in (R/'report/experimental-resume-v1/R3E_POSTPRIMARY_DIAGNOSTIC_v1.md',
              R/'scripts/experimental_resume/run_r3e_cv_clean_v1.py',
              S/'runs/resume-v1/r3e-cv-clean-v1/full.json'):
        inputs[str(p)]=sha(p)
    body={'schema':'r3e-postprimary-exploratory-price-v1','status':'READY_EXPLORATORY_ONLY',
          'registered_primary_seal':primary['sha256'],'freeze_seal':prior['freeze_seal'],
          'plan_sha256':inputs[str(R/'report/experimental-resume-v1/R3E_POSTPRIMARY_DIAGNOSTIC_v1.md')],
          'driver_sha256':sha(R/'scripts/experimental_resume/run_r3e_diagnostics_v1.py'),
          'input_sha256':inputs,
          'candidate_count':8,'comparison_count':8,'outer':5,'inner':3,'repeats':5,
          'ridge_grid':[10.0**x for x in range(-4,5)],
          'max_core_hours':8.0,
          'slurm':{'account':'iscrc_miosr','partition':'boost_usr_prod','qos':'normal',
                   'cpus':8,'memory':'64G','gpus':0,'time':'01:00:00'},
          'role':'diagnose baseline harm on reused primary cohort; no confirmatory improvement claim'}
    value={**body,'sha256':d(body)}
    if OUT.exists():
        if json.loads(OUT.read_text())!=value:
            raise ValueError('existing diagnostic price differs')
    else:
        OUT.write_text(json.dumps(value,indent=1)+'\n')
    print(json.dumps({'price':str(OUT),'seal':value['sha256'],'max_core_hours':8}))


if __name__=='__main__':
    main()
