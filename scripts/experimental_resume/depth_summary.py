"""DP-S3: verify and summarize existing caches; never fit or produce tensors."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import NormalDist
import time

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
OUT = ROOT / 'depth-paths/results/resume-v1'
MODELS = ('gpt','glm','gemma','nemotron','qwen35','qwen330b','qwen36')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seal(value):
    return {**value,'sha256':hashlib.sha256(json.dumps(value,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()}


def run():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('saved-results analysis must run in CPU Slurm')
    start = time.monotonic()
    old = ROOT/'depth-paths/results'
    attestation = ROOT/'forum/tests/r3_integrity/FROZEN.json'
    inventory = json.loads(attestation.read_text())['inputs']
    files = [old/'s3/nmi'/f'{m}.json' for m in MODELS]
    files += sorted((old/'s3/readout').glob('*.json'))
    files += [old/'s3/f1prime.json',old/'part2/results.json']
    files += sorted((old/'validation').glob('nullcal_gpt_*.json'))
    bindings={}
    for p in files:
        if not p.is_file():
            continue
        relative=str(p.relative_to(ROOT))
        previous=inventory.get(relative)
        actual=sha(p)
        if previous is not None and actual != previous['sha256']:
            raise ValueError('historically attested output changed: '+relative)
        bindings[relative]={'sha256':actual,'historical_attestation':previous is not None}
    freeze=seal({'schema':'DP-S3-resume-v1','inputs':bindings,'attestation_sha256':sha(attestation),
        'driver_sha256':sha(__file__),'CPU_core_hour_ceiling':6.,'fits':0,
        'analysis_units':'saved question readout losses; saved token NMI with original jackknife',
        'multiplicity':'seven-model readout and 28 model-by-gap NMI; Bonferroni normal approximations explicitly labelled',
        'equivalence':'requires adjusted CI inside +/- .003 AND training-only same-scale power >= .8; unavailable => NOT EXCLUDED'})
    OUT.mkdir(parents=True,exist_ok=True)
    fp=OUT/'FROZEN.json'
    if fp.exists() and json.loads(fp.read_text()) != freeze:
        raise ValueError('DP summary bindings changed; use a new version')
    fp.write_text(json.dumps(freeze,indent=1)+'\n')
    nmi,readout,rows={}, {}, []
    z28=NormalDist().inv_cdf(1-.05/(2*28))
    for m in MODELS:
        p=old/'s3/nmi'/f'{m}.json'
        if not p.exists():continue
        v=json.loads(p.read_text())
        status='EXACT_ID_CLEAN' if v.get('split_restriction') else 'HISTORICAL_CONFIRM_EXPOSED'
        nmi[m]={'scope':status,'n_attempts':v['n_attempts'],'n_tokens':v['n_tokens'],
                'split_restriction':v.get('split_restriction'),'gaps':v['gaps']}
        for gap in ('1','2','4','8'):
            cell=v['gaps'].get(gap)
            if cell is None:
                rows.append({'model':m,'gap':gap,'status':'MISSING'});continue
            lm=cell['layer_mean'];j=lm['jackknife']['excess_over_lex']
            estimate=lm['excess_over_lex']
            rows.append({'model':m,'gap':gap,'status':status,'n_tokens':v['n_tokens'],
                'NMI':lm['nmi_observed'],'lexical_null':lm['null_lex'],
                'excess':estimate,'nominal_lo':j['ci95'][0],'nominal_hi':j['ci95'][1],
                'adjusted_lo':estimate-z28*j['se'],'adjusted_hi':estimate+z28*j['se'],
                'interval_method':'normal jackknife approximation, Bonferroni 28'})
    z7=NormalDist().inv_cdf(1-.05/(2*7))
    for m in MODELS:
        p=old/'s3/readout'/f'{m}.json'
        if not p.exists():continue
        v=json.loads(p.read_text());g=v['gain_aug_vs_base'];e=v['null_within_token_id']['excess_obs_over_null']
        readout[m]={'scope':'EXACT_ID_CLEAN' if v.get('split_restriction') else 'HISTORICAL_CONFIRM_EXPOSED',
            'questions':v['n_questions'],'sentences':v['n_sentences'],'source':str(p),
            'P_nats_per_sentence':-g['gain'],'nominal_P_CI':[-g['ci_hi'],-g['ci_lo']],
            'adjusted_P_CI_normal_approximation':[-g['gain']-z7*g['boot_sd'],-g['gain']+z7*g['boot_sd']],
            'excess_gain':e['gain'],'excess_nominal_CI':[e['ci_lo'],e['ci_hi']],
            'excess_adjusted_CI_normal_approximation':[e['gain']-z7*e['boot_sd'],e['gain']+z7*e['boot_sd']],
            'multiplicity':'Bonferroni 7 using saved bootstrap SD; original draws unavailable',
            'context_control':'NOT TESTED in this historical cache; see separate R3-D fits'}
    p2=json.loads((old/'part2/results.json').read_text())
    primary=p2['parts']['gpt_A']['comparisons']['primary_C_vs_C+F1F2F3']
    power=[]
    for p in sorted((old/'validation').glob('nullcal_gpt_*.json')):
        v=json.loads(p.read_text());s=v['summary']['primary_C_vs_C+F1F2F3']
        power.append({'source':str(p),'plant_scale':'logit coefficient, not a calibrated conditional information gain',
            'effect':v['effect'],'mean_OOF_gain':s['mean_gain'],'power_p_lt_05':s['p_lt_05'],
            'equivalence_qualification':False})
    equiv={'scope':'HISTORICAL_CONFIRM_EXPOSED GPT A', 'P':-primary['gain'],
        'nominal_P_CI':[-primary['ci_hi'],-primary['ci_lo']],
        'same_scale_training_only_power':'MISSING','adjusted_CI':'MISSING original joint draws',
        'equivalence':'NOT EXCLUDED','historical_plants':power,
        'interpretation':'The nominal matched-population interval is not an equivalence certificate. No clean correctness claim is derived from this cache.'}
    f1=json.loads((old/'s3/f1prime.json').read_text())
    result=seal({'schema':'DP-S3-resume-results-v1','frozen':freeze['sha256'],'nmi':nmi,
        'readout':readout,'missing_readout':[m for m in MODELS if m not in readout],
        'f1prime':{'scope':'HISTORICAL_CONFIRM_EXPOSED','result':f1,
            'historical_hash_attestation':bindings['depth-paths/results/s3/f1prime.json']['historical_attestation']},
        'correctness_precision':equiv,'job_id':os.environ['SLURM_JOB_ID'],
        'seconds':time.monotonic()-start,'new_fits':0})
    (OUT/'results.json').write_text(json.dumps(result,indent=1)+'\n')
    with (OUT/'depth-gap-table.csv').open('w') as h:
        cols=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(h,fieldnames=cols);w.writeheader();w.writerows(rows)
    lines=['# DP-S3 resume v1','',
        'Verified cached summaries are reused without new fits. Six NMI populations and three cached readouts retain historical confirm exposure; Qwen3.6 caches are exact-ID clean. Family-clean sensitivity is separate.',
        '', '| model | scope | P (nats/sentence) | nominal 95% CI |', '|---|---|---:|---|']
    for m,r in readout.items():
        lines.append(f"| {m} | {r['scope']} | {r['P_nats_per_sentence']:.5f} | {r['nominal_P_CI']} |")
    lines += ['', 'Seven-model and context-controlled clean readouts are running separately in R3-D. Missing cached cells remain missing here.',
        '', 'Correctness equivalence at +/-0.003 nats/attempt is **not excluded**: historical plants are not a training-only, same-scale power qualification.',
        '',f"Slurm job {os.environ['SLURM_JOB_ID']}; final state and actual allocation must be checked in the job audit.",'']
    (OUT/'summary.md').write_text('\n'.join(lines))
    plot(rows)
    print(json.dumps({'status':'COMPLETE_SAVED_SUMMARY','frozen':freeze['sha256'],'output':str(OUT)}))


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for m in MODELS:
        rr=[r for r in rows if r['model']==m and 'excess' in r]
        if not rr:continue
        ax=axes[1] if m=='qwen36' else axes[0]
        ax.plot([int(r['gap']) for r in rr],[r['excess'] for r in rr],marker='o',label=m)
        ax.fill_between([int(r['gap']) for r in rr],[r['adjusted_lo'] for r in rr],
                        [r['adjusted_hi'] for r in rr],alpha=.12)
    for ax,title in zip(axes,['Historical confirm exposure','Qwen3.6 exact-ID clean']):
        ax.set(title=title,xlabel='Layer gap',ylabel='NMI excess over lexical null')
        ax.set_xticks([1,2,4,8]);ax.legend(fontsize=8)
    fig.savefig(OUT/'depth-gap.pdf');fig.savefig(OUT/'depth-gap.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();run()
