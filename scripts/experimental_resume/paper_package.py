"""Versioned results package from saved artifacts; missing cells remain explicit."""
from __future__ import annotations
import csv
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

R=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DOC=REPO/'report/experimental-resume-v1'


def load(p):return json.loads(Path(p).read_text())


def readout_coverage(root):
    models=('gpt','qwen36','gemma','glm','nemotron','qwen330b','qwen35')
    # Use the actual registered model order rather than guessing model aliases.
    freeze=root/'FROZEN.readout.v3.json'
    if freeze.is_file():models=tuple(load(freeze)['models'])
    cells={}
    for model in models:
        for mode in ('token_identity','judge_context'):
            path=root/'readout'/model/mode/'result.json'
            cells[model+'/'+mode]='CLEAN' if path.is_file() and load(path).get('status')=='CLEAN' else 'INCOMPLETE'
    return {'status':'COMPLETE' if all(v=='CLEAN' for v in cells.values()) else 'INCOMPLETE',
            'cells':cells,'completed':sum(v=='CLEAN' for v in cells.values()),'required':len(cells)}


def claim(identifier,population,intervention,estimate,uncertainty,family,status,source,note=''):
    return dict(id=identifier,population=population,intervention=intervention,estimate=estimate,
        uncertainty=uncertainty,multiplicity_family=family,status=status,source=str(source),interpretation=note)


def binary_reporting_bounds(x2):
    helper=Path(__file__).with_name('uncertainty.py')
    if not helper.is_file():helper=REPO/'src/moe_exp/routing_control/uncertainty.py'
    spec=importlib.util.spec_from_file_location('paper_binary_uncertainty',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    observations=load(R/'steering-v1/runs/x2-resume-v1/analysis/observations.json')
    frozen=load(R/'steering-v1/runs/x2-resume-v1/FROZEN.json')
    families={q:f for f,qs in frozen['families'].items() for q in qs}
    rows={(r['policy'],r['question'],r['seed_k']):r for r in observations}
    native=next(r['policy'] for r in observations if r['arm']=='N')
    questions=sorted({r['question'] for r in observations})
    baseline=[[rows[native,q,k]['degeneration'] for k in (0,1)] for q in questions]
    return {cell['policy']:module.paired_binary_bounds(
        [[rows[cell['policy'],q,k]['degeneration'] for k in (0,1)] for q in questions],baseline,
        [families[q] for q in questions],contrasts=564) for cell in x2['cells']}


def run():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('paper figures and summaries require CPU Slurm')
    now=datetime.datetime.now(datetime.timezone.utc)
    out=R/'paper/resume-v1'/now.strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True,exist_ok=False)
    rkpath=R/'reasoning-kinematics/rk1c/results.json'
    bpath=R/'forum/tests/r3_dynamics_validity/estimates.json'
    dpath=R/'depth-paths/results/resume-v1/results.json'
    hist=R/'forum/tests/r3_integrity/estimates.v2.json'
    audit=DOC/'JOB_AUDIT.json'
    x2path=R/'steering-v1/runs/x2-resume-v1/analysis/estimates.json'
    x2=load(x2path) if x2path.is_file() else None
    g3path=DOC/'G3_DOSE_SUPPORT_AMENDMENT.json'
    g3=load(g3path) if x2 is not None else None
    if x2 is not None:
        if not g3path.is_file() or g3['source_file_sha256']!=hashlib.sha256(x2path.read_bytes()).hexdigest():
            raise ValueError('the registered G3 dose-support amendment is required for X2 reporting')
        if g3['registered_selected']!=x2['selected_policy_per_sign']:
            raise ValueError('G3 selection changed; review and re-freeze downstream selection')
    source_paths=[rkpath,bpath,dpath,hist,audit,DOC/'STUDY_GATES.json',DOC/'PREREG_SEAL.json',
        R/'steering-v1/qualification/h14/resume-v1-recovery/H14.json']
    if x2 is not None:source_paths.append(x2path)
    if x2 is not None:
        source_paths.extend([R/'steering-v1/runs/x2-resume-v1/analysis/observations.json',
            R/'steering-v1/runs/x2-resume-v1/FROZEN.json',
            R/'steering-v1/runs/x2-resume-v1/analysis/G3_SELECTION.json',g3path])
        helper=Path(__file__).with_name('uncertainty.py')
        source_paths.append(helper if helper.is_file() else REPO/'src/moe_exp/routing_control/uncertainty.py')
    for p in [DOC/'RUNBOOK_v1.md',DOC/'RESOURCE_CHECKPOINT_v2.md',DOC/'QUALIFICATION_RECOVERY_PROPOSAL.json']:
        if p.is_file():source_paths.append(p)
    for p in [R/'forum/tests/r3_context/readout-results.v3.json',R/'forum/tests/r3_context/anticipation-results.json']:
        if p.is_file():source_paths.append(p)
    readout_root=R/'forum/tests/r3_context'
    source_paths.extend(sorted((readout_root/'readout').glob('*/*/result.json')))
    if (readout_root/'FROZEN.readout.v3.json').is_file():source_paths.append(readout_root/'FROZEN.readout.v3.json')
    inventory={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    freeze={'schema':'resume-paper-inputs-v1','time_UTC':now.isoformat(),'inputs':inventory,
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'job_id':os.environ['SLURM_JOB_ID'],
        'no_new_fits':True,'pending_results':'not inferred from an active job or absent artifact'}
    (out/'FROZEN.json').write_text(json.dumps(freeze,indent=1)+'\n')
    claims=[];tables={};dynamics=[]
    rk=load(rkpath)
    tables['class_transitions']={name: {key: cell['weighted'][key] for key in ('P','pi','counts')}
        for name,cell in rk['parts']['sparse_descriptive']['cohorts'].items()}
    tables['class_persistence']={name: {key:v for key,v in cell['weighted']['metrics'].items()
        if key.startswith(('persist_','dwell_','ret2_','loop2_'))}
        for name,cell in rk['parts']['sparse_descriptive']['cohorts'].items()}
    for name,cell in rk['parts']['sparse_descriptive']['cohorts'].items():
        metrics=cell['weighted']['metrics']
        for metric in ('switch_rate','EP'):
            if metric not in metrics:continue
            m=metrics[metric]
            claims.append(claim('rk1c:'+name+':'+metric,'A exact-ID-clean dev+tune, sparse adjacent labels',
                'native observational',m['est'],[m['lo'],m['hi']],'descriptive nominal intervals','CLEAN_DESCRIPTIVE',rkpath,
                'Sparse gaps are excluded. Transition association does not establish causal relaxation or an optimal sequence.'))
            dynamics.append({'model':name.split('/')[0],'metric':metric,'estimate':m['est'],'lo':m['lo'],'hi':m['hi']})
    b=load(bpath)
    tables['composition_controls']={m:r['primary'] for m,r in b['models'].items()}
    for m,cell in b['models'].items():
        v=cell['primary']
        claims.append(claim('R3B:'+m,'A dev+tune, '+str(v['n_questions'])+' questions',
            'correct-minus-wrong Explore persistence, within-attempt composition reference',v['estimate'],
            v['simultaneous_ci'],'seven-model Holm family','IMPRECISE_COMPOSITION_CONTROL',bpath,
            'An interval spanning zero neither proves composition explains the association nor establishes equivalence.'))
    h=b['human_measurement']['conditions']['0.5']['judge_minus_gold']['switch_rate']
    claims.append(claim('R3B:measurement','six non-test validation documents, 407 units / 401 adjacent pairs',
        'LLM judge versus human labels',h['estimate'],[h['lo'],h['hi']],'descriptive document bootstrap',
        'MEASUREMENT_LIMITATION',bpath,'This is not seven-model or target-domain semantic validation.'))
    d=load(dpath);tables['depth_readouts']=d['readout']
    for m,v in d['readout'].items():
        claims.append(claim('S3:cached:'+m,str(v['questions'])+' questions; '+v['scope'],
            'routing features added to token identity and position baseline',v['P_nats_per_sentence'],
            v['adjusted_P_CI_normal_approximation'],'seven-model Bonferroni normal approximation',
            v['scope'],v['source'],'Cached baseline lacks judge-context control; this does not establish controllability.'))
    e=d['correctness_precision']
    claims.append(claim('DP:historical-correctness',e['scope'],'selected F1/F2/F3 beyond historical controls',
        e['P'],e['nominal_P_CI'],'historical primary, nominal interval','NOT_EXCLUDED',dpath,e['interpretation']))
    coverage=readout_coverage(readout_root)
    tables['clean_readout_coverage']=coverage
    for path in sorted((readout_root/'readout').glob('*/*/result.json')):
        v=load(path)
        if v.get('status')!='CLEAN':continue
        model=path.parent.parent.name;mode=path.parent.name;routing=v['routing']
        claims.append(claim('R3D:S3:'+model+':'+mode,
            'exact-ID-clean dev+tune; '+str(v['n_questions'])+' questions / '+str(v['n_families'])+' frozen families',
            'routing added to '+mode+' baseline',routing['P'],routing['simultaneous_ci7'],
            'seven-model Bonferroni; full registered family incomplete','CLEAN_CELL_REGISTERED_FAMILY_INCOMPLETE',path,
            'Completed checkpoint only; confirm-connected-family exclusion sensitivity and full seven-model family remain incomplete.'))
    claims.extend([
        claim('K1:bound','original matched native K1 population','matched transition-associated turnover',None,None,
            'original K family','NARROW_MATCHED_POPULATION_BOUND',hist,'MDE does not establish equivalence or causal relaxation.'),
        claim('K2:anticipation','original K2 pipeline and evaluated populations','16/32-token kinematic augmentation',None,None,
            'original within-K2 then K family','INCONCLUSIVE',hist,'Missing or rounded cells do not pass the original gate.'),
        claim('G2:baseline','accepted X1 benchmark reading','native versus card-v3 routing',None,None,
            'registered G2 benchmark checks','ACCEPTED_NON_REJECTION',R/'steering-v1/DECISIONS.md',
            'Accepted non-rejection does not establish engine equivalence; raw Q3 failure and adjudication remain separate.'),
        claim('G3:dose','39 eligible of 48 dev-cal questions; two seeds; 6630 same-prefix branches','legacy X2 dose grid',
            g3['registered_selected'] if g3 else None,None,
            'registered point-estimate dose-selection screens',
            'REGISTERED_MATCHING_COMPLETE_DOSE_ONLY' if g3 else 'GENERATION_COMPLETE_PENDING_NATIVE_NLL_AND_ANALYSIS',
            g3path if g3 else x2path,
            'Strict ±10% random-dose support leaves 2 of 42 target cells passing G3; the earlier saved analysis counted 6. Dose selection does not establish semantic steering.'),
        claim('new:trajectory','family-disjoint discovery / mechanism parent pools','frozen local template versus native, random and reverse order',
            None,None,'nine mechanism contrasts','HELD_QUALIFICATION_ELIGIBILITY_PRICING',DOC/'STUDY_GATES.json',
            'No causal semantic-control result or universal optimal trajectory is established.'),
        claim('new:utility','96-family parent pool, original prompts, 16k endpoint','frozen policy versus native',None,None,
            'three joint utility contrasts','HELD_PRICING',DOC/'STUDY_GATES.json',
            'Similar observed accuracy cannot establish retention without a chosen noninferiority margin; do not extrapolate past 16k.')])
    if x2 is not None:
        tables['X2_dose_diagnostics']=x2
        tables['X2_registered_G3_dose_support']=g3
        supported={c['policy']:c['registered_random_dose_support'] for c in g3['cells']}
        binary_bounds=binary_reporting_bounds(x2)
        tables['X2_binary_uncertainty_addendum']={'intervals':binary_bounds,
            'supersedes_reporting_only':'zero-width degeneration bootstrap/t intervals are dispersion diagnostics; use these conservative simultaneous bounds for claims',
            'G3_change':False,'sample_change':False,'selection_change':False}
        for cell in x2['cells']:
            for field in ('marker_E_minus_N','degeneration_E_minus_N','native_NLL_E_minus_N','marker_E_minus_matched_M'):
                estimate=cell[field]
                if estimate is None:continue
                bounds=binary_bounds[cell['policy']]['simultaneous_interval'] if field=='degeneration_E_minus_N' else estimate.get('family_clustered_Bonferroni_t_approximation')
                family='564 diagnostic contrasts; conservative independent-family binary bound' if field=='degeneration_E_minus_N' else '564 potential diagnostic contrasts, approximate family-clustered Bonferroni t intervals'
                offband=field=='marker_E_minus_matched_M' and not supported[cell['policy']]
                claims.append(claim('X2:'+cell['policy']+':'+field,
                    '39 eligible dev-cal questions, same prefix and paired seeds; nine ineligible questions not replaced',
                    cell['policy'],estimate['estimate'],bounds,family,
                    'UNMATCHED_RANDOM_DOSE_DIAGNOSTIC' if offband else 'DIAGNOSTIC_DOSE_SELECTION',x2path,
                    ('Random expert dose fell outside registered ±10% support; this E-minus-M estimate is not a matched-control result. '
                     if offband else '')+
                    'Literal lexical markers are not substantive semantic verification. Boundary intervals and dose screens do not establish accuracy retention or steering.'))
    (out/'claim-ledger.json').write_text(json.dumps(claims,indent=1)+'\n')
    (out/'tables.json').write_text(json.dumps(tables,indent=1)+'\n')
    historical=load(hist)
    (out/'historical-scoreboard.json').write_text(json.dumps(historical.get('scoreboard',historical),indent=1)+'\n')
    with (out/'clean-dynamics.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['model','metric','estimate','lo','hi']);writer.writeheader();writer.writerows(dynamics)
    plot(out,dynamics,tables['composition_controls'])
    plot_transitions(out,tables['class_transitions'])
    if x2 is not None:plot_x2(out,x2)
    jobs=load(audit)
    with (out/'compute-expenditure.csv').open('w') as handle:
        fields=['job_id','task','state','exit_code','unit','actual_resource_hours','verified_completion']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows({k:j[k] for k in fields} for j in jobs['jobs'])
    plot_costs(out,jobs['jobs'])
    experiment_inventory={'H14':'COMPLETE_PASS','caps':'COMPLETE_CPU_REGRESSION_AND_CAPPED_MANIFEST',
        'preregistration_v0.3':'SEALED','rk1c':'COMPLETE','R3B':'COMPLETE','DP-S3':'COMPLETE_SAVED_CACHE_SUMMARY',
        'R3D_readout':coverage,
        'R3D_anticipation':'COMPLETE' if (R/'forum/tests/r3_context/anticipation-results.json').exists() else 'INCOMPLETE_CPU_CEILING',
        'R3D_B1':'PENDING_CLEAN_REFIT','R3E':'PENDING_R3D; generator helper CPU-tested, no real-data power result',
        'X2':x2['status'] if x2 else 'GENERATION_COMPLETE; ANALYSIS_PENDING',
        'G3_registered':{'passing_cells':g3['registered_passing_cells'],
            'selected':g3['registered_selected'],'amendment':str(g3path)} if g3 else 'PENDING',
        'native_NLL':'COMPLETE_IDENTICAL_PARITY_VALIDATED' if x2 and x2['native_NLL_validation'] else 'PENDING',
        'X3':'builder / full cost gate; new study has priority',
        'M9':'separate closure driver prepared; replay and full grading price required','M10_M8':'matched-prefix / fold / replay gate',
        'new_study':load(DOC/'STUDY_GATES.json')['qualification_stage']['status'],
        'jobs':jobs['jobs'],'paper_snapshot':str(out)}
    (out/'experiment-inventory.json').write_text(json.dumps(experiment_inventory,indent=1)+'\n')
    (out/'README.md').write_text('# Resume evidence snapshot\n\nAll tables and figures are generated from saved inputs inventoried in FROZEN.json. Historical and clean populations retain separate status. Active jobs and absent cells are not completed results. Causal action, trajectory and accuracy–token figures remain unavailable while the new study is held. See claim-ledger.json and experiment-inventory.json.\n')
    (DOC/'CLAIM_LEDGER.json').write_text(json.dumps(claims,indent=1)+'\n')
    (DOC/'EXPERIMENT_INVENTORY.json').write_text(json.dumps(experiment_inventory,indent=1)+'\n')
    (DOC/'PAPER_SNAPSHOT.json').write_text(json.dumps({'path':str(out),'job_id':os.environ['SLURM_JOB_ID']},indent=1)+'\n')
    print(json.dumps({'path':str(out),'claims':len(claims),'incomplete_cells_preserved':True}))


def plot(out,rows,composition):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,metric in zip(axes,['switch_rate','EP']):
        r=[x for x in rows if x['metric']==metric]
        ax.errorbar([x['estimate'] for x in r],range(len(r)),xerr=[
            [x['estimate']-x['lo'] for x in r],[x['hi']-x['estimate'] for x in r]],fmt='o')
        ax.set_yticks(range(len(r)),[x['model'] for x in r]);ax.set_xlabel(metric)
        ax.set_title('Clean native observations, nominal 95% CI')
    fig.savefig(out/'clean-dynamics.pdf');fig.savefig(out/'clean-dynamics.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    m=list(composition);v=[composition[k] for k in m]
    ax.errorbar([x['estimate'] for x in v],range(len(m)),xerr=[
        [x['estimate']-x['simultaneous_ci']['lo'] for x in v],
        [x['simultaneous_ci']['hi']-x['estimate'] for x in v]],fmt='o')
    ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(len(m)),m)
    ax.set_xlabel('Composition-adjusted correct-minus-wrong Explore persistence')
    ax.set_title('Seven-model simultaneous intervals')
    fig.savefig(out/'composition-control.pdf');fig.savefig(out/'composition-control.png',dpi=160);plt.close(fig)


def plot_transitions(out,transitions):
    import matplotlib.pyplot as plt
    import numpy as np
    classes=['Read','Analyze','Plan','Implement','Explore','Verify','Monitor']
    cells=[(name,v) for name,v in transitions.items() if name.endswith('/A')]
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for ax,(name,value) in zip(axes.flat,cells):
        im=ax.imshow(np.asarray(value['P']['est']),vmin=0,vmax=1,cmap='viridis')
        ax.set_xticks(range(7),classes,rotation=60,ha='right',fontsize=8)
        ax.set_yticks(range(7),classes,fontsize=8);ax.set_title(name.split('/')[0])
        ax.set_xlabel('Next contiguous sentence');ax.set_ylabel('Current sentence')
    for ax in list(axes.flat)[len(cells):]:ax.set_visible(False)
    fig.colorbar(im,ax=[ax for ax in axes.flat if ax.get_visible()],label='Conditional transition probability',shrink=.7)
    fig.suptitle('Exact-ID-clean native class transitions; sparse gaps excluded; observational')
    fig.savefig(out/'clean-class-transitions.pdf');fig.savefig(out/'clean-class-transitions.png',dpi=160);plt.close(fig)


def plot_x2(out,x2):
    import matplotlib.pyplot as plt
    colors={'bias':'#1b6ca8','reweight':'#b3541e','force':'#6a4795'}
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    fields=('marker_E_minus_N','native_NLL_E_minus_N')
    for column,scope in enumerate(('L1','BAND','ALL')):
        for row,field in enumerate(fields):
            ax=axes[row,column]
            for operator in ('bias','reweight','force'):
                for sign in (-1,1):
                    cells=sorted([c for c in x2['cells'] if c['scope']==scope and c['operator']==operator and c['sign']==sign and c[field] is not None],key=lambda c:c['magnitude'])
                    if not cells:continue
                    x=[c['dose']['estimate'] for c in cells];y=[c[field]['estimate'] for c in cells]
                    ci=[c[field].get('question_bootstrap_95') for c in cells]
                    if any(v is None for v in ci):continue
                    ax.errorbar(x,y,yerr=[[e-v[0] for e,v in zip(y,ci)],[v[1]-e for e,v in zip(y,ci)]],
                        fmt='o-' if sign>0 else 's--',color=colors[operator],capsize=2,
                        label=operator+(' promote' if sign>0 else ' suppress'))
            ax.axhline(0,color='gray',lw=1)
            if row==1:ax.axhline(.1,color='gray',lw=1,ls=':')
            ax.set_xlabel('Executed TV dose over policy layers');ax.set_title(scope)
            if column==0:ax.set_ylabel('Marker rate difference / 1,000 opportunities' if row==0 else 'Native NLL difference (nats/token)')
    axes[0,2].legend(fontsize=7)
    fig.suptitle('X2 diagnostic dose curves: all 42 cells, nominal question-bootstrap 95% intervals\nLexical markers do not establish semantic control; simultaneous intervals are in the tables')
    fig.savefig(out/'X2-dose-diagnostics.pdf');fig.savefig(out/'X2-dose-diagnostics.png',dpi=160);plt.close(fig)


def plot_costs(out,jobs):
    import matplotlib.pyplot as plt
    gpu={'H1–H4':0.,'X2 generation / NLL':0.,'New qualification':0.}
    cpu={'R3-D':0.,'Other resume CPU jobs':0.}
    for job in jobs:
        if job['unit']=='GPU-h':
            key='H1–H4' if job['task'].startswith('h14') else 'New qualification' if job['task'].startswith('ordered-qualify') else 'X2 generation / NLL'
            gpu[key]+=job['actual_resource_hours']
        else:cpu['R3-D' if job['task'].startswith('r3d-') else 'Other resume CPU jobs']+=job['actual_resource_hours']
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,values,unit in zip(axes,(gpu,cpu),('Allocated GPU-hours','Allocated CPU core-hours')):
        bars=ax.barh(list(values),list(values.values()),color='#357894')
        ax.bar_label(bars,fmt='%.3f',padding=4);ax.set_xlabel(unit);ax.margins(x=.2)
    fig.suptitle('Verified allocation costs, including failures\nSnapshot excludes its own final accounting until sacct completes')
    fig.savefig(out/'compute-expenditure.pdf');fig.savefig(out/'compute-expenditure.png',dpi=160);plt.close(fig)


if __name__=='__main__':run()
