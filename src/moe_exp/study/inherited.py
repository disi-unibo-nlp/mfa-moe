"""Original task and ownership joins; saved-data supplements retain seven models."""
from pathlib import Path
import re
import time
from moe_exp.io import (
    submission_paths,
    reservation_paths,
    MODELS,
    REPO,
    digest,
    file_sha,
    load,
    publish,
    require,
    verified,
)


def roots(binding):
    from moe_exp.research_campaign_v3_r6.resume import ancestry
    parent = load(Path(binding['source_campaign'])/'CAMPAIGN.json')
    return [Path(parent['root'])]+[Path(a['root']) for a in ancestry(parent)]


def inventory(binding):
    from moe_exp.research_campaign_v3_r6.observations import pair_tasks
    source = Path(binding['source_campaign'])
    parent = load(source/'CAMPAIGN.json')
    tasks, manifests, owners = [], [], []
    for model in MODELS[:-1]:
        # Bind ALL inherited manifests, including unsubmitted and now excluded
        # physical-compaction assignments. Their identities are not relabelled
        # as final-routing execution or silently marked complete.
        paths = sorted(set((source/model).glob('*TASKS*.json')) | set((source/model).glob('*SHARDS*.json')) |
                       set((source/model).glob('*ASSIGNMENTS*.json')))
        for path in paths:
            value = load(path)
            manifests.append({'model':model,'path':str(path),'sha256_file':file_sha(path),
                'manifest_sha256':value.get('sha256'),'records':len(value.get('records',value.get('shards',[]))),
                'execution_scope':'historical identity retained; only native-analysis supplements continue'})
        for row in pair_tasks(parent,model):
            path = source/model/'coordination'/('pair-%03d'%row['shard_index'])/'RESULT.json'
            tasks.append({'stage':'coordination','model':model,**row,'source_root':str(source),
                'receipt_path':str(path),'observed_status':load(path).get('status') if path.exists() else 'unsubmitted_or_owned',
                'ownership':'original until accepted replacement','stochastic_redraw':False})
        for stage,name in [('supplementary','SUPPLEMENTARY.json'),('analysis_finalize','CACHE_RETIRED.json')]:
            path=source/model/name
            tasks.append({'stage':stage,'model':model,'source_root':str(source),'receipt_path':str(path),
                'observed_status':load(path).get('status') if path.exists() else 'unsubmitted_or_owned',
                'ownership':'original until accepted replacement','stochastic_redraw':False})
    for ancestor in roots(binding):
        for path in (ancestor/'submissions').glob('*/ACCEPTED.json'):
            receipt=load(path)
            owners.append({'job_id':str(receipt['job_id']),'path':str(path),'sha256_file':file_sha(path),
                          'source_root':str(ancestor),'card_sha256':receipt.get('card_sha256')})
    return publish(Path(binding['root'])/'INHERITED_TASKS.json',{'records':tasks,'manifests':manifests,'ownership_journals':owners,
        'protected_report_jobs':binding['config']['protected_report_jobs'],'unsubmitted_tasks_retained':True,
        'historical_anticipation_and_coordination_models':list(MODELS[:-1]),'physical_compaction':'outside completion path'})


def accounting(binding, run=None):
    from .scheduler import command
    run=run or command
    ids=set()
    for root in [Path(binding['root']),*roots(binding)]:
        for path in submission_paths(root):
            ids.add(str(load(path)['job_id']))
    rows={}
    ids=sorted(ids)
    for offset in range(0,len(ids),100):
        raw=run(['sacct','-nP','-X','-j',','.join(ids[offset:offset+100]),
                 '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'])
        for line in raw.splitlines():
            jid,state,code,seconds,tres=line.split('|')[:5]
            if '.' in jid:continue
            billing=re.search(r'(?:^|,)billing=([\d.]+)',tres)
            require(billing or not int(seconds or 0),'allocated ancestor has no billing evidence')
            rows[jid]={'job_id':jid,'state':state,'exit_code':code,'seconds':int(seconds or 0),
                'billing_hours':float(billing[1])*int(seconds or 0)/3600 if billing else 0.,'tres':tres}
    return {'records':list(rows.values()),'actual_hours':sum(r['billing_hours'] for r in rows.values()),
            'deduplicated_by_physical_job_id':True,'requested_ids':ids}


def parent_campaign(binding):
    """Verify inherited scientific proofs without installing scheduler adapters."""
    from moe_exp.research_campaign_v3_r6_reader_price_failure_id_r1 import check_binding
    parent = load(Path(binding['source_campaign']) / 'CAMPAIGN.json')
    check_binding(parent)
    return parent


def price(binding,deadline,models=MODELS[:-1]):
    from moe_exp.research_campaign_v3_r6.analysis import pilot
    parent=parent_campaign(binding)
    result=[]
    for model in models:
        target=Path(parent['root'])/model/'ANALYSIS_PRICE.json'
        if not target.exists():
            require(time.time()+60<deadline,'inherited CPU pricing deadline; committed extraction remains reusable')
            pilot(parent,model)
        result.append({'model':model,'path':str(target),'sha256_file':file_sha(target)})
    if tuple(models) == MODELS[:-1]:
        return publish(Path(binding['root'])/'INHERITED_PRICES.json',{'status':'complete','records':result})
    return {'status':'complete','records':result,'original_assignments_preserved':True}


def execute(binding,model,indices,deadline):
    from moe_exp.research_campaign_v3_r6.observations import coordination,supplementary
    from moe_exp.research_campaign_v3_r6.analysis import storage_price
    parent=parent_campaign(binding)
    supplementary(parent,model)
    for index in indices:
        require(time.time()+60<deadline,'inherited coordination deadline')
        path=Path(parent['root'])/model/'coordination'/('pair-%03d'%index)/'RESULT.json'
        if not path.exists():coordination(parent,model,index)
        require(load(path)['status']=='complete','original pair failed; do not replace its identity')
        storage_price(parent,model,index)
    return {'status':'complete','model':model,'original_pair_indices':indices,'stochastic_draws':0}
