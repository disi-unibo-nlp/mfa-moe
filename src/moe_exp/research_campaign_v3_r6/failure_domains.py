"""Finite failure dispositions follow capability dependencies, not a global switch."""
from pathlib import Path
from .contracts import digest, load, publish

DEPENDENTS = {
    'reader': ('reader_price','annotate'),
    'saved_preparation': ('measure','freeze'),
    'compact': ('memory_qualify','memory_price','qualify'),
    'analysis': ('analysis_pilot','coordination','supplementary'),
    'contexts': ('contexts','native_contexts'),
    'score': ('score',),
    'inference': ('execute','price','memory_price','measure','freeze','qualify','memory_qualify','reader_price','annotate'),
}


def capability(stage):
    for name,stages in DEPENDENTS.items():
        if stage in stages:return name
    return stage


def record(campaign,result):
    if result.get('failure_scope') not in ('shared','model'):return None
    stage=result['stage'];name=result.get('capability') or capability(stage)
    if name not in DEPENDENTS:return None
    return publish(Path(campaign['root'])/'failure_dispositions'/(digest(result)+'.json'),{
        'capability':name,'model':result.get('model','all'),'scope':result['failure_scope'],
        'dependent_stages':DEPENDENTS[name],'failure':result,
        'bounded_recovery_exhausted':True,'independent_branches_continue':True})


def blocked(campaign,stage,model):
    for path in (Path(campaign['root'])/'failure_dispositions').glob('*.json'):
        value=load(path)
        if stage in value['dependent_stages'] and (value['scope']=='shared' or value['model']==model):
            return value
    return None
