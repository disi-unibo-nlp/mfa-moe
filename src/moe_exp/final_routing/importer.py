"""Reference existing science and derive new COUNT nominations on allocated CPUs."""
from collections import defaultdict
from pathlib import Path
from .common import MODELS, check_files, file_sha, load, publish, require
from .design import allocate, assignments


def inventory(config):
    root, source = Path(config['root']), Path(config['source_campaign'])
    parent = load(source/'CAMPAIGN.json')
    require(file_sha(source/'CAMPAIGN.json'), 'missing source campaign')
    references, rosters = [], {}
    for model in MODELS[:-1]:
        for name in ('PREFIXES.json', 'SOURCES.json', 'NOMINATIONS.json', 'DERIVED.json', 'DYNAMICS.json'):
            path = source/model/name
            require(path.is_file(), f'missing required existing evidence: {path}')
            references.append({'model': model, 'artifact': name, 'path': str(path), 'sha256_file': file_sha(path)})
        roster = load(source/model/'PREFIXES.json')
        require(len(roster['records']) == 129, 'original prefix roster differs')
        rosters[model] = roster['records']
        publish(root/model/'PREFIXES.json', {'records': roster['records'], 'source_path': str(source/model/'PREFIXES.json'),
                                           'source_sha256': roster['sha256']})
    common = [{'benchmark': r['benchmark'], 'family': r['family'], 'fold': r['fold']} for r in rosters['gpt']]
    for model, rows in rosters.items():
        require({(r['benchmark'], r['family'], r['fold']) for r in rows} ==
                {(r['benchmark'], r['family'], r['fold']) for r in common}, 'models disagree on frozen slots')
    allocation = publish(root/'ALLOCATION.json', allocate(common, config['seed']))
    return publish(root/'IMPORT.json', {'status': 'complete', 'references': references,
        'prefix_slots': {m: len(rows) for m, rows in rosters.items()},
        'available_prefixes': {m: sum(r['status'] == 'enrolled' for r in rows) for m, rows in rosters.items()},
        'allocation_sha256': allocation['sha256'], 'native_reused': True,
        'lfm_native_status': 'unmeasured', 'historical_failures_preserved': True})


def derive_model(binding, model):
    import numpy as np
    import pandas as pd
    from .selection import nominate
    from moe_exp.research_campaign_v3_r6.inputs import counts_for
    config = binding['config']
    root, source = Path(config['root']), Path(config['source_campaign'])
    if (root/model/'COUNT_NOMINATIONS.json').exists():
        result=load(root/model/'COUNT_NOMINATIONS.json')
        finish_assignments(binding,model,result)
        return result
    if model == 'lfm25':
        data_root = root/'lfm25'
        require((data_root/'DISCOVERY_INPUTS.json').exists(), 'LFM native discovery extraction not complete')
        data = load(data_root/'DISCOVERY_INPUTS.json')
        sources = data['sources']
        with np.load(data['counts'], allow_pickle=False) as archive:
            counts, tokens = archive['counts'], archive['tokens']
        sentences = pd.read_parquet(data['sentences']).to_dict('records')
    else:
        parent = load(source/'CAMPAIGN.json')
        sources = load(source/model/'SOURCES.json')['records']
        folds = load(source/'FOLDS.json')
        sources = [{**s, 'fold': folds['folds'][s['family']]} for s in sources]
        counts, tokens = counts_for(parent, model, sources)
        df = pd.read_parquet(parent['saved_inputs'][model]['sentences'])
        qf = folds['question_families']
        sentences = [{'benchmark': r.dataset, 'family': qf[r.question],
                      'fold': folds['folds'][qf[r.question]], 'n_tokens': int(r.n_tokens)} for r in df.itertuples()]
    profile = binding['models'][model]
    result = nominate(model, counts, tokens, sources, profile['layers'], profile['native_k'], sentences, config['seed'])
    from .selection import confirmation
    publish(root/model/'COUNT_CONFIRMATION.json',confirmation(result,counts,tokens,sources,profile['layers'],profile['native_k']))
    publish(root/model/'DISCOVERY_CONTEXT_COUNTS.json', {'records': context_inventory(binding, model, sources),
                                                       'outcomes_used': False})
    result = publish(root/model/'COUNT_NOMINATIONS.json', result)
    finish_assignments(binding,model,result)
    return result


def finish_assignments(binding,model,result):
    """Finish the deterministic tail even if the nomination receipt already committed."""
    config=binding['config'];root=Path(binding['root']);profile=binding['models'][model]
    allocation = load(root/'ALLOCATION.json')
    prefixes = load(root/model/'PREFIXES.json')['records']
    for mode in ('full', 'fallback'):
        rows = assignments(model, profile['profile_sha256'], prefixes, result['records'], allocation, config['seed'], mode)
        publish(root/model/('ASSIGNMENTS-'+mode+'.json'), {'records': rows, 'mode': mode,
                'nominations_sha256': result['sha256'], 'allocation_sha256': allocation['sha256'],
                'available_executions': sum(r['prefix_status'] == 'enrolled' and r['nomination_status'] == 'nominated' for r in rows)})


def context_inventory(binding, model, sources):
    """Discovery-only native context density; no primary continuation is read."""
    from moe_exp.research_campaign_v3_r6.inputs import read_source
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from .common import digest
    root = Path(binding['root'])
    excluded = {r['family'] for r in load(root/'ALLOCATION.json')['records']}
    result=[]
    for source in sources:
        if source['family'] in excluded:continue
        layout=saved_layout(TraceRecord(**read_source(source['source_location'])))
        owned=[p for p in layout['unit_tokens'] if p]
        # Store positions, not labels. The fold-specific H is applied when
        # forecasting each design; all potential first sentences are retained.
        result.append({'model':model,'benchmark':source['benchmark'],'family':source['family'],
            'fold':source['fold'],'contexts':len(owned),'starts':[min(p) for p in owned],
            'ends':[max(p)+1 for p in owned],'completion_tokens':source['completion_tokens'],
            'source_identity':source['attempt_id']})
    return result


def prepare(binding, deadline):
    import time
    root = Path(binding['root'])
    if not (root/'IMPORT.json').exists():
        inventory(binding['config'])
    for model in MODELS[:-1]:
        require(time.time()+180 < deadline, 'bounded CPU preparation deadline; resume untouched models')
        derive_model(binding, model)
    return {'status': 'complete', 'models': list(MODELS[:-1]), 'lfm_discovery_pending': True}
