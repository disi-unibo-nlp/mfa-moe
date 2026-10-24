"""Derive COUNT nominations and whole-reasoning candidates from the frozen discovery inputs."""
from collections import defaultdict
from pathlib import Path
from store import MODELS, check_files, digest, file_sha, load, publish, require
from steering.allocation import allocate, assignments


def derive_model(binding, model):
    import numpy as np
    import pandas as pd
    from steering.selection import nominate
    from routing.expert_selection import counts_for
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
    from steering.selection import confirmation
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


def discovery_inputs(binding, model):
    """Load one model's frozen discovery sources (with folds), class counts and token denominators.

    LFM2.5's counts live in its own native discovery receipt; every other model reads the
    seven-model source campaign.

    Returns:
        sources, counts, tokens and the provenance hashes of those inputs.
    """
    import numpy as np
    import pandas as pd
    config = binding['config']
    if model == 'lfm25':
        from routing.expert_selection import validate_counts
        from store import CLASSES
        data = load(Path(config['root'])/'lfm25/DISCOVERY_INPUTS.json')
        sources = data['sources']
        # The same archive checks as routing.expert_selection.counts_for.
        with np.load(data['counts'], allow_pickle=False) as archive:
            ids, counts, tokens = (archive[k] for k in ('attempt_id', 'counts', 'tokens'))
            require(list(archive['classes']) == list(CLASSES)+['ALL_REASONING', 'ALL_LABELLED'], 'class order mismatch')
        index = pd.Index(ids).get_indexer([s['attempt_id'] for s in sources])
        require((index >= 0).all() and len(index) == len(ids), 'archive row join failed')
        counts, tokens = counts[index], tokens[index]
        validate_counts(counts, tokens, binding['models'][model]['native_k'])
        return sources, counts, tokens, {'discovery_inputs_sha256': data['sha256'],
                                         'counts_sha256_file': file_sha(data['counts'])}
    from routing.expert_selection import counts_for
    source = Path(config['source_campaign'])
    parent = load(source/'CAMPAIGN.json')
    folds = load(source/'FOLDS.json')
    sources = [{**row, 'fold': folds['folds'][row['family']]}
               for row in load(source/model/'SOURCES.json')['records']]
    counts, tokens = counts_for(parent, model, sources)
    return sources, counts, tokens, {'source_campaign_sha256': parent['sha256'],
                                     'counts_sha256_file': file_sha(parent['saved_inputs'][model]['counts'])}


def derive_whole(binding, model, destination):
    """Publish global RD candidates from the same frozen discovery inputs.

    Args:
        binding: Study binding, used only for discovery data/profile.
        model: Any study model.
        destination: Fresh whole experiment model directory.

    Returns:
        WHOLE_NOMINATIONS receipt; no main-family outcome is read.
    """
    from steering.selection import global_pairs
    sources, counts, tokens, provenance = discovery_inputs(binding, model)
    profile = binding['models'][model]
    # The capped selection stored here is a preview; manifests apply each model's own dose to the candidates.
    records = global_pairs(counts, tokens, sources, profile['layers'], profile['native_k'],
                           10 if model == 'gpt' else 15, 50 if model == 'gpt' else 500)
    return publish(Path(destination)/'WHOLE_NOMINATIONS.json', {
        'model': model, 'selector': 'global-inclusion-risk-difference-sign-stable-four-refits',
        'release_commit': (Path(__file__).resolve().parents[2]/'RELEASE_COMMIT').read_text().strip(),
        'sources_sha256': digest(sources), **provenance,
        'outcomes_used': False, 'layers': profile['layers'], 'experts': int(counts.shape[-1]),
        'records': records})


def context_inventory(binding, model, sources):
    """Discovery-only native context density; no primary continuation is read."""
    from store import read_source
    from schemas import TraceRecord
    from capture.dynamics.classes import saved_layout
    from store import digest
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
