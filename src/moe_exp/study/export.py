"""Measured artifacts and honest collection/results/integration completion states."""
from pathlib import Path
from .common import MODELS, digest, file_sha, load, publish

ARTIFACTS = (
    ('native_profiles', 'Eight pinned native profiles and actual context/geometry qualifications', 'all8'),
    ('native_benchmarks', 'Six native benchmark panels; original attempts and unknown denominators', 'all8'),
    ('readout_results', 'Present-state text/routing comparison plus five lexical null draws', 'all8'),
    ('selection_coverage', '75% retained-expert coverage; three methods by five folds', 'all8'),
    ('native_dynamics', 'Seven states, transitions, dwell, gaps and uncertainty', 'all8'),
    ('hypothesis_cards', 'Discovery COUNT nominations, stability, support and confirmation', 'all8'),
    ('primary_effects', 'Four central assigned-policy effects and simultaneous bound intervals', 'all8'),
    ('response_matrices', 'Full target-by-observed-class responses plus termination and unknowns', 'all8'),
    ('quality_cost', 'Paired final-answer accuracy, reasoning tokens and generated tokens', 'all8'),
    ('sensitivity_dynamics', 'All assigned outer-dose and pulse/postpulse diagnostics', 'all8'),
    ('legacy_supporting', 'Original coordination/transfer obligations and terminal dispositions', 'original7'),
)


def frame(path, rows):
    from moe_exp.research_campaign_v3_r6.artifacts import frame as write
    return write(path, rows)


def export(binding, mode):
    from .analysis import assemble,paired_summaries
    from .delivery import catalogue,diagnostics,figures,legacy_complete
    root = Path(binding['root'])
    data = assemble(binding, mode)
    native=catalogue(binding)
    identity = digest({'mode': mode, 'observed': sorted(data['outcomes']), 'catalogue':native,
                       'labels': [{k: v for k, v in r.items() if k in ('uid', 'first', 'occupancy')}
                                  for r in data['outcomes'].values()]})
    folder = root/'exports'/identity
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    written = {}
    for name, rows in [('primary_effects', data['primary']['contrasts']),
                       ('primary_cells', data['primary']['cells']), ('response_matrices', data['responses']),
                       ('quality_cost', data['quality']), ('sensitivity_dynamics', data['secondary'])]:
        if rows:
            written[name] = frame(folder/(name+'.parquet'), rows)
    publish(folder/'PRIMARY.json', data['primary'])
    paired=paired_summaries(binding,data)
    publish(folder/'PAIRED_SUMMARIES.json',paired)
    publish(folder/'DIAGNOSTICS.json',{'records':diagnostics(binding,data)})
    plots=figures(folder,data)
    references = []
    for name, description, scope in ARTIFACTS:
        sources = list(native.get(name,[]))
        if name in written:
            sources.append({**written[name],'models':[m for m,v in data['models'].items() if v['observed']]})
        if name == 'hypothesis_cards':
            sources=list(native[name])
        if name=='quality_cost':sources.append({'path':str(folder/'PAIRED_SUMMARIES.json'),
            'sha256_file':file_sha(folder/'PAIRED_SUMMARIES.json'),'models':[m for m,v in data['models'].items() if v['observed']]})
        covered=set(m for source in sources for m in source.get('models',[]))
        expected=set(MODELS[:-1] if scope=='original7' else MODELS)
        complete=covered==expected and (name!='legacy_supporting' or legacy_complete(binding))
        references.append({'artifact': name, 'claim': description, 'model_scope': scope,
                           'sources': sources, 'status': 'measured' if complete else 'incomplete',
                           'missing_models':sorted(expected-covered),
                           'population': '129 enrolled family slots; finite fixed models' if name in written else 'source-defined'})
    states = completion(binding, mode, data, references)
    manifest = publish(folder/'paper_results_manifest.json', {'schema': 'final-routing-paper-results-v1',
        'mode': mode, 'model_measurements': data['models'], 'artifacts': references,
        'completion': states, 'figures':plots,'primary_interpretation': 'assigned targeting policy versus count/frequency controls',
        'manuscript_changes': [
            'Replace the four-policy mock with five native/targeted/random signed arms and actual endpoints.',
            'Replace additive beta with hard inclusion/exclusion, original affinities and native weights.',
            'Remove physical compaction and unsupported latency/resident-memory savings claims.',
            'Describe fixed prefixes, local median pulses, fixed-horizon outcomes and missingness bounds.',
            'Retain null, harmful and inconclusive estimates; do not imply noninferiority.',
            'LFM joins main native analyses; legacy anticipation and coordination keep their original scopes.'],
        'canonical_manuscript_edited': False, 'remote_delivery': 'pending-user-push'})
    publish(folder/'MANUSCRIPT_REPLACEMENT_MAP.json',{'records':[{'claim':r['claim'],'artifact':r['artifact'],
        'sources':r['sources'],'status':r['status'],'integration':'pending canonical manuscript task'} for r in references],
        'manuscript_integrated':False})
    for reference in references:
        for source in reference['sources']:
            expected=source.get('sha256_file',source.get('sha256'))
            if expected:
                from .common import require
                require(file_sha(source['path'])==expected,'export provenance differs: '+source['path'])
    return {'path': str(folder), 'manifest_sha256': manifest['sha256'], **states}


def completion(binding, mode, data, references):
    from .common import Units, TERMINAL
    from .delivery import lfm_complete
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    pending = 0
    unresolved=0
    for row in data['assignments']:
        if row['prefix_status'] != 'enrolled' or row['nomination_status'] != 'nominated':
            continue
        state = Units(bulk/row['model']/'executions').state(row['uid'])
        if not state or state['status'] not in TERMINAL:
            pending += 1
        elif state['status']!='complete':unresolved+=1
        elif not (bulk/row['model']/'executions'/row['uid']/'CAPTURE.json').exists():unresolved+=1
    all_models = all((root/m/('ASSIGNMENTS-'+mode+'.json')).exists() for m in MODELS)
    annotation_pending = 0
    reader_units = Units(bulk/'reader'/'annotations')
    for row in data['assignments']:
        p = bulk/row['model']/'contexts'/(row['uid']+'.json')
        if p.exists():
            for task in load(p)['tasks']:
                state=reader_units.state(task['uid'])
                if state is None:
                    annotation_pending += 1
                elif state['status']!='complete':unresolved+=1
    terminal = all_models and pending == 0 and annotation_pending == 0
    required = terminal and not unresolved and all(v['status'] == 'measured' for v in references)
    required &= all(v['observed'] > 0 for v in data['models'].values())
    required &= lfm_complete(binding)
    return {'collection_terminal': terminal, 'required_results_complete': bool(required),
            'manuscript_integrated': False, 'pending_executions': pending,
            'pending_annotation_references': annotation_pending,'unresolved_evidence_references':unresolved}
