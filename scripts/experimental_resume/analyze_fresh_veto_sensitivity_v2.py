"""Separate pretreatment format-adjudicated strict-start sensitivity; CPU only."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import socket

import analyze_overnight_semantics_v2 as frozen
import adjudicate_extension_veto_format_v1 as veto
import dispatch_overnight_readers_v1 as shared

require = frozen.require
sealed = shared.base.sealed
DOC = shared.DOC
PLAN = DOC / 'FRESH_VETO_OUTCOME_SENSITIVITY_PLAN_v2.json'
PROTOCOL = DOC / 'OVERNIGHT_FRESH_COMPARISON_PROTOCOL_v1.json'
MANIFEST = DOC / 'OVERNIGHT_FRESH_COMPARISON_MANIFEST_v2.json'
PRIMARY = DOC / 'OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2'
DEFINITIONS = DOC / 'FRESH_VETO_OUTCOME_SENSITIVITY_DEFINITIONS_v2.md'
FORMAT_RESULT = veto.contract.ROOT / 'veto-format-v1-6b2fe84a01e81700/RESULT.json'
FORMAT_SHA = '29b86bbe7ccbfc07131e6caa0d4a9df91a4a70bc5a7111d828634b718a1af3a0'
PROTOCOL_SHA = '2864678d94e44bb7d9d105ea8af780db36a4e9bdf6c60d458180f4f550f3c80f'


def code_files():
    paths = [Path(__file__), Path(frozen.__file__), Path(frozen.builder.__file__),
             Path(frozen.rating.__file__), Path(shared.__file__), Path(veto.__file__),
             Path(__file__).with_suffix('.sbatch'),
             Path(__file__).with_name('submit_fresh_veto_sensitivity_v2.py'),
             Path(__file__).with_name('submit_fresh_veto_sensitivity_v2.sbatch'),
             Path(__file__).with_name('dispatch_generated_dense_v1.py')]
    return {str(p): shared.base.file_sha(p) for p in paths}


def plan_body():
    protocol, statuses, enrollment = sealed(PROTOCOL), sealed(FORMAT_RESULT), sealed(veto.enrollment.OUT)
    require(protocol['sha256'] == PROTOCOL_SHA and statuses['sha256'] == FORMAT_SHA and
            enrollment['sha256'] == veto.ENROLLMENT_SHA and
            statuses['enrollment_sha256'] == enrollment['sha256'] and
            statuses['plan_sha256'] == sealed(veto.PLAN)['sha256'] and
            len(statuses['records']) == len(enrollment['rows']) == 134,
            'fixed format sensitivity sources differ')
    return {'schema': 'fresh-veto-outcome-sensitivity-plan-v2', 'prior_draft_plan_sha256': sealed(DOC/'FRESH_VETO_OUTCOME_SENSITIVITY_PLAN_v1.json')['sha256'], 'version_change': 'Exact UID joins accommodate the runner deterministic transition/hash ordering; no scientific or enrollment change.', 'code_files': code_files(),
        'definitions_sha256': shared.base.file_sha(DEFINITIONS),
        'protocol_sha256': protocol['sha256'], 'format_result_sha256': statuses['sha256'],
        'enrollment_sha256': enrollment['sha256'], 'source_primary_analysis': str(PRIMARY),
        'primary_contrasts': protocol['primary_contrasts'], 'bootstrap_replicates': 50000,
        'analysis_seed': 20261004, 'horizon': 1024, 'seeds': [0, 1],
        'freeze_state': 'Frozen before fresh generated outcomes or primary analysis were read. Native veto format results are known and explicitly bound.',
        'population': 'Known strict starts among the same fixed 134 native starts: both stopped veto readers valid and false after format adjudication. Unknown veto ratings remain unknown; no replacement enrollment.',
        'multiplicity': 'All 28 frozen contrasts remain in the semantic sensitivity family; continuation tokens have their own 28-contrast secondary family. Neither family replaces the original primary family.',
        'resources': {'analysis': {'CPU_cores': 2, 'memory_GiB': 8, 'wall_seconds': 1800, 'CPU_core_hour_ceiling': 1., 'GPU_hours': 0.},
                      'maximum_two_one_shot_attachments': {'CPU_cores': 2, 'wall_seconds_each': 600, 'CPU_core_hour_ceiling': 2/3, 'GPU_hours': 0.}}}


def validate_plan():
    plan = sealed(PLAN)
    require(plan['binding'] == plan_body(), 'frozen fresh sensitivity plan changed')
    return plan


def validate_manifest(manifest, enrollment, protocol):
    require(manifest['schema'] == 'overnight-routing-manifest-v2' and
            manifest['source_enrollment_sha256'] == enrollment['sha256'] == veto.ENROLLMENT_SHA and
            Path(manifest['source_enrollment_path']).resolve() == veto.enrollment.OUT.resolve() and
            manifest['horizon'] == protocol['horizon'] == 1024 and manifest['seeds'] == protocol['seeds'] == [0,1] and
            manifest['bootstrap_replicates'] == protocol['bootstrap_replicates'] == 50000 and
            manifest['analysis_seed'] == protocol['analysis_seed'] == 20261004 and
            len({r['uid'] for r in manifest['rows']}) == len(manifest['rows']) == len(enrollment['rows']) and
            {r['uid'] for r in manifest['rows']} == {r['uid'] for r in enrollment['rows']},
            'fresh manifest no longer matches the complete fixed enrollment')
    originals = {r['uid']: r for r in enrollment['rows']}
    transitions = manifest['transitions']
    expected_order = sorted(enrollment['rows'], key=lambda r: (transitions.index(r['transition']), shared.base.digest(['overnight-v2-enrollment', r['uid']])))
    require([r['uid'] for r in manifest['rows']] == [r['uid'] for r in expected_order], 'manifest deterministic request order changed')
    for actual in manifest['rows']:
        original = originals[actual['uid']]
        require(all(actual[k] == original[k] for k in ('uid','family','transition','prompt_ids_sha256','prefix_ids_sha256',
                                                       'strict_veto_sensitivity_eligible')), 'native identity or original veto flag changed')
    require({t: [a['name'] for a in arms] for t, arms in manifest['arms_by_transition'].items()} == protocol['arms_by_transition'],
            'fresh assigned arm inventory changed')
    pairs = [{'scope': p['transition'], 'left': p['a'], 'right': p['b']} for p in protocol['primary_contrasts']]
    require(frozen.frozen_pairs(manifest)[0] == pairs and len(pairs) == 28,
            'fresh frozen 28-contrast inventory changed')


def start_statuses(manifest, format_rows):
    require(len({r['uid'] for r in format_rows}) == len(format_rows) == len(manifest['rows']) and
            {r['uid'] for r in format_rows} == {r['uid'] for r in manifest['rows']},
            'format status UIDs duplicate, omit or add fixed starts')
    by_uid = {r['uid']: r for r in format_rows}
    result = []
    for native in manifest['rows']:
        status = by_uid[native['uid']]
        require(all(native[k] == status[k] for k in ('uid','family','transition')) and
                native['strict_veto_sensitivity_eligible'] == status['original_strict_veto_sensitivity_eligible'],
                'veto status changes native family, transition or original flag')
        votes = status['veto_readers']
        require(len(votes) == 2 and [v['reader'] for v in votes] == [0,1] and
                all(type(v['adjudicated_valid']) is bool and
                    (type(v['adjudicated_value']) is bool if v['adjudicated_valid'] else v['adjudicated_value'] is None)
                    for v in votes), 'malformed or reordered veto reader status')
        valid_pair = all(v['adjudicated_valid'] for v in votes)
        known_strict = valid_pair and all(v['adjudicated_value'] is False for v in votes)
        require(valid_pair == status['format_adjudicated_veto_pair_complete'] and
                known_strict == status['format_adjudicated_strict_veto_sensitivity_eligible'],
                'format-adjudicated flag contradicts the saved votes')
        category = 'KNOWN_STRICT' if known_strict else 'KNOWN_PRIOR_COMPLETION' if valid_pair else 'UNKNOWN_VETO'
        result.append({'prefix_uid': native['uid'], 'family': native['family'], 'transition': native['transition'],
            'veto_status': category, 'original_strict_veto_sensitivity_eligible': native['strict_veto_sensitivity_eligible'],
            'format_strict_veto_sensitivity_eligible': known_strict,
            'valid_veto_ratings': sum(v['adjudicated_valid'] for v in votes),
            'unknown_veto_readers': [v['reader'] for v in votes if not v['adjudicated_valid']]})
    return result


def subset_join(manifest, statuses, records):
    by_uid = {r['prefix_uid']: r for r in statuses}
    require(len(by_uid) == len(statuses) == len(manifest['rows']) and
            set(by_uid) == {r['uid'] for r in manifest['rows']}, 'start-status UID set differs')
    require(len({r['uid'] for r in records}) == len(records), 'duplicate outcome assignment UID')
    for record in records:
        status = by_uid.get(record['prefix_uid'])
        require(status is not None and all(record[k] == status[k] for k in ('family','transition')),
                'outcome assignment does not match exact native status UID/family/transition')
    # Validate the COMPLETE assigned population before taking any subset.
    frozen.cells(manifest, records, ('both_positive','emitted_tokens','measurement_unknown'))
    selected = {r['prefix_uid'] for r in statuses if r['veto_status'] == 'KNOWN_STRICT'}
    subset = [r for r in records if r['prefix_uid'] in selected]
    restricted = {**manifest, 'rows': [r for r in manifest['rows'] if r['uid'] in selected]}
    require(frozen.frozen_pairs(restricted) == frozen.frozen_pairs(manifest), 'sensitivity changed frozen contrasts')
    frozen.cells(restricted, subset, ('both_positive','emitted_tokens','measurement_unknown'))
    return restricted, subset


def primary_sources(manifest):
    assigned, primary = sealed(PRIMARY/'ASSIGNED_RESULTS.json'), sealed(PRIMARY/'ANALYSIS.json')
    require(assigned['schema'] == 'overnight-semantic-assigned-results-v2' and primary['schema'] == 'overnight-semantic-itt-v2' and
            assigned['manifest_sha256'] == primary['manifest_sha256'] == manifest['sha256'] and
            primary['assigned_results_sha256'] == assigned['sha256'] and
            all(assigned[k] == primary[k] for k in ('arm_map_sha256','frame_sha256','reader_stage_sha256')) and
            primary['analysis_driver_sha256'] == shared.base.file_sha(frozen.__file__) and
            primary['horizon'] == manifest['horizon'] and
            primary['assigned'] == len(assigned['records']) == manifest['expected_requests'],
            'primary fresh analysis or assigned results are incomplete or rebound')
    expected = frozen.builder.expected_assignments(manifest)
    require(len(expected) == len(assigned['records']) and all(all(row.get(k) == v for k,v in original.items())
            for row,original in zip(assigned['records'],expected,strict=True)), 'primary assigned request identities differ')
    expected_pairs = frozen.frozen_pairs(manifest)[0]
    for name, endpoint in (('primary','both_positive'),('emitted_token_secondary','emitted_tokens')):
        family = primary[name]
        require(family['multiplicity'] == 28 and family['replicates'] == 50000 and family['seed'] == 20261004 and
                family['endpoints'] == [endpoint] and
                [{'scope':c['scope'],'left':c['arm'],'right':c['reference']} for c in family['contrasts']] == expected_pairs,
                'primary result contrast family differs from frozen protocol')
    return assigned, primary


def run():
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
            os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz' and getpass.getuser() == 'lmolfett' and
            not socket.gethostname().startswith('login'), 'fresh sensitivity requires CPU Slurm step')
    plan = validate_plan()
    manifest, enrollment, protocol, formatted = map(sealed,(MANIFEST,veto.enrollment.OUT,PROTOCOL,FORMAT_RESULT))
    validate_manifest(manifest,enrollment,protocol)
    assigned, primary = primary_sources(manifest)
    statuses = start_statuses(manifest,formatted['records'])
    restricted, subset = subset_join(manifest,statuses,assigned['records'])
    out = DOC / ('FRESH_VETO_OUTCOME_SENSITIVITY_v2-' + manifest['sha256'][:16] + '-' + plan['sha256'][:12])
    out.mkdir(exist_ok=True)
    source = {'plan_sha256':plan['sha256'],'manifest_sha256':manifest['sha256'],
        'primary_analysis_sha256':primary['sha256'],'primary_assigned_results_sha256':assigned['sha256'],
        'format_result_sha256':formatted['sha256'],'enrollment_sha256':enrollment['sha256']}
    status_result = shared.save(out/'START_STATUS.json',{'schema':'fresh-veto-start-status-v1',**source,'records':statuses})
    subset_result = shared.save(out/'SUBSET_ASSIGNED_RESULTS.json',{'schema':'fresh-veto-subset-assigned-v1',**source,
        'records':subset,'scope':'Every assigned arm/seed within known strict starts, including errors, caps, closures and invalid outcome ratings.'})
    result = shared.save(out/'ANALYSIS.json',{'schema':'fresh-veto-outcome-sensitivity-v2',
        'status':'COMPLETE_SEPARATE_FORMAT_STRICT_SENSITIVITY',**source,
        'start_status_sha256':status_result['sha256'],'subset_assigned_results_sha256':subset_result['sha256'],
        'all_start_status_counts':dict(Counter(r['veto_status'] for r in statuses)),
        'all_start_status_counts_by_transition':{t:dict(Counter(r['veto_status'] for r in statuses if r['transition']==t)) for t in manifest['arms_by_transition']},
        'unknown_veto_ratings':sum(len(r['unknown_veto_readers']) for r in statuses),
        'selected_starts':len(restricted['rows']),'selected_families':len({r['family'] for r in restricted['rows']}),
        'selected_assignments':len(subset),'horizon':manifest['horizon'],
        'semantic_sensitivity':frozen.clustered_contrasts(restricted,subset),
        'emitted_token_secondary':frozen.clustered_contrasts(restricted,subset,('emitted_tokens',)),
        'unknown_outcome_identification_bounds':frozen.missing_sensitivity(restricted,subset),
        'arm_summary':frozen.arm_summary(restricted,subset),
        'interpretation':'Separate exploratory sensitivity conditional on known pretreatment strict starts after a transparent format amendment. Six capped veto ratings remain unknown. Does not replace primary ITT or original strict-veto flags; no claim about unknown strict membership, original-prompt utility, human validation, equivalence or optimal routing.'})
    print(json.dumps({'result':str(out/'ANALYSIS.json'),'sha256':result['sha256'],
        'selected_starts':result['selected_starts'],'selected_families':result['selected_families'],
        'selected_assignments':result['selected_assignments']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze-plan','analyze'))
    args=parser.parse_args()
    if args.mode=='freeze-plan':
        value=shared.save(PLAN,{'schema':'fresh-veto-outcome-sensitivity-plan-envelope-v2',
            'frozen_utc':datetime.now(timezone.utc).isoformat(),'binding':plan_body()})
        print(value['sha256'])
    else:
        run()


if __name__=='__main__':
    main()
