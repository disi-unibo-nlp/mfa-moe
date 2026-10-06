"""Freeze fresh-family enrollment and recover its exact qualified native prefixes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import mechanism_extension_reader_contract_v2 as contract
import rate_mechanism_extension_start_readers_v2 as rating
import prepare_mechanism_validation_v1 as native
import prepare_mechanism_extension_220_v1 as extension
from freeze_mechanism_extension_220_v1 import digest, sealed

PLAN = contract.DOC / 'MECHANISM_EXTENSION_OVERNIGHT_ENROLLMENT_PLAN_v1.json'
OUT = contract.DOC / 'MECHANISM_EXTENSION_OVERNIGHT_ENROLLMENT_v1.json'
WRAPPER = extension.REPO / 'scripts/experimental_resume/prepare_extension_overnight_enrollment_v1.sbatch'
ARMS_PER_TRANSITION = {'candidate_to_verify': 14, 'approach_to_commit': 10}
PROTOCOL = contract.DOC / 'OVERNIGHT_FRESH_COMPARISON_PROTOCOL_v1.json'


def plan_body():
    family, selection, frame, price = contract.source_inputs()
    protocol = sealed(PROTOCOL)
    contract.require(protocol['sha256'] == '2864678d94e44bb7d9d105ea8af780db36a4e9bdf6c60d458180f4f550f3c80f' and
                     {t: len(arms) for t, arms in protocol['arms_by_transition'].items()} == ARMS_PER_TRANSITION and
                     protocol['horizon'] == 1024 and protocol['seeds'] == [0, 1],
                     'fresh comparison protocol differs from enrollment plan')
    return {
        'schema': 'extension-overnight-enrollment-plan-v1',
        'extension_family_freeze_sha256': family['sha256'],
        'family_freeze_sha256': family['source_family_freeze_sha256'],
        'comparison_protocol_path': str(PROTOCOL),
        'comparison_protocol_sha256': protocol['sha256'],
        'selection_sha256': selection['sha256'], 'frame_sha256': frame['sha256'],
        'source_reader_price_sha256': price['sha256'],
        'reader_output_root': str(contract.OUTPUT_ROOT),
        'reader_contract_sha256': contract.file_sha(contract.__file__),
        'reader_driver_sha256': contract.file_sha(rating.__file__),
        'preparation_driver_sha256': contract.file_sha(__file__),
        'replay_driver_sha256': contract.file_sha(native.__file__),
        'slurm_wrapper_sha256': contract.file_sha(WRAPPER),
        'families': family['families'], 'transitions': list(extension.SUPPORTED),
        'selection_rule': 'All220 frozen families in fixed hash order; first primary-accepted start per family/transition among the <=3 original event-hash-ordered proposals.',
        'acceptance': 'Both primary readers parsed start=true with natural stop. No post-rating replacements beyond original proposals.',
        'strict_veto': 'Chosen-start sensitivity flag only; never changes primary enrollment or selects an alternative start.',
        'generation_horizon': 1024, 'generation_seeds': [0, 1],
        'generation_arm_counts': ARMS_PER_TRANSITION,
        'scope': 'Fresh220-family exploratory extension comparing all frozen routing methods with shared controls. No winner selected from13-family pilot; no original128-family feasibility rescue.',
    }


def primary_accepted(row):
    votes = row['primary_readers']
    contract.require(len(votes) == 2, 'missing primary reader')
    return all(v['rating'] == {'start': True} and v['finish_reason'] == 'stop' for v in votes)


def strict_sensitivity(row):
    votes = row['veto_readers']
    contract.require(len(votes) == 2, 'missing strict-veto reader')
    return primary_accepted(row) and all(v['already_completed'] is False and
                                         v['finish_reason'] == 'stop' for v in votes)


def choose(selection_rows, rating_rows, family_order):
    contract.require([r['uid'] for r in selection_rows] == [r['uid'] for r in rating_rows],
                     'extension readers differ from exact frozen proposal order')
    family_rank = {family: index for index, family in enumerate(family_order)}
    contract.require(len(family_rank) == len(family_order) and
                     {r['family'] for r in selection_rows} <= set(family_order),
                     'duplicate family or foreign start proposal')
    ordered = sorted(selection_rows, key=lambda r: (family_rank[r['family']],
                     extension.SUPPORTED.index(r['transition']),
                     extension.event_key(r['transition'], r)))
    contract.require(ordered == selection_rows, 'selection no longer follows frozen event hashes')
    ratings = {r['uid']: r for r in rating_rows}
    contract.require(len(ratings) == len(rating_rows), 'duplicate reader UID')
    grouped = defaultdict(list)
    for row in selection_rows:
        grouped[row['family'], row['transition']].append(row)
    chosen, accounting = [], []
    for family in family_order:
        for transition in extension.SUPPORTED:
            candidates = grouped[family, transition]
            contract.require(len(candidates) <= 3, 'too many preselected family candidates')
            positives = [r for r in candidates if primary_accepted(ratings[r['uid']])]
            picked = positives[0] if positives else None
            strict = bool(picked is not None and strict_sensitivity(ratings[picked['uid']]))
            if picked is not None:
                chosen.append({**picked, 'strict_veto_sensitivity_eligible': strict})
            accounting.append({'family': family, 'transition': transition,
                               'proposed_start_uids': [r['uid'] for r in candidates],
                               'primary_accepted_start_uids': [r['uid'] for r in positives],
                               'selected_uid': None if picked is None else picked['uid'],
                               'selected_strict_veto_sensitivity_eligible': strict})
    return chosen, accounting


def completed_reader_rows(frame, price):
    stage = rating.seal_stage()  # Verify all committed channels, attempts and summaries.
    completed = []
    for shard_index, shard in enumerate(price['shards']):
        directory = rating.shard_directory(shard_index)
        for offset in range(shard['start_row'], shard['end_row'], contract.BATCH):
            completed.extend(sealed(directory / 'batches' / f'{offset:06d}.json')['records'])
    contract.require([r['uid'] for r in completed] == [r['uid'] for r in frame['records']] and
                     len(completed) == stage['counts']['rows'],
                     'complete extension reader stage differs from frame')
    return completed, stage


def enroll():
    contract.require(bool(os.environ.get('SLURM_JOB_ID')) and
                     bool(os.environ.get('SLURM_STEP_ID')) and
                     os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz',
                     'native prefix recovery requires authorized viz CPU Slurm step')
    plan = sealed(PLAN)
    contract.require(plan == {**plan_body(), 'sha256': digest(plan_body())},
                     'frozen extension enrollment plan changed')
    family, selection, frame, source, _, _, price = rating.expected_final()
    completed, stage = completed_reader_rows(frame, price)
    chosen, accounting = choose(selection['records'], completed, family['families'])
    import pandas as pd
    attempts = pd.read_parquet(extension.ATTEMPTS)
    attempt_index = {str(r['attempt_id']): r for r in attempts.to_dict('records')}
    replayed = native.replay_rows(chosen, frame, attempt_index)
    chosen_by_uid = {r['uid']: r for r in chosen}
    rows = [{**row,
             'question': family['representative_questions'][row['family']],
             'canonical_question': family['representative_questions'][row['family']],
             'sentence_index': chosen_by_uid[row['uid']]['sentence_index'],
             'strict_veto_sensitivity_eligible': chosen_by_uid[row['uid']]['strict_veto_sensitivity_eligible']}
            for row in replayed]
    counts = dict(Counter(row['transition'] for row in rows))
    requests = sum(counts.get(t, 0) * arms * 2 for t, arms in ARMS_PER_TRANSITION.items())
    body = {
        'schema': 'extension-overnight-enrollment-v1',
        'status': 'READY_EXACT_NATIVE_PREFIXES' if rows else 'NO_ELIGIBLE_STARTS',
        'plan_sha256': plan['sha256'],
        'extension_family_freeze_sha256': family['sha256'],
        'family_freeze_sha256': family['source_family_freeze_sha256'],
        'selection_sha256': selection['sha256'], 'frame_sha256': frame['sha256'],
        'reader_stage_completion_sha256': stage['sha256'],
        'reader_final_price_sha256': price['sha256'],
        'reader_output_root': str(contract.OUTPUT_ROOT),
        'source_reader_price_sha256': source['sha256'],
        'preparation_driver_sha256': contract.file_sha(__file__),
        'replay_driver_sha256': contract.file_sha(native.__file__),
        'frozen_families': family['families'], 'frozen_family_count': len(family['families']),
        'family_pool': family['families'],
        'comparison_protocol_path': str(PROTOCOL),
        'comparison_protocol_sha256': plan['comparison_protocol_sha256'],
        'accepted_family_count': len({r['family'] for r in rows}),
        'accepted_start_count': len(rows), 'accepted_starts_by_transition': counts,
        'strict_veto_sensitivity_start_count': sum(r['strict_veto_sensitivity_eligible'] for r in rows),
        'strict_veto_sensitivity_families': len({r['family'] for r in rows if r['strict_veto_sensitivity_eligible']}),
        'family_transition_accounting': accounting, 'rows': rows,
        'generation_horizon': 1024, 'generation_seeds': [0, 1],
        'generation_arm_counts': ARMS_PER_TRANSITION, 'planned_generation_requests': requests,
        'scope': plan['scope'],
    }
    value = extension.write_once(OUT, body)
    print(json.dumps({'enrollment': str(OUT), 'sha256': value['sha256'],
                      'status': value['status'], 'accepted_families': value['accepted_family_count'],
                      'starts_by_transition': counts, 'planned_generation_requests': requests}), flush=True)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze-plan', 'enroll'))
    args = parser.parse_args()
    if args.stage == 'freeze-plan':
        value = extension.write_once(PLAN, plan_body())
        print(json.dumps({'plan': str(PLAN), 'sha256': value['sha256']}))
    else:
        enroll()
