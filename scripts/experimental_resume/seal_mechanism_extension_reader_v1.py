"""Seal a small live reader qualification, then rebind the exact extension price."""
from __future__ import annotations

import argparse
import json

import mechanism_extension_reader_contract_v1 as contract
import prepare_mechanism_extension_220_v1 as extension
from freeze_mechanism_extension_220_v1 import sealed


def qualification_indices(frame):
    first = {}
    for index, row in enumerate(frame['records']):
        first.setdefault(row['transition'], index)
    contract.require(set(first) == set(extension.SUPPORTED),
                     'qualification must sample both supported transitions')
    return sorted(set([first[k] for k in extension.SUPPORTED] + [frame['rows'] - 1]))


def prepare_body(family, selection, frame, price):
    indices = qualification_indices(frame)
    ratings = 4 * len(indices)
    previous = sealed(extension.PRIOR_PRICE)
    load = previous['components_seconds']['two_cold_loads'] / 2
    shutdown = previous['components_seconds']['two_shutdowns'] / 2
    conservative_wall = (
        load + shutdown + 900 + price['retry_factor'] * (
            ratings * price['prompt_tokens_max'] /
            price['bounded_prefill_tokens_per_second'] +
            ratings * contract.MAX_TOKENS /
            price['bounded_decode_tokens_per_second']))
    contract.require(conservative_wall <= 3600,
                     'qualification sample exceeds its one-hour complete price')
    return {
        'schema': 'mechanism-extension-reader-qualification-manifest-v1',
        'status': 'READY_FOR_LIVE_GPU_QUALIFICATION',
        'extension_family_freeze_sha256': family['sha256'],
        'selection_sha256': selection['sha256'], 'frame_sha256': frame['sha256'],
        'source_exact_price_sha256': price['sha256'],
        'contract_sha256': contract.file_sha(contract.__file__),
        'rating_driver_sha256': contract.file_sha(contract.DRIVER),
        'qualification_sealer_sha256': contract.file_sha(__file__),
        'primary_message_source_sha256': contract.file_sha(contract.primary.__file__),
        'veto_message_source_sha256': contract.file_sha(contract.veto.__file__),
        'model_profile': contract.PROFILE,
        'sample_indices': indices,
        'sample_uids': [frame['records'][i]['uid'] for i in indices],
        'sample_transitions': [frame['records'][i]['transition'] for i in indices],
        'reader_channels': [{'kind': kind, 'reader': reader}
                            for kind in contract.KINDS for reader in contract.READERS],
        'ratings': ratings, 'max_decode_tokens': ratings * contract.MAX_TOKENS,
        'qualification_wall_seconds_ceiling': conservative_wall,
        'qualification_GPU_h_ceiling': 2 * conservative_wall / 3600,
        'qualification_wall_limit_seconds': 3600,
        'output_root': str(contract.OUTPUT_ROOT),
        'scope': ('Small live engine/profile and exact chat-template prompt-ID check '
                  'for both extension reader types. This qualification is separate '
                  'from the full 51.443 GPU-hour rating-stage price.'),
    }


def prepare():
    family, selection, frame, price = contract.source_inputs()
    body = prepare_body(family, selection, frame, price)
    value = extension.write_once(contract.QUAL_MANIFEST, body)
    print(json.dumps({'manifest': str(contract.QUAL_MANIFEST),
                      'sha256': value['sha256'],
                      'ratings': value['ratings'],
                      'GPU_h_ceiling': value['qualification_GPU_h_ceiling']}))
    return value


def check_qualification(result, manifest):
    channels = manifest['reader_channels']
    expected = [(kind, reader, uid) for kind in contract.KINDS
                for reader in contract.READERS for uid in manifest['sample_uids']]
    observed = [(r['kind'], r['reader'], r['uid']) for r in result['records']]
    contract.require(
        result['schema'] == 'mechanism-extension-reader-qualification-result-v1' and
        result['status'] == 'PASS' and result['pass'] is True and
        result['qualification_manifest_sha256'] == manifest['sha256'] and
        result['source_exact_price_sha256'] == manifest['source_exact_price_sha256'] and
        result['rating_driver_sha256'] == contract.file_sha(contract.DRIVER) and
        result['contract_sha256'] == contract.file_sha(contract.__file__) and
        result['model_profile'] == contract.PROFILE and
        result['sample_uids'] == manifest['sample_uids'] and
        result['reader_channels'] == channels and
        result['ratings'] == manifest['ratings'] and
        result['prompt_ids_exact_pass'] is True and
        result['model_output_count_pass'] is True and
        result['cap_and_finish_pass'] is True and
        observed == expected and
        all(r['seed'] == contract.rating_seed(r['uid'], r['kind'], r['reader'])
            for r in result['records']) and
        result['generated_tokens'] == sum(r['generated_tokens']
                                          for r in result['records']),
        'live extension reader qualification is absent or differs')


def rebind_body(family, selection, frame, source, manifest, result):
    contract.require(manifest['schema'] ==
                     'mechanism-extension-reader-qualification-manifest-v1' and
                     manifest['source_exact_price_sha256'] == source['sha256'] and
                     manifest['frame_sha256'] == frame['sha256'] and
                     manifest['selection_sha256'] == selection['sha256'] and
                     manifest['extension_family_freeze_sha256'] == family['sha256'] and
                     manifest['qualification_sealer_sha256'] == contract.file_sha(__file__) and
                     manifest['rating_driver_sha256'] == contract.file_sha(contract.DRIVER) and
                     manifest['contract_sha256'] == contract.file_sha(contract.__file__) and
                     manifest['output_root'] == str(contract.OUTPUT_ROOT),
                     'qualification manifest differs from final extension reader code')
    check_qualification(result, manifest)
    previous = sealed(extension.PRIOR_PRICE)
    load = previous['components_seconds']['two_cold_loads'] / 2
    shutdown = previous['components_seconds']['two_shutdowns'] / 2
    contract.require(all(s['estimated_work_seconds'] + load + shutdown + 900 <=
                         source['max_wall_seconds_per_job'] for s in source['shards']) and
                     abs(source['complete_stage_projected_GPU_h'] -
                         2 * source['complete_stage_projected_wall_seconds'] / 3600) < 1e-9,
                     'complete priced shard exceeds live wall or total price differs')
    return {
        **{key: value for key, value in source.items() if key != 'sha256'},
        'schema': 'mechanism-extension-start-reader-price-v2',
        'status': 'PASS_COMPLETE_STAGE',
        'source_exact_price_sha256': source['sha256'],
        'qualification_manifest_sha256': manifest['sha256'],
        'qualification_result_sha256': result['sha256'],
        'rating_driver_sha256': contract.file_sha(contract.DRIVER),
        'contract_sha256': contract.file_sha(contract.__file__),
        'rebind_driver_sha256': contract.file_sha(__file__),
        'qualification_price_separate_GPU_h_ceiling':
            manifest['qualification_GPU_h_ceiling'],
        'recovery_amendment': ('Append-only attempted-channel ledgers and per-UID '
                               'receipts; one committed result per UID/type/reader. '
                               'Uncommitted model calls may be recomputed and are '
                               'counted in physical attempts. The source 1.25 factor '
                               'and recovery load remain in the complete-stage price.'),
        'scope': ('Exploratory 220-family reader audit, separate from registered '
                  '128-family mechanism validation; all 758 frozen starts remain '
                  'assigned for primary and strict-veto sensitivity ratings.'),
    }


def rebind():
    family, selection, frame, source = contract.source_inputs()
    manifest, result = sealed(contract.QUAL_MANIFEST), sealed(contract.QUAL_RESULT)
    value = extension.write_once(contract.FINAL_PRICE,
                                 rebind_body(family, selection, frame,
                                             source, manifest, result))
    print(json.dumps({'price': str(contract.FINAL_PRICE), 'sha256': value['sha256'],
                      'status': value['status'],
                      'complete_stage_GPU_h': value['complete_stage_projected_GPU_h'],
                      'shards': len(value['shards'])}))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'rebind'))
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else rebind()
