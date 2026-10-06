"""Frozen source and receipt contract for exploratory extension start readers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import prepare_mechanism_extension_220_v1 as extension
import rate_transition_v22_fullprefix_starts_v2 as primary
import mechanism_extension_strict_veto_v1 as veto
from freeze_mechanism_extension_220_v1 import digest, sealed

DOC = extension.DOC
ROOT = extension.ROOT
SOURCE_PRICE = extension.PRICE
QUAL_MANIFEST = DOC / 'MECHANISM_EXTENSION_READER_QUAL_MANIFEST_v1.json'
QUAL_RESULT = DOC / 'MECHANISM_EXTENSION_READER_QUAL_RESULT_v1.json'
FINAL_PRICE = DOC / 'MECHANISM_EXTENSION_START_READER_PRICE_v2.json'
OUTPUT_ROOT = ROOT / 'ratings-v1-a7a9ea5d1e59cf76'
DRIVER = extension.REPO / 'scripts/experimental_resume/rate_mechanism_extension_start_readers_v1.py'
SEALER = extension.REPO / 'scripts/experimental_resume/seal_mechanism_extension_reader_v1.py'
KINDS = ('primary', 'veto')
READERS = (0, 1)
BATCH = 16
MAX_TOKENS = 1024
PROFILE = {
    'model_snapshot': str(primary.MODEL),
    'model_revision': primary.MODEL.name,
    'tensor_parallel_size': 2, 'dtype': 'bfloat16',
    'kv_cache_dtype': 'bfloat16', 'max_model_len': 49152,
    'max_num_seqs': 32, 'max_num_batched_tokens': 8192,
    'gpu_memory_utilization': .85, 'enforce_eager': True,
    'generation_config': 'vllm', 'language_model_only': True,
    'attention_backend': 'FLASH_ATTN',
    'temperature': .2, 'top_p': .95, 'enable_thinking': True,
    'reasoning_effort': 'low', 'max_tokens': MAX_TOKENS,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inputs():
    """Check the complete frozen source contract without replaying native traces."""
    _, family = extension.frozen_extension()
    selection, frame, price = (sealed(extension.SELECTION), sealed(extension.FRAME),
                               sealed(SOURCE_PRICE))
    prior = sealed(extension.PRIOR_PRICE)
    dictionary = sealed(extension.ACTION_DICTIONARY)
    require(selection['schema'] == 'mechanism-extension-start-selection-v1' and
            selection['extension_family_freeze_sha256'] == family['sha256'] and
            selection['families'] == 220 and
            selection['selection_driver_sha256'] == file_sha(extension.__file__) and
            selection['source_selection_driver_sha256'] == file_sha(
                extension.prior_frame.__file__) and
            selection['action_dictionary_sha256'] == dictionary['sha256'] and
            selection['detector_sha256'] == file_sha(
                extension.REPO / 'src/moe_exp/routing_control/transitions_v2.py') and
            frame['schema'] == 'mechanism-extension-start-frame-v1' and
            frame['selection_sha256'] == selection['sha256'] and
            frame['units_sha256'] == selection['units_sha256'] and
            frame['extension_family_freeze_sha256'] == family['sha256'] and
            frame['driver_sha256'] == file_sha(extension.__file__) and
            frame['source_frame_builder_sha256'] == file_sha(
                extension.prior_frame.__file__) and
            frame['families_in_frozen_pool'] == 220 and
            frame['visible_input_allowlist'] ==
            ['problem', 'emitted_prefix', 'triggering_sentence'] and
            frame['rows'] == len(frame['records']) == len(selection['records']) and
            frame['rows'] > 0 and
            [r['uid'] for r in frame['records']] ==
            [r['uid'] for r in selection['records']] and
            len({r['uid'] for r in frame['records']}) == frame['rows'],
            'extension frame or selection differs from frozen 220-family source')
    require(price['schema'] == 'mechanism-extension-start-reader-price-v1' and
            price['status'] == 'HOLD_EXTENSION_READER_DRIVER_AND_GPU_QUALIFICATION' and
            price['extension_family_freeze_sha256'] == family['sha256'] and
            price['frame_sha256'] == frame['sha256'] and
            price['selection_sha256'] == selection['sha256'] and
            price['pricing_driver_sha256'] == file_sha(extension.__file__) and
            price['reader_message_source_sha256'] == file_sha(primary.__file__) and
            price['reader_rubric_sha256'] == file_sha(primary.RUBRIC) and
            price['strict_veto_message_source_sha256'] == file_sha(veto.__file__) and
            price['prior_price_sha256'] == prior['sha256'] and
            price['rows'] == frame['rows'] and
            price['ratings'] == 4 * frame['rows'] and
            price['primary_start_ratings'] == 2 * frame['rows'] and
            price['strict_veto_sensitivity_ratings'] == 2 * frame['rows'] and
            price['max_decode_tokens'] == price['ratings'] * MAX_TOKENS and
            price['prompt_tokens_all_ratings_exact'] ==
            price['primary_prompt_tokens_exact_twice'] +
            price['strict_veto_prompt_tokens_exact_twice'] and
            price['prompt_tokens_min'] > 0 and
            price['max_model_len'] == PROFILE['max_model_len'] and
            price['prompt_tokens_max'] + MAX_TOKENS <= PROFILE['max_model_len'] and
            price['complete_stage_projected_GPU_h'] > 0 and
            price['max_wall_seconds_per_job'] > 0 and
            price['retry_factor'] >= 1.25,
            'extension complete-stage source price differs')
    require(OUTPUT_ROOT.name == 'ratings-v1-' + price['sha256'][:16],
            'extension output directory is not tied to exact source price')
    shards = price['shards']
    require(bool(shards) and shards[0]['start_row'] == 0 and
            shards[-1]['end_row'] == frame['rows'] and
            all(shards[i]['end_row'] == shards[i + 1]['start_row']
                for i in range(len(shards) - 1)) and
            all(s['start_row'] < s['end_row'] and s['start_row'] % BATCH == 0 and
                (s['end_row'] % BATCH == 0 or s['end_row'] == frame['rows'])
                for s in shards),
            'reader price shards are not complete contiguous batch ranges')
    for row in frame['records']:
        require(row['transition'] in extension.SUPPORTED,
                'unsupported extension transition in reader frame')
        messages(row, 'primary'), messages(row, 'veto')
    return family, selection, frame, price


def messages(row, kind):
    if kind == 'primary':
        return primary.messages(row)
    if kind == 'veto':
        return veto.messages(row)
    raise ValueError('unknown extension reader kind')


def parse(text, kind):
    if kind == 'primary':
        return primary.parse_rating(text)
    if kind == 'veto':
        result = veto.parse_veto(text)
        return None if result is None else {'already_completed': result}
    raise ValueError('unknown extension reader kind')


def rating_seed(uid, kind, reader):
    require(kind in KINDS and reader in READERS, 'reader channel differs')
    return int(digest(['mechanism-extension-rating-v1', uid, kind, reader])[:8], 16) % 2_000_000_000


def save(path, body, *, existing_ok=False):
    """Atomic sealed result, called under a single-writer directory lock."""
    path = Path(path)
    value = {**body, 'sha256': digest(body)}
    if path.exists():
        if existing_ok and sealed(path) == value:
            return value
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f'.partial-{os.getpid()}-{time.monotonic_ns()}')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return value
