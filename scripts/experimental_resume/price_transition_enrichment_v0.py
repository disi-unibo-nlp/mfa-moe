"""Count prospective discovery-only full-prefix starts; no ratings or enrollment.

Only already-emitted source-sentence text is read for the stronger numeric
candidate screen. Later sentences, gold, correctness, and class labels are not
inputs. The late-prefix stratum is kept separate for exact-token pricing.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
BASE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
            'claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery')
SENSITIVITY = REPO / 'report/experimental-resume-v1/TRANSITION_PREFIX_ATTRIBUTION_SENSITIVITY_v2.2.json'
UNITS = BASE / 'UNITS.json'
FRAME = BASE / 'TRANSITION_V22_FULL_PREFIX_START_FRAME.json'
OUT = REPO / 'report/experimental-resume-v1/TRANSITION_ENRICHMENT_COUNT_PRICE_v0.json'
TRANSITIONS = ('candidate_to_verify', 'approach_to_commit', 'failed_check_to_revise')
MAX_PER_FAMILY_PER_STRATUM = 6

sys.path.insert(0, str(REPO / 'src'))
from moe_exp.routing_control.prefix import numeric_expression


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed sealed source: ' + str(path))
    return value


def strict_numeric_claim(sentence):
    if re.search(r'\b(?:check|verify|verification|substitut\w*)\b', sentence, re.I):
        return False
    math = re.findall(r'\$([^$\n]{1,240})\$', sentence)
    for expression in math:
        if '=' in expression and numeric_expression(expression.rsplit('=', 1)[-1].strip()) is not None:
            return True
    for boxed in re.findall(r'\\boxed\s*\{([^{}]{1,160})\}', sentence):
        if numeric_expression(boxed) is not None:
            return True
    return False


def stratum(tokens):
    if tokens <= 8192:
        return 'at_most_8192'
    if tokens <= 16384:
        return '8193_to_16384'
    return 'over_16384'


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('frozen dense-unit counting belongs on CPU Slurm')
    sensitivity, units, frame = (sealed(path) for path in (SENSITIVITY, UNITS, FRAME))
    if units['attempts'] != 48 or frame['units_sha256'] != units['sha256'] or sensitivity['units_sha256'] != units['sha256']:
        raise ValueError('discovery source/units/frame differ')
    source = {(r['attempt_id'], r['sentence_index']):
              {'family': r['family'], 'sentence': r['inputs']['sentence'],
               'prefix_tokens': r['token_end']} for r in units['records']}
    already = {(r['transition'], r['attempt_id'], r['sentence_index'])
               for r in frame['records']}
    counts = Counter()
    proposed = defaultdict(list)
    for transition in TRANSITIONS:
        key = 'v2.2|delimiter_aware|' + transition
        for event in sensitivity['eligible_discovery_events'][key]:
            row = source[(event['attempt_id'], event['sentence_index'])]
            if event['family'] != row['family'] or event['prefix_tokens'] != row['prefix_tokens']:
                raise ValueError('event and native dense-unit boundary differ')
            band = stratum(row['prefix_tokens'])
            counts[f'{transition}|{band}|all_fires'] += 1
            if (transition, event['attempt_id'], event['sentence_index']) in already:
                counts[f'{transition}|{band}|already_in_372_frame'] += 1
                continue
            if transition == 'candidate_to_verify' and not strict_numeric_claim(row['sentence']):
                counts[f'{transition}|{band}|strict_numeric_abstain'] += 1
                continue
            uid = digest(['enrichment-count-v0', transition, event['family'],
                          event['attempt_id'], event['sentence_index']])
            proposed[(transition, band, event['family'])].append({
                'uid': uid, 'family': event['family'], 'attempt_id': event['attempt_id'],
                'sentence_index': event['sentence_index'],
                'native_prefix_tokens': row['prefix_tokens']})
    selected = []
    for key, records in sorted(proposed.items()):
        ordered = sorted(records, key=lambda r: digest(['enrichment-selection-v0', r['uid']]))
        selected.extend(ordered[:MAX_PER_FAMILY_PER_STRATUM])
        counts[f'{key[0]}|{key[1]}|pre_screen_rows'] += len(records)
        counts[f'{key[0]}|{key[1]}|capped_rows'] += min(len(records), MAX_PER_FAMILY_PER_STRATUM)
    by_stratum = Counter(stratum(r['native_prefix_tokens']) for r in selected)
    early = by_stratum['at_most_8192']
    # Provisional scenario using the largest Qwen prompt observed in the prior
    # 372-row frame. It is not a guaranteed bound for these new prompts.
    prior_max_qwen_prompt = 8358
    reader_decode_cap = 1024
    early_ratings = 2 * early
    early_prefill_ceiling = early_ratings * prior_max_qwen_prompt
    early_decode_ceiling = early_ratings * reader_decode_cap
    prefill_rate = 4914.0
    decode_rate = 43.71
    cold_load_seconds = 600.0
    shutdown_seconds = 196.0
    repeat_factor = 1.25
    early_wall_estimate = (cold_load_seconds + shutdown_seconds + repeat_factor *
                           (early_prefill_ceiling / prefill_rate +
                            early_decode_ceiling / decode_rate))
    body = {'schema': 'transition-discovery-enrichment-count-price-v0',
            'status': 'COUNT_ONLY_NO_RATING_SUBMISSION',
            'source_sensitivity_sha256': sensitivity['sha256'],
            'units_sha256': units['sha256'], 'existing_frame_sha256': frame['sha256'],
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'fixed_hypotheses': list(TRANSITIONS),
            'screen': 'v2.2 contiguous completed source event, excluding existing 372-frame rows; candidate additionally requires current-sentence closed numeric equality RHS or numeric box and abstains on check/verify/substitution language',
            'selection': 'discovery-only deterministic SHA rank, at most six proposed events per family and prefix stratum; no future sentence/class/correctness fields read',
            'per_family_per_stratum_cap': MAX_PER_FAMILY_PER_STRATUM,
            'counts': dict(counts), 'selected_count_by_stratum': dict(by_stratum),
            'selected_source_rows': selected,
            'at_most_8192_provisional_2_reader_price': {
                'ratings': early_ratings, 'maximum_prompt_tokens_assuming_prior_max': early_prefill_ceiling,
                'maximum_decode_tokens': early_decode_ceiling,
                'prior_max_qwen_prompt_tokens': prior_max_qwen_prompt,
                'prefill_tokens_per_second_stress': prefill_rate,
                'decode_tokens_per_second_stress': decode_rate,
                'cold_load_seconds': cold_load_seconds, 'shutdown_seconds': shutdown_seconds,
                'repeat_factor': repeat_factor,
                'provisional_wall_seconds': early_wall_estimate,
                'provisional_gpu_hours_2_a100': early_wall_estimate * 2 / 3600},
            'late_prefix_price': 'Exact Qwen prompt tokenization and max-context qualification required separately; never truncate 8193–16384 or >16384 emitted prefixes.',
            'gate': 'No judge job may launch from this count-only artifact; first freeze exact full-prefix reader inputs, tokenizer token counts, independent 2-reader rubric and complete-stage ceiling. No causal enrollment until full-prefix eligibility audit completes.',
            'interpretation': 'prospective discovery enrichment to find true starting conditions in saved 48 native families; neither action selection nor validation/confirm effect estimate'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing enrichment count differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'selected_count_by_stratum': dict(by_stratum),
                      'provisional_early_gpu_hours': early_wall_estimate * 2 / 3600}))


if __name__ == '__main__':
    main()
