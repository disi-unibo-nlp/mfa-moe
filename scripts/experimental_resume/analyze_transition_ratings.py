"""Reconcile blinded discovery ratings, dense labels, and detector proposals.

This is a descriptive discovery audit. It cannot qualify causal control or
promote an LLM reader to human semantic ground truth.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
BASE = ROOT / 'steering-v1/runs/routing-control-v1/dense-discovery'
FRAME = BASE / 'TRANSITION_AUDIT_CANDIDATES.json'
LABELED = BASE / 'TRANSITION_AUDIT_FIXTURES.json'
RATINGS = BASE / 'ratings-f5b2e28c-74c16a5d'
OUTPUT = REPO / 'report/experimental-resume-v1/TRANSITION_DETECTOR_DISCOVERY_AUDIT.json'
TRANSITIONS = ('candidate_to_verify', 'approach_to_commit', 'failed_check_to_revise')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed audit seal: ' + str(path))
    return value


def reconcile(frame, labeled, binding, summary, parts):
    """Require complete UID identity and forbid changing the reader's visible text."""
    if frame['schema'] != 'transition-audit-candidates-unlabeled-v1':
        raise ValueError('unrecognized candidate frame')
    if labeled['schema'] != 'transition-audit-fixtures-v1':
        raise ValueError('unrecognized labeled frame')
    if binding['schema'] != 'transition-LLM-audit-binding-v2' or binding['fixtures_sha256'] != frame['sha256']:
        raise ValueError('ratings bound to another frame')
    if summary['schema'] != 'transition-LLM-audit-summary-v2' or summary['binding_sha256'] != binding['sha256']:
        raise ValueError('rating summary bound to another batch set')
    if (frame['families'] != labeled['families'] or frame['contiguous_pairs'] != labeled['contiguous_pairs'] or
        frame['frame_counts'] != labeled['frame_counts'] or
        frame['detector_sha256'] != labeled['detector_sha256']):
        raise ValueError('label-dependent frame differs from label-free enrollment')
    original = frame['records']
    enriched = labeled['records']
    if len(original) != len(enriched) or summary['rows'] != len(original):
        raise ValueError('rating and fixture row counts differ')
    result = []
    rated = [r for part in parts for r in part['records']]
    if len(rated) != len(original):
        raise ValueError('rating batches incomplete')
    for a, b, c in zip(original, enriched, rated, strict=True):
        if (a['uid'] != b['uid'] or a['uid'] != c['uid'] or
            a['transition'] != b['transition'] or a['transition'] != c['transition']):
            raise ValueError('rating or labeled fixture UID/order differs')
        if a['reader_input'] != b['reader_input']:
            raise ValueError('reader-visible text changed after rating frame freeze')
        if any(a['analysis_meta'].get(key) != b['analysis_meta'].get(key) for key in a['analysis_meta']):
            raise ValueError('labeled fixture changed detector metadata')
        if len(c['readers']) != 2:
            raise ValueError('two ratings required per row')
        result.append((a, b, c))
    return result


def summarize_transition(rows, frame_counts):
    n = Counter()
    families = defaultdict(set)
    token_positions = []
    source_classes = Counter()
    class_pairs = Counter()
    for candidate, labeled, rated in rows:
        fired = candidate['analysis_meta']['detector_fired']
        band = 'fired' if fired else 'nonfire'
        n[band] += 1
        if fired:
            families['all_fire'].add(candidate['analysis_meta']['family'])
            source_classes[labeled['analysis_meta']['source_class_audit']] += 1
            class_pairs[(labeled['analysis_meta']['source_class_audit'],
                         labeled['analysis_meta']['later_class_audit'])] += 1
        ratings = rated['readers']
        valid = all(x['finish_reason'] == 'stop' and x['rating'] is not None for x in ratings)
        if not valid:
            n[band + '_unresolved'] += 1
            continue
        n[band + '_covered'] += 1
        start = [bool(x['rating']['start']) for x in ratings]
        target = [bool(x['rating']['target']) for x in ratings]
        n[band + '_start_agree'] += start[0] == start[1]
        n[band + '_target_agree'] += target[0] == target[1]
        n[band + '_start_reader0'] += start[0]
        n[band + '_start_reader1'] += start[1]
        n[band + '_start_both'] += all(start)
        n[band + '_start_either'] += any(start)
        n[band + '_target_both'] += all(target)
        n[band + '_local_transition_both'] += all(start) and all(target)
        if fired and all(start):
            families['accepted_fire'].add(candidate['analysis_meta']['family'])
            token_positions.append(candidate['analysis_meta']['prefix_tokens'])
        if not fired and all(start):
            families['sampled_miss'].add(candidate['analysis_meta']['family'])
    if n['fired'] != frame_counts['fired']:
        raise ValueError('not every detector fire was rated')
    if n['nonfire'] > frame_counts['nonfire'] or n['nonfire'] > 200:
        raise ValueError('nonfire sample exceeds frozen frame')
    denominator = n['nonfire']
    weight = frame_counts['nonfire'] / denominator if denominator else None
    # Misses are sampled: this is a descriptive Horvitz-Thompson estimate,
    # conditional on the LLM audit's conservative two-reader definition.
    estimated_misses = n['nonfire_start_both'] * weight if weight is not None else None
    estimated_recall = (n['fired_start_both'] / (n['fired_start_both'] + estimated_misses)
                        if estimated_misses is not None and n['fired_start_both'] + estimated_misses else None)
    return {
        'frame': frame_counts,
        'rated_counts': dict(n),
        'families_all_fires': len(families['all_fire']),
        'families_with_both_reader_valid_fires': len(families['accepted_fire']),
        'families_with_sampled_missed_starts': len(families['sampled_miss']),
        'all_fire_family_ids': sorted(families['all_fire']),
        'both_reader_valid_fire_family_ids': sorted(families['accepted_fire']),
        'valid_fire_prefix_token_positions': sorted(token_positions),
        'fired_source_class_counts': dict(source_classes),
        'fired_class_pair_counts': {a + '→' + b: count for (a, b), count in class_pairs.items()},
        'reader_coverage': (n['fired_covered'] + n['nonfire_covered']) / (n['fired'] + n['nonfire']),
        'fired_start_agreement': n['fired_start_agree'] / n['fired_covered'] if n['fired_covered'] else None,
        'fired_start_precision_both_readers': n['fired_start_both'] / n['fired'] if n['fired'] else None,
        'fired_start_precision_either_reader': n['fired_start_either'] / n['fired'] if n['fired'] else None,
        'sampled_nonfire_inverse_probability_weight': weight,
        'estimated_missed_starts_from_nonfire_sample': estimated_misses,
        'estimated_start_recall_two_reader_audit': estimated_recall,
        'immediate_next_sentence_transition_count_both_readers': n['fired_local_transition_both'],
        'interpretation': 'LLM-audited local starts and immediate next sentence only; sampled misses are descriptive, not validated population truth',
    }


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('transition rating analysis requires CPU Slurm')
    frame, labeled = sealed(FRAME), sealed(LABELED)
    binding, summary = sealed(RATINGS / 'BINDING.json'), sealed(RATINGS / 'SUMMARY.json')
    parts = []
    for start in range(0, len(frame['records']), binding['batch_size']):
        part = sealed(RATINGS / 'batches' / f'{start:06d}.json')
        if part['binding_sha256'] != binding['sha256'] or part['start'] != start:
            raise ValueError('rating batch identity differs')
        parts.append(part)
    reconciled = reconcile(frame, labeled, binding, summary, parts)
    by_transition = defaultdict(list)
    for item in reconciled:
        by_transition[item[0]['transition']].append(item)
    analyses = {t: summarize_transition(by_transition[t], frame['frame_counts'][t]) for t in TRANSITIONS}
    body = {'schema': 'transition-detector-discovery-audit-v1',
            'job_id': os.environ['SLURM_JOB_ID'],
            'candidate_frame_sha256': frame['sha256'],
            'dense_fixture_sha256': labeled['sha256'],
            'rating_summary_sha256': summary['sha256'],
            'rating_binding_sha256': binding['sha256'],
            'analysis_driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'class_coverage_gate': labeled['class_coverage_gate'],
            'unknown_label_rows': labeled['unknown_label_rows'],
            'families': frame['families'], 'contiguous_pairs': frame['contiguous_pairs'],
            'by_transition': analyses,
            'status': 'DESCRIPTIVE_DISCOVERY_ONLY_THRESHOLDS_AND_ELIGIBILITY_NOT_YET_SEALED',
            'scope': 'same-model independent blinded LLM draws; no human truth or causal routing effect'}
    value = {**body, 'sha256': digest(body)}
    if OUTPUT.exists():
        if sealed(OUTPUT) != value:
            raise ValueError('existing discovery audit differs')
    else:
        OUTPUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'families': frame['families'],
                      'supported_start_families': {t: analyses[t]['families_with_both_reader_valid_fires']
                                                   for t in TRANSITIONS}}), flush=True)


if __name__ == '__main__':
    main()
