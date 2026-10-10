"""Prefix-only six-decision proposals and frozen two-reader audit adaptation."""
from __future__ import annotations

from dataclasses import asdict
import json
import re
import time

from moe_exp.routing_control.prefix import PrefixInput, candidates
from moe_exp.routing_control.transitions_v24 import classify_sentence_v2, _completed_sentences, _closed_math
from moe_exp.routing_control.study_v1 import digest, require, STATES
from moe_exp.routing_control.study_v2 import RECOGNIZES, seal, verify, qualify_recognizer

VERSION = 'routing-six-decision-proposals-v2.1'
CRITERIA = {
    'intermediate': 'The prefix contains a completed proposed mathematical intermediate value, expression or relation that does not answer all parts of the original problem.',
    'complete_unchecked': 'The prefix contains a completed proposed answer to all requested parts of the original problem and has not yet substantively checked that candidate against an original condition.',
    'complete_checked': 'A completed proposed answer to all requested parts is visibly checked against an original condition with substantive calculation or constraint detail.',
    'identified_approach': 'The current completed sentence tentatively identifies a specific method, construction, case split, formula or representation and what operation it would perform.',
    'substantive_verification': 'The prefix visibly performs a substantive computation or constraint comparison to check a proposed candidate against an original condition. Merely intending to check is false.',
    'substantive_commitment': 'The prefix visibly executes the previously identified tentative approach in a substantive mathematical step. Merely naming or intending to use the approach is false.',
}


def proposals(text):
    # The corrected detector protects open symbolic math. General symbolic
    # equalities are added explicitly; the numeric candidate parser is retained.
    units = list(_completed_sentences(text))
    sentence = units[-1][2] if units else ''
    boundary = bool(units and not text[units[-1][1]:].strip())
    labels = classify_sentence_v2(sentence)
    closed = _closed_math(sentence)
    symbolic = bool(closed and any(re.search(r'(?<![<>=])=(?![=])|\\(?:equiv|approx|to)\b', x) for x in closed))
    candidate = bool(candidates(sentence) or 'candidate_to_verify' in labels or symbolic)
    complete = bool(candidate and re.search(r'\b(?:final\s+answer|answer\s+is|the\s+answer|solution\s+is)\b|\\boxed\s*\{', sentence, re.I))
    verification = bool(re.search(r'\b(?:checking|substituting|substitute|verify|satisfies|satisfy|checks\s+out)\b', text, re.I) and
                        re.search(r'(?<![<>=])=(?![=])|\\(?:leq|geq|equiv)|≠', text))
    commitment = bool(re.search(r'\b(?:using|applying|we\s+(?:apply|substitute|factor|differentiate|integrate)|expanding)\b', text, re.I) and
                      re.search(r'(?<![<>=])=(?![=])', text))
    return {'boundary': boundary, 'symbolic_candidate': symbolic, 'triggering_sentence': sentence,
        'intermediate': candidate and not complete, 'complete_unchecked': complete and not verification,
        'complete_checked': complete and verification, 'identified_approach': 'approach_to_commit' in labels,
        'substantive_verification': verification, 'substantive_commitment': commitment}


class CheapRecognizer:
    """Qualification determines whether these proposals may control routing."""
    def __call__(self, prefix):
        started = time.monotonic()
        values = proposals(prefix.emitted_text)
        return {'prefix_sha256': digest(asdict(prefix)), 'boundary': values['boundary'],
            'source_state': None, 'candidate_status': 'complete_checked' if values['complete_checked'] else
                'complete_unchecked' if values['complete_unchecked'] else 'intermediate' if values['intermediate'] else 'unresolved',
            'approach_identified': values['identified_approach'],
            'substantive_acts': [transition for decision, transition in
                (('substantive_verification', 'candidate_to_verify'), ('substantive_commitment', 'approach_to_commit')) if values[decision]],
            'controller_tokens': 0, 'controller_input_tokens': len(prefix.emitted_token_ids),
            'controller_seconds': time.monotonic() - started}


def messages(row):
    visible = row['reader_input']
    require(set(visible) == {'problem', 'emitted_prefix', 'triggering_sentence'} and
            all(isinstance(v, str) for v in visible.values()), 'reader input allowlist differs')
    system = ('Rate mathematical reasoning using only the original problem and the already emitted prefix. '
        'Never infer later text, gold answers or correctness. Symbolic answers count exactly like numeric answers. '
        'Each decision is true, false, or null for ambiguous evidence. '
        'Return exactly one JSON object with these six decisions and source_state (one of ' + ', '.join(STATES) +
        ', or null). No explanation.\n' + '\n'.join(k + ': ' + v for k, v in CRITERIA.items()))
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(visible, ensure_ascii=False)}]


def parse_rating(text):
    text = text.rsplit('</think>', 1)[-1].strip()
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if (not isinstance(value, dict) or set(value) != {*RECOGNIZES, 'source_state'} or
            any(value[d] is not None and type(value[d]) is not bool for d in RECOGNIZES) or
            value['source_state'] not in (*STATES, None)):
        return None
    return value


def wilson(yes, count):
    if count == 0:
        return [0., 1.]
    z, p = 1.959963984540054, yes / count
    denominator = 1 + z*z / count
    middle = (p + z*z / (2*count)) / denominator
    half = z * ((p*(1-p) + z*z / (4*count)) / count)**.5 / denominator
    return [max(0., middle-half), min(1., middle+half)]


def audit(frame, ratings):
    verify(frame)
    require(frame.get('splits_frozen_before_ratings') is True, 'audit splits not frozen')
    indexed = {}
    for row in ratings:
        require(row['uid'] in {r['uid'] for r in frame['records']} and row['reader'] in (0, 1) and
                (row['uid'], row['reader']) not in indexed and row['frame_sha256'] == frame['sha256'],
                'duplicate, foreign or unbound reader')
        indexed[row['uid'], row['reader']] = row
    audits = []
    for decision in RECOGNIZES:
        apparent, yes, held, held_yes, held_usable, eligible, held_families = 0, 0, 0, 0, 0, set(), set()
        symbolic_proposed, symbolic_usable = 0, 0
        disagreements, missing = [], []
        for row in frame['records']:
            pair = [indexed.get((row['uid'], r)) for r in (0, 1)]
            usable = all(r and r['finish_reason'] == 'stop' and r['parsed'] is not None and
                         type(r['parsed'].get(decision)) is bool for r in pair)
            agreement = usable and pair[0]['parsed'][decision] == pair[1]['parsed'][decision]
            confirmed = agreement and pair[0]['parsed'][decision] is True
            if not usable:
                missing.append(row['uid'])
            elif not agreement:
                disagreements.append(row['uid'])
            proposed = row['proposals'][decision]
            if row['split'] == 'apparent' and proposed:
                apparent += 1
                yes += bool(confirmed)
                eligible.add(row['family'])
            elif row['split'] == 'held_out':
                held_families.add(row['family'])
                held_usable += bool(agreement)
                if proposed:
                    held += 1
                    held_yes += bool(confirmed)
            if row['proposals']['symbolic_candidate'] and proposed:
                symbolic_proposed += 1
                symbolic_usable += bool(agreement)
        audits.append(seal({'decision': decision, 'population': 'discovery', 'prefix_only': True,
            'frame_sha256': frame['sha256'], 'eligible_families': sorted(eligible),
            'apparent_positive_count': apparent, 'confirmed_positive_count': yes,
            'held_out_families': sorted(held_families), 'held_out_precision_ci95': wilson(held_yes, held),
            'held_out_coverage': held_usable / max(1, sum(r['split'] == 'held_out' for r in frame['records'])),
            'held_out_positive_count': held, 'held_out_confirmed_count': held_yes,
            'symbolic_candidates_audited': symbolic_proposed > 0 and symbolic_usable == symbolic_proposed,
            'symbolic_proposed': symbolic_proposed, 'symbolic_usable': symbolic_usable,
            'missing': missing, 'disagreement': disagreements,
            'truth_scope': 'Two same-model readers; apparent precision, family support and held-out uncertainty; no new held-out threshold.'}))
    qualification = qualify_recognizer(audits, family_boundary=frame['family_boundary'],
        supported_transitions=['candidate_to_verify', 'approach_to_commit', 'checked_candidate_to_close'])
    return audits, qualification
