"""Pre-intervention, arm-blind prior-completion sensitivity for extension starts.

The primary extension start criterion remains the original two-reader audit.
This separate two-reader veto is priced for every preselected native start.
"""
from __future__ import annotations

import json

SYSTEM = (
    'You audit a proposed reasoning starting condition. Use only the original '
    'problem, the emitted native reasoning prefix, and the triggering sentence. '
    'Identify the specific candidate value or proposed approach named in the '
    'triggering sentence. Decide whether that same candidate or substantively '
    'equivalent approach was already checked or committed to earlier in the '
    'prefix, before the triggering sentence. Earlier work on a different '
    'candidate or approach does not count. A mere intention, restatement, or '
    'setup is not completion; if identity or prior completion is ambiguous, '
    'answer false. Return exactly one JSON object '
    'with key "already_completed" and a boolean value. Do not assess answer '
    'correctness or any future continuation.'
)
TARGET = {
    'candidate_to_verify': 'substantive checking or verification of the candidate',
    'approach_to_commit': 'substantive commitment to the chosen approach',
}


def messages(row):
    if row['transition'] not in TARGET:
        raise ValueError('unsupported extension start condition')
    if set(row['reader_input']) != {'problem', 'emitted_prefix', 'triggering_sentence'}:
        raise ValueError('reader-visible native input allowlist differs')
    visible = row['reader_input']
    trigger = visible['triggering_sentence']
    prefix = visible['emitted_prefix']
    if not trigger or not prefix or not prefix.rstrip().endswith(trigger.rstrip()):
        raise ValueError('trigger is not at the end of the native prefix')
    user = ('Target act: ' + TARGET[row['transition']] + '\n\n' +
            'Original problem:\n' + visible['problem'] + '\n\n' +
            'Emitted native prefix:\n' + prefix + '\n\n' +
            'Triggering sentence:\n' + trigger + '\n\n' +
            'Was the target act already substantively completed before the '
            'triggering sentence?')
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user}]


def parse_veto(text):
    try:
        if '</think>' in text:
            text = text.rsplit('</think>', 1)[1]
        value = json.loads(text.strip())
    except (ValueError, TypeError, AttributeError):
        return None
    if (isinstance(value, dict) and set(value) == {'already_completed'} and
            type(value['already_completed']) is bool):
        return value['already_completed']
    return None


def strict_valid(primary_votes, veto_votes):
    """Only valid stopped pairs can enter the prespecified sensitivity set."""
    if len(primary_votes) != 2 or len(veto_votes) != 2:
        raise ValueError('exactly two primary and two veto readers required')
    return all(v['rating'] == {'start': True} and v['finish_reason'] == 'stop'
               for v in primary_votes) and all(
        v['already_completed'] is False and v['finish_reason'] == 'stop'
        for v in veto_votes)
