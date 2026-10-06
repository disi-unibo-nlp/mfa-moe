"""M9's separate single-cut closure request preparation; no routing action is applied."""
from __future__ import annotations

CUT = 14336
TOTAL_BUDGET = 16384
CLOSURE_TEXT = '</think>\n\n'
EMISSION_TEMPLATE = 'Final answer: {candidate}'


def prepare_integer_comparator(completion_ids, *, native_natural_stop, decode, encode, candidates):
    """The registered M9 integer control, using only text at the same fixed cut.

    The caller supplies the frozen legacy parser. Its incomplete-lookahead behavior
    is a historical measurement limitation; the new controller uses its own adapter.
    Native fallbacks are comparison outcomes, never inputs to a closure decision.
    """
    completion = list(completion_ids)
    native = {'completion_token_ids': completion[:TOTAL_BUDGET],
              'tokens_charged': min(len(completion), TOTAL_BUDGET)}
    if native_natural_stop and len(completion) <= CUT:
        return {'status': 'retain_native_finished', **native}
    if len(completion) < CUT:
        return {'status': 'failed_source_prefix', **native, 'operational_correct': False}
    found = candidates(decode(completion[:CUT]))
    if not found:
        return {'status': 'fallback_native_no_integer', **native}
    candidate = max(found, key=lambda c: (c['end'], c['start']))['value']
    emission = list(encode(EMISSION_TEMPLATE.format(candidate=candidate)))
    if not emission or any(type(t) is not int or t < 0 for t in emission):
        raise ValueError('integer comparator tokenizer returned invalid IDs')
    if CUT + len(emission) > TOTAL_BUDGET:
        return {'status': 'failed_emission_budget', 'candidate': candidate,
                'tokens_charged': CUT, 'operational_correct': False}
    return {'status': 'integer_emission', 'candidate': candidate,
            'completion_token_ids': completion[:CUT] + emission,
            'emission_token_ids': emission, 'tokens_charged': CUT + len(emission)}


def prepare_closure(prompt_ids, completion_ids, *, native_natural_stop: bool, encode):
    """Prepare a matched prefix and charge both injection and new answer tokens to 16k.

    A native trace that finished at or before the cut is retained, including incorrect answers.
    Execution must still qualify replay/sampler behavior; this helper does not establish it.
    """
    prompt, completion = list(prompt_ids), list(completion_ids)
    if any(type(t) is not int or t < 0 for t in prompt + completion):
        raise ValueError('saved prompt/completion IDs must be nonnegative integers')
    if native_natural_stop and len(completion) <= CUT:
        return {'status': 'retain_native', 'completion_token_ids': completion,
                'tokens_charged': len(completion), 'injection_tokens': 0}
    if len(completion) < CUT:
        return {'status': 'incomplete_prefix', 'completion_token_ids': completion,
                'tokens_charged': len(completion), 'injection_tokens': 0}
    injected = list(encode(CLOSURE_TEXT))
    if not injected or any(type(t) is not int or t < 0 for t in injected):
        raise ValueError('closure tokenizer returned invalid IDs')
    remaining = TOTAL_BUDGET - CUT - len(injected)
    if remaining < 1:
        raise ValueError('closure injection exhausts the single answer allowance')
    prefix = completion[:CUT] + injected
    return {'status': 'generate_closure', 'prompt_token_ids': prompt + prefix,
            'native_prefix_len': CUT, 'prefix_len': len(prefix),
            'prefix_presence_start': len(prompt), 'injection_token_ids': injected,
            'injection_tokens': len(injected), 'max_new_tokens': remaining,
            'total_budget': TOTAL_BUDGET, 'closure_text': CLOSURE_TEXT}
