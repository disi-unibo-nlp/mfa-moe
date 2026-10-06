"""Prefix-only transition proposals abstain on incomplete or forbidden evidence."""
from moe_exp.routing_control.transitions import StreamingTransitionDetector


def record(text, **extra):
    return {'problem': 'Find x with 2x=4.', 'emitted_token_ids': list(range(len(text))),
            'emitted_text': text, **extra}


def names(result):
    return [event.transition for event in result['events']]


def test_candidates_require_complete_visible_evidence_and_dedupe_repeated_calls():
    detector = StreamingTransitionDetector()
    assert names(detector.observe(record('The answer is 2'))) == []
    complete = 'The answer is 2. Therefore.'
    first = detector.observe(record(complete))
    assert names(first) == ['candidate_to_verify']
    assert names(detector.observe(record(complete))) == []


def test_approach_and_failed_check_require_specific_operation_and_evidence():
    detector = StreamingTransitionDetector()
    vague = 'Maybe another way. This seems wrong.'
    assert names(detector.observe(record(vague))) == []
    approach = vague + ' We can use substitution to solve the equation.\n'
    assert names(detector.observe(record(approach))) == ['approach_to_commit']
    failed = approach + ' Checking x=3 against 2x=4 fails the check.\n'
    assert names(detector.observe(record(failed))) == ['failed_check_to_revise']


def test_future_and_forbidden_fields_do_not_change_decisions_or_reopen_closure():
    text = 'We can use induction to prove the recurrence.\n'
    first = StreamingTransitionDetector().observe(record(text, gold_answer='42',
                                                         future_completion='wrong'))
    second = StreamingTransitionDetector().observe(record(text, gold_answer='3',
                                                          future_completion='right'))
    assert first == second
    detector = StreamingTransitionDetector()
    assert names(detector.observe(record(text))) == ['approach_to_commit']
    assert detector.observe(record(text + '</think>\nFinal answer: 3'))['events'] == []


def test_incomplete_sentence_and_symbolic_answer_abstain():
    detector = StreamingTransitionDetector()
    assert names(detector.observe(record('We can use factoring to solve'))) == []
    assert names(detector.observe(record('We can use factoring to solve x^2=4.\n'))) == ['approach_to_commit']
    assert names(detector.observe(record('We can use factoring to solve x^2=4.\nThe answer is x.'))) == []


def test_numeric_sentence_period_is_complete_but_decimal_point_is_not():
    detector = StreamingTransitionDetector()
    assert names(detector.observe(record('The value 3.14 does not satisfy the condition'))) == []
    assert names(detector.observe(record('The value 3.14 does not satisfy the condition.'))) == ['failed_check_to_revise']


def test_two_complete_candidates_in_one_increment_are_not_silently_skipped():
    text = 'The answer is 2. More text. The answer is 3. More text.'
    events = StreamingTransitionDetector().observe(record(text))['events']
    assert names({'events': events}) == ['candidate_to_verify', 'candidate_to_verify']
