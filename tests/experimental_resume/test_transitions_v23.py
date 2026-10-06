"""Regression for display-math spans in the next discovery detector version."""
from moe_exp.routing_control.transitions_v23 import classify_sentence_v2


def test_closed_display_math_candidate_is_visible():
    assert 'candidate_to_verify' in classify_sentence_v2(r'We obtain \[x=2\].')
    assert classify_sentence_v2(r'We obtain \[x=2') == ()
