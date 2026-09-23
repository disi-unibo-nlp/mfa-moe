import pytest
from moe_exp.utils import extract_model_answer
from moe_exp.correlation_pipeline import scoring

@pytest.mark.parametrize('text,expected', [
    (r'Use \boxed{ }. Final: \boxed{7\text{ stuffed goats and }4\text{ toy helicopters}}',
     r'7\text{ stuffed goats and }4\text{ toy helicopters}'),
    (r'\boxed{1} corrected: \boxed{\frac{2}{3}}', r'\frac{2}{3}'),
    (r'\boxed{7} then \boxed{ }', '7'),
    (r'\boxed {\{1,2\}}', r'\{1,2\}'),
    (r'\boxed{7} unfinished \boxed{', '7'),
])
def test_last_nonempty_balanced_box(text, expected):
    assert extract_model_answer(text) == expected

def test_missing_prediction_is_false_but_missing_gold_is_unscored(monkeypatch):
    monkeypatch.setattr(scoring, '_math_verify', lambda *a: None)
    assert scoring.score_completion({'gold_answer': '7'}, answer_type='math', model_text='')[1] is False
    assert scoring.score_completion({}, answer_type='math', model_text='')[1] is None
