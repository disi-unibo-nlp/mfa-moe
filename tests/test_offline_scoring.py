import pytest
from moe_exp.correlation_pipeline.offline_scoring import equivalent, rescore

@pytest.mark.parametrize('prediction,gold', [
    ('14/15', r'\frac{14}{15}'), ('East', r'\text{east}'),
    (r'\dfrac{2}{4}', '0.5'), (r'\sqrt{8}', r'2\sqrt{2}'),
    ('1.24e-8', r'1.24\times10^{-8}'), ('17.5%', '0.175'),
    (r'\{2,1\}', r'\{1,2\}'), ('0.3333333333', '1/3'),
    (r'(2,32)\text{ and }(8,18)','(2,32),(8,18)'),
    (r'0.192\text{ (approximately)}','0.192'), ('x+x','2x'), (r'\$1250','1250'), (r'45\text{ minutes}','45'),
    (r'12\pi\text{ inches per second}',r'12\pi'), ('E(2020,1993)=76','76'), (r'15\mbox{ cm}^2', '15'),
])
def test_equivalent_answers(prediction, gold):
    assert equivalent(prediction, gold)[0]

@pytest.mark.parametrize('prediction,gold', [
    ('East', 'west'), ('east', 'seat'), ('1/2','1/3'),
    ('(2,1)','(1,2)'), ('(1,2)', '12'), ('x','y'),
    ('10000001','10000000'), ('0','1.325e-27'), ('2.65e-27','1.325e-27'),
    (r'7\text{ stuffed goats and }4\text{ toy helicopters}', '7'),
    ('','0'), ('0.33','1/3'),
])
def test_distinct_answers(prediction,gold):
    assert not equivalent(prediction,gold)[0]


def test_final_candidate_only_and_balanced_fraction():
    answer, correct, _ = rescore(r'\boxed{14/15}, corrected: \boxed{1/2}', r'\frac{14}{15}')
    assert answer == '1/2' and not correct
    assert rescore(r'\boxed{\frac{14}{15}}', '14/15')[1]
    assert not rescore('', '7')[1]


def test_missing_gold_is_not_silently_wrong():
    with pytest.raises(ValueError):
        rescore('7', '')
