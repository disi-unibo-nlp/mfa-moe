import pytest

from moe_exp.correlation_pipeline.outcomes import _num, tolerance_match


@pytest.mark.parametrize("text,value", [
    ("\\frac{3}{4}", 0.75),
    ("-\\dfrac{1}{2}", -0.5),
    ("v = \\frac{3}{4}", 0.75),
    ("x = -\\frac{1}{8}\\ \\text{m}", -0.125),
    ("7.26 \\times 10^6", 7.26e6),
    ("1.22\\times10^{-18}\\ \\text{J}", 1.22e-18),
    ("R \\approx 3\\times10^{9}\\ \\text{cm}", 3e9),
    ("3.275e12 \\text{ cm}", 3.275e12),
    ("1,002,001", 1002001.0),
    ("t = 2 s", 2.0),
    ("no number", None),
    ("\\frac{1}{0}", None),
])
def test_num_takes_the_first_numeric_form(text, value):
    assert _num(text) == pytest.approx(value) if value is not None else _num(text) is None


def test_tolerance_match_uses_relative_error():
    assert tolerance_match("v = \\frac{3}{4}", "0.75") is True
    assert tolerance_match("4.49171457", "4.49") is True
    assert tolerance_match("1.36 \\times 10^{-10}", "1.39e-10") is False
    assert tolerance_match("E_{1}", None) is None
