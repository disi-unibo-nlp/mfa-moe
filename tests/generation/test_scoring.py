"""Answer extraction and thread-safe math_verify scoring."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from generation import scoring
from generation.scoring import score_completion_detailed
from generation.answers import extract_model_answer, last_boxed


@pytest.mark.parametrize("text, expected", [
    (r"so \boxed{\frac{11}{2}}.", r"\frac{11}{2}"),
    (r"first \boxed{3} then \boxed{\dfrac{\sqrt{3}}{3}}", r"\dfrac{\sqrt{3}}{3}"),
    (r"\boxed {(-\infty, 2) \cup (3, \infty)}", r"(-\infty, 2) \cup (3, \infty)"),
    (r"\boxed{\text{Evelyn}}", r"\text{Evelyn}"),
    (r"\fbox{42}", "42"),
    (r"\boxed{7} and an unclosed \boxed{\frac{1}{2}", "7"),
    ("no box here", None),
])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


def test_extract_model_answer_uses_last_balanced_box():
    assert extract_model_answer(r"try \boxed{1}; final \boxed{\frac{243}{625}}") == r"\frac{243}{625}"
    assert extract_model_answer("The answer is 12.") == "12"


CASES = [
    (r"Thus the answer is \boxed{\frac{11}{2}}.", r"\frac{11}{2}", True),
    (r"\boxed{\dfrac{\sqrt3}{3}}", r"\frac{\sqrt{3}}{3}", True),
    (r"\boxed{[-2,\,7]}", r"x \in [-2,7]", True),
    (r"\boxed{\text{Evelyn}}", r"\text{Evelyn}", True),
    (r"\boxed{2\sqrt5}", r"2\sqrt{5}", True),
    (r"\boxed{14}", "12", False),
    (r"\boxed{\frac{1}{3}}", r"\frac{1}{2}", False),
]


@pytest.mark.parametrize("text, gold, expected", CASES)
def test_main_thread_scoring_uses_math_verify(text, gold, expected):
    result = score_completion_detailed({"gold_answer": gold}, answer_type="math", model_text=text)
    assert result["method"] == "math_verify"
    assert result["is_correct"] is expected


def test_threaded_scoring_matches_main_thread():
    def score(case):
        result = score_completion_detailed({"gold_answer": case[1]}, answer_type="math", model_text=case[0])
        return result["model_answer"], result["is_correct"], result["method"]

    main = [score(case) for case in CASES]
    with ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(score, CASES))
    assert threaded == main
    assert all(method == "math_verify" for _, _, method in threaded)


def test_missing_answer_is_not_correct():
    result = score_completion_detailed({"gold_answer": "5"}, answer_type="math", model_text="")
    assert result["is_correct"] in (None, False)


def test_missing_gold_is_unscored():
    result = score_completion_detailed({"gold_answer": ""}, answer_type="math", model_text=r"\boxed{1}")
    assert result["method"] == "unscored_no_gold_answer" and result["is_correct"] is None


def test_large_input_is_bounded():
    text = "x" * 500_000 + r" \boxed{\frac{3}{2}}"
    result = score_completion_detailed({"gold_answer": r"\frac32"}, answer_type="math", model_text=text)
    assert result["is_correct"] is True


def test_outer_timeout_recovers(monkeypatch):
    scoring.shutdown_scoring_pool()
    monkeypatch.setenv("MOE_EXP_SCORING_TEST_DELAY_S", "5")
    monkeypatch.setattr(scoring, "MATH_VERIFY_TIMEOUT_S", 1.0)
    with ThreadPoolExecutor(1) as pool:
        result = pool.submit(score_completion_detailed, {"gold_answer": "2"},
                             answer_type="math", model_text=r"\boxed{2}").result()
    assert result["math_verify_status"] == "timeout"
    assert result["method"] == "normalized_exact_numeric_fallback" and result["is_correct"] is True
    monkeypatch.delenv("MOE_EXP_SCORING_TEST_DELAY_S")
    monkeypatch.setattr(scoring, "MATH_VERIFY_TIMEOUT_S", 60.0)  # fresh pool pays spawn + import
    scoring.shutdown_scoring_pool()
    with ThreadPoolExecutor(1) as pool:
        result = pool.submit(score_completion_detailed, {"gold_answer": "2"},
                             answer_type="math", model_text=r"\boxed{2}").result()
    assert result["method"] == "math_verify"
    scoring.shutdown_scoring_pool()


def test_choice_scoring_unchanged():
    result = score_completion_detailed({"gold_answer": "b"}, answer_type="choice", model_text=r"\boxed{B}")
    assert (result["model_answer"], result["is_correct"], result["method"]) == ("B", True, "exact_multiple_choice")
