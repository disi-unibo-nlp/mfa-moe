"""Full-source duplicate exclusions retain transitive prior families."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts/experimental_resume")]
from prepare_independent_math_v1 import duplicate_edges, normalize
from moe_exp.routing_control.study_v1 import freeze_independent_families


def test_exact_normalization_and_prior_family_transitive_exclusion():
    assert normalize(" x\n  = １ ") == "x = 1"
    texts = {"old1": "old question", "old2": " x = 1", "new0": "x\n= １"}
    tokens = {"old1": list(range(100, 140)), "old2": list(range(40)), "new0": list(range(40))}
    edges, _ = duplicate_edges(tokens, texts, [("old1", "old2")])
    questions = [{"id": "new0", "subject": "algebra", "level": 1}]
    result = freeze_independent_families(questions, edges, ["old1", "old2"], revision="pinned")
    assert result["eligible"] == 0


def test_near_duplicate_containment_connects_without_exact_text_match():
    tokens = {"a": list(range(50)), "b": list(range(50)) + list(range(100, 110)),
              "c": list(range(200, 250))}
    edges, exact = duplicate_edges(tokens, {"a": "first", "b": "second", "c": "third"})
    assert ("a", "b") in edges
    assert not any("c" in edge for edge in edges)
    assert exact == []
