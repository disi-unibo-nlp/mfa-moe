from complete_independent_math_v2 import reservation


def test_reservation_ignores_solution_text_and_preserves_transitive_exclusions():
    source = {"sha256": "source", "revision": "revision", "dataset": "full-MATH"}
    graph = {"source_sha256": "source", "outcomes_used": False, "matched_prior_math500": 500,
             "edges": [["old", "q0"], ["q0", "q1"]], "previously_used": ["old"], "sha256": "graph"}
    questions = [{"id": f"q{i}", "subject": "algebra", "level": i%5+1, "solution": "answer"} for i in range(140)]
    result = reservation(questions, graph, source)
    changed = reservation([{**q, "solution": "different gold"} for q in questions], graph, source)
    assert result["validation"] == changed["validation"] and result["final"] == changed["final"]
    assert result["eligible"] == 138
    assert not {r["question_id"] for r in result["validation"] + result["final"]} & {"q0", "q1"}
