from scripts.experimental_resume.evaluate_candidate_veto_v1 import prefix_features


def test_veto_features_ignore_future_unit_field():
    unit = {"inputs": {"problem_statement": "Find x.",
                       "previous_sentence": "Let x be positive.",
                       "sentence": "We obtain $x=5$.",
                       "next_sentence": "future secret"}}
    before = prefix_features(unit)
    unit["inputs"]["next_sentence"] = "changed future and answer"
    unit["gold_answer"] = "secret"
    assert prefix_features(unit) == before
    assert "future secret" not in before
