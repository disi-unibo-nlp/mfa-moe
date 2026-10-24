"""Check scientific vetoes and blinded selection before spending ELM requests.

Run with PYTHONPATH=src python -m unittest discover -s tests/paper -p test_paper_optimization.py.
These tests require no network, credentials or generated manuscript edits.
"""

import json
import unittest

from paper.optimize_sections import _complete_response, protected_changes, select_candidate


def review(scores_a: float, scores_b: float, choice: str, regression: str = '') -> dict:
    """Construct an independent pairwise judgment for a selection contract.

    Args:
        scores_a: Score shared by draft A's five dimensions.
        scores_b: Score shared by draft B's five dimensions.
        choice: Explicit pairwise preference.
        regression: Optional scientific defect in draft A.
    Returns:
        A judge result with all selection-relevant fields.
    """
    return {
        'scores_A': {str(index): scores_a for index in range(5)},
        'scores_B': {str(index): scores_b for index in range(5)},
        'choice': choice,
        'scientific_regressions_A': [regression] if regression else [],
        'scientific_regressions_B': [],
    }


class PaperOptimizationTests(unittest.TestCase):
    """A higher style score cannot overrule the fixed scientific evidence."""

    def setUp(self) -> None:
        """Bind opposite candidate positions for the independent judges."""
        self.mapping = [{'candidate_label': 'A', 'incumbent_label': 'B'},
                        {'candidate_label': 'B', 'incumbent_label': 'A'}]
        self.clear_guard = {'vetoes': {}, 'added_numeric_literals': []}

    def test_counterbalanced_scores_follow_identity(self) -> None:
        """The second judge's B score belongs to the same candidate."""
        result = select_candidate([review(9, 7, 'A'), review(7, 9, 'B')],
                                  self.mapping, self.clear_guard)
        self.assertEqual(result['selected'], 'candidate')
        self.assertEqual(result['mean_scores'], {'candidate': 9, 'incumbent': 7})

    def test_scientific_regression_vetoes_higher_score(self) -> None:
        """An invented benefit defeats both judges' numerical preference."""
        result = select_candidate([review(10, 7, 'A', 'Invented latency gain'), review(7, 10, 'B')],
                                  self.mapping, self.clear_guard)
        self.assertEqual(result['selected'], 'incumbent')
        self.assertFalse(result['admissible'])

    def test_tie_preserves_incumbent(self) -> None:
        """A flat iteration counts without introducing a gratuitous replacement."""
        result = select_candidate([review(8, 8, 'tie'), review(8, 8, 'tie')],
                                  self.mapping, self.clear_guard)
        self.assertEqual(result['selected'], 'incumbent')

    def test_changed_interval_and_missing_citation_are_detected(self) -> None:
        """Plausible looking measurements still require exact evidence fidelity."""
        baseline = r'Effect \cid{2.5}{0.2}{0.3} from \citep{source}. \label{result}'
        candidate = r'Effect \cid{2.6}{0.2}{0.3}. \label{result}'
        changes = protected_changes(baseline, candidate)['vetoes']
        self.assertIn('intervals', changes)
        self.assertEqual(changes['citations']['removed'], ['source'])

    def test_removing_a_link_does_not_lose_numeric_measurements(self) -> None:
        """Removing a URL version number is not a changed scientific number."""
        baseline = r'\href{https://example.org/model/2024}{Model 2.5} & 64'
        candidate = r'Model 2.5 & 64'
        self.assertFalse(protected_changes(baseline, candidate)['vetoes'])

    def test_uncounterbalanced_judges_are_rejected(self) -> None:
        """A duplicated draft order cannot masquerade as the two-judge design."""
        with self.assertRaisesRegex(ValueError, 'counterbalanced'):
            select_candidate([review(9, 7, 'A'), review(9, 7, 'A')],
                             [self.mapping[0], self.mapping[0]], self.clear_guard)

    def test_http_success_with_truncated_json_is_not_a_completed_review(self) -> None:
        """Reasoning-token exhaustion must trigger recovery, never count a vote."""
        response = {'custom_id': 'abstract-iteration-01-judge-1',
                    'error': None, 'output_text': '{"scores_A":'}
        self.assertFalse(_complete_response(response))

    def test_schema_shaped_placeholder_is_not_a_completed_review(self) -> None:
        """A model's refusal cannot be converted into a zero-score vote."""
        payload = review(0, 0, 'tie')
        payload['reason'] = 'Review not performed; zeros are placeholders.'
        response = {'custom_id': 'abstract-iteration-02-judge-1',
                    'error': None, 'output_text': json.dumps(payload)}
        self.assertFalse(_complete_response(response))


if __name__ == '__main__':
    unittest.main()
