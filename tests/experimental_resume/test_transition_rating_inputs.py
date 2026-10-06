"""The semantic reader sees only the fixed local text and frozen transition name."""
import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / 'scripts/experimental_resume/rate_transition_candidates.py'
SPEC = importlib.util.spec_from_file_location('transition_rating_inputs_test', SOURCE)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_rating_input_is_arm_blind_and_excludes_analysis_metadata():
    row = {'uid': 'a', 'transition': 'candidate_to_verify',
           'reader_input': {'problem': 'P', 'previous_sentence': 'Earlier.',
                            'triggering_sentence': 'Candidate is 2.',
                            'later_sentence': 'Substituting 2 gives 4.'},
           'analysis_meta': {'gold': 'SECRET_GOLD', 'arm': 'SECRET_ARM',
                             'later_class_audit': 'SECRET_LABEL'}}
    rendered = str(R.messages(row))
    assert 'SECRET_GOLD' not in rendered
    assert 'SECRET_ARM' not in rendered
    assert 'SECRET_LABEL' not in rendered
    assert 'Candidate is 2.' in rendered


def test_strict_boolean_extraction_and_distinct_reader_seeds():
    assert R.parse_rating('</think> {"start": true, "target": false}') == {'start': True, 'target': False}
    assert R.parse_rating('{"start": true, "start": false, "target": true}') is None
    assert R.parse_rating('uncertain') is None
    assert R.rating_seed('uid', 0) != R.rating_seed('uid', 1)


def test_v2_uses_qualified_judge_limit_and_remains_arm_blind():
    path = SOURCE.with_name('rate_transition_candidates_v2.py')
    spec = importlib.util.spec_from_file_location('transition_rating_inputs_v2_test', path)
    version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version)
    assert version.MAX_TOKENS == 1024
    row = {'transition': 'failed_check_to_revise',
           'reader_input': {'problem': 'P', 'previous_sentence': 'Earlier.',
                            'triggering_sentence': '2 != 3, so this fails.',
                            'later_sentence': 'Use the other branch.'},
           'analysis_meta': {'gold': 'FORBIDDEN'}}
    assert 'FORBIDDEN' not in str(version.messages(row))
