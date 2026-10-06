"""Protect registered sentence counts and dose matching against boundary/selection errors."""
import importlib.util
from pathlib import Path

from moe_steer.trigger import Lexicon, VocabBytes, THINK_END_ID


def driver():
    path = Path(__file__).resolve().parents[2]/'scripts/experimental_resume/x2_analysis.py'
    spec = importlib.util.spec_from_file_location('saved_x2_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_matching_sentences_preserve_onset_sensitivity_and_exclude_prefix():
    d = driver()
    lexicon = Lexicon(('let us',), max_prefix_tokens=32)
    vocab = VocabBytes.from_strings({0: 'Let us finish.\n', 1: 'Let us check.\n', 2: 'Let us try.\n'})
    state = d.new_fsm(lexicon, vocab)
    state.fast_forward([0])
    assert d.marker_counts(state, [1, 2], lexicon, vocab) == (2, 0)
    assert state.n == 1  # replay must not mutate the prefix shared by arms.
    assert d.marker_counts(state, [THINK_END_ID, 1], lexicon, vocab) == (0, 0)


def test_unfinished_match_requires_emitted_boundary_and_fixed_window():
    d = driver()
    lexicon = Lexicon(('let us',), max_prefix_tokens=32)
    vocab = VocabBytes.from_strings({0: 'Let us', 1: ' check.\n', 2: '\n', 3: 'x'})
    state = d.new_fsm(lexicon, vocab)
    state.fast_forward([])
    assert d.marker_counts(state, [0], lexicon, vocab)[0] == 0
    assert d.marker_counts(state, [0, 1], lexicon, vocab)[0] == 1
    assert d.marker_counts(state, [2]*511+[0, 1], lexicon, vocab)[0] == 0
    state = d.new_fsm(lexicon, vocab)
    state.fast_forward([0])
    assert d.marker_counts(state, [1], lexicon, vocab)[0] == 0  # sentence begins in prefix.


def test_nearest_log_dose_tie_and_operator_support():
    d = driver()
    candidates = [{'policy': 'low', 'magnitude': .5, 'D': .5}, {'policy': 'high', 'magnitude': 1., 'D': 2.}]
    assert d.choose_match(1., candidates)['policy'] == 'low'
    assert not d.choose_match(1., candidates)['eligible']
    off = d.choose_match(1., [{'policy': 'near', 'magnitude': 1., 'D': 1.2}])
    assert not off['eligible'] and off['off_band']
    assert not d.choose_match(1., [{'policy': 'force', 'magnitude': 0, 'D': 1.2}], force=True)['eligible']
    assert not d.choose_match(0., candidates)['eligible']
