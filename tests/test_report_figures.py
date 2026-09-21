"""Check bin boundaries and the report's problem-level resampling unit."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'descriptive_figures', Path(__file__).resolve().parents[1] / 'report/descriptive_figures.py'
)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def test_bins_include_zero_and_one_without_double_counting_boundaries():
    rows = [dict(dataset='d', source_problem_id=str(i), share=x, y=x)
            for i, x in enumerate([0, .2, .4, .6, .8, 1])]
    result = figures.binned(rows, 'share', 'y', [0, .2, .4, .6, .8, 1])
    assert [b['n'] for b in result] == [1, 1, 1, 1, 2]
    assert result[-1]['mean'] == .9


def test_bootstrap_resamples_whole_problems_not_independent_attempts():
    rows = [dict(dataset='d', source_problem_id=p, x=.5, y=y)
            for p, y in [('a', 0), ('a', 0), ('a', 0), ('b', 1)]]
    result = figures.binned(rows, 'x', 'y', [0, 1], bootstrap=True)[0]
    assert result['mean'] == .25
    assert result['ci'] == [0, 1]
    assert figures.binned(rows, 'x', 'y', [0, 1], bootstrap=True)[0] == result


def test_single_problem_has_no_bootstrap_interval():
    rows = [dict(dataset='d', source_problem_id='p', x=.5, y=y) for y in [0,1]]
    assert figures.binned(rows, 'x', 'y', [0,1], bootstrap=True)[0]['ci'] is None
