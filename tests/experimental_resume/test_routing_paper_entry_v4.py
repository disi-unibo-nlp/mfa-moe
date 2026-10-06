"""Paper recovery adapts explicit args without changing or submitting v3 code."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/experimental_resume'))
import routing_paper_entry_v4 as entry


@pytest.mark.parametrize('stage,phase', [('C', 'attach'), ('FRESH', 'attach'), ('C', 'dispatch'), ('FRESH', 'build')])
def test_named_args_delegate_in_canonical_frozen_order(monkeypatch, stage, phase):
    seen = []
    original_argv = list(sys.argv)
    module = SimpleNamespace(__file__='/frozen/routing_paper_v3.py',
                             main=lambda: seen.append(list(sys.argv)))
    monkeypatch.setitem(sys.modules, 'routing_paper_v3', module)
    entry.main(['--phase', phase, '--stage', stage])
    assert seen == [['/frozen/routing_paper_v3.py', stage, phase]]
    assert sys.argv == original_argv


def test_old_inverted_positional_call_is_rejected_before_delegate():
    with pytest.raises(SystemExit) as error:
        entry.arguments(['attach', 'C'])
    assert error.value.code == 2


def test_source_bound_adapter_rejects_any_changed_frozen_source(monkeypatch):
    monkeypatch.setattr(entry, 'FROZEN', {'routing_paper_v3.py': '0'*64})
    with pytest.raises(ValueError, match='frozen paper source changed'):
        entry.validate_sources()
