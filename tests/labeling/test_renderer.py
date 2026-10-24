"""Where the judge's renderer child finds the pinned DSPy packages, without DSPy or a judge."""
from __future__ import annotations

import io
from pathlib import Path
import sys
from unittest.mock import patch

from labeling import batch_predictor, renderer, renderer_child

PINNED = '/somewhere/else/.research-runtime/original-renderer'


def run_child(monkeypatch, *arguments):
    """Run the child's main() on an empty request stream and return its sys.path."""
    monkeypatch.setattr(sys, 'argv', ['renderer_child', 'program.json', *arguments])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))
    monkeypatch.setattr(batch_predictor, 'load_program_and_adapter', lambda program: (None, None))
    renderer_child.main()
    return sys.path


def test_the_renderer_hands_the_child_the_pinned_copy_its_parent_record_names(monkeypatch, tmp_path):
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    parent = {'renderer': {'files': {}, 'path': PINNED}, 'annotator': {'program': 'program.json'}}
    with patch.object(renderer.subprocess, 'Popen') as popen:
        renderer.Renderer(parent)
    assert popen.call_args.args[0][-3:] == ['labeling.renderer_child', 'program.json', PINNED]


def test_the_child_uses_the_given_pinned_copy_and_otherwise_the_one_beside_the_checkout(monkeypatch):
    assert PINNED in run_child(monkeypatch, PINNED)
    beside = str(Path(renderer_child.__file__).resolve().parents[2]/'.research-runtime/original-renderer')
    assert beside in run_child(monkeypatch)
