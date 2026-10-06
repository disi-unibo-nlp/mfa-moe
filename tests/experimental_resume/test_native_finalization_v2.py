"""Regression for validator imports mutating a loaded native runner's class."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experimental_resume'))
from native_finalization_worker_v2 import NativeFinalizationWorker, V1


def test_native_instance_callable_survives_validator_class_hook(monkeypatch):
    class Runner:
        def _model_forward(self, value):
            return value + 7
    worker = NativeFinalizationWorker(); worker.model_runner = Runner()

    def importing_v1_install(self):
        def guard(runner, value):
            raise RuntimeError('absent controller')
        monkeypatch.setattr(Runner, '_model_forward', guard)
        self._nf = {'status': {'answer': self.model_runner._model_forward(5)}}
        return self._nf['status']
    monkeypatch.setattr(V1, 'finalization_install', importing_v1_install)
    assert worker.finalization_install()['answer'] == 12
    assert worker.model_runner._model_forward(8) == 15
    assert worker.finalization_install()['answer'] == 12
    with pytest.raises(RuntimeError): Runner()._model_forward(5)


@pytest.mark.parametrize('kind', ['guard', 'controller'])
def test_rejects_an_existing_intervention(kind):
    def forward(value): return value
    if kind == 'guard': forward.steer_wrapped = True
    worker = NativeFinalizationWorker()
    worker.model_runner = SimpleNamespace(_model_forward=forward,
        steer_ctl=object() if kind == 'controller' else None)
    with pytest.raises(ValueError, match='uninstrumented'):
        worker.finalization_install()
