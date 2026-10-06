"""Preserve the native bound forward before importing frozen validators.

The validators' module installs class hooks on import. A loaded native engine
has no steering controller. Its existing instance forward must remain native.
"""
from native_finalization_worker_v1 import NativeFinalizationWorker as V1


class NativeFinalizationWorker(V1):
    def finalization_install(self):
        runner = self.model_runner
        if hasattr(self, '_nf'):
            return self._nf['status']
        native = runner._model_forward
        if getattr(native, 'steer_wrapped', False) or getattr(runner, 'steer_ctl', None) is not None:
            raise ValueError('native capture requires an uninstrumented loaded forward')
        # Instance shadowing retains the exact native callable when importing
        # vLLM validators changes the class. No frozen hooks are altered.
        runner._model_forward = native
        result = V1.finalization_install(self)
        result['adapter_correction'] = 'v2: pin native instance forward before validator import'
        return result
