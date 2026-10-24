# tests/capture

Tests of `src/capture` on synthetic data: the manifest, resume markers, verification receipt and completeness check on a tiny shard, and the start of `capture_routers.main` (the model comes from the manifest, and gpt gets the newer kernels); the sliding-window routing reducer; the capture buffers, memory budget, token check and agreement gates; and `reduce_capture` on a synthetic trace. The GPU path itself (`backends.py`, `vllm_capture.py`, the rest of `capture_routers.main`) has no test.

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/capture
```

Light environment: `test_capture_manifest.py` and `test_capture_routing_reducer.py` run; `test_capture_reduce.py` and `test_capture_signals.py` need torch (and transformers) and are skipped there. Use the vLLM environment to run everything: `python -B -m unittest discover -s tests/capture -t tests/capture` (67 tests, CPU only, about 15 s). Set `TMPDIR` to a folder you own.
