# tests/generation

Tests of `src/generation` and `scripts/serve`: the request fields a model card sets and the job deadline (`test_client.py`), the generation contract that must not drift (`test_generate_contract.py`), answer extraction and thread-safe `math_verify` scoring (`test_scoring.py`), that the server command line built from `model_profiles.py` (`serve_flags.py`) is the one each earlier server script started (`test_serve_flags.py`), and real tiny architectures with exact token replay and native chat markers (`test_correlation_model_support.py`).

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/generation --ignore=tests/generation/test_correlation_model_support.py
```

The first four run in the light environment. `test_correlation_model_support.py` needs torch and transformers together with pytest, so it runs with the combined path described in [`tests/README.md`](../README.md) (option 3).
