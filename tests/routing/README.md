# tests/routing

Tests of `src/routing`, `src/store.py` and `src/stats.py`: the receipt hashes and the family bootstrap reproduce the numbers of the modules they replaced (`test_store.py`), the selection coverage table comes from the stored receipts (`test_selection.py`), the frozen pieces of the readout: its null function, family folds and numerical helpers, and the rule that a rerun may change code hashes but not inputs (`test_readout.py`), and the lexical classifier's fold pickles (`test_lexical.py`).

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/routing/test_store.py tests/routing/test_selection.py   # light
PYTHONPATH=src python -B -m unittest discover -s tests/routing -t tests/routing                                      # all, vLLM environment
```

`test_readout.py` and `test_lexical.py` need scipy, so run them in the vLLM environment (16 tests, a few seconds). Set `TMPDIR` to a folder you own.
