# tests/routing

Tests of `src/routing`, `src/store.py` and `src/stats.py`: the receipt hashes and the family bootstrap reproduce the numbers of the modules they replaced, and a receipt or data file already on disk is accepted only when a rerun reproduces it (`test_store.py`), the selection coverage table comes from the stored receipts (`test_selection.py`), the frozen pieces of the readout: its null function, family folds and numerical helpers, and the rule that a rerun may change code hashes but not inputs (`test_readout.py`), the lexical classifier's fold pickles (`test_lexical.py`), and the next-sentence labels of the state-transition figure: which successors need a judge, that the rebuilt judge contexts equal the recorded ones, that added labels change only next-state rates, that a relaunched table rebuild finishes (`test_transition_pairs.py`).

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/routing/test_store.py tests/routing/test_selection.py   # light
PYTHONPATH=src python -B -m unittest discover -s tests/routing -t tests/routing                                      # all, vLLM environment
```

`test_readout.py` and `test_lexical.py` need scipy, so run them in the vLLM environment (16 tests, a few seconds). `test_transition_pairs.py` (14 tests) also runs in the light environment. Set `TMPDIR` to a folder you own.
