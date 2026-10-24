# tests/routing

Tests of `src/routing`, `src/store.py` and `src/stats.py`: the receipt hashes and the family bootstrap reproduce the numbers of the modules they replaced, and a receipt or data file already on disk is accepted only when a rerun reproduces it (`test_store.py`), the selection coverage table comes from the stored receipts of the models a campaign lists (`test_expert_selection.py`), the dynamics command takes its models from the campaign and needs `--out` (`test_state_dynamics.py`), the frozen pieces of the readout: its null function, family folds and numerical helpers, the rule that a rerun may change code hashes but not inputs, the analysis folder the two drivers read, and the rosters the helper packages take from `models.roster` (`test_readout.py`), the lexical classifier's fold pickles, the stored ones included (`test_lexical.py`), and the next-sentence labels of the state-transition figure: which successors need a judge, that the rebuilt judge contexts equal the recorded ones, that added labels change only next-state rates, that a relaunched table rebuild finishes, and that the commands name the study folder and the model (`test_next_sentence_labels.py`).

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/routing/test_store.py tests/routing/test_expert_selection.py   # light
PYTHONPATH=src python -B -m unittest discover -s tests/routing -t tests/routing                                      # all, vLLM environment
```

`test_readout.py` and `test_lexical.py` need scipy, so run them in the vLLM environment (16 tests, a few seconds). `test_transition_pairs.py` (14 tests) also runs in the light environment. Set `TMPDIR` to a folder you own.
