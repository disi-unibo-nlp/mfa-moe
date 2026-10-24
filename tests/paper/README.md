# tests/paper

One test file: `test_paper_optimization.py` checks the scientific vetoes and the blinded, counterbalanced selection of `paper.optimize_sections` before any ELM request is spent. It needs no network, credentials or generated manuscript edits.

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/paper
```

Runs in the light environment (also `python -B -m unittest discover -s tests/paper -t tests/paper`). The table checks of `src/paper` have their own `python -m paper.verify_cl_tables --self-test`.
