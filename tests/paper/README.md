# tests/paper

Two test files. `test_paper_optimization.py` checks the scientific vetoes and the blinded, counterbalanced selection of `paper.optimize_sections` before any ELM request is spent. `test_table_order.py` checks the model display order of `paper.cl_models` and that the table source comparison of `paper.verify_cl_printed_tables` accepts a row reorder and rejects swapped model values and changed rank marks. Neither needs network, credentials, a manuscript or generated edits.

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/paper
```

Runs in the light environment (also `python -B -m unittest discover -s tests/paper -t tests/paper`). The table checks of `src/paper` have their own `python -m paper.verify_cl_tables --self-test`.
