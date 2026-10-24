# tests/labeling

Tests of `src/labeling` on small synthetic traces: the sentence splitter and its version-1 mapping (`test_splitter_v2.py`), sentence segmentation of cohorts (`test_cohort_segmentation.py`), the stratified sampler (`test_sample_stratified.py`), the judge client and its tolerance to bad answers (`test_batch_judge_client.py`, `test_batch_predictor_failures.py`), resumable labeling and the `--include-traces` relabel (`test_annotation_batch_resume.py`, `test_annotation_include_traces.py`), and GEPA (`test_gepaLLMAsJudge.py`).

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/labeling --ignore=tests/labeling/test_gepaLLMAsJudge.py
```

That runs in the light environment. `test_gepaLLMAsJudge.py` needs DSPy, and `test_batch_predictor_equivalence.py` needs DSPy and the saved judge program (`results/gepaLLMAsJudge/.../selected_program_20260827_173300.json`; it is skipped without it): use option 3 of [`tests/README.md`](../README.md). No test calls a live judge.
