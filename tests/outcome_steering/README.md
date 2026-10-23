# Outcome-steering tests

CPU-only unit tests of `src/moe_exp/outcome_steering`. They load no model and read no capture.

- `test_select.py`: r1 selection on synthetic routing with a planted wrong-heavy expert (within-question
  z-scores, mixed-outcome filter, leave-one-question-out stability, protection, matched controls).
- `test_manifest.py`: nudge actions per layer, where the reasoning closes, the length-limit prefix.
- `test_analysis.py`: the length-limit curve and the bounded contrasts on hand-made families.
- `test_capture.py`: the r2 roster (fixed, stratified, one question per family) and capture rows.
- `test_r2.py`: the r2 window layers, the per-class score and its held-out versions, the set choice with
  two disjoint matched random draws, the fixed GPT-OSS set, the signal table on a planted effect, and
  attempt selection.
- `test_sweep.py`: halves A/B, sweep rows (target and both draws per cell), the per-half frame against the
  mean of six attempts and two draws, and the frozen pick rule.

The nudge operator, its counters and the native-id capture are tested with the other routing
operators in `tests/study/test_stream_batch.py`.

```bash
E=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python
TMPDIR=<WORK dir> PYTHONPATH=src:scripts/experimental_resume $E -m unittest discover -s tests/outcome_steering
```
