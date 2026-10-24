# tests/steering/outcome

CPU-only unit tests of `src/steering/outcome`. They load no model and read no capture.

- `test_select.py`: synthetic routing with a planted wrong-heavy expert. Covers within-question z-scores, the mixed-outcome filter, leave-one-question-out stability, protection, the chosen set and its matched controls.
- `test_manifest.py`: nudge actions per layer, where the reasoning closes, and the length-limit prefix (cut + suffix, refused if the source changed).
- `test_analysis.py`: the length-limit curve and the bounded contrasts on hand-made families.
- `test_decide.py`: the declared rule for extending the experiment to more models.
- `test_capture.py`: the r2 roster (fixed, stratified, one question per family) and the capture rows (distinct seeds, capped allowance, native-id capture).
- `test_r2.py`: the r2 window layers, the per-class score and its held-out versions, the set choice with two disjoint matched random draws (also the nearest-usage fallback for the fixed GPT-OSS set), the signal table on a planted effect, and attempt selection.
- `test_sweep.py`: halves A/B, sweep rows (target and both draws per cell), the per-half frame against the mean of six attempts and two draws, the frozen pick rule, and the exit codes of `steering.outcome pick`.

The nudge operator, its counters and the native-id capture are tested with the other routing operators in `tests/steering/test_stream_batch.py`; sealing the capture and sweep releases is tested in `tests/steering/test_release.py`, and the job graph of a capture or sweep release in `tests/steering/test_whole_launch.py`.

```bash
E=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python     # or the vllm-cu129 python
TMPDIR=<a folder you own> PYTHONPATH=src $E -m unittest discover -s tests/steering/outcome
```
