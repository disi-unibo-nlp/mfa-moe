# tests/steering/outcome

CPU-only unit tests of `src/steering/outcome`. They load no model and read no capture.

- `test_select.py`: synthetic routing with a planted wrong-heavy expert. Covers within-question z-scores, the mixed-outcome filter, leave-one-question-out stability, protection, the chosen set and its matched controls.
- `test_manifest.py`: nudge actions per layer, where the reasoning closes, and the length-limit prefix (cut + suffix, refused if the source changed).
- `test_analysis.py`: the length-limit curve and the bounded contrasts on hand-made families.
- `test_decide.py`: the declared rule for extending the experiment to more models.

The nudge operator and its counters are tested with the other routing operators in `tests/steering/test_stream_batch.py` (`NudgeTests`).

```bash
E=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python     # or the vllm-cu129 python
TMPDIR=<a folder you own> PYTHONPATH=src $E -m unittest discover -s tests/steering/outcome
```
