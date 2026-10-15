# Study tests

```bash
PYTHONPATH=src python -m unittest discover -s tests/study -v
```

Tests cover request identities, admission failures, cleanup, durable recovery,
submission reconciliation, accounting and the frozen scientific calculations.
CPU tests load no model. The allocated validation also runs the real-tokenizer
trace-to-export fixture; a separate two-GPU check exercises LFM and GPT.
