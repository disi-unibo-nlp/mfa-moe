# tests/models

One test file: the memory-safe streaming extraction must give the same signals as a full forward pass (`test_streaming_routing_extraction.py`, on tiny stand-in model classes). It checks the router hooks of `src/models/router_adapters.py`, that they are removed on failure, and that only the last token's logits are requested.

```bash
PYTHONPATH=src:... python -m pytest -q -p no:cacheprovider tests/models      # needs torch and pytest together
```

Use option 3 of [`tests/README.md`](../README.md): the light interpreter with the vLLM environment's packages on the path.
