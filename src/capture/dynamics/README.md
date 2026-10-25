# src/capture/dynamics

The reductions that capture applies to every trace: how the routing moves from token to token, and the sentence table that says which tokens belong to which sentence. They are used inside `capture.reduce`; `classes.saved_layout` is also used by `routing.lfm25_analyses`, `routing.labelled_sentence_table` and the lexical-table builder `routing.lexical_data.build_native`.

| File | What it does |
| --- | --- |
| `routing.py` | `RoutingReducer`: for one layer it keeps a bounded trailing window of the selected experts and emits entropy, switching, overlap with earlier tokens (lags 1, 4, 16, 64), drift between the two halves of a window, and Jensen-Shannon distance. Windows are 64 and 256 tokens with a stride of 64. All logarithms are natural. Shuffle controls use only the window's own past, so no future token changes a saved value. |
| `classes.py` | The sentence table of a trace: the token ids each sentence owns (from the exact saved offsets), its state label and the labels of its neighbours. |
| `common.py` | The window settings, hashes, trace keys, and how a generation ended (`generation_cap`, `natural_termination` or `unknown_termination`). |

`SCHEMA_VERSION` (1) in `__init__.py` is stored in every packet. Change a reduction only together with that version, because stored packets are compared across runs.

Paper use: none directly. The token ownership decided here says which routing belongs to which state in every state-based exhibit.

Tests: `tests/capture/test_capture_routing_reducer.py` (see [`tests/capture`](../../../tests/capture/README.md)).
