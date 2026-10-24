# src/routing/rkin

Routing kinematics: how fast the selected experts change over the last tokens of a sentence, and whether that predicts the next state. A helper package for `routing.anticipation`. Its plan was frozen, so change nothing here.

| File | What it does |
| --- | --- |
| `common.py` | Paths, constants and provenance helpers. Inputs are read only. |
| `k2.py` | The anticipation hazard from the last 16 and 32 reasoning tokens of a sentence: pair tables, lexical counts, kinematic columns, a baseline-only inner calibration, the cross-fit of the baseline against the routing-augmented model, and a null made of random columns. Targets are "does the state switch" and "which state comes next". |

`k2.py` reuses the fitting machinery of `routing.dynrt` unchanged (multinomial ridge, question-weighted log-loss, paired question bootstrap). It is a library; `routing.anticipation` calls it.

Paper use: through `routing.anticipation` (figure `next_state`). See [`src/routing`](../README.md).
