# src/routing/dynrt

Routing against reasoning dynamics: a table of routing per labelled sentence, how well a sentence's routing matches the profile of each state, and the cross-validation machinery that tests whether routing adds information. A helper package for `routing.readout` and `routing.anticipation`. Its plan was frozen, so change nothing here.

| File | What it does |
| --- | --- |
| `common.py` | Paths, model specs and shared constants. Inputs are read only. |
| `data.py` | Loaders for the stage-1 tables. Outcomes are loaded only through `load_outcomes`. |
| `features.py` | Class-mismatch features, fitted on training questions only: state profiles shrunk toward the marginal, and a distance of a sentence to each profile. |
| `cv.py` | The cross-validation machinery: ridge logistic, question-grouped folds, penalty chosen on inner folds, paired question bootstrap. |
| `d3a_pairs.py` | The table of adjacent labelled sentence pairs (self-transitions included); the target is the state of the next sentence. |
| `d3a_text.py` | Hashed text features (2^18 buckets). `routing.readout_text` has the same recipe. |
| `d3a_softmax.py` | Weighted multinomial ridge with dense and sparse blocks. |
| `d3a_cv.py`, `d3a_eval.py` | The cross-fitted multinomial fit and the paired question bootstrap of held-out log-loss. |

These modules are libraries, imported by the drivers. Paper use: through `routing.readout` (figure `readout`) and `routing.anticipation` (figure `next_state`). See [`src/routing`](../README.md).
