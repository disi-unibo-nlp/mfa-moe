# src/routing/dynamics_routing

Routing against reasoning dynamics: a table of routing per labelled sentence, how well a sentence's routing matches the profile of each state, and the cross-validation machinery that tests whether routing adds information. A helper package for `routing.present_state_readout` and `routing.next_state_anticipation`. Its plan was frozen, so change nothing here.

| File | What it does |
| --- | --- |
| `common.py` | Paths, model specs and shared constants. Inputs are read only. |
| `data.py` | Loaders for the saved cohort tables: attempts without their outcome columns, sentences and class histograms. |
| `features.py` | Class-mismatch features, fitted on training questions only: state profiles shrunk toward the marginal, and a distance of a sentence to each profile. |
| `cv.py` | The cross-validation machinery: ridge logistic, question-grouped folds, penalty chosen on inner folds, paired question bootstrap. |
| `d3a_pairs.py` | The judge label records of the sentences: `load_unit_texts` returns the text, label and span of each labelled sentence by its identity. |
| `d3a_text.py` | Hashed text features (2^18 buckets). `routing.readout_text` has the same recipe. |
| `d3a_softmax.py` | Weighted multinomial ridge with dense and sparse blocks. |

These modules are libraries, imported by the drivers. Paper use: through `routing.present_state_readout` (figure `readout`) and `routing.next_state_anticipation` (figure `next_state`). See [`src/routing`](../README.md).
