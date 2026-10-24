# src/routing/depth_paths

Depth-wise routing paths: the expert a token takes at each layer, followed from the first routed layer to the last. A helper package for the present-state readout (`routing.present_state_readout`, `routing.next_state`). Its plan was frozen on 2026-09-29, so change nothing here; a rerun must reproduce the first run.

| File | What it does |
| --- | --- |
| `common.py` | Paths, model specs, frozen constants, and the loading of the outcome-free attempt tables and the question split. |
| `s3_readout.py` | The path-based readout: `load_model` reads the labelled sentences and token routing of one model. The readout and next-state drivers call it. |
| `cv.py` | Question-grouped ridge-logistic cross-validation (4 outer folds times 5 repeats, 3 inner folds, penalty grid 10^-3 to 10^2). |

These modules are libraries. Nothing in the live launchers starts them directly. Inputs are the saved analysis caches that `ANALYSIS` in `src/store.py` points to, read only.

Paper use: through `routing.present_state_readout` (figure `readout`) and `routing.next_state`. See [`src/routing`](../README.md).
