# Saved-result figures, 2026-10-09

Run `python plot_results.py` with NumPy and Matplotlib to regenerate the three
PDF/PNG figure pairs and the descriptive retention summary. The bundled CSVs
are exact copies of existing compact evidence; `SOURCES.json` records their
repository locators and hashes. `FIGURES.json` binds inputs, script and outputs.
No fitting, new bootstrap, generation, relabelling or policy selection occurs.

- `routing_readout`: all seven CLEAN judge-context readouts and their lexical-null
  excess comparisons. Confidence limits are the original model-specific
  `simultaneous_ci7` limits. They do not establish simultaneous coverage across
  all models and both displayed panels. Negative differences favor routing.
- `heldout_retention`: all 105 model/fold/arm cells at 75% retained expert
  identities per routed layer. Left: minimum class coverage for state balancing,
  labelled frequency and the original fixed random set. Right: all five paired
  fold differences; bars terminate at the descriptive arithmetic fold mean.
  Fold spread is not a confidence interval. The 1,486 families, rather than the
  five folds, are the independent problem units. Coverage averages layers within
  each class before taking the minimum. The result measures saved selected slots,
  not accuracy under removal, executed-weight mass or resident memory.
- `next_class`: the original `composition16` next-class endpoint and its
  contained-window sensitivity for GPT-OSS and Qwen3.6. Limits retain the original
  model-specific `simultaneous_ci6` procedure. Questions and adjacent-pair counts
  are shown separately; pairs within a question are not independent. Other
  endpoints remain in the complete bundled `anticipation.csv`; the five other
  models have no equivalent completed next-class result in this evidence set.

Exploratory same-cohort expert/correctness correlations, conditional candidate
identities and in-sample concentration curves remain in the manuscript's saved-
data appendix. Historical uncertain and harmful intervention outcomes remain
in their original appendices. They do not support a general pruning or
accuracy--memory benefit.
