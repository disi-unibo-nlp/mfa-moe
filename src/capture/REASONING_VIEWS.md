# Reasoning views

`capture.views.compute_views` reduces the router and hidden tensors of one captured trace to per-layer features in three kinds of
view. Every capture packet (`<id>.json.gz`) holds them under `cohorts/<cohort>/views`, next to the sentence spans with the
token ranges each sentence owns.

- **full**: all selected reasoning tokens of the trace.
- **class**: the tokens of the sentences with one label (Read, Analyze, Plan, Implement, Explore, Verify, Monitor). Only
  cohorts that carry labels have class views.
- **position**: fixed intervals of reasoning tokens, with boundaries `ceil(model_mean_reasoning_tokens * i / 10)`, and one
  overflow window beyond the mean. The mean comes from `position_reference` in the manifest, computed from all traces of the
  model's cohorts; it is not a per-trace percentage.

How the tokens are assigned:

- **Reasoning text**: the saved reasoning content or the `<think>` body; for a model without either, the whole completion.
- **Segmentation**: deterministic punctuation and newline units with exact character offsets; protected LaTeX and decimal
  points stay whole (`labeling.spans`, schema version stored in the packet). Older annotations keep the first splitter
  (`capture.sentence_schema`).
- **Token ownership**: the offsets come from the saved token ids of the prompt and completion. The unit with the largest
  character overlap owns a boundary token; the earlier unit wins a tie and whitespace gaps go to the preceding unit. Every
  reasoning token is counted once.
- **Transitions**: switching rate, top-k overlap and hidden-step distance use only adjacent tokens of the same sentence, so
  disconnected spans create no artificial transitions.
- **Missing scopes**: a scope without tokens has no features (never zero); coverage is reported in `label_coverage`.
