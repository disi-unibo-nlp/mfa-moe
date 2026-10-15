# Eight-model study

Python implementation for the final routing study. Run it with
`python -m moe_exp.study`; use `--help` for commands.

- `session.py` owns request admission, internal IDs, stepping and cleanup.
- `runtime.py`, `observer.py` and `reader.py` generate, capture routing and annotate.
- `qualification.py` and `validation.py` check the real engine before production.
- `workflow.py`, `scheduler.py` and `reservations.py` submit the finite job graph.
- Frozen design, endpoint, routing and selection helpers are imported normally
  from the original source package; the study does not install its old schedulers.

Inputs are the frozen `configs/final_routing.yaml` and existing campaign receipts.
Compact results stay in `report/final-routing-20261015-r1/`; bulk data stays in
its approved private FAST directory. Control records live under `study/<commit>/`
inside that campaign. Jobs run from the committed source snapshot there. Historical `final_routing*` packages remain unchanged.
