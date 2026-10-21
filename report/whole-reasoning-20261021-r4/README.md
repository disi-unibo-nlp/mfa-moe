# Whole-reasoning release r4

Continue F-0003 and D-0002 under the unchanged 72 GPU-hour / 16 CPU-core-hour ceilings; every earlier
whole job is charged in PRIOR_USAGE.json. Source 0e4717d runs the batched whole hook eagerly
(no torch.compile), builds row tensors once per engine step and defers audit transfer to the next step.
Canary rows share a 32,768-token cap (pre-outcome correction; relative gates unchanged) and keep production
audit flags. Ranks must agree exactly on routing and integer counters; float telemetry within 1e-6.
Nested sbatch drops the caller's SLURM_* state. Released and launched by the external reviewer (Claude)
after the Codex session stalled; nominations, doses, cells, seeds and gates are unchanged.
