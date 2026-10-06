Two specific corrections to your follow-up need adjudication against source:

1. `seal_eligible_micro_serial_v2.py` lines 60-65 sorts all `(family, seed)` slots together within a transition before assigning `index % 4`. It does not sort and rotate separately within each seed. Observed set counts [0,1,2,3] are candidate seed0 [2,3,2,1], seed1 [2,1,2,3]; approach seed0 [2,2,1,0], seed1 [1,1,1,2]. Please verify and correct your statement that v2 achieved per-transition-per-seed balance by construction. The planned v3 replacement will enforce max-min <=1 separately within each transition and seed.

2. You proposed a 16k utility extension "from the same starts." `PROTOCOL_v0.1.md` requires 96 disjoint utility families and generation from original prompts, not same-prefix starts. Please distinguish the registered utility scout from any optional exploratory same-start timing probe. Preserve the registered endpoint and population in your recommendation.

Please respond only to these corrections and any consequence for the minimal next experimental route. Do not infer unobserved v3 contents as complete until its seal and CPU preflight pass.
