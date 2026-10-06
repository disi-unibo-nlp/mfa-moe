## Correction 1: verified — my "by construction" claim was wrong

Source confirms your reading. `seal_eligible_micro_serial_v2.py` 0060–0064 builds slots as `[(r.family, seed) for r in rows if transition ... for seed in (0,1)]`, sorts that combined list once per transition, and assigns `index % 4` across the whole list. There is no per-seed stratification step in the assignment. So balance is guaranteed only at the transition level (the exact property the validator enforces), not per transition-seed.

The observed counts are consistent with that mechanism: for verify, the combined multiset `[4,4,4,4]` can split into `[2,3,2,1]` / `[2,1,2,3]`; for approach, the combined `[3,3,2,2]` can split into `[2,2,1,0]` / `[1,1,1,2]` (sum `[3,3,2,2]`). These are plausible outputs, not something I can independently confirm from the packet; what is verifiable is that nothing in the seal constrains the split, and the current validator would not catch it — `run_eligible_micro_serial_v2.py` 0108–0117 checks per transition over both seeds pooled, so `[4,4,4,4]` passes with `max − min = 0` even though per seed it is `[2,3,2,1]`.

Note the exact observed approach seed0 `[2,2,1,0]` has max − min = 2, so it violates the planned per-transition-per-seed rule; seed1 `[1,1,1,2]` has max − min = 1. The v3 requirement is therefore a real change, not a restatement of what v2 did.

Consequences:
- The v3 seal should compute assignment as `assignment[(transition, seed, family)] = index % 4` over slots sorted *within each transition-seed stratum*, and the validator should add a per-seed loop: for each transition, for each seed, build counts over that transition's families and require `max − min ≤ 1`. Without the stratified sort, the validator alone can only reject a bad draw, not produce a balanced one.
- Enforcing per transition-seed balance reduces the chance that one random set dominates a seed's random arm; it does not make the random contrast per-family or per-seed reproducible. For that, the schedule also needs to be frozen per family (same or deliberately counterbalanced sets across seeds) — a separate choice from balance.
- If the deactivation diagnostic copies `positive['random_set_by_family_seed']` (`seal_eligible_deactivation_serial_v3.py` 0074), the positive replacement's new schedule must be sealed first and the deactivation seal re-run against it, or the paired diagnostics will use different random assignments.

## Correction 2: accepted — registered utility scout must keep its own population and prompt source

`PROTOCOL_v0.1.md` 0137–0139 is explicit: "native versus frozen policy, two seeds, from original prompts, capped at 16,384 tokens," and the population is 96 further families, not the 13 discovery families. My "from the same starts" phrasing implied reusing the pilot prefixes and families, which would violate both the disjointness and original-prompt requirements. Corrected split:

- **Registered utility scout (the only source of utility claims):** 96 disjoint utility families from `parent_pools['utility']`, original prompts, native vs frozen policy, 2 seeds, 16,384 cap. No trajectory or dose conclusions may be drawn from it beyond what the protocol registers, and it remains subject to the pricing/authorization gate.
- **Optional same-start timing probe (exploratory only):** extending pilot families from their same-prefix starts is useful for measuring decode throughput, cap behavior and injected-token accounting, but it is not the registered utility end point and cannot be reported as accuracy/token utility for the intervention in the registered sense. If run, it should carry its own versioned exploratory label, share the pilot's sealed binding, and not substitute for the 96-family scout.

## Consequence for the minimal route

1. Seal the replacement pilot and run CPU preflight; include the stratified per transition-seed assertion and the arm-order permutation assertion. Do not touch the unsealed v3 driver's behavior on the basis of expectations — only the sealed manifest and passing preflight make it real.
2. After generation, bind the frame to the later post-generation rating price and rate; the pilot's endpoint stays the 256-token immediate target.
3. Separately freeze the 1,024-token ordered-trajectory confirmation with the reversed-order arm; that is where trajectory claims live.
4. Keep the registered utility scout as a distinct 96-family, original-prompt stage, priced and run only after the pilot shows nonzero dose and after the rating cost is measured; do not fold the same-start probe into it or infer utility from it.