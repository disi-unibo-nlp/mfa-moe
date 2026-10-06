Verified the code paths. Findings below.

## Corrections to my earlier audit

**7 A11 was wrong as stated.** `run_eligible_micro_serial_v2.build_requests` reads `manifest['random_set_by_family_seed'][row['family']][seed]` directly (v2 lines 0153–0156; identical in `run_eligible_micro_serial_v3.py` 0155). `design.balanced_random_assignment` is called only by `manifests.py` (import line 0011, call line 0080) and tests (`tests/test_routing_control.py` 0008, 0193) — not by any serial driver. My residual point survives in weakened form: the schedule is manifest data validated only for aggregate per-transition balance, so the replacement must bind the exact schedule and order, not recompute it ad hoc. Retract the `(i+j)` claim.

**A2 was a false positive in context.** The v3 price is explicitly pre-generation ("RESOURCE_PRICE_BEFORE_GENERATION;...REQUIRE_POSTGENERATION_HASH_BINDING"), and no rating job ran; the new rating driver's requirement of a later frame-bound price is by design. What actually blocked the old run was the cancel/hold, not a latent contract bug. Reclassify A11 and A2 as caveats, not findings.

**A3/A10 do not invalidate the serial pilot's ITT.** A10 is qualification, not evidence: 1/8 native-duplicate text equality, 0/8 first-token unordered top-k equality means exact hidden-state/counterfactual pairing is dead; 8/8 first-token target-mask and 5/8 pre-divergence target-mask equality mean the routing edit itself is correctly applied. ITT over family-level arm contrasts remains the right estimand under run-order counterbalancing; it is not a fixed-seed paired effect.

## Classification for the 256-token pilot

The pilot can answer only: does the immediate target appear within 256 tokens, with first-stage routing/dose? It cannot answer trajectory or utility. Blocking/comparability items and caveats are:

| Issue | Blocks pilot? | Blocking later claims? |
|---|---|---|
| Run-order counterbalancing + random-set rotation | No (proposed replacement handles) | No |
| A1: no ordered/dense trajectory at 1,024 | Not for pilot (deferred) | Yes, hard |
| A6: no answer grading, no injected-token cost | No | Yes, hard |
| A3: deactivation = necessity, not control | No (label separately) | Yes for "control" language |
| A4: live gate not used; stays backlog | No | Yes for online-detector claims |
| A8: rater told the transition | No | Caveat for primed-vs-naive debate |
| A11/A2 | No | No |

I recommend explicitly freezing the confirmatory pilot estimand as: immediate target presence, both-positive reader, ITT with all six arms; and freezing the trajectory/utility questions as separate later stages, so the pilot's misclassification risk is confined to what it measures.

## Is the replacement plan sufficient?

**Run-order counterbalancing: necessary, not yet sufficient.** The current drivers build `cases` as `row → seed → arm` and slice contiguous blocks (base lines 0291–0292; v2 lines 0169–0172). Nothing in the v3 driver currently reorders arms; `arm['name']` order is still manifest order (v2 lines 0133–0163; v3 lines 0133–0168). If you add the rotation, bind it explicitly: a sealed `arm_order` per (family, seed) block, and require every block's six UIDs to equal the frozen permutation (UIDs are digest-derived from `manifest_sha256, row.uid, seed, arm.name` at v2 0147, so they are stable against reordering). With 26 blocks and 6 arms, each arm must occupy each position within ±1 across blocks.

**Random-set rotation: sufficient only if per transition and seed, and frozen per family.** The proposal exactly matches what the v2 seal script already computes (`seal_eligible_micro_serial_v2.py` 0061–0064: sort hashed slots, assign `index % 4`), which yields verify 2/2/2/2 and approach 2/1/1/1 per seed. So per-transition-per-seed balance is achieved by construction. The residual question is whether the random arm should use the same set at seed 0 and seed 1 for a given family; the current design (v2 line 0155) lets it differ. For a paired 2-seed contrast, freeze one set per family and put the rotation over the other 4 sets at the family level; aggregate rotation across families gives the same balance. If you keep per-seed rotation, the random arm is a mixture over two differently matched controls and the "random" contrast is a distributional one, not a fixed-control one.

**Essential validation assertions (smallest set):**
1. Block composition: each `(family, seed)` block contains exactly the six frozen arms, same `prompt_ids`, `prefix_ids`, `canonical_question`, and `common_seed` for all six (v2 0137–0148).
2. Arm-order binding: sealed `arm_order[b]` is a permutation of the six; driver replay must rebuild cases in that order; batch assignment/receipt UIDs must match (base 0218–0219; 0295–0306).
3. Random-set rule: for each transition and seed, set counts differ by at most 1; each set is disjoint from that transition's target experts, and the random arm uses only the set selected for its `(family, seed)` — assert explicitly rather than relying on `% 4`.
4. Policy identity: each random set equals the dictionary's frozen `matched_random_control_sets` (v2 0097–0107) and the target set equals `target_templates`.
5. No outcome feedback: the sealing script must be the only producer of the schedule; assert the driver never recomputes it.

## Smallest credible tests for the later claims

- Ordered trajectory: freeze a two-action template with slots `[0, 512]` (design.py 0193, 0205–0208), run the same pilot's families with native, template, and random at 1,024 tokens, dense-label, and call `trajectory_completion` (analysis.py 0012–0034). Also run the reversed action multiset at the same horizon (design.py 0208–0209; protocol 0119–0123) so ordering, not just presence, is testable. That is the direct test of the objective's "ordered" claim; the current 256-token endpoint cannot reach the second pulse.
- Utility: extend exactly one frozen policy and native from the same starts to the protocol cap (16k), freeze an answer parser and a cost schema containing emitted and injected tokens, and compute accuracy and token contrasts jointly. Run this only after the pilot shows a nonzero first-stage dose; otherwise it is an expensive null.
- Order/mechanics confirmation: run the six-arm block through all six arm permutations on the four pilot families used for engineering qualification, and report the arm-position × outcome interaction. This directly measures the carryover magnitude that counterbalancing is meant to neutralize; scale the pilot only if that interaction is non-negligible.

Do not enlarge the 13-family pilot because of a near-significant contrast. The pilot's value is detecting large immediate effects and estimating variance; if it is near-null, the correct next step is a precisely powered independent mechanism run, per the protocol's own precision language.