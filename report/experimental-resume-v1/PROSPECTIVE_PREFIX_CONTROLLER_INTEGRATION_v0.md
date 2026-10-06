# Prospective prefix-controller integration gate (discovery only)

The completed 372-window audit rates cached native prefixes. It does not
qualify a controller that starts from the original question and makes decisions
on a new, possibly steered continuation. No utility arm should be submitted
until the following executable checks pass on the **same frozen** three
transition definitions.

1. **Streaming source:** Instantiate `StreamingTransitionDetectorV2` from a
   fresh state per assigned question. Pass only the original problem and
   actually emitted token IDs. Decode those IDs at each complete sentence;
   compare replay after batching, batch permutation and preemption. The
   detector output must be unchanged when gold, correctness, future native
   completion, future dense labels and final-answer fields are mutated or
   removed. Reject non-append-only history and never reconstruct a live
   trigger from a saved native trace offset.
2. **Eligibility:** A lexical fire is only a proposal. Freeze the discovery
   trained cheap screen and semantic-veto threshold before validation. If the
   evidence for a transition is inadequate, that transition abstains. The
   Qwen3.8 two-reader full-prefix ratings and the GPT-OSS panel are offline
   measurement audits; neither may be consulted using future text online.
   Close the detector at `</think>` and never re-enter after closure.
3. **Side-query boundary:** At a live trigger, capture the exact emitted prefix
   through the complete triggering sentence. An unsteered native-model
   side-query may see only `problem`, `emitted_prefix` and
   `triggering_sentence`; store hashes of input IDs/text and all returned
   votes. Qualify separate 8k and 8k–16k prefixes, maximum side context,
   batch isolation, failure/timeout behavior and strict JSON parsing. Count
   prefill, generated tokens, cold load and delay for every side-query,
   including vetoes and failures. A missing or ambiguous vote abstains.
4. **Resume and routing:** Stop generation on the exact sentence boundary,
   save token IDs, KV-cache/recompute choice, detector state and side-query
   decision, then resume from the same emitted prefix. Verify identical
   native continuation with inactive routing; no neighbor request changes;
   action pulse begins at the intended branch-relative slot; 256-token pulse
   ends exactly or at reasoning closure; second slot remains at +512.
   Recompute/preemption and crash recovery must not duplicate or skip an
   assignment, side-query or pulse. Version and hash manifest, code, detector,
   rule, model cache and output directory.
5. **Intention-to-treat accounting:** Enroll original questions and families
   before generation. Record every assigned arm, lexical nonfire, veto,
   early finish, cap, exception and restart. Accuracy and total token cost
   include side-query and replay costs. Report exact detector coverage and
   failure counts for each arm; native cached-prefix eligibility is a
   discovery diagnostic, not the utility denominator.

The sealed all-fire discovery price currently projects 1,549 native side
requests through the 16,384-token output horizon and 6,572,311 prompt tokens
for a naive query-at-every-fire policy. A fixed first-eight candidate-fire cap
would retain only 7 of 25 Qwen two-reader approved candidate starts in the
stratified rated sample, so it is not a qualified savings rule. Evaluate a
cheap prefix-only gate on family-held-out discovery ratings and price its
remaining live calls before freezing a prospective utility policy. The
proposed 32,768-token side context needs separate GPU execution qualification;
8,192 is only the earlier audit-frame limit, not a model context limit.
