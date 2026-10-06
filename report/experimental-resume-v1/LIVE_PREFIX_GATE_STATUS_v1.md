# Live prefix gate and side-query cost, discovery status

`src/moe_exp/routing_control/live_gate_v1.py` is an integration boundary for
the routing controller. It takes the original problem and token IDs already
emitted by the native model, decodes those IDs, and emits a side-query request
only at a completed sentence with no following nonspace text. Its semantic
reader input has exactly `problem`, `emitted_prefix`, and
`triggering_sentence`. The gate stops at reasoning closure and suppresses
duplicate request IDs. A discovery-frozen cheap screen must be supplied;
without one, it abstains. Tests mutate future text, gold, correctness and
labels while holding the emitted tokens fixed, and obtain the same decision.

This interface has **not** qualified an adaptive controller. The existing
character-gram cheap screen selected 0 of 103 resolved, family-held-out
candidate fires at its required 70% apparent precision threshold. Its own
report is `CANDIDATE_VETO_FULLPREFIX_DISCOVERY_v1.json`. The 372-window Qwen
and native-reader audit remains an offline discovery sample; neither its
saved offsets nor its future labels may be used to trigger a utility arm.

The sealed all-fire burden in `NATIVE_VETO_ALL_FIRE_BURDEN_v0.json` gives the
cost of asking the native side model at every lexical fire through the 16,384
token output horizon: 1,549 calls, 6,572,311 exact prompt tokens and at most
99,136 response tokens. Candidate fires account for 1,505 calls and 6,443,641
prompt tokens; approach fires account for 39 calls and 82,487 prompt tokens;
failed-check fires account for five calls and 46,183 prompt tokens. These are
cached-native discovery counts, not a price for steered trajectories.

`QueryCostLedger` records the actual prefill tokens, generated tokens,
elapsed side-query time and outcome for accepted, vetoed, unresolved and
failed calls. A prospective utility run must also count native generation,
recompute, cold load and failures, using its actual live requests. The later
8,193–16,384-token context stratum needs separate qualification; an 8,192
token audit limit is not a model limit.

The exact-ID discovery pool `JOINT_QWEN_NATIVE_EXACT_POOL_v1.json` contains
14 Qwen/native two-reader agreed starts and a preselected globally
family-disjoint 13-row subset: eight candidate and five approach starts.
The stricter Qwen/native/GPT-OSS vote conjunction leaves seven rows in seven
families: two candidate and five approach. GPT-OSS accepted both frozen
Qwen-agreed negative controls, so its positive vote alone is not a validity
certificate. All these are LLM measurement audits, not human truth or
evidence of a causal transition effect. They support a local approach action
screen and motivate independent eligibility enrichment before a larger
trajectory test.
