# Continuation-cap implementation and qualification

The frozen s2 tree is `9a61e32f48c04c242acccc89c529bd750776c553cdfc776347151d359bc53430`.
It differs from the old `f2ded3957eb54fd5a3b4ffa132e0f68f57170c53b6013d885767ac83c466dc17`
tree only in two hunks of `moe_steer/manifests.py`. Every other runtime and
`moe_exp_src` file remains byte-identical. The old keeper remains drained and its
`code/latest.json` pointer was restored byte-for-byte after freezing.

`make_request(max_new_tokens=None)` retains the legacy request dictionary. A valid
positive integer cap clips the remaining cumulative budget. Booleans, nonintegers,
zero and negative values are rejected; a manifest cannot claim more than the
remaining budget. Requests reach the actual runner's sampler with the capped
`max_tokens`. Fake-engine integration preserves natural-stop and length-stop records,
routed windows, crash recovery and same-manifest resume.

The revised X2 builder requires the explicit frozen tree, defaults its CLI N arm
to the registered sham and refuses the old uncapped API. Cached enumeration gives
39 eligible questions, 85 conditions per seed, two seeds and **6,630 requests**.
All continuations are **1,024 tokens**, maximum decoding **6,789,120 tokens**.
The table has 128 policies including zero and sham; the old 127-policy arithmetic
omitted zero. No grid cells were removed. The dry manifest is an engineering
artifact, not a production experiment seal.

The guarded runner separately binds manifest bytes, code-tree and wrapper hashes
to an output directory. It rejects changed caps/code and unbound nonempty outputs.
Atomic binding publication can recover an interrupted pending-only binding.
Per-UID receipts and the legacy shard lock prevent duplicate completed requests.
The old legacy UIDs omit caps, so this external binding is required.

Default-request differential checks passed for original prompts and all three
real branch forms. Same-name X2 comparison retains UIDs and every other request
field; a differently named dry run necessarily changes salted UIDs and assignment.
The frozen legacy compatibility suite passed 60 tests, with seven GPU tests skipped.
The unamended suite's old assertion that max_tokens=5 is invalid failed as expected
under the new API. Its raw failure is retained. A separate follow-up copy changes
only that assertion to max_tokens=0 and passed 60 tests, seven skipped.

The final cap suite and its JUnit receipt are under
`steering-v1/runs/s2prop/final-cap-tests.xml` in the external analysis tree.
GPU execution qualification remains separate: CPU tests do not establish force,
neighbor isolation, prefix or recompute behavior on the loaded model.
