# Counterfactual score measurement amendment v0.2

The 148-assignment qualification, Slurm 59189255, completed every request and
passed fixed-k force inclusion/exclusion, inactive routing, reasoning closure
and public-reset recovery. It **failed** its registered absolute native versus
appended-token teacher-forcing alignment gate: mean 0.001365 and maximum
0.021752 nats, versus limits 0.0001 and 0.001. Native repeats also differed
(mean 0.000718, maximum 0.011340). Preserve that result and its failed status.

Before collecting new scores, qualify the designated-token API against the
full-vocabulary raw-score API using the **identical prediction input**. The
same four engineering discovery prefixes and first four native positions are
retained, even though the separate semantic audit found no jointly eligible
unchecked-candidate start among these four prefixes. They cannot rank semantic
experts or establish steering.

For each of 16 positions, interleave designated A, full-vocabulary A,
designated B and full-vocabulary B in fixed order. All 64 requests select zero
under the exact populated pilot policy table, preserving layer-28 hooks and
native top-k=8. Another 16 appended-token teacher-forcing requests remain
diagnostics. Neither a future scoring token nor its log probability enters
routing metadata or an online decision. Save only the designated scalar from
each full-vocabulary reference, along with routes and both-rank receipts.

Freeze the pre-existing Q3-style numerical rule prospectively: for both mean
and p99 absolute score difference, the larger cross-API A/B difference must
be at most 1.25 times the larger same-API repeat difference plus 1e-6. Report
every per-case difference and route disagreement. This is a new score-API
calibration estimand, not a reversal of the old teacher-forcing gate and not
engine equivalence. Any subsequent expert screen must include native repeat
noise and target/non-target semantic controls; a local loss change alone does
not establish useful free generation.

The unchanged qualified native engine remains in use. A batch-invariant
backend is a separate engineering experiment, with its own qualification.
The complete new 80-request job is priced at 45 minutes on two A100s (1.5
GPU-hours), including cold load, all prefill, full-vocabulary extraction,
telemetry, shutdown and margin. Driver, helpers, worker and table are frozen
by digest; changed inputs cannot reuse completed UIDs. Necessary recovery
spending is covered by RESOURCE_AUTHORIZATION_2026-10-02_v2.json.
