# M9 replay adjudication, 2026-10-01

The original card-sampled replay remains a failed engineering test. Job
`59117029` ended `FAILED 3:0` in 1,002 seconds on two GPUs, 0.55667 GPU-hour.
Its sealed `REPLAY.json` reports exact prefixes, closure injection and CPU mask
checks passing, but isolated and mixed output sequences differed in four of
twelve requests, including an unedited native request. The earlier launcher
failure `59116442` cost 0.11167 GPU-hour. Neither is a semantic result.

The causal ambiguity is whether batch scheduling changes stochastic samples
without cross-request state leakage. Versioned driver
`qualify_m9_replay_deterministic.py` (SHA-256
`ea7f3a9334bab006b9131da2afc916015a9aba4cec287cd769027f8c9bb45575`)
uses the same four frozen prefixes, modes, model, logits processor and 24-token
limit, with temperature zero for a request-local deterministic isolation test.
It verifies the raw failed receipt before loading. The old script and output
stay unchanged. A passing greedy test would show deterministic request isolation;
it would **not** make the stochastic identity test pass. Report that limitation
for all M9 results.

One 25-minute allocation on two A100s costs at most 0.83334 GPU-hour, making
the full M9 qualification expenditure at most 1.50167 GPU-hours including both
failed attempts. The user's current broad paper-workload authorization covers
this bounded adjudication. M9 production remains held until this result passes
and the full generation/replay plus blind-grading resource price is sealed and
amended. Greedy failure or another request-local defect holds production.

**Observed adjudication:** Job `59118998` ended `FAILED 3:0` in 727 seconds
on two GPUs (0.40389 GPU-hour). The sealed deterministic replay
`replay-deterministic-63772b26-ea7f3a93/REPLAY.json` reports native requests
matching in all four fixtures, but the edited mode differed in one fixture and
the large-dose mode differed in three. The processor's CPU mask, exact prompt
IDs, closure tokens and large-dose activation passed. This narrows the failure
to the edited GPU path or batch-dependent numerical effects; it does not
distinguish them. M9 generation is **HOLD_ENGINE_ISOLATION**, so no amended price,
production closure job, strict grading job or M9 utility claim is supported by
the engineering result. The prepared amended launcher and scoring code remain
inactive. The three qualification attempts total 1.07222 GPU-hours.

A separate diagnostic driver, `diagnose_m9_batch.py` (SHA-256
`eab702b33c8ed0eaeb1bf2a8397e36b04fcf015a07cc71a288fd161087ec7f6f`),
records the first greedy token divergence, top-two logprob gaps there and an
isolated repeat for every edited fixture. It is explicitly incapable of setting
a qualification pass. One two-GPU, 25-minute attempt costs at most 0.83334
GPU-hour, bringing all M9 replay/diagnostic allocations to at most 1.90556
GPU-hours. This targeted test asks whether edited outputs vary even in repeated
isolation and whether batch divergences occur at near ties. A production fix
would still require a new versioned launcher and a passing independent replay;
the existing M9 hold remains in force.

**Diagnostic result:** Job `59120703` completed `0:0` in 1,100 seconds on
two GPUs (0.61111 GPU-hour), producing sealed diagnostic output. Completion
means the measurement ran; its qualification fields remain `pass: false` and
`HOLD_ENGINE_ISOLATION`. In the greedy run with top-two logprobs, two of twelve
isolated-versus-mixed requests differed, both large-dose cases. One first
divergence had reported top-two gaps of about 6.8 and 6.6 logprob units,
which is inconsistent with a simple near tie explanation for that instance.
Repeating edited requests in isolation changed one of four ordinary-dose
outputs and two of four large-dose outputs. The data do not identify whether
the cause is the custom processor, runtime state or model-kernel variability.
Request-local batch behavior is not qualified; further M9 production is held.
All four M9 replay/diagnostic allocations spent 1.68333 GPU-hours.
