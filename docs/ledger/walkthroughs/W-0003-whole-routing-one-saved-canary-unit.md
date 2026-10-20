# W-0003: One saved whole-routing canary unit
- date: 2026-10-20
- commit: 16cf59973f27a066d6f1a47a620d02c5c519928e
- source: report/whole-reasoning-20261020-r3-batched/deterministic/stores/gpt/main/canary/d110fdc51c1f33901e79e28217594b4654f1ae3a69f15a03aee8435067a09618/; native job61581237
- paths: src/moe_exp/study/whole.py, src/moe_exp/study/driver.py, src/moe_exp/study/stream.py, src/moe_exp/final_routing/operator.py
- keywords: whole, batching, routing, audit, dose, canary

**Input:** Actual Minerva family
60e8b726e6566124e257983eb62d37173a5df88f0c7196d6f76052195338f2ed,
fold4, target_suppress/Read. Its original prompt has191tokens, seed2186601647,
and allowance130881. The canary UID begins d110fdc5; it is separate from the
production UID and never enters production analysis.

## 1. Admission (`whole.canary_rows`, `driver.Engine.admit`)

The frozen eight-family roster includes this family. The row retains the
production allowance and enables a full audit. Forty-eight rows enter the
64-slot engine without a family barrier. The engine records the external UID,
the returned internal request alias, prompt history and seed before decoding.

## 2. Scope (`stream.Clock`, `StreamCapture._wrap`)

Absolute prompt positions remain active even though the GPT prompt ends in a
Harmony suffix. Generated closure is recognized only after the prompt. This
row deactivates20experts across20layers until reasoning closes. Layer1 removes
expert17; layer0 has no action. The final saved capture records closed=True.

## 3. Layer execution (`StreamCapture._whole_batch`)

The layer batches request-owned token rows, gathers each row's activation and
deactivation masks, and calls the same dual score override. The implementation
uses compiled project kernels on CUDA, then returns the executed weights and
identities to vLLM. One lossless batch transfer supplies the per-unit audit.

For this unit's generated position0 at layer1, native/executed IDs are both
[18,29,12,6]; the removed expert17 was absent. Native weights are
[0.5410026908,0.2987700403,0.0962416083,0.0639856309]. Executed weights are
[0.5410026908,0.2987700701,0.0962416083,0.0639856383]. Membership stays the same
at that position; score normalization can change floating weights slightly.

## 4. Counters (`stream.batched_dose`)

Contiguous request segments supply additive counters into each unit's own
buffer slot. Layer1 reports191active prompt positions and738active generated
positions,929total. Membership changes at325positions, including284generated
positions verified from the stored arrays. Native deactivated weight sums to
70.95339203, executed deactivated weight is0, and L1 sums to141.90676880.
Membership failures are0; all739layer observations are finite. Layer0 records
zero active positions. These are same-state routing measurements, not effects
on correctness or length.

## 5. Release (`driver.Engine.release`, `compress_audit_capture`)

Both ranks agreed on this unit's complete capture hash. Its805output tokens
finish with stop; RESULT.json records complete, one segment and no redraw.
The compressed audit contains144arrays (24layers times6fields), including
generated positions0through737 for layer1. Its620456-byte archive has SHA256
c3caac8761200ccfd831af97d12b56fc8daf936837cbe6ee534678f603533588.
The stored CAPTURE.json binds that archive and the per-layer counters.

## Worth noticing

This unit completed, but the allocation later failed on another unit's strict
tensor-parallel capture agreement. Three saved units do not qualify the
48-row canary. GPT production was withheld; the original failure, partials,
source archive and every charge remain preserved. The cleanup's subsequent
unknown-released-request error must not obscure the first capture disagreement.
