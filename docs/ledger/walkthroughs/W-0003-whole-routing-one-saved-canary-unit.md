# W-0003: One saved whole-routing canary unit
- date: 2026-10-24
- commit: 610ed2a6b1f5089b828de755c0eb951df4df7e08
- source: report/whole-reasoning-20261020-r3-batched/deterministic/stores/gpt/main/canary/d110fdc51c1f33901e79e28217594b4654f1ae3a69f15a03aee8435067a09618/ (AUDIT.npz, CAPTURE.json, OUTPUT.json, PARTIAL.json, RESULT.json), gpt/WHOLE.json (8b78678e), gpt/CANARY.json, GPT_SOFTWARE_FAILURE.json; native job 61581237; every value below comes from a run on 2026-10-24 of the code at this commit over those saved files
- paths: src/steering/whole.py, src/steering/pulse.py, src/steering/stream.py, src/steering/operators.py, scripts/steering/gpu.sh, tests/steering/test_stream_batch.py
- keywords: whole, batching, routing, audit, dose, canary

**Input:** The saved unit of the actual Minerva family
60e8b726e6566124e257983eb62d37173a5df88f0c7196d6f76052195338f2ed, fold 4, arm `target_suppress`, class
Read, from the GPT canary of the whole-reasoning release (job 61581237). Its original prompt has 191 tokens,
its seed is 2186601647 and its allowance 130,881. The unit is a canary: it never enters a production
analysis. This is the retrace of the walkthrough first written on 2026-10-20 at commit 16cf599, when the
modules were `study.whole`, `study.driver`, `study.stream` and `final_routing.operator`.

## 1. Admission (`whole.canary_rows`, `pulse.intervention_builder`, `pulse.Engine.admit`)

`canary_rows` on the saved GPT main manifest picks 8 families (one per benchmark plus two more) and, per
family, the two native arms and the Verify promote and Read suppress cells: 48 rows, of which 32 steer.
Today's row for this unit has uid e70dffcb, the saved unit has uid d110fdc5. The only difference is the
audit flag: the 2026-10-20 canary forced `audit_capture` on every row, today only the 32 production units
chosen in advance carry it (32 of 3,096). Setting the flag on the manifest row gives uid d110fdc5 and the
assignment hash 1949593e that `RESULT.json` stores.

| Row of the unit | Value |
|---|---|
| layers with an action | 20 (all but 0, 2, 3 and 19), one deactivated expert each, no activation |
| layer 1 | deactivate expert 17 |
| prompt, from `prompt_prefix` | 191 tokens, its digest equals `prompt_sha256` |
| seed of segment 0, `max_tokens` | 2186601647, 130,881 (the whole allowance) |
| closure marker | token ids [200005, 17196, 200008] = `<|channel|>final<|message|>` |

The builder returns the request that `Engine.admit` registers on both tensor-parallel ranks (fields
`uid, prompt, prefix, closure, actions, capture_layers, window, whole, prompt_scope, audit_capture`). The
engine itself needs vLLM and was not run: `PARTIAL.json` on disk records the same seed, `max_tokens`, history 0
and job 61581237.

## 2. Scope (`stream.Clock.observe`)

The clock sees every position the engine runs. Replaying the 191 prompt tokens and the saved 805 output
tokens (the last one is never fed, so 995 positions) gives one pass, `closed` True, and the closure marker
completes at generated position 738. The hook keeps every prompt row and every generated row before
closure: 191 + 738 = 929 rows. The stored layer-1 counter `active_positions` is also 929 (191 prompt, 738
generated), and the stored capture says `observed_tokens` 995, `passes` 1, `closed` True.

## 3. Layer execution (`StreamCapture._whole_batch`, `operators.score_override_dual_masked`)

For every row the layer applies the dual score override: the removed expert's log-score goes to the row
minimum minus 0.01, the router takes its own top-4 and renormalizes. The router logits are not kept, only the
audited top-4 ids and weights, so the operator cannot be re-run on this unit; the operator test
(`test_dual_matches_extrema_reference_and_preserves_inactive_bits`) passes instead. What the archive shows at
generated position 0 of layer 1:

| | ids | weights |
|---|---|---|
| native | [18, 29, 12, 6] | 0.5410026908, 0.2987700403, 0.0962416083, 0.0639856309 |
| executed | [18, 29, 12, 6] | 0.5410026908, 0.2987700701, 0.0962416083, 0.0639856383 |

Expert 17 was absent, so the membership is unchanged; renormalizing over a changed score vector moves the
weights in the eighth digit. Across the 738 stored generated positions of layer 1, native routing had expert
17 in the top 4 at 284 positions and the executed routing at none.

## 4. Counters (`stream.batched_dose`)

The additive counters of one request are summed from the layer's rows. Feeding the 738 stored generated rows
of layer 1 to `batched_dose` with the role "deactivate 17" gives the generated part of the stored totals:

| Counter, layer 1 | Recomputed from the archive (generated rows) | Stored in `CAPTURE.json` (prompt + generated) |
|---|---|---|
| active positions | 738 | 929 (191 + 738) |
| changed membership | 284 | 325 (so 41 are prompt rows) |
| native deactivated weight | 61.9238586 | 70.9533920 |
| executed deactivated weight | 0 | 0 |
| L1 distance of the weights | 123.8477249 | 141.9067688 |
| membership failures | 0 | 0 |

The archive holds no prompt rows, so those totals are checked in part. The stored finite and observation
counters are 739 on every layer; layers without an action (0, 2, 3, 19) store zero active positions. These
are same-state routing measurements, not effects on correctness or length.

## 5. Release and archive (`Engine.release`, `pulse.compress_audit_capture`)

The unit finishes after 805 output tokens with `stop`; `RESULT.json` says `complete`, one segment, no
stochastic redraw, and binds the raw output hash e53ba782 that `CAPTURE.json` repeats. The audit is a
compressed archive of 144 arrays (24 layers times 6 fields: positions int32, ids uint8, weights float32,
active bool), layer 1 covering generated positions 0 to 737. Its 620,456 bytes hash to c3caac87…3533588,
which equals `full_capture.sha256_file` in `CAPTURE.json`.

## Worth noticing

- A saved canary store cannot be matched by rebuilding the 48 rows with today's code: only 1 of the 48 saved
  uids is reproduced (a row whose audit flag was already on). With the flag forced on, all 48 are reproduced.
- The scope is checkable on a CPU: replaying saved tokens through `Clock` gives exactly the 929 rows the
  GPU job counted, with the closure at generated position 738.
- 41 of the 325 changed positions are prompt positions (the intervention covers the prompt); the audit
  archive keeps generated rows only, so the prompt part of each counter rests on the stored totals.
- This unit completed, but job 61581237 failed after 545 s because the capture of another unit
  disagreed between tensor-parallel ranks (`GPT_SOFTWARE_FAILURE.json`: 3 completed units, 45 partial). The
  later cleanup error (`unknown released request`) hides that first error. Three saved units do not qualify the
  48-row canary, and GPT production was withheld; the failure and every charge remain recorded.
- The tests of the batched counters (`tests/steering/test_stream_batch.py`, with `test_whole.py`) ran on the
  login node: 32 tests, OK, 33 s. The canary now starts with `sbatch scripts/steering/gpu.sh steering.whole
  canary --release report/NAME --model gpt`.
