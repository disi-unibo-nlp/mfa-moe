# Gemma full eligibility audit review

Read-only Sol review, 2026-10-08. The existing protected controller's Gemma qualification **FAIL reproduces exactly** from the original 128-prefix / 256-draw audit. Both decisions meet the apparent-family support minimum and have held-out support; their apparent precision fails the unchanged 70% gate. No definite operational or interpretation defect was found.

## Original coverage and execution

All 48 original discovery receipts retain their manifest assignments and model/runtime/scientific executor bindings, and their seals match the audit frame's full source-receipt roster. The frame has 128 unique prefix identities and no missing discovery assignments. Every reader receipt passes its original full-frame, UID, draw, record-input, checkpoint, contract, executor and invocation/runtime joins. The source closure checks pass for all 35 scientific files and r5 readiness's 29 source files / 24 program files.

The original reader cards reconstruct their unchanged scientific parents under transport r3. Exact card-selected UIDs match invocation runtimes and cover the frame once:

| Original indices | Job | Draws | Actual terminal accounting |
| --- | --- | --- | --- |
| 0–46 | 59675163 | 94 | COMPLETED 0:0, 4,954 s, 16 CPUs |
| 47, retained first measurement | 59674666 | 2 | COMPLETED 0:0, 764 s, 16 CPUs |
| 48–127 | 59675165 | 160 | COMPLETED 0:0, 8,539 s, 16 CPUs |

All invocation results are COMPLETE without errors. Submission/intent identities and actual scheduler comments match these accepted cards. Original seeds remain determined by the unchanged [Reader.rate](../../../src/moe_exp/verification_study/readers.py) function from each original record digest and fixed draw index; seeds are not separately serialized in raw receipts. No inference or competing recognizer fit was performed.

## Reader output and qualification

The full ledger has **209 STOP_PARSED**, **30 LENGTH_UNKNOWN** and **17 STOP_UNPARSEABLE** draws. All length draws reached the original 1,024-token cap and remain unknown. Sixteen stopped invalid outputs violate the required implication from `eligible=true` to candidate-valid, constraint-bound and unchecked; one uses a nonboolean judgment. The declared strict whole-suffix JSON-fence amendment recovers 62 stopped draws while preserving all raw receipts, original caps and validation rules. Total annotation output is 171,972 tokens.

Both draws are usable for 91/128 prefixes. Eligibility pairs comprise 24 agreeing true, 56 agreeing false, 11 disagreements and 37 unknowns. Already-checked pairs comprise 16 agreeing true, 59 agreeing false, ten disagreements and 43 unknowns; six of those unknowns are null judgments despite otherwise usable parses.

| Apparent decision | Proposal rows | Confirmed | Rejected | Nonstop | Invalid | Ambiguous | Disagree | Families | Family-weighted precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Eligible | 50 | 14 | 14 | 11 | 5 | 0 | 6 | 32 | 25.5208% |
| Already checked | 12 | 5 | 2 | 2 | 2 | 1 | 0 | 12 | 41.6667% |

These row counts explain the failure but are not the family-weighted precision statistic. Both family counts exceed the frozen minimum of ten. Held-out support exists for 11 eligible families and four already-checked families; the frozen family-bootstrap intervals are [0, 0.3636] and [0, 0]. The latter reflects two agreeing-false and two ambiguous proposed-check pairs, not proof that substantive verification is absent. No held-out precision threshold is imposed by the frozen protocol.

## Scientific meaning and continuation

This is a qualification limit for the current cheap recognizer under the common frozen two-draw reader. It combines actual stopped-reader rejections, disagreement, truncation and invariant-invalid output. It does not establish absence of candidate-specific verification or a zero causal routing effect. The reader reference is two draws from one checkpoint, not independent human ground truth.

The sealed [r5 stage runner](../../../scripts/experimental_resume/verification_study_v3_autostages_r5.py) correctly records recognition LIMIT before nomination, calibration and later causal allocations. Preserve Gemma's result and unavailable downstream claim cells in the all-seven report. No new draws, parser relaxation, threshold change or additional allocation follows from this review.

Exact seals: qualification `7538fd8e821cdc057aeb549af054d54c63a24c6c3bbd3bbce403301034bd311b`; frame `40c8b06a92632b44934723fcfe290fdb5d501007ff6cf93529b887ecb8b33046`; full ledger `2e361c9cee2608b314cc0348aaec1c05e06d0e3c5349477e59ecd034f08e1091`; aggregate semantic inputs `f1f0655fd8a95a0a711a2bd3f03a11abe869833a894163bf0bb4bcad172feb94`; format annex `02a806cf73686302916783cdc39585c4bc31a353c3b0a4bcbabed4d414e9d40c`; r5 readiness `68c4db773948a12b277e421824fa8f2ff6515c3f5970944834fd717483225a41`.

The pure full-ledger reconstruction equals the saved ledger. The original [qualification function](../../../src/moe_exp/verification_study/semantics.py), evaluated under the declared scoped format adapter and annotated with the controller's provenance, equals the saved qualification including its seal. This review changed only this document.
