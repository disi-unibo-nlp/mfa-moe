# Qwen3-30B full eligibility audit review

Read-only Sol review, 2026-10-08. The protected controller's full qualification **FAIL reproduces exactly** from the original 128-prefix, 256-draw audit. Both decisions meet the minimum apparent-family support, but their family-weighted precision fails the frozen 70% gate. No definite execution, deployment, input, resume or interpretation defect was found.

## Execution and provenance

The original full frame has 128 unique prefix identities with both fixed draws present. The three accepted reader cards reconstruct their unchanged scientific parents under the sealed transport binding, and their exact selected UIDs match their invocation runtimes. Coverage is disjoint and complete:

| Original frame indices | Job | Draw receipts | Actual terminal accounting |
| --- | --- | --- | --- |
| 0–55 | 59674100 | 112 | COMPLETED 0:0, 5,686 s, 16 CPUs |
| 56, retained first measurement | 59673381 | 2 | COMPLETED 0:0, 720 s, 16 CPUs |
| 57–127 | 59674102 | 142 | COMPLETED 0:0, 7,358 s, 16 CPUs |

Every original receipt passes UID/draw/input/full-frame, checkpoint/reader-contract/runtime/executor and invocation binding. All three invocation RESULT artifacts are COMPLETE without errors; submission, intent and exact scheduler-comment joins pass. The 35 scientific sources and r5 readiness's 29 source files / 24 program files remain unchanged. Seeds are determined by the unchanged [Reader.rate](../../../src/moe_exp/verification_study/readers.py) function from the original record digest and draw index. Receipts do not serialize seeds, so this is verified source and input provenance rather than a separate empirical seed readback. No inference or fitting was rerun.

## What the reader measured

Of 256 draws, **193 are STOP_PARSED**, **39 are LENGTH_UNKNOWN**, and **24 are STOP_UNPARSEABLE**. All 39 length draws reached the original 1,024-token cap and remain unknown. The stopped invalid responses comprise 22 violations of the eligibility implication and two nonboolean values. The declared whole-suffix JSON-fence amendment recovers 33 stopped draws; it does not salvage length draws or weaken schema checks. Total annotation output is 159,114 tokens.

Both draws are usable for 80/128 prefixes. Across all prefixes, eligibility agrees true for 10, false for 63, disagrees for seven, and is unknown for 48. Already-checked judgments agree true for 21, false for 48, disagree for seven, and are unknown for 52, including four ambiguous boolean judgments.

For the apparent proposals actually used by the qualification:

| Decision | Proposal rows | Both confirm | Both reject | Nonstop | Invalid stopped pair | Disagree | Apparent families | Family-weighted precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Eligible | 50 | 5 | 22 | 9 | 9 | 5 | 33 | 10.1010% |
| Already checked | 12 | 4 | 5 | 3 | 0 | 0 | 12 | 33.3333% |

The family-weighted precision is not the raw proposal-row confirmation fraction. Held-out support is available for 10 eligible families and four already-checked families, with frozen family-bootstrap intervals [0, 0.4] and [0, 0.75]. There is no held-out precision threshold in this protocol. The two recorded SUPPORT_OR_PRECISION reasons here arise from apparent precision, not missing family support or missing held-out coverage.

## Interpretation and continuation

The failure combines genuine stopped-reader rejection, disagreement, cap-related unknowns and invariant-invalid output. It is a measurement and recognizer qualification limit; it does not establish that candidate-specific verification is absent or that routing interventions have zero effect. The reader truth scope remains two fixed draws from one frozen checkpoint, not independent human judgments.

The sealed [autostages r5](../../../scripts/experimental_resume/verification_study_v3_autostages_r5.py) records this model's recognition LIMIT before nomination, calibration or later causal allocations. That is safe continuation under the existing gates. Preserve the Qwen3-30B result and its unavailable downstream claim cells in the all-seven report; this result does not justify dropping the model or changing the threshold.

## Exact seals

- Qualification: `393580428dfd985aa39f0e085c84a974740c82cb64712ca9290f67e50b79cc45`.
- Original frame: `1e7cad70b74c88ffd00bf952d7a4c424eb89a61bc9788db3735c94abfe2e3f0c`.
- Full decision ledger: `517f995ffc857aba9d0699e0a3e2f0dce1d08540c38d66c03fee298c5e26a45f`.
- Aggregate semantic inputs: `a838cef23943c2247799e198f0dd2d13a3b83d01b9425c9f16e42963b546c37f`.
- Declared format annex: `02a806cf73686302916783cdc39585c4bc31a353c3b0a4bcbabed4d414e9d40c`.
- r5 readiness: `68c4db773948a12b277e421824fa8f2ff6515c3f5970944834fd717483225a41`.

The pure full-ledger reconstruction equals the saved ledger, and original qualification under the declared scoped parser, with the controller's provenance annotations, equals the saved qualification including its seal. This review changed only this document; the protected controller retained sole ownership of scientific derivation.
