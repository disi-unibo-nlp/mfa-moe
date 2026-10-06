# Independent Opus 5.5 review and source check — 2026-10-02

The one-shot, tool-free Claude CLI returned a successful response from
`claude-opus-5-5` in `OPUS_SECOND_OPINION_RAW_v0.2.json`. Earlier high/medium
effort attempts timed out without a response; only v0.2 is an opinion.
This review is design advice, not evidence from a run.

Useful advice retained for a versioned discovery amendment:

- Repair closed-TeX equality and formula-based start detection, with all online
  inputs restricted to the original problem and already emitted tokens.
- Audit semantic ratings with a different model family and, where feasible,
  human adjudication. Two independently sampled Qwen ratings share a model
  family and do not establish human truth.
- Include actual target-expert top-k entry, gate displacement, and realized
  dose alongside behavioral effects; native activation ranks are only proposals.
- Compare selected experts against exposure/layer-matched random controls and
  consider a separate fixed-k ablation screen. Treat routing velocity and
  acceleration as manipulation/readout measures, not target-selection scores.
- Freeze detector, experts, doses, families, outcomes and analysis before
  independent validation. A SHA-bound source snapshot meets this project's
  freeze requirement; no Git commit is authorized.

Advice **not adopted as stated**:

- The review claimed that legacy X2 showed router-logit bias +2 as a sweet
  spot and +0.5/+1 at the noise floor. The saved G3 receipt instead selected
  **reweight +2 over a layer band** under its registered dose/support rule.
  Bias +2 is a different intervention, and several bias cells lacked matched
  random-dose support. `G3_DOSE_SUPPORT_AMENDMENT.json` expressly permits
  dose selection, not semantic or bias-dose efficacy. A +2 or negative-bias
  action would require a separately versioned exploratory arm and worker
  qualification.
- The proposed 80 prefixes × 5 arms × 3 seeds per hypothesis is an unpriced
  workload. Its eligibility and runtime must be measured; it cannot substitute
  for the frozen 48/128/96 family pools or silently make validation families
  part of expert selection.
- Its suggested primary outcome required both detector and rater agreement.
  For causal validation, arm-blind semantic ratings must remain separately
  reported from the online detector, whose mistakes are themselves an outcome.
- Truncated/capped/failed assigned continuations remain in intention-to-treat
  accounting under `PROTOCOL_v0.1.md`; they are not excluded merely because
  a reviewer suggested it.

The immediate next steps are the discovery-only detector v2 audit, a
family-balanced native expert shortlist, and a priced same-prefix causal
micro-screen with native and matched-random controls. The independent review
does not alter the sealed legacy v0.3 study or original v0.1 trajectory study.
