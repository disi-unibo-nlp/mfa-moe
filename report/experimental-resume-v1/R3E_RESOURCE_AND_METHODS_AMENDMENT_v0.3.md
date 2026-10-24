# Clean R3-E confirm-connected-family sensitivity — 2026-10-02

This version extends `R3E_RESOURCE_AND_METHODS_AMENDMENT_v0.2.md` without replacing the registered 509-question clean primary. The sensitivity removes every development/tuning question whose frozen duplicate family contains a confirm question, leaving the 483 questions and 479 families sealed in `R3E_CLEAN_PREFLIGHT_v1.json`. It preserves the original split and does not claim the historical confirm set was untouched.

All 100 nested feature training sets are recomputed within the 483-question population. The B1 routing deltas, marginal class profiles, and ten token-ID × position pseudo-class reference draws are fitted on each sensitivity training set only. The 509-question B1 fit is provenance, not a consistency target for this different population. Correctness remains unread until the 483-question feature freeze is sealed.

The feature stage has a maximum 32.5 CPU core-hours: 2 CPUs ×15 minutes for a one-pair/one-draw smoke; then 16 CPUs ×2 hours for all 43 pairs and ten draws, conditional on smoke verification. The 483-question outcome CV is priced and run separately after the full feature freeze. This sensitivity will report its own estimate and interval; it does not update the 509-question primary or its Holm family.
