Outputs of the failed LFM materialize job 60844283 (2026-10-17 13:30). The stratified sampler selected 1,534 of
the 1,547 designated questions: 13 single-attempt olympiad traces had no submitted answer (12 hit the
131,072-token cap in repetition loops, 1 stopped without closing </think>) and were marked is_correct=None,
hence ineligible. The seven source models' original sampling admitted such traces (scored incorrect at sampling
time), so lfm.trace_record now marks no-answer traces strictly incorrect for sampling eligibility while the native
estimands keep them as unknown bounds. Bulk traces moved to <bulk>/lfm25/failed-materialize-60844283/.
