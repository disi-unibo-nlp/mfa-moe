# Nemotron recognition qualification review — 2026-10-08

GPT-6.1 Sol xhigh independently reproduced the completed Nemotron qualification and its entire original 256-draw interpretation ledger. Every raw receipt passed the unchanged finite-reader checks for full-frame identity, UID, fixed draw, input, deployment and invocation/runtime binding. The declared format annex and frozen scientific/r5 operational source closures also passed validation. This review changes no source, gate, rating or assignment.

All 256 fixed draws are present for 128 prefixes. Invocation results are COMPLETE with no errors: pilot 59665246 produced two draws at frame index 49; shard 59666571 produced 98 draws for indices 0–48; shard 59666574 produced 156 draws for indices 50–127. Receipt presence and completed execution are distinct from semantic qualification.

Of the 256 draws, 219 stopped and 37 reached the length cap. There are 191 usable stopped judgments and 28 stopped outputs rejected by unchanged validation: 25 assert `eligible=true` without satisfying the required candidate-valid/constraint-bound/unchecked conjunction, and three use non-boolean judgment types. The annex recovered 97 complete fenced stopped responses; these format repairs are already included in the result.

Only 81 of 128 prefixes have two usable judgments. Eligibility pairs comprise 23 agreeing true, 43 agreeing false, 15 disagreements and 47 unknowns. Already-checked pairs comprise 13 agreeing true, 50 agreeing false, 14 disagreements and 51 unknowns; four additional unknowns arise from null judgments despite otherwise usable parses. A missing, truncated, malformed, null or disagreeing draw counts as unconfirmed under the frozen gate.

| Apparent decision | Families | Family-weighted confirmation precision | Frozen gate |
| --- | ---: | ---: | --- |
| Eligible | 34 | 25% | At least 10 families and 70% precision |
| Already checked | 10 | 10% | At least 10 families and 70% precision |

Both decisions meet the family-count minimum and fail precision. The 48 apparent lexical eligibility proposals contain 13 confirmations, 16 agreeing rejections, 10 pairs with a non-stopped draw, four invalid pairs and five disagreements. The 13 already-checked proposals contain one confirmation, five agreeing rejections, three pairs with a non-stopped draw, one invalid pair, one ambiguous pair and two disagreements. These raw counts are not the family-weighted gate statistic. Even an explanatory best-case calculation treating every unknown apparent pair as confirmed, while retaining observed disagreements, reaches only 57.35% eligibility and 45% already-checked precision. This calculation does not amend any outcome or gate.

Held-out support exists: 11 eligibility families and two already-checked families. No unavailable-held-out-support failure was triggered. Eligibility's held-out interval is [0, 0.4545]; the already-checked interval is [0, 0], derived from one agreeing-false and three unknown proposed-check pairs. The latter describes unconfirmed judgments, not proof that substantive verification is absent. The reference remains two fixed draws from one reader, without independent human ground truth.

No definite operational defect was found. This is a scoped qualification limit for the current cheap controller, not evidence that Nemotron routing cannot causally affect verification. The sealed r5 runner correctly records recognition LIMIT before nomination, calibration or final treatment. All seven required model cells remain represented; the other model cells continue through their own gates without treating this failure as a zero effect. No new inference, submission or gate change is justified by this review.

Exact provenance: recognizer `4c713a226fd3256b9c037ead711b9c37396c7051380e2ed9a471bc614b99cac3`; full frame `b7c117261ed90c15fe447a4f5bf446eb0c20d9b1eecedd74743bbd2866548f9a`; interpretation ledger `b896c59cc524671191c1bf3bf29c9c1c0619d63daf492e74ecedceda99833518`; action-input aggregate `dc4fcb1fa08287cf87dc74ff2c468b5a796be2def944485036c01d0adaff8b06`; format annex `02a806cf73686302916783cdc39585c4bc31a353c3b0a4bcbabed4d414e9d40c`.

Sources: the active [campaign](../../../report/verification-study-v3/CAMPAIGN.json), [format annex](../../../report/verification-study-v3/SEMANTIC_FORMAT_BINDING-r1.json), `RESULT_ROOT/nemotron/RECOGNIZER.json` and its complete bound raw-draw ledger, unchanged [semantic qualification](../../../src/moe_exp/verification_study/semantics.py), [stage consumer](../../../src/moe_exp/verification_study/stages.py), and sealed [r5 stage runner](../../../scripts/experimental_resume/verification_study_v3_autostages_r5.py). `RESULT_ROOT` is the campaign's unchanged private result root.
