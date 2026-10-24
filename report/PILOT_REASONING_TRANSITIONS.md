# Exploratory reasoning-class exchanges

This post-hoc follow-up uses the 64 saved question-level summaries. Each question contributes equally. The table conditions on an observed class change: self-transitions and natural termination are excluded. Thus percentages describe where a trace switches after a class, not the fraction of all reasoning sentences.

| Exchange | Mean rate, correct | Mean rate, incorrect | Correct-minus-incorrect (95% question-bootstrap CI), percentage points | Raw r | BH q | Length/cap-adjusted r | Adjusted BH q |
|---|---:|---:|---|---:|---:|---:|---:|
| Read → Analyze | 25.66% | 36.72% | -11.06 [-18.06, -3.52] | -0.324 | 0.251 | -0.235 | 0.462 |
| Read → Implement | 27.61% | 16.39% | +11.21 [+2.53, +19.32] | +0.294 | 0.251 | +0.241 | 0.462 |
| Implement → Analyze | 16.64% | 25.66% | -9.02 [-15.87, -2.62] | -0.312 | 0.251 | -0.161 | 0.744 |
| Verify → Monitor | 34.29% | 23.22% | +11.08 [+1.52, +20.62] | +0.239 | 0.346 | +0.079 | 0.951 |
| Plan → Explore | 0.79% | 2.35% | -1.56 [-2.85, -0.15] | -0.267 | 0.251 | -0.193 | 0.616 |
| Explore → Monitor | 11.42% | 12.96% | -1.54 [-7.95, +4.67] | -0.059 | 0.936 | +0.082 | 0.951 |
| Explore → Analyze | 36.40% | 34.21% | +2.19 [-9.16, +13.88] | +0.047 | 0.936 | +0.022 | 0.951 |
| Explore → Verify | 20.54% | 22.95% | -2.41 [-12.41, +7.42] | -0.060 | 0.936 | -0.064 | 0.951 |

The patterns motivate hypotheses about monitoring after verification and returning from implementation to analysis. They do not establish helpful or harmful exchanges. None of the 42 class-change contrasts, or the separate 49 next-sentence contrasts including self-transitions, survives BH 5%; neither do their length/cap-adjusted counterparts. Plan→Explore occurs in only 16 questions, so its low rate deserves particular caution.

The earlier dynamics pass also found lower transition entropy after Explore (r = −0.369, q = 0.088) and shorter Monitor return intervals (r = −0.344, q = 0.088) associated with correctness. These weaken to −0.170 and −0.176 after length/cap adjustment. Entropy estimates can be lower simply because fewer transitions were observed; no cognitive-control interpretation is established.

This follow-up describes label sequences, not expert-routing changes aligned to individual transition boundaries. Source-transition counts, position, content and question difficulty remain possible confounders. Labels are retrospective.

Bootstrap intervals use 1,000 outcome-stratified question resamples for raw group differences. Correlation intervals use 1,000 question resamples, refitting adjustments per replicate. Inference requires at least ten source-eligible questions in each outcome and ten exhibiting the edge. BH is separate per mode and adjustment; this is an exploratory extension, not a prespecified confirmatory test.

[All 182 comparisons](../results/pilot-metrics/transitions-v1/comparisons.jsonl) · [Method record](../results/pilot-metrics/transitions-v1/summary.json)
