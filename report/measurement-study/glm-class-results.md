# GLM complete-label reasoning dynamics

Source: completed CPU job 58487905, all four 25,000-record label shards. There are 1,390 selected completions spanning all six benchmarks, 99,999 canonical labels and one retained unknown label. Exact-source matching rejected no traces.

The table reports exploratory question-level associations with correctness, adjusted for log completion length and cap status. BH correction is applied within metric family, adjustment and benchmark. These data have one available selected completion per question; they cannot establish within-question outcome effects or causal routing effects. Monitor stay rate and Monitor→Monitor are the same statistic and are one finding.

| Benchmark | Metric | Questions | r | BH q |
|---|---|---:|---:|---:|
| amc23 | class_fraction/Verify | 40 | 0.461 | 0.0219 |
| olympiad | class_fraction/Monitor | 539 | -0.329 | 3.51e-14 |
| olympiad | dynamics/Monitor/stay_rate | 354 | -0.271 | 1.06e-05 |
| olympiad | next_sentence/Monitor/Monitor | 354 | -0.271 | 1.24e-05 |
| olympiad | class_fraction/Analyze | 539 | 0.144 | 0.00276 |
| olympiad | existing_events/self_correction_steps | 539 | 0.104 | 0.0477 |

These are label-based class dynamics. Complete all-layer routing/class associations still need the production routing collection. No adjusted association passes BH at 0.05 in the AIME24/AIME25, MATH500 or Minerva subsets of this GLM cohort. AIME remains in the reporting scope.
