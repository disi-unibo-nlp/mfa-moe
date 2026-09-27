# Quick pinned-expert cache analysis

Reproduce: `python3 report/analyze_pinned_expert_cache.py`

Qwen fixed positive strength +1, full stochastic evaluation. This is a new analysis of saved counters, not a new generation or SSD timing experiment.

## Fixed resident-cache proxy

Pin eight experts per steered layer (48 layer–expert identities across six layers). Count every selection outside that set as a potential expert load. Both columns use the same hidden states from the guided run.

| Layer | Target-cache hits before bias | Target-cache hits after bias | Relative reduction in nonresident selections |
|---|---:|---:|---:|
| 27 | 7.00% | 37.39% | 32.68% |
| 34 | 3.52% | 36.72% | 34.41% |
| 35 | 3.15% | 51.95% | 50.38% |
| 36 | 1.08% | 38.26% | 37.58% |
| 38 | 1.62% | 50.73% | 49.91% |
| 39 | 1.66% | 36.42% | 35.35% |
| All six | 3.00% | 41.91% | 40.11% |

An equal-budget cache chosen by calibration frequency covers 13.68% of pre-bias selections. Guided target pinning reduces nonresident selections by 32.70% against this control on the same guided states.

Even a hindsight-optimal static eight-expert cache covers only 14.67% of pre-bias selections. Against that static oracle, the reduction is 31.93%. This oracle is an analytical control, not a deployable baseline or a bound on adaptive caching.

## Separately generated, matched quality evaluation

Across 1395 attempts on 465 problems, rescored accuracy is 84.95% baseline and 84.66% guided: -0.29 percentage points.
The saved paired problem-bootstrap 95% interval is [-1.15, +0.65] points; this is not proof of non-inferiority.
Mean generated tokens: 13862.9 baseline versus 13706.1 guided. AIME24 alone changes by -2.78 points; aggregate quality is not uniform across tasks.

## Interpretation and limits

The bias demonstrably concentrates hook-selected experts into a small fixed set, alongside a small aggregate accuracy difference in the matched generation experiment. This supports testing an SSD-streaming trade-off. It does not establish reduced actual I/O or faster inference.

- Pre-bias histograms are not separately generated baseline routing.
- Counters include prefill, decode and engine padding.
- Hook top-k is not separately observed native dispatch; ties can differ.
- No temporal order, per-attempt routing, or unsteered-layer counters.
- No measured SSD traffic, wall-clock speed, or cache-effect confidence interval.
- Aggregate accuracy can conceal benchmark-specific losses.

The six-layer reduction must not be reported as a whole-model reduction. Prefill and batched tokens can share one expert load, so selection counts are not disk-read counts. Cache warm-up and the remaining layers also cost time.

Next validation: collect ordered decode-only expert IDs for paired baseline/guided generations; compare equal-memory target pinning and LRU caches; then benchmark actual expert reads and end-to-end answer latency. Fix the acceptable accuracy loss before choosing the deployment policy.
