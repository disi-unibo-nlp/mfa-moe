# Qwen3.6 routing tradeoffs

Observed frontier is descriptive. Supported improvements require paired intervals and BH-adjusted prespecified comparisons.

Small AIME question counts can yield wide uncertainty even with 32 attempts per question.
Matching seeds do not imply identical stochastic trajectories.

| Policy | Accuracy | Mean tokens | Raw total tokens | Infrastructure failures |
|---|---:|---:|---:|---:|
| b7112a905e51 | 0.3750 | 5.5 | 264 | 0 |
| e07786ed703e | 0.3750 | 6.2 | 300 | 0 |

Full per-benchmark intervals, paired reversals, latency and realized intervention rates are in comparison.json.
Failed-attempt tokens are included only when their exact cost is known; incomplete cost ledgers are rejected.
Seven-class pilot labels are interpretive context and never an online routing judge.
