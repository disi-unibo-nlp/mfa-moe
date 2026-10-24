# Recovered routing associations

Job 58471939 completed all four interrupted CPU bundles with exit 0:0 and zero rejected rows. All 15 saved-routing bundles are now processed. These recovered files contain legacy routing captures with partial layer coverage and unverified reasoning-only token scope. They do not establish complete all-model measurement coverage.

Associations average attempts within questions. Adjusted correlations control question-mean log completion length and cap fraction. BH correction is applied separately within benchmark, metric family and adjustment; related metric/layer tests are not independent discoveries. GPQA is excluded from the six-math-benchmark aggregate.

## Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 — aime25

960 attempts; 30 questions; 144 capped attempts retained. Adjusted BH-significant tests: 0 / 1284.

| Metric | Adjusted r | BH q | 95% question-bootstrap interval |
|---|---:|---:|---|

Source: `results/available-metrics/recovery-v1/routing-435bd20516aa/aime25/correlations.jsonl`

## Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 — olympiad

675 attempts; 675 questions; 94 capped attempts retained. Adjusted BH-significant tests: 223 / 1284.

| Metric | Adjusted r | BH q | 95% question-bootstrap interval |
|---|---:|---:|---|
| saved_full/reasoning/router_margin_std_layers | -0.261 | 3.77e-10 | [-0.330555609091097, -0.19667272285158105] |
| saved_full/router_entropy_l27 | 0.261 | 3.77e-10 | [0.20095160461431694, 0.3269110610211052] |
| saved_full/router_confidence_l27 | -0.261 | 3.77e-10 | [-0.3269104988812931, -0.200950838131546] |

Source: `results/available-metrics/recovery-v1/routing-7f8969729d83/olympiad/correlations.jsonl`

## Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 — aime24

960 attempts; 30 questions; 106 capped attempts retained. Adjusted BH-significant tests: 4 / 1284.

| Metric | Adjusted r | BH q | 95% question-bootstrap interval |
|---|---:|---:|---|
| saved_full/hidden_step_distance_mean_layers | -0.605 | 0.0499 | [-0.844754011381674, -0.08653351037353003] |
| saved_full/reasoning/hidden_step_distance_mean_layers | -0.603 | 0.0499 | [-0.8363236272438608, -0.08576535158813217] |
| saved_full/hidden_step_distance_l35 | -0.589 | 0.0499 | [-0.7837518282281668, -0.17840766773224187] |

Source: `results/available-metrics/recovery-v1/routing-c0f91d66bb58/aime24/correlations.jsonl`

## qwen3.5-35b-a3b-mtp-ud-q4-k-xl — gpqa_diamond

1980 attempts; 198 questions; 292 capped attempts retained. Adjusted BH-significant tests: 46 / 228.

| Metric | Adjusted r | BH q | 95% question-bootstrap interval |
|---|---:|---:|---|
| saved_full/router_boundary_margin_l36 | -0.333 | 0.000157 | [-0.4487492769600671, -0.20607876381796542] |
| saved_full/hidden_norm_l27 | -0.306 | 0.000384 | [-0.41900646954710963, -0.1821247431237687] |
| saved_full/hidden_norm_l34 | -0.300 | 0.000384 | [-0.42637741597033246, -0.18044825005556797] |

Source: `results/available-metrics/recovery-v1/routing-ee07a2f16ddd/gpqa_diamond/correlations.jsonl`

