# Question-balanced fixed-budget baselines

Descriptive estimates from the frozen original-budget inventory. Correctness and termination are separate.
The 95% intervals jointly resample question identities across models; they do not treat sibling attempts as independent questions.
Unresolved correctness is omitted only from the correctness denominator. Cost retains all attempts.

| Model | Cohort | Questions scored | Accuracy (95% CI) | Mean completion tokens | Cap fraction |
|---|---|---:|---|---:|---:|
| gemma | A | 1545 | 0.909 (0.895–0.924) | 8750 | 0.010 |
| gemma | B | 15 | 0.736 (0.581–0.861) | 30167 | 0.089 |
| glm | B | 14 | 0.728 (0.566–0.867) | 49233 | 0.007 |
| gpt | A | 1546 | 0.856 (0.840–0.873) | 3935 | 0.000 |
| gpt | B | 37 | 0.749 (0.676–0.811) | 12031 | 0.000 |
| nemotron | A | 1547 | 0.895 (0.879–0.910) | 8750 | 0.001 |
| nemotron | B | 14 | 0.701 (0.510–0.877) | 36320 | 0.019 |
| qwen330b | A | 1545 | 0.871 (0.853–0.888) | 7989 | 0.008 |
| qwen330b | B | 31 | 0.729 (0.622–0.817) | 18442 | 0.023 |
| qwen35 | A | 1546 | 0.915 (0.900–0.928) | 14700 | 0.002 |
| qwen35 | B | 7 | 0.696 (0.426–0.912) | 46622 | 0.000 |
| qwen36 | B | 6 | 0.568 (0.351–0.832) | 54613 | 0.000 |

B results use arm inclusion probabilities within each question and apply only to that model's historically eligible questions. They are not benchmark accuracy estimates.

Natural-completion sensitivity tables, denominators, unresolved counts, configuration and input/code bindings are retained in the adjacent JSON artifact.

No routing candidate, prospective gain, or causal improvement is established by these tables.
