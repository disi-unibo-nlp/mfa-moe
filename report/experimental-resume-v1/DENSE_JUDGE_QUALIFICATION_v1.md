# Direct seven-class judge qualification v1

Frozen before the result of Slurm job 59112590 was read. The fixed audit contains
200 discovery-family sentences, balanced across the seven historical classes
(28 or 29 each). The current direct vLLM prompt is a new measurement, not the
original DSPy prompt. Its qualification controls whether it can supply dense
class labels for discovery; it does not validate semantic transition endpoints.

The direct judge qualifies for exploratory dense labeling only if at least 95%
of outputs stop with exactly one parsed seven-class label and at least 70% of
all 200 assigned fixtures agree with their historical class. Unparsed or
non-stop outputs count as disagreement. Report the complete confusion matrix,
per-class recall, family coverage, generated tokens, wall time, and failures.
No class-specific minimum is imposed because sparse historical labels are not
ground truth, but any class with less than 50% historical agreement must be
flagged as weak; hypotheses depending on it need a separate semantic audit.

Passing these operational thresholds permits dense discovery labels only.
Substantive verification, approach identification, failed checks, and ordered
trajectory success require the separately frozen behavioral rubric and arm-blind
ratings. This judge sees next-sentence context and must never serve as the online
prefix detector. If either operational threshold fails, no dense labeling using
this exact prompt is submitted; a revised, separately versioned judge must be
qualified on a new prospective fixture before use.
