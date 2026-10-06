# Arm-blind candidate-to-substantive-check rubric (v1)

Judge only the original problem, the full emitted native reasoning prefix,
its final triggering sentence, and one assigned 256-token reasoning
continuation. Do not use a gold answer, arm identity, other continuations or
unemitted native future text. The triggering sentence is already part of the
prefix; do not count it as an outcome.

Return two booleans:

- `start`: The triggering sentence presents a specific computed candidate,
  value or answer that can be tested, and the full preceding prefix has not
  already substantively checked it against an original constraint or an
  independent calculation. A tentative expression or generic plan is false.
- `target`: In the new continuation, the model substantively tests that same
  candidate against an original condition or by an independent calculation.
  Merely saying “check,” repeating the candidate, checking an unrelated
  value, or announcing a final answer is false. If the continuation closes
  before a substantive check, use false.

When the evidence is ambiguous, mark the relevant boolean false. The reader
must return exactly a JSON object with keys `start` and `target` and no
explanation. Two independent seeded draws from the same local Qwen3.8 judge
are LLM audits of visible text, not human ground truth. Primary agreement
requires both readers to return valid stopped JSON and both booleans true.
