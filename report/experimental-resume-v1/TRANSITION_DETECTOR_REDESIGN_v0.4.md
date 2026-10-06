# Prefix-only transition detector qualification, v0.4

This version supplements v0.3. The same three transition hypotheses remain
frozen. The v2.2 rules are candidate screens. Their fires are not semantic
starts, and a semantic start is not evidence that an intervention changes the
following class or improves utility.

## New full-context scout

The 12-prefix context-stratified Qwen3.8 scout used two same-model readers
on 12 previously unrated discovery families. All prompts contained only the
original problem and native reasoning already emitted through the triggering
sentence. The output parser accepted only an exact one-field boolean JSON
object after the reasoning delimiter. Job 59180589 completed with 24/24
assigned ratings, 23/24 parsed natural stops, and 8/8 long-prefix ratings
parsed. Cold load took 479.4 seconds. The 24 ratings consumed 72,524 prompt
tokens and generated 4,244 tokens in 70.94 generation seconds. The effective
59.82 generated tokens/second includes prefill time and is a batch-specific
measurement, not a general decode speed.

Six sampled candidate windows fired under v2.2. Among five pairs whose two
readers returned valid starts, one was jointly positive, three jointly
negative, and one disagreed; one more pair was unresolved because a reader
did not return parseable JSON. All six sampled nonfires were jointly
negative. This tiny stratified sample warns that v2.2 fires can be ordinary
algebra, already checked conclusions, problem restatements, or plans to
calculate. It cannot estimate population precision. The native emitted
prefix is necessary for the rating: local current-sentence wording alone
cannot tell whether a candidate was previously checked.

The complete-stage price now uses exact 1,518,938 prompt tokens and at most
761,856 output tokens for all 744 assigned ratings. It uses the lower of
historical stressed Qwen speed and 0.8 times the scout's context-matched
effective speed (43.71 tokens/second), plus two cold loads, prefill,
shutdown and repeat work. The sealed price projects 13.20 GPU-hours within
a 16 GPU-hour ceiling; scout parse qualification passed. CPU price job
59183247 completed 0:0. The first 2×A100, six-hour slice is Slurm job
59186342; it was pending for priority at this note's writing. Its output
directory is bound to the frozen frame, code, rubric and price hashes; each
of the 744 assignments has a separate pre-generation attempt receipt.
One two-hour same-manifest recovery is budgeted if the first slice cannot
finish. No rating result is claimed until the job, exit code and artifacts
are verified.

## Veto feasibility findings

A narrow explicit-proposal lexical gate retained only 7 of v2.2's 1,816
native candidate fires and reached only 4 of 48 discovery families. On the
five resolved fresh scout fires it retained the sole jointly accepted start,
but this sample is far too small and the full native burden fails enrollment.
That gate is an exploratory negative control, not a controller.

A family-held-out char-gram/logistic baseline used only the original problem,
previous completed sentence and current completed sentence. On 70 resolved
old rated v2.2 fires, 44 were jointly positive under the *old* Qwen prompt;
seven more had reader disagreement. Nested family folds found no threshold
that reached 70% apparent precision with at least four inner-fold fires,
so the policy abstained in every outer fold. This result is diagnostic only:
the old readers also saw the later sentence, and those labels cannot qualify
an online prefix-only veto.

The frozen full-context 372-window audit is needed to measure candidate
precision, sampled misses, family coverage and timing without that prompt
leak. It retains the registered strata and every assigned window, including
nonfires, failures and unparsed responses. A planned CPU analysis fits a
candidate veto only on discovery families, chooses thresholds within
training families, and evaluates on held-out discovery families. It must
report all-family trigger burden from the 1,816 native fires separately from
held-out rated precision. Approach and failed-check starts remain unsupported
if their rated support is insufficient.

If the cheap text model still cannot qualify, a bounded semantic veto can
ask the native generation model to judge the already emitted prefix through
the trigger as a separate, unsteered request in the same allocated engine.
It must use the same explicit input allowlist, strict JSON/abstain behavior,
batch isolation, per-family trigger accounting and full side-request token
costs. Its rubric and threshold would be frozen on discovery before
family-disjoint evaluation. This is a proposed experiment, not an achieved
detector.

## Different-model-family audit

Two Qwen draws are correlated LLM audits. The cached GPT-OSS 20B snapshot
has complete local mxfp4 weight shards, tokenizer and chat template; Gemma 4
has tokenizer/config but no local weights. A one-A100, four-prefix GPT-OSS
parity pilot is prepared with a 512-token response cap, strict JSON parser,
model-cache guard, separate output binding and a one GPU-hour wall ceiling.
It will enroll two valid Qwen-reader disagreements and two agreed controls
from distinct discovery families after the complete 372-window audit. CPU
tokenization of all 372 potential prefixes puts any selected four at no
more than 32,712 total GPT-OSS prompt tokens and 8,690 context tokens per
request including the response cap. The selected frame and exact price are
still pending. A different model family checks reproducibility of the rubric;
it is not independent human truth.
