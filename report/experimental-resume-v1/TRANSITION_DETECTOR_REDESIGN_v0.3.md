# Discovery-only transition detector redesign, v0.3

The three registered transition hypotheses are unchanged. This note records
why the original online proposal rule underfired, what the broader rule
recovers, and what must qualify before adaptive action.

## Failure mechanism

V1 recognized complete boxed numeric forms and literal “answer is” numeric
forms. Native reasoning often expresses a proposed intermediate result as a
closed mathematical equality. For example, “Final check of calculation
$20^2/40$.” was followed by “$400/40 = 10$.”; both blinded Qwen readers
called the second sentence a candidate start, but v1 did not fire. Another
pair was “$264 / 24 = 11$.” followed by “$408 / 24 = 17$.” V1 also required
a method, intent, and operation in one sentence for approach proposals; a
reader-rated tentative branch such as “If $b = \sqrt{40}$,
$a = 20/\sqrt{40} = \sqrt{10}$.” did not match that conjunction.

The frozen audit's source span ends before a following newline. A streaming
event may be recognized one or two characters after the source text while
still using only emitted tokens. Exact replay on 5,523 contiguous discovery
pairs separated this bookkeeping effect from rule coverage:

| Rule and attribution | Candidate fires | Candidate families | Approach fires | Failed-check fires |
|---|---:|---:|---:|---:|
| v1, original source-span bound | 15 | 9 | 2–3* | 1 |
| v1, delimiter aware | 15 | 9 | 3 | 1 |
| v2.1, delimiter aware | 1,823 | 48 | 42 | 7 |
| v2.2, delimiter aware | 1,816 | 48 | 42 | 7 |

*Fresh incremental replay counted two strict approach events; the original
fresh-prefix frame and corrected delimiter-aware replay each counted three.
The candidate conclusion is unaffected. V2.1 recognized 395 candidate events
after the source span, mostly a one-character newline lag. Its candidate
fire burden was 15–74 per family, median 35.5; 1,227 occurred by 8,192
emitted tokens. V2.2 fixes a lookahead spillover that could attach an earlier
complete candidate to a later sentence such as “Perfect.” The exact v2.2
replay removed seven candidate fires. Historical receipts and code hashes
are preserved separately.

## Limits of the old audit

On the previously rated 214 resolved candidate windows, v2.2's
delimiter-aware screen has 44 jointly accepted starts, 33 rejected fires,
18 accepted nonfires, and 119 rejected nonfires. Four accepted fires had a
jointly accepted substantive check in the immediate next sentence; v1 had
zero. These comparisons are diagnostic only: the v2 rules were developed
after inspecting the same discovery examples, the 200 old nonfires per
hypothesis were selected to spread families rather than sampled uniformly,
and both readers were independent draws from the same model. The old prompt
also showed the later sentence while asking about the start, so start ratings
could be contaminated by future text.

The two jointly reader-rated old failed-check “misses” were ordinary math
statements (“Simplification leads to $x(x-5)=0$.” and three equal angles);
neither visibly computed a failed check. This directly demonstrates rating
error. No failed-check detector or transition is qualified by those votes.

The rule is a high-recall proposal screen. A formal start must be judged
separately, and a substantive target must evaluate an original constraint
or independent computation. Lexical verification words or routing speed alone
do not pass.

## Frozen next measurements

A new 372-window frame excludes all 619 previously rated windows, stays
within the same 48 discovery families, caps native emitted prefixes at 8,192
tokens, and preserves one to four contiguous later sentences that begin
within 256 tokens for a separate target audit. It contains 142 candidate
fires from 48 families and 96 candidate nonfires from 48, 35 approach fires
from 20 and 48 approach nonfires, and three failed-check fires from two and
48 failed-check nonfires. No cell receives replacement sampling if support
is sparse. The sealed source is
TRANSITION_V21_INDEPENDENT_AUDIT_FRAME.json; every sampled v2.1 candidate
fire also fires under v2.2.

The corrected start-only frame TRANSITION_V22_FULL_PREFIX_START_FRAME.json
binds exact native tokens and text through each trigger. Its reader input
allowlist is exactly original problem, already emitted reasoning prefix,
and triggering sentence. Later sentences, correctness, gold answers,
classes, and detector fire state are excluded. The driver independently
checks that the trigger ends the prefix. The two-draw Qwen audit remains an
LLM audit, not human truth. Fresh sampled windows do not constitute a new
family-disjoint validation population.

Exact tokenizer pricing for 744 ratings yields 1,518,938 prefill tokens,
median 889, p90 7,036, maximum 8,358 prompt tokens, and at most 761,856
decode tokens under the 1,024 cap. Prior Qwen ratings had median 270, p99
954 generated tokens; lowering the cap to 768 would truncate at least 28
successfully parsed old responses. A conservative 2× long-context stress
estimates 13.54 GPU-hours within a proposed 16-hour complete-stage ceiling.
The sealed price status is HOLD_CONTEXT_MATCHED_THROUGHPUT: the 12-prefix GPU
scout must measure long-context rate and recovery before the stage is
submitted. See TRANSITION_V22_FULL_PREFIX_START_RATING_PRICE.json.

Locally cached GPT-OSS 20B or Gemma 4 26B weights could provide a
different-family second reader. Either needs a small rubric/parsing
qualification and complete-stage price. Disagreements should be preserved
and independently adjudicated; a model-family switch does not create
human ground truth.

The 12 sealed scout prefixes are reader-agreed candidate starts from
distinct families, at most 7,549 emitted tokens; all still fire under
v2.2. Exact prompt and prefix token IDs, tokenizer hash, native trace hash,
and text hash are in CANDIDATE_PREFIX_SCOUT_v1.json. Same-prefix causal
micro-screening can proceed on that offline-enrolled set while online trigger
qualification continues. Any resulting effect is local to those prefixes
and cannot be presented as adaptive-controller utility.
