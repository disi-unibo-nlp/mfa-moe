# Early-cut candidate start audit v1

Rate only the original problem and reasoning actually emitted through the
current **fragment**. The fragment ends at an exact native-token boundary
after a fully closed mathematical expression. The rest of its original
sentence, any later reasoning and the answer are not visible and must not be
inferred.

`start=true` only if the visible fragment states a complete, concrete proposed
value or answer that has not already been substantively checked in the visible
prefix. A computed intermediate value may count. A mere equation from the
problem statement, setup relation, partial algebraic chain, unfinished
expression, check result or already checked conclusion does not count.

When ambiguous, answer `false`. Return exactly one JSON object with the
single boolean key `start`, and no explanation. This is a separate diagnostic
of candidate starting-condition timing. The original transition primary still
excludes the full triggering sentence and is unchanged.
