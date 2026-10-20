# Deterministic whole-reasoning batching

Continue the same Alignment v3 experiment and the parent batching correction.
CPU61580365 completed0:0 after149seconds and passed190tests on source3a848b4.
No GPU was submitted from that source. The final code audit replaced atomic
floating grouped sums with fixed-order contiguous segment reductions, retaining
the original strict equality of tensor-parallel capture hashes. Single-row
decode needs no grouped reduction. Seven focused tests cover full-graph
tracing, row-wise operator and counter equivalence, complete audit captures,
prompt/reasoning scope and recomputation.

All scientific settings, rows, seeds, nomination/control rules, doses and gates
remain fixed. This source needs allocated qualification before GPU release.
The cumulative prior is0.842222 new GPU-hours and1.637778 CPU core-hours,
including every validation, canceled engineering allocation and conservative
local preparation charge. Retained rescue0.346111GPU-hours is separate.

The budget-bound shard admission, independent afterany extraction/analysis,
relative canary gates and purged-dependency reconciliation remain as documented
in the parent README. The original165/240-minute shard values are upper limits;
actual admitted maxima include all predecessor costs and stay within72GPU-h.
No manuscript changes or remote pushes are included.
