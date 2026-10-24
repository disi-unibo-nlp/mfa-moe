# Dense discovery class-label launch, 2026-10-01

The prospective direct-judge operational gate in
`DENSE_JUDGE_QUALIFICATION_v1.md` passed: 199/200 parsed stop outputs
(99.5% coverage), 174/200 agreement across all assigned fixtures (87%), and
no historical class below 50% agreement. GPU job 59112590 completed, exit
0:0, in 19:13 on two A100 GPUs. Its direct vLLM prompt is a new LLM
measurement; it does not reproduce the historical DSPy prompt exactly and
cannot supply an online trigger or independently validate substantive checks.

`DENSE_DISCOVERY_LABEL_PRICE.json` binds the 5,659 contiguous discovery
sentences, parity output, and label driver. Observed generation was 413.45 s
for 200 items after 722.56 s of load/setup. The complete-stage price scales
all inputs by 28.295 with a twofold generation allowance, a second cold load,
and two 196-second shutdown allowances: **14.019 GPU-hours**, with a revised
**14.25 GPU-hour ceiling** under the user's expanded compute authorization.
The audit inputs averaged 505 characters across the four fields; full dense
inputs averaged 465. All 48 families and all selected sentences are retained.

Submit the first checkpointed slice for **at most six hours on two GPUs**
(12 GPU-hours maximum). This is shorter than the price driver's seven-hour
first-job suggestion so a bounded resume can fit within the 14.25 ceiling.
The source, unit seal and output directory are SHA bound. A time limit leaves
atomic batch receipts; resume only after verifying progress, exact job state,
remaining cost, and unchanged binding. Six hours is an allocation bound,
not an expected runtime. The study remains exploratory until independent
behavioral ratings and the prefix detector pass their own qualifications.
