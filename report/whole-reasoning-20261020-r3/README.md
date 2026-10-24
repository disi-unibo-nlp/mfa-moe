# Whole-reasoning Alignment v3

The user's October 20 Alignment v3 authorizes this session as the sole owner,
all corrections, one allocated CPU validation, both canaries, production and
fresh corrected analyses. It supersedes the conflicting v2 launch instructions.
Commits abbcffd, 385ea15 and 0c0243b, both r2 feasibility records, every earlier
allocation and all receipts remain historical. No whole GPU outcome existed
when these decisions were fixed. No manuscript edit or push is authorized.

Use control-feasible absolute-RD nominations with the original layer caps,
quintiles and role separation. Reserve distinct activation controls before
accepting deactivations. The greedy rule must reproduce all 70 saved GPT
promote DP selections exactly; every Qwen and GPT suppress target keeps its
exact rank. Each manifest cell records retained RD, replacements and mean
training-native frequency. No tertiles or efficacy-driven tuning are allowed.

Run 64 in flight, continuously. Tier 1 includes native, native_repeat, promote
Verify/Monitor and all suppress cells for every shard family. Tier 2 contains
the remaining promote rows. Each tier orders by family, arm and class, with
no family completion barrier. Deadline drains retain unfinished rows as unknown.

Canaries use the same eight frozen families and 48 rows (16 native/repeat and
32 steered), each with its production allowance. Software gates are unchanged.
Against same-run native/native_repeat references, fail if steered cap excess
exceeds0.15, geometric mean length ratio exceeds1.40, or mean16-gram repetition
excess exceeds0.10. Early gates use completed same-family pairs with both
native references observed. Only a fluency failure permits the frozen half
dose; its full allocation must fit the remaining charged budget and maxima.

| Stage | Count | GPUs | CPUs/job | Walltime/job | Maximum GPU-h | Maximum CPU-only core-h |
|---|---:|---:|---:|---:|---:|---:|
| CPU validation/prepare |1|0|4|30min|0|2|
| GPT main canary |1|2|16|45min|1.5|0|
| Qwen main canary |1|2|16|90min|3|0|
| GPT production |2|2|16|165min|11|0|
| Qwen production |7|2|16|240min|56|0|
| Monitor |1|0|1|16h|0|16|
| Extraction |2|0|8|90min|0|24|
| Analysis, including fresh main/strong |1|0|8|120min|0|16|

The base GPU reservation is71.5hours, below the72-hour new-work ceiling.
Retained rescue engineering0.346111GPU-hours is reported separately. Failures,
startup and all earlier whole CPU allocations remain charged. The requested
CPU walltimes total58core-hours if exhausted; they do not by themselves enforce
the16-hour actual-use allowance. The monitor reserves8core-hours for tail work
and exits without stopping GPUs when actual CPU usage reaches7.5hours. CPU
workers supervise cumulative native allocation time, stop only CPU work at
15.5hours and retain0.5hours for concurrent shutdown/accounting latency.
Extraction uses8separate tokenizers and8scoring helper processes. Detailed
allocation accounting, including conservative local-development charge, is
in RELEASE.json and ACCOUNTING.json; no historical budget is reset.

All shards are unheld afterok on the passing model canary. Each model's
extraction is queued with its production shards, afterany all its shards;
analysis is queued afterany both extractions. Passing canaries can queue their
own dependent graph under the same schedule lock, so monitor exit cannot
prevent their launch. The monitor's deadline exits normally. Transient sacct,
scontrol or quota errors log and retry after60seconds. Scientific STOP is
restricted to failed45/75-minute relative gates, a quarter-completion projection
above72GPU-hours, the existing storage reserve or October23 05:30Rome cutoff.
The other model's frozen projection uses its actual planned shard maxima.

The six primary contrasts, weights, signs, bounds, shared10,000-draw bootstrap
(seed20261015), max-|t|envelope and decision rule remain as frozen in D-0002.
Non-nominated cells are not_in_design and excluded from composites. Every
efficiency cell must be nominated. Extraction failures remain explicit in
EFFECTS.json and REPORT.md. Coverage failures bound only that unit. Reasoning
closed before a cap is observed; capped fixed-budget accuracy remains0.

Planning uses main-study native_repeat−native differences: sigma_w=SD(d)/sqrt2,
composite SD=sigma_w*sqrt(2/m). Accuracy uses driver outcomes, never legacy
trace is_correct. Transport is SUSPECTED and never gates execution.

Manipulation checks, by model and arm, report mean training-native frequency
of steered experts, changed-membership fraction, L1 and deactivated native
weight. Targets remain the larger frequency perturbation. Historical generic
disruption lengthened reasoning and cut accuracy, so the residual mismatch
works against an efficiency claim. The exploratory link uses the shared
10,000draws/20261015, differing from the preregistered2,000/20261017. Fresh main
analysis reports qualification of activate:log_total and every bounded row.

Execution uses only the archived committed release and fresh stores. This
document is an authorization/design record; validation, gate and native job
receipts determine observed completion.
