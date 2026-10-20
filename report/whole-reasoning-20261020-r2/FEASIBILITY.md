# Whole experiment: pre-outcome feasibility

No whole canary or production output exists. The native queue is empty.

| Native job | Terminal state | Allocated CPU core-hours | GPU-hours |
|---|---|---:|---:|
| 61565142 and first release closeout jobs | FAILED / CANCELLED, preserved | 0.407778 | 0 |
| 61566298 | FAILED during control construction, before suite | 0.047778 | 0 |
| 61566300 / 61566301 / 61566302 | CANCELLED pending, no allocation | 0 | 0 |
| 61566702 | COMPLETED 0:0; Qwen preparation only | 0.086111 | 0 |

Cumulative new allocated CPU: **0.541667 core-hours**. The conservative local
preparation charge is 0.1 core-hours. Retain r3 engineering 0.346111 GPU-hours
and every older receipt separately; no budget reset or new rescue submission.

## Matching decision required

The exact full-dose GPT targets exist in all cells, but quintile controls are
impossible in all 35 promote fold/class cells. All 20 suppress cells are
feasible. At fold 0 / Read / layer 14, the top native-frequency quintile is
experts [9, 20, 3, 26, 13, 25]. Targets deactivate [3, 25, 20]; the current-cell
and opposite-polarity exclusions remove [20, 3, 13, 25], leaving [9, 26] for
three required distinct random experts. No sampler can supply that match.

The outcome-independent alternatives are:

| GPT frequency bins | Feasible full-dose promote cells |
|---:|---:|
| 5, requested quintiles | 0 / 35 |
| 4 | 9 / 35 |
| 3, tertiles | 35 / 35 |

The requested scientific rule requires the user's decision before GPT generation:
use tertiles for GPT controls with all other matching constraints retained, or
retain quintiles and explicitly authorize unavailable-cell drops. Half dose is
a fluency fallback only, so control feasibility has not selected it. Even the
prebuilt GPT half dose has three infeasible Explore fold/class cells.

See CONTROL_FEASIBILITY.json, CONTROL_MATCHING_ALTERNATIVES.json and
GPT_CONTROL_WALKTHROUGH.json. These use only discovery nominations.

## Qwen result

Qwen needs no matching amendment: all 50 cells qualify with exact counts, main
promote 15/0 and suppress 0/500; fallback promote 8/0 and suppress 0/250. Both
manifests have 2,838 planned, executable, unique UIDs and prompt scope enabled.
This is preparation evidence, not allocated suite or canary qualification.

The native-only planning fallback has log-reasoning SD 0.905930 and accuracy
SD 0.329707 over 128 observed correctness labels among 129 families. The
transported MDEs are 0.277564 log units and 0.101018 accuracy, marked SUSPECTED.
The claimed composite SDs were not reproduced; no composite covariance is
invented. See qwen-feasibility/SD_MDE.json.

## Correction readiness

The completed-output extraction worker now imports the replay helper in its
own scope. A real replay-layout fixture extracts seven reasoning tokens and
a correct answer despite a usage value of zero. Nine focused extraction,
bounded closeout and execution-gate checks passed in 41.043 seconds. The
current source correction is not a released or qualified execution binding.

Independent live-tree edits and staged release records were observed while
this work continued. They were preserved. No duplicate Slurm job was observed;
clarification about a second submitting session remains pending, so no GPU
work was admitted. All prior immutable materializations remain unchanged.
