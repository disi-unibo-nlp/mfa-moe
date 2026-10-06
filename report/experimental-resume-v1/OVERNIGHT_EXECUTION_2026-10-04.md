# Parallel routing execution — 2026-10-04

The user authorized implementation, the required compute, and comparison of all
three approaches on the fresh 220-family extension. Each stage still requires
complete pricing, qualified execution, immutable artifacts, and verification by
Slurm job ID and saved outputs.

## Frozen experiments

The following designs were frozen before the completing mechanism outcomes were
read. Fresh outcomes cannot choose the candidate methods or enrollment.

| Group | Question tested | Size and horizon |
|---|---|---|
| A | Do the selected Verify experts help separately or as a pair? | 8 starts × 8 arms × 2 seeds; 1,024 tokens |
| B | Does a second pulse improve the effect of the same action? | 13 starts × 6 arms × 2 seeds; 1,024 tokens |
| C | Does logit bias, forced inclusion, or selected-weight reweighting work best? | 13 starts × 8 arms × 2 seeds; 256 tokens |
| Fresh comparison | Compare all methods with shared native and matched random controls. | First accepted native start per transition and family from the 220-family pool; 14 Verify / 10 Approach arms; 1,024 tokens |

Designs are `OVERNIGHT_DISCOVERY_{A,B,C}_DESIGN_v1.json` and
`OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json`. The fresh protocol freezes 28
semantic contrasts and deterministic enrollment. Strict-veto results are a
sensitivity; they do not select replacement starts. Discovery groups remain
exploratory. Repeated pulses test exposure, not ordering of different actions.

## Submitted execution chains

Status observations below were verified on October 4 at 23:07 CEST.
Successful submission does not establish successful execution.

| Stage | Job IDs | Observation / next dependency |
|---|---|---|
| Existing semantic readers | 59342858_0–7 and 59342983_8 | All completed; original failures preserved. |
| Existing semantic analysis | 59343015 | COMPLETED, 0:0; 760 assignments, 95 starts, 78 families. |
| Extension token preflight | 59344064 | COMPLETED, 0:0; 1,516 exact prompts for 758 candidate starts. |
| Extension reader qualification | 59344077 | COMPLETED, 0:0; 12 fixed ratings passed. |
| Extension eligibility readers | 59344192_0–23 | First 16 shards completed, remaining shards running at the observation. |
| Extension reader seal | 59344202 | After all eligibility-reader shards. |
| Fresh enrollment | 59344223 | After the complete reader seal; accounts for the entire parent pool. |
| A/B CPU preflight | 59344179 | COMPLETED, 0:0; exact requests, shards, and price envelopes. |
| A generation | 59344807_0–3 | First shard running; remaining shards awaiting allocation. |
| B generation | 59344808_0–6 | All seven shards running. First worker passed the repaired startup check. |
| A semantic chain | 59344822 → 59344824 | CPU frame/price, then automatic full reader array and ITT analysis. |
| B semantic chain | 59344826 → 59344828 | CPU frame/price, then automatic full reader array and ITT analysis. |
| Operator qualification | 59344841 | Awaiting allocation; checks force, reweight, isolation, pulse boundaries, closure and recompute. |
| C preparation and launch | 59345112 → 59345508 | After operator qualification passes; automatic generation and semantic chain. |
| Fresh preparation and launch | 59345113 → 59345513 | After qualification and enrollment; automatic complete priced comparison. |

A's generation plus semantic reserve is 23.3929 GPU-hours; B's is 31.0006.
Extension eligibility has a 63.371711 GPU-hour complete-stage projection.
These are projections, not measured expenditure. Semantic grading is repriced
using exact generated prompts before submission. Dense sentence labeling has
its own complete-stage price. All model loads, prefill, decoding, and reserved
recovery work are included. Actual expenditure must come from final Slurm
accounting, including failed attempts.

## Automatic secondary measurements and paper outputs

| Chain | Jobs | Result |
|---|---|---|
| A dense trajectory chain | Attach 59345815; preparation 59345851 → dispatch 59345855 | Attach completed 0:0; preparation waits for the semantic frame/price job to seal generation. |
| B dense trajectory chain | Attach 59345817; preparation 59345850 → dispatch 59345854 | Attach completed 0:0; same generation-seal dependency. |
| C dense attachment | 59345819 | After C generation dispatch; derives exact child dependencies from receipts. |
| Fresh dense attachment | 59345821 | After fresh generation dispatch; derives exact child dependencies from receipts. |
| Completed-result paper addendum | 59345662 | COMPLETED, 0:0; 12 new claim records, TSV table, PDF/SVG effect figure, reader-coverage figure. |
| A / B paper chains | 59345664 / 59345666 | Submit report generation after their exact semantic-analysis jobs. |
| C / fresh paper chains | 59345668 / 59345670 | Attach through generation and reader receipts, then build after analysis. |

Dense measurement is frozen in `GENERATED_DENSE_MEASUREMENT_PLAN_v1.json`.
It retains missing labels as gaps and reports seven-class transitions, dwell,
re-entry, loops, expert turnover, and selection-frequency motion. Full gate
weights were not saved in the route arrays; aggregate intervention telemetry
is separate. These descriptive class measurements do not replace substantive
semantic outcomes or establish an optimal trajectory.

Submission receipts are retained under `overnight-*-submissions-*` and
`generated-dense-chain-*`. The verified first paper package is
`routing-paper-v3-legacy-104ee9fbd9387320/SUMMARY.json`.

## Preserved startup failure and recovery

Original arrays 59344221 (A) and 59344222 (B) failed or were canceled before any
completed generation batch. Their directories contained bindings and policy
tables only. The runner had imported the repository package before pinning the
qualified worker overlay. The recovery entry point pins that overlay before
all runner imports and records source hashes and exceptions. Exact CPU startup
paths passed for both groups before relaunch; the first running B worker has
also passed the check. Sealed manifests, policies, prefixes and UIDs are
unchanged. Failed attempts remain in accounting.

Original postprocessing chains 59344411 → 59344413 and 59344415 → 59344417 were
superseded by new, recorded dependencies. Submission helpers write durable
attempt and receipt records; an unresolved attempt stops for reconciliation
instead of duplicating a submission.

## Current evidence and remaining work

The previous 1,024-token stage is exploratory because its registered
128-family feasibility criterion failed. Target bias minus native semantic
completion is −2.105 percentage points with a simultaneous 95% interval of
[−10.417, +5.978]. Target minus matched random is +0.526 points
[−8.211, +8.560]. These results are inconclusive.

Of 760 assignments, 128 lack two valid stopped reader votes. There are 179
individual ratings capped at 1,024 tokens and one further stopped rating that
could not be parsed. A separate larger-reader-cap sensitivity is being
prepared. It selects ratings by this measurement failure rule across every
arm, preserves original prompts and seeds, and does not replace the original
primary analysis.

The later 16k original-prompt utility controller uses the first semantically
accepted episode, with no re-entry. Its selected template has one or two
pulses. Selection follows a frozen rule applied to the fresh comparison;
engineering qualification and runtime pricing are separate. No 16k utility
run, accuracy benefit, token saving, or accuracy–token confidence region is
available yet.

The semantic readers are arm-blind LLM audits. Historical confirm exposure,
K1's matched-population scope, K2's uncertainty, G2's non-rejection reading, and
G3's dose-selection role remain disclosed. Completed observational analyses
and X2 are reused. The failed serial/eager NLL qualification is preserved and
is not rerun unchanged. Legacy X3 remains lower priority than the new steering
comparison and requires its existing qualification and full-stage gates.
