# Legacy X3 context-timing pilot proposal (2026-10-02)

CPU job `59182720` completed `0:0` in nine seconds and produced the frozen
manifest `steering-v1/runs/resume-v1/x3-timing-pilot-v1/x3-timing-pilot-v1-8f7c67d6b2a06e42.json`
(manifest seal `8f7c67d6b2a06e425c4f65d2f7c7fecca1b035a9f207500ca53900df890b8377`).
The earlier attempt `59181837` failed `1:0` in 12 seconds because it inherited
the original X3 builder's wrong World-path accessor. Its failure log remains;
the corrected code uses `world.lexicon_path` and `world.vocab_path`.

This is a **cost-only** pilot from the 96 registered dev-disc X3 questions,
not a change in X3 enrollment or an efficacy look. Four distinct families
were selected by a frozen rule from the 50 first-64 long-endpoint firing
questions: each saved native trace exceeded 32,768 tokens, none belonged to a
confirm-connected family, and selected prefixes span 4,251 to 12,671 tokens.
Each question receives N/E−/M−/E+/M+ at seeds 0/1, using the registered G3
cells and 32,768 minus prefix-length cap: 40 requests, 1,040,440 maximum
decode tokens and 274,110 maximum prefill tokens. There are no labels, grades,
native NLL measurements or semantic readouts in this pilot. The output
analysis may use only loads, elapsed time, prompt/decoded tokens, finish
reason and batch shape; generated content stays embargoed. Pilot requests
have a separate manifest name and are not added to X3's 96-question results.

The sealed `X3_TIMING_PILOT_PRICE_v1.json` (seal
`01273a36850d73dbdc52932334dbeca10a2ced440c5eba8892be91e69391ac7c`)
uses 25% derated X1 steady generation, a further 2× context slowdown,
2× slower Q10 prefill, 25% repeated work, two 838-second cold loads and a
196-second shutdown. Its maximum-workload projection is **6,080 seconds / 
3.378 two-A100 GPU-hours**. The receipt rounds to 3.5 GPU-hours, which
leaves only four minutes of walltime margin. A **two-hour, two-A100 job with
a 4.0 GPU-hour ceiling** is the safer proposed envelope, leaving roughly
19 minutes while still stopping at the cap. The separate sealed
`X3_TIMING_PILOT_PRICE_v2.json` (seal
`e0afcc80e0ae538c58014a214f6fb95c972cdc6c7d9216bcab4bc2d4b8dbbab0`)
records this amendment and retains a GPU hold until new-study priority and
the exact Slurm plan are resolved. This context slowdown is a stress assumption,
not a measured 32k throughput result. No GPU pilot has been submitted.

The pilot's role is to measure long-prefix prefill, later-context decode,
load/shutdown and actual early-stop distribution before building a complete
X3 stage price. A separate frozen GEPA-labeler throughput calibration and
grade/audit/NLL scope remain needed. The registered two-sign X3 maximum has
770 firing requests, 14,049,070 maximum decode tokens and 4,132,590 prefill
tokens; its historical generation-plus-prefill scenario alone is 13.104
GPU-hours, already above the original 12-hour ceiling. After the timing and
label calibrations, publish an all-in bound covering every load, prefill,
decode, label, grade, NLL, retry and shutdown. If it exceeds 12 GPU-hours,
seal a versioned X3 resource amendment and update the guarded builder before
any full X3 launch. The new causal routing-action study retains priority.
