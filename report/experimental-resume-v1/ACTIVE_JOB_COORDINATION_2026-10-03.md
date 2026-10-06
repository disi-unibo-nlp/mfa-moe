# Routing-control job reconciliation, 2026-10-03

The two discovery-internal, same-prefix 256-token generation pilots have
completed successfully. Positive routing job **59216285** produced 156/156
assigned continuations across 13 discovery families, six arms and two seeds;
its sealed summary is
`eligible-micro-serial-v3-320ecca116c7ab63/SUMMARY.json` (seal
`aa604558ff42e873368406950d4de2616b868d8bcb86ef1c322a0d43016c4110`).
Deactivation job **59216601** produced the matching 156/156 continuations;
its sealed summary is
`eligible-deactivation-serial-v4-85b7f90ae7295d85/SUMMARY.json` (seal
`b920264bb1746a635f694801c7e311e2164931bd9a80c0028d6ba63b4b89a955`).
Both output directories are under
`/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/`.
Their arm-blind frames contain all 156 assigned rows each. The duplicate
positive-v4 submission **59216368** was cancelled before allocation, as
documented in the 2026-10-02 coordination record.

The paired first-stage audit is sealed in
`CAUSAL_ELIGIBLE_ROUTING_FIRST_STAGE_AUDIT_v1.json` (seal
`87aea9ad1b1dc57c8353a0844f366910e219dd5b356f2251969e3e3f66b12fec`).
It validates both tensor-parallel dose receipts and reports typed routing
effects. In the approach-to-commit stratum, target +1.0 changed selected
expert membership in 28.6% of active layer-token rows; matched random +1.0
changed 8.4%. Force-off eliminated target hits as designed. These are
descriptive first-stage findings; semantic and utility effects remain ungraded.

Price jobs **59267238** and **59267241** completed with exit 0. Their sealed
price records both say `PASS_COMPLETE_STAGE`, each for 312 independent-reader
ratings on two A100s, with a 4-hour Slurm limit and 8 GPU-hour ceiling. The
conservative projected completion is 10,153 seconds per job; this is a price
estimate, not observed runtime. Positive rating job **59267527** and
deactivation rating job **59267532** were verified running on separate GPU
nodes at 2026-10-03 16:51 CEST. Dependent CPU analysis jobs **59267665** and
**59267672** were verified pending on `afterok` dependencies, respectively.
Check `sacct` final state, exit code and sealed rating/analysis artifacts
before using any semantic result.

The CPU analysis driver now gates on a Slurm step ID rather than hostname:
the CINECA serial partition executes on a `login`-named node. Its positive
dose summary reports receipt coverage only; heterogeneous routing fields are
summarized by the separate typed first-stage audit. Nine focused analysis
tests passed after this edit.

Mechanism start preparation job **59267739** completed 0:0 in 3:16. It
reconstructed 454 exact native-prefix starts across the frozen 128-family
mechanism pool and sealed `MECHANISM_START_FRAME_v1.json` (seal
`7497e6aafbaf689cc5ae221e8564a97ed4e8adafd1e12a55b1887812a73a415d`).
The exact Qwen3.8 reader price is
`MECHANISM_START_READER_PRICE_v1.json` (seal
`9ea35133da6022e5a5dda261bb690b05dfa6275b7316b4379ca7b98731bd7a87`):
908 ratings, `PASS_COMPLETE_20_GPUH`, projected 15.894 GPU-hours. The v2
reader driver makes interrupted attempts explicit and commits one result per
UID and reader. Its exact-driver reprice is
`MECHANISM_START_READER_PRICE_v2.json` (seal
`497b2e8f57646826c3e27d3876b512c93f8ebaf14ef85dc64997a6b84f0ce4d6`),
also `PASS_COMPLETE_20_GPUH`. After binding validation and recovery tests,
mechanism reader job **59268130** was submitted and then verified running on
`lrdn0281` at 2026-10-03 17:09 CEST, on two A100s with a 10-hour limit. Its bound output is
`dense-mechanism/ratings-v2-7497e6aafbaf689c-497b2e8f57646826`.

The independent 1,024-token serial/eager engineering qualification was
sealed as `MECHANISM_ENGINE_1024_QUAL_MANIFEST_v1.json` (seal
`0a027bf6d511055edd01d4aba2e582fb1180b4cb46b25680cc8fd3cb87453f68`)
and `MECHANISM_ENGINE_1024_QUAL_PRICE_v1.json` (seal
`bd965dc934770b08b6d842ca9320ee66de0797e5b7f8690d24154054e2b4722c`).
It prices 12 engineering requests, 11,392 maximum decode tokens, 17,221
prefill tokens, two A100s for at most one hour (2 GPU-hours), with a
2,716-second modeled completion. Four focused tests and the CPU preflight
passed. Job **59269372** was submitted and verified pending for priority;
its eventual `MECHANISM_ENGINE_1024_QUAL_RESULT_v1.json` must pass all
checks before the longer mechanism generation can be sealed.

Deactivation rating job **59267532** completed 0:0 in 37:03 and sealed all
26 blind reader batches. Its first dependent analysis job **59267672** wrote
the sealed v1 analysis but exited 1:0 because the final status `print`
incorrectly passed `flush` to `json.dumps`. The print call was fixed and
nine regression tests passed. The corrected analysis was written to a fresh
v2 path by job **59269463**, which completed 0:0 in 0:08. The v2 analysis
seal is `6dd3c7123fe543fd10e8fcaa14f57c14bff2907cb6702870ac80f2a7e28aeacb`;
156/156 assignments are retained, 144/156 have two parsed reader stops.
Its exploratory negative-bias versus native semantic estimate is +0.077
(simultaneous interval [0, 0.231]); target force-off versus native is
−0.038 (interval [−0.231, 0.115]). These small discovery-internal cells do
not establish beneficial or necessary semantic steering.

Positive routing rating job **59267527** completed 0:0 in 42:53; its
dependent CPU analysis job **59267665** completed 0:0 in 0:05. The sealed
analysis `CAUSAL_ELIGIBLE_MICRO_SERIAL_SEMANTIC_ANALYSIS_v1.json` has seal
`6e556a38ddc6c54d16acd66038fed53b39f457c6cedb50eeafd118fdd50ff628`.
It retains 156/156 assignments, with 140/156 two-reader parsed stops.
In this discovery-internal screen, target +1.0 versus native is +0.115
(simultaneous interval [−0.115, 0.385]); versus matched random +1.0 it is
+0.038 (interval [−0.154, 0.308]). Target +0.5 versus native is −0.038
(interval [−0.154, 0]). The pilot has a strong measured routing first stage
but does not establish semantic control or utility. The independent
1,024-token mechanism stage is the next test.

The mechanism validation sealer and runner passed seven focused tests.
Live `scontrol show partition boost_usr_prod` reports `MaxTime=1-00:00:00`;
the sealer will price complete same-prefix shards against that 86,400-second
limit. CPU job **59269714** was submitted and verified pending on
`afterok:59268130:59269372`. It can run only after both the v2 start audit
and full-horizon engine qualification exit successfully, and it verifies
both public seals before writing a manifest and complete-stage price. It
does not submit generation jobs itself.

The v2 mechanism start audit **59268130** subsequently completed 0:0 in
47:02. Its sealed `SUMMARY.json` has digest
`9c851018ad07253765da6433debe29d001afa4062f62ad3b2e3a7021621de7c5`:
454/454 proposed starts, 908/908 reader assignments, 403 two-reader parsed
stops, 122 both-reader-positive starts and 58 committed physical reader
attempt blocks, with no uncommitted attempts. Applying the frozen first
accepted start per family and transition selects 95 starts across 78 of the
128 frozen mechanism families (66 candidate-to-verify, 29
approach-to-commit). This yields 760 assigned four-arm/two-seed continuations
and 778,240 maximum decode tokens. Thus the registered 128-family
eligibility target is infeasible in this pool; the manifest must label its
stage exploratory. These counts are pre-intervention eligibility, not a
semantic effect estimate.

An independent extension freeze now reserves the 220 previously unallocated
confirm-clean parent families in
`MECHANISM_EXTENSION_220_FAMILY_FREEZE_v1.json` (digest
`222f996be8f3a62b4ed71672ad776fa8649764bfc6d8f78da348abb0bc63b83c`).
Its CPU preparation and separate reader qualification are in progress; it
cannot retroactively turn the 78-family exploratory stage into the
registered validation. A separate strict prior-check veto is being specified
before extension intervention outcomes because an inspected Qwen start vote
showed a potentially already checked candidate. This is a measurement
sensitivity, not an outcome-based exclusion.

The CPU stage-completion wrapper
`scripts/experimental_resume/seal_mechanism_validation_v1.sbatch` has passed
`bash -n` and Slurm test-only. It will be submitted with dependencies on all
priced GPU generation shards after those jobs are launched; no completion
job was submitted at this point.

The 1,024-token engine qualification **59269372** completed 0:0 in 32:03.
Its public result `MECHANISM_ENGINE_1024_QUAL_RESULT_v1.json` is sealed with
digest `c77f8392b6041ee70a5b06ff47f477477856f04667aa25dd03b9dfe465063b20`.
All 12 case checks passed, including same-prefix native/target/random
isolation, AB/BA pulse order, preemption recomputation and reasoning closure.
Observed full-length decoding was about 11.4 tokens/s; the stage price still
uses the more conservative measured 8 tokens/s reference.

Dependent CPU sealer **59269714** started, then failed 1:0 in 17 seconds
before writing a manifest: its wrapper selected the GPU inference venv, which
does not contain `pandas`, for CPU Parquet trace-index reading. The already
working CPU preparation venv at `mfa-moe/envs/vllm-cu129` has pandas 3.0.5
and pyarrow 25.0.1. The wrapper alone was corrected to use that venv; pinned
driver hashes were unchanged. `bash -n` and Slurm test-only passed, and
recovered CPU sealer **59270959** was submitted and verified pending with
the intended serial resources. No mechanism generation submission preceded
its complete-stage price.

Recovered sealer **59270959** completed 0:0 in 0:37. It wrote
`MECHANISM_VALIDATION_MANIFEST_v1.json` (seal
`40ea298e178d08631d7b46c58c71ef1a19b2072ba087b29896df20ada6e59548`)
and `MECHANISM_VALIDATION_PRICE_v1.json` (seal
`c952ab8147d61fc493ecd576047117f5544e3f117f39287675e8c1debc676a37`).
The price is `PASS_COMPLETE_STAGE`: 95 accepted starts, 78 families,
760 assignments, 1,107,256 prefill and at most 778,240 decode tokens.
It budgets 69.793 two-A100 GPU-hours including 1.25 retry factor, three
cold loads and two complete row shards (rows 0–64 and 65–94), each under a
24-hour allocation. The manifest records
`registered_128_family_feasibility=FAIL` and stage label
`exploratory_accepted_family_subset`.

After `bash -n` and Slurm test-only, priced mechanism generation shards
**59271052** (index 0) and **59271059** (index 1) were submitted with two
A100s, 16 CPUs, 120 GiB and a 24-hour limit each. Both were verified
running on separate nodes (`lrdn0009`, `lrdn0800`); shard 0 printed
`PASS_CPU_STRUCTURE_PREFLIGHT`. Dependent serial CPU completion job
**59271086** was submitted and verified with exact
`afterok:59271052:59271059`; it will check every assigned UID, batch
receipt and route array before writing `STAGE_COMPLETION.json`. No semantic
effect is known from this generation startup.

Both first mechanism GPU shard attempts failed identically before any
assigned request or generation batch: **59271052** failed 3:0 in 3:28 on
`lrdn0009`, and **59271059** failed 3:0 in 3:29 on `lrdn0800`. Each passed
the CPU structure preflight and wrote only same-manifest `BINDING.json` and
`policy-table.json`; no continuation or outcome exists. Slurm automatically
canceled dependent completion job **59271086** with
`DependencyNeverSatisfied`. A logging-only real-Python wrapper was tested and
submitted as **59272045** for shard 0. It failed 3:0 in 3:25 but exposed the
previously swallowed exception: `ValueError: worker import resolves outside
the qualified overlay` at `prepare_worker_import_path`. The diagnostic
changed no sealed generation, base or worker source and produced no batch.
An import-order recovery is being prepared before any further generation
submission. These failed allocations count toward actual compute expenditure;
the original complete-stage price included one recovery load, so observed
recovery overhead must be reported separately.

The prospectively frozen 220-family extension CPU pipeline passed seven
focused tests and ran under serial job **59271965**, completed 0:0 in 6:01.
Its sealed detector selected 758 proposed starts across 218 families with a
fire, including 633 candidate-to-verify and 125 approach-to-commit starts.
`MECHANISM_EXTENSION_START_SELECTION_v1.json` has seal
`23758cee43febfc61a941024e2141554d3aef86bb91601e224b4ab61f48c00f6`;
the exact arm-blind frame has seal
`6a8b55ff69995bb7337bb97a52f6f9e6b056bb9b2de75bb64830be1e20ffc2d7`.
Its reader price `MECHANISM_EXTENSION_START_READER_PRICE_v1.json` has seal
`a7a9ea5d1e59cf763b913c54856810f71f9b8d6d7af9c2d1961b3eeb8319a299`:
3,032 total primary plus strict-veto ratings, 5,594,490 exact prompt tokens,
3,104,768 maximum decode tokens, and a conservative 51.443 GPU-hour
complete-stage projection. It remains
`HOLD_EXTENSION_READER_DRIVER_AND_GPU_QUALIFICATION`; no extension GPU reader
was submitted. The observed family count is a detector fire count, not a
two-reader accepted-start count or intervention effect.

The logging-only diagnostic **59272045** captured the exact suppressed
startup failure at `run_boundary_micro_screen.prepare_worker_import_path`:
`ValueError: worker import resolves outside the qualified overlay`. It
failed before model load and before any assignment. A versioned import-order
entry script now imports and verifies `moe_exp` and `ordered_vllm` from the
qualified immutable overlay before importing the unchanged sealed runner.
Its final digest is `a06b81c4a03184be6e13bfc6984e04a1854873efe95e2d8af4e634f21573149e`;
fresh-process CPU tests checked the package provenance after runner/world
imports. `bash -n` and Slurm test-only passed. Same-manifest recovery shards
**59272540** (index 0) and **59272547** (index 1) were submitted with two
A100s each and verified pending for priority. New serial completion job
**59272552** is verified pending on exact
`afterok:59272540:59272547`. No recovery success or generated batch is
claimed until logs and sealed receipts show it.

Both v2 recovery shards failed before model loading: **59272540** exited
1:0 after 0:15 and **59272547** exited 1:0 after 0:06. The traceback from
shard 0 identifies `ModuleNotFoundError: No module named
'moe_exp.routing_control.transitions_v2'` during manifest validation. The
qualified worker overlay predates that repository-only detector module. The
dependent completion job **59272552** was canceled; neither shard wrote a
generation batch. These allocations also count as actual compute.

The v3 recovery entry script keeps the qualified overlay first for
`moe_exp.routing_control.ordered_vllm`, while exposing repository-only
validation modules through package search paths. Its source digest is
`c90534682b4a9ac7a1ef5608785c4e8f2962ed8ccbe450a842ab4524dd398c5b`.
A fresh-process CPU check confirmed the worker still resolves to the
qualified overlay and the detector to `repo/src`; full sealed-manifest
validation passed for all 95 rows, ten action definitions and four arms.
Independent source review reached the same conclusion. `bash -n` and
Slurm test-only passed. New GPU shards **59272676** (index 0) and
**59272700** (index 1) were submitted and verified; shard 0 began running
on `lrdn1263`, while shard 1 was pending for priority at last check.
Dependent serial stage-completion job **59272726** was submitted and
verified with exact `afterok:59272676:59272700`. No semantic result or
batch receipt is claimed yet.

The blinded semantic-frame and exact two-reader pricing wrapper was checked
with `bash -n` and Slurm test-only after reducing its serial CPU allocation
to two CPUs and 16 GiB to fit live QoS memory. It is queued as **59272814**
with `afterok:59272726`; it cannot read or grade partial generation. At the
last check both v3 GPU shards were running (`lrdn1263`, `lrdn0574`) and had
passed CPU structure preflight, but only bindings and policy tables existed.
The 95 pretreatment reader-frame prefixes also passed the suffix invariant
used by the future blind-frame builder. No semantic effect is known yet.

The independent 220-family extension reader now has a separate sealed live
qualification manifest (`MECHANISM_EXTENSION_READER_QUAL_MANIFEST_v1.json`,
logical seal `beb73ba42fe5cbc14844751bd88355806345c8b7cbe0521efe57de7abbea856a`).
It fixes three starts spanning both supported transitions, two readers and
both primary/strict-veto channels: 12 capped Qwen3.8 ratings, priced at at
most 1.151 GPU-hours. The source, prompt contract and sealer hashes are
bound by `qualify_mechanism_extension_reader_v1.sbatch`. Four new focused
receipt/qualification tests and seven CPU extension tests passed; `bash -n`
and Slurm test-only passed. One-hour, two-A100 qualification **59273283**
was submitted and verified pending for priority. This is an engine/prompt-ID
qualification only; full extension reader shards remain held until live PASS
and exact price rebind.
