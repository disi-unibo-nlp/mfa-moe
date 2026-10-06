# Serial/eager 1,024-token engineering qualification readout

Job **59269372** was submitted by the root orchestrator from
`qualify_mechanism_engine_1024_v1.sbatch`. Its sealed manifest is
`0a027bf6d511055edd01d4aba2e582fb1180b4cb46b25680cc8fd3cb87453f68`;
the price is `bd965dc934770b08b6d842ca9320ee66de0797e5b7f8690d24154054e2b4722c`.
The 2-A100 allocation has a 60-minute/2.0 GPU-hour ceiling. No automatic
retry is included.

Before accepting any result, verify `sacct -j 59269372` reports the parent and
batch step **COMPLETED, exit 0:0**, and record elapsed time, AllocTRES and
actual GPU-hours. A `QUALIFICATION.json` written before a subsequent job
failure does not establish PASS. The output directory is
`S/runs/routing-control-v1/mechanism-engine-1024-qual-0a027bf6d511055e`,
where `S` is the user-owned `tools/tmp/claude-analysis-2026-09-24/steering-v1`.
Its `BINDING.json` must name the manifest/driver hashes and job ID. Verify
`routed.npz` and `raw-results.json` bytes against the hashes in the sealed
`QUALIFICATION.json`. The public copy is
`report/experimental-resume-v1/MECHANISM_ENGINE_1024_QUAL_RESULT_v1.json`;
it must match the raw result byte-for-byte or by full parsed content and seal.

PASS requires exactly 12 assigned request outcomes, 12 passing individual
checks, and all six flags true: `same_prefix_four_arm_pass`,
`native_isolation_pass`, `ordered_pulse_pass`,
`preemption_recompute_pass`, `closure_pass`, and
`inherited_h14_force_pass`. Inspect both GPU-rank telemetry records for every
request: zero inactive expert/weight mismatches; no active rows in either
native arm; correct +1 target/random dose in the 128-token and 2,048-token
prefix blocks; 256-token action windows at branch positions 0 and 512 in
both AB and BA orders; one successful deliberate reset and positive preemption
and recompute counts on both ranks; and zero action rows after a closed
reasoning prefix. Confirm token-array shape `(emitted_tokens, 40, 8)` and the
per-request cap of 1,024 (128 for the closed fixture). The two native repeats
need not have identical tokens or top-k routes; their variability is reported,
not interpreted as failed exact pairing.

If the job fails, is cancelled, times out, lacks either rank's telemetry, has
missing outputs, or any flag fails, do not use the qualification as a
mechanism-stage gate. Preserve `FAILURE.json`, logs and partial artifacts,
diagnose the concrete failure, then prepare a separately versioned and priced
recovery. Do not reuse this manifest's output directory or silently change the
thresholds. This is an **engineering qualification** on fixed fixtures. It
does not establish engine equivalence, semantic trajectory control, accuracy
benefit, or utility. The mechanism validation stage must independently bind
the PASS result seal and verify its own Slurm job and artifacts.
