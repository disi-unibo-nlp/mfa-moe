# Original-prompt utility production implementation

The production adapter preserves the frozen 96-family, 384-assignment, 16,384-token plan and reuses the qualified v2 generator/controller unchanged. All 208 scientific source hashes remain required. The eight pilot assignments retain their canonical UIDs and original receipt, route and attempt paths; they are never regenerated as new production observations.

## Execution and pricing

`utility_production_v1.py config` seals the new orchestration sources, scientific source closure, pilot proposal, v3 offline outcome plan and maximum-384 J1 envelope. The intended configuration uses a 16,000 billing-core-hour generation allowance, a separate 216 GPU-hour grading reservation, a 2× empirical stress factor, up to two complete families per shard and 32 concurrent four-GPU shards. These are allocation-management choices within the user's existing authorization. They are not a claimed runtime guarantee or a new cap on authorized research.

After the exact eight-cell pilot finishes, the CPU gate measures both joint cold loads and each request's generator/controller time, reader latency, query density and emitted tokens. Nonfire/veto queries remain charged. The stress forecast uses the largest observed generator seconds per token, query density extrapolated to 16k, and explicit reader output/context scaling. It is a forecast from two families, not a statistical upper confidence bound. Missing side-reader timings, incomplete pilot costs, unaccounted prior pilot allocations or an oversized complete family produce a concrete HOLD proposal.

The proposed array walltime is the complete-shard stress forecast rounded up to five minutes, with a one-hour floor and the configured maximum as its ceiling. Every family retains all four arm/seed cells on one shard. Initial requested GPU hours, stress-projected GPU hours and recovery reserve are reported separately. The aggregate recovery reserve covers `ceil(0.25 × initial shard count)` full allocations, in one wave. Missing work beyond that reserve produces an exact additional-work proposal; the dispatcher does not choose a favorable subset or silently submit a larger retry array. Committed errors, caps and nonfires are never rerun to improve outcomes.

Before GPU submission, the dispatcher re-reads `saldo`, live partition/QoS associations and the complete project's queued/running resource commitments. Compressed array ranges are counted correctly. For the four-GPU production nodes, 32 billing cores correspond to eight billing-core-hours per allocated GPU-hour. The 16,000-core-hour generation allowance and the separately reserved grading resources must fit the live account after queued commitments. Site accounting can lag, so the saved preflight is evidence of the observed balance rather than a promise about future project usage.

The separate J1 envelope prices up to 384 strict-rejected natural-stop items, three votes per item and at most three transport attempts per vote, with the frozen 8192-token output cap. Its 216 GPU-hour reservation is distinct from the smaller empirical historical-item forecast. The exact offline item/prompt price is still checked before grading. No side-reader or offline grading time is added twice to the four-GPU generation allocation.

## One-shot chain and outputs

The root orchestrator can submit `dispatch_utility_production_v1.sbatch` after the initial v4 pilot attachment with:

- `UTILITY_PRODUCTION_CONFIG`: the sealed production configuration path.
- `UTILITY_PILOT_ATTACHMENT`: the v4 pilot attachment directory.
- `UTILITY_PRODUCTION_PHASE=follow-pilot-generation`.

The finite chain follows saved pilot generation, analysis, selector and accounting receipts, then prepares the complete production price. It submits the GPU array only when the configuration enables dispatch and every gate passes. Each child submission uses sanitized placement variables, a durable attempted receipt, `sbatch --test-only`, a submitted job-ID receipt and Slurm readback. Completed predecessors are accepted only with successful accounting and the matching sealed artifact, avoiding dependencies on purged controller records.

The CPU receipt directory is `report/experimental-resume-v1/utility-production-submissions-v1-<config-digest16>`. Prepared price/manifest artifacts are under `utility-production-prepared-v1-<config-digest16>`. GPU artifacts are under the existing routing-control run root at `utility-production-v1-<manifest-digest16>/shard-NNN`.

`INITIAL_CHAIN.json` gives the initial array and accounting job IDs. An optional `RECOVERY_CHAIN.json` gives the single bounded infrastructure-recovery array and final accounting job. `FINAL_CHAIN.json` identifies the reconciled index and its seal. There is no resident polling process or unlimited retry loop.

Each shard uses a writer lock, immutable canonical assignments and an attempt journal. Route arrays and result receipts commit atomically. A later allocation can execute only the exact missing-receipt set in a sealed recovery manifest after reconciling prior attempts. Partial token histories are not resumed through an unqualified checkpoint mechanism: infrastructure recovery restarts the original prompt with the same assignment/seed, preserving every prior attempt and cost. Committed errors remain errors.

## Grading handoff and validation

The index schema is `utility-production-reconciled-index-v1`; it includes all 384 canonical assignments in their original order. Every row records `COMMITTED_GENERATION`, `GENERATION_ERROR` or `MISSING`, its origin, exact receipt path/seal, source manifest/binding, route path/seal and attempt provenance. Imported pilot artifacts remain at their original paths. Final accounting includes failed loads, killed tasks, all recovery allocations and the original pilot cost.

The unchanged offline outcome adapter grades only natural stops; caps and committed errors are operationally wrong, while missing executions remain explicit unknown endpoints. The production index matches that adapter's contract. Offline grading and final utility inference are attached separately through the frozen v3 measurement/J1 pipeline.

Eight focused no-submit tests cover canonical pilot reuse, complete shard partitioning, nonfire query costs, insufficient-runtime holds, committed-error preservation, exact missing-cell recovery, all-384 ITT reconciliation, failed/unallocated job accounting and compressed-array billing. Wrapper syntax and the qualified GPU entrypoint's import path were checked without loading a model. No Slurm submission was performed by the implementation agent.
