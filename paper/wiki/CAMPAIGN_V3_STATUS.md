# Campaign v3 observed evidence

Observed protected launch — 2026-10-10: **research-campaign-v3-20261009-r2**.
[Launch and resources](../../report/research-campaign-v3-20261009-r2/LAUNCH.json),
[ownership transfer](../../report/research-campaign-v3-20261009-r2/OWNERSHIP_TRANSFER.json),
and [69 passing contracts](../../report/research-campaign-v3-20261009-r2/READINESS.json)
are durable. Source resolution was observed RUNNING; GPU qualification and
scientific effects are pending, with ordinary continuation owned by Slurm.

| Owner | Job | Verified request | Dependency |
|---|---|---|---|
| Transitive source import | 59900682 | 4 CPUs, 24 GiB, 180 min | None |
| October 15 snapshot | 59900687 | 4 CPUs, 24 GiB, 180 min | BeginTime October 15 21:59:59 UTC |
| Controller | 59900703 | 2 CPUs, 4 GiB, 60 min | afterany:59900682 |
| First failure owner | 59900706 | 2 CPUs, 4 GiB, 60 min | afternotok:59900703 |
| Terminal failure owner | 59900709 | 4 CPUs, 24 GiB, 180 min | afternotok:59900706 and 59900703 |

Both failure owners were accepted and verified before release. V2 owners
59783334/59794746/59794749/59794753 were retired only after pending,
never-started ownership checks; fresh accounting proves CANCELLED, zero
elapsed time and zero allocated CPUs for all four. No scientific job was cancelled.
Later IDs belong to immutable protected handoffs, not another dispatcher.

The requested r1 root preserves an aborted prelaunch attempt: held source job
59900049 was rejected by an integer-only parser of Slurm's `NumNodes=1-1`.
It was cancelled with zero time/CPUs before ownership transfer. Its source/runtime
closure (794 bindings), intent, exception and accounting are preserved in
[prelaunch provenance](../../report/research-campaign-v3-20261009-r1/PRELAUNCH_ABORT.json).
The corrected verifier accepts only exact one-node geometry, and r2 supplies
the fresh seal required after repair. The public entry point remains
`scripts/research_campaign_v3/run.py`; use `--campaign report/research-campaign-v3-20261009-r2`.

All seven models, fixed enrollment, the original 42,050 local-hour ceiling,
October 15 snapshot and terminal export remain required. The pre-production
decision and final scientific decision are separate. Historical findings,
source seals and failed receipts remain preserved.
Final scientific outputs will be under `report/research-campaign-v3-20261009-r2/output/`;
the milestone package under `snapshots/2026-10-15/`.
