# R3-D Qwen anticipation parallel shard, 2026-10-02

The canonical frozen R3-D job `59111360` is processing GPT cells. At 09:34
CEST it had 346/1,650 checkpointed tasks, including two complete GPT `next`
cells. Its observed 328 new tasks in 15.94 hours imply about 20.6 tasks/hour;
1,304 remaining tasks would take about 63 hours at that rate. Later cells may
change this estimate. Six guarded canonical continuation jobs are already
dependent on it.

The isolated Qwen shard's first submission was job `59172945`, with
16 CPUs, 128 GiB, zero GPUs and a 24-hour ceiling. It began immediately but
failed after 52 seconds before any fit checkpoint because the scheduling
wrapper imported the frozen module under a name that multiprocessing could
not pickle. The import has been corrected and picklability tested. This
failure spent about 0.23 CPU core-hours and did not touch canonical files.
The scheduler's test-only estimate of October 6 was provisional and did not
predict this immediate start.
The corrected first slice was submitted as job `59173234` at 09:53 CEST,
and began at 09:54:19 CEST on `lrdn2568`. At 10:03 CEST it was running
with 16 fit checkpoints in 8 minutes 45 seconds and no fit error. A sampled
checkpoint has the exact frozen binding, 492 held-out rows and eight model
predictions. The first-wave throughput is about 110 tasks/hour, implying
roughly seven hours for 750 tasks if sustained; allow 7–12 hours because
later response cells may differ. Live resource use reached about 16 busy
cores and a 31.3 GiB step peak, below the 128 GiB reservation.
The shard runs only the six Qwen response/containment cells (750 fit tasks),
loads the original frozen estimator at SHA256
`4f1e0e1aa57126eb3757a554560992314ffabbaf65310e007be40cb638f717ca`,
checks the original frozen specification and all recorded inputs, and uses
the same frozen family folds, seeds, model variants and noise controls. Its
scheduling source SHA256 is
`82d1bd619bd73dd738a9a887e0d2c98499f019176af99d833620221afd295900`.
It writes only under
`steering-v1/runs/resume-v1/r3d-anticipation-qwen-shard-v1`, never the
canonical checkpoint tree. If the canonical Qwen tree has already started
when the shard launches, it exits without fitting to avoid late duplicate
work.

The shard's first-slice maximum is 384 CPU core-hours. The measured rate
suggests roughly 9 hours on 16 cores if task scaling is close to linear;
startup, memory bandwidth and later-cell variance make that uncertain. A
conservative 2,500 seconds per fit task plus 20% overhead prices all 750
tasks at 625 CPU core-hours, at most two 24-hour slices (768 core-hours).
The previously bounded canonical chain and historical work total at most
704 CPU core-hours, so even two full shard slices would remain below the
1,500-core-hour R3-D stage ceiling (1,472 maximum). A second slice is not
submitted and needs a fresh progress/receipt check.

Consolidation is gated. Once a Qwen shard cell completes, verify its sealed
receipt, frozen driver/input hashes, 25-fold checkpoint count per model chunk,
prediction dimensions and finite values. Compare the copied cell-construction
logic against the frozen driver, including selected rows, group weights,
response labels and folds. Treat it as a separate analysis artifact until
that check passes. If all Qwen cells finish before canonical Qwen begins,
coordinate at a stopped canonical slice boundary: hold the exact dependent
continuation briefly, verify no writer is active, atomically copy the verified
checkpoint set into previously absent canonical Qwen cell directories, then
release that continuation. The unchanged frozen driver can then validate each
checkpoint binding and assemble its own final result. Do not merge during an
active canonical writer, and do not claim two independent Qwen replications
from the same copied fits. If the shard starts too late or overlaps canonical
Qwen, preserve isolated results and do not copy into canonical directories.

The fits are SciPy sparse softmax models with grouped out-of-fold class
controls; their present bottleneck is CPU optimization. Replacing them with
GPU solvers would require numerical-equivalence and tuning checks, so the
separate shard accelerates wall time without changing the estimator.
