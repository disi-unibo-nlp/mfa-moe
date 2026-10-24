# src/steering

Code that pushes the routers of the eight models toward or away from chosen experts while a model reasons, and measures what that did to the length and the correctness of the reasoning. Launchers are in [`scripts/steering`](../../scripts/steering/README.md). Python decides what runs; the scripts decide where and in which order.

There are three experiments:

- Pulsed steering (`pulsed_steering.py`, `pulsed_outcomes.py`, `pulsed_effects.py`): a short push on the experts of one reasoning state at the start of a continuation. Activating a state forces its nominated experts into the selection; deactivating removes them. It runs at a central strength and, for four models, a stronger one. Each edit is compared with matched random experts at the same layers and with native routing. Its tables are the main and strong steering results of the paper (figure `routing`).
- Whole-reasoning steering (`whole_reasoning.py`, `whole_reasoning_effects.py`, `release.py`, `plan_continuation.py`): the push is applied at every prompt and reasoning token until the reasoning closes, and the final answer is left to native routing. States are promoted or suppressed according to whether they precede shorter or longer remaining reasoning (the efficiency link), at quarter, half and full strength, with matched random experts as controls. The sealed result is `EFFECTS_WHOLE.json` (figures `whole_states`, `whole_dose`).
- Outcome-selected nudge (`outcome/`): not in the paper, the active next line. Two campaigns: r1 (experts picked by outcome on cohort B) and r2 (native attempts captured on all eight models, experts picked inside each reasoning class). See [its README](outcome/README.md).

The same routing operators serve all three.

## Files

| File | What it does |
| --- | --- |
| `pulsed_steering.py` | The pulsed design: its manifests, one engine per model shard that admits requests while others finish, and the judge engine (`annotate_rows`) that `routing.next_sentence_labels` also uses. The engine, the stores and the `--model` handling are shared with the whole-reasoning code. |
| `lfm25_generation.py` | The GPU stages of LFM2.5 in the steering engine: its native traces (`native`), their judge labels (`annotate`) and the router replay (`replay`). |
| `pulsed_outcomes.py` | Reads the saved continuations and writes one table of outcomes and one of sentences per model. |
| `pulsed_effects.py` | The paired effects with bounds for unknown outcomes, the bootstrap, and the efficiency-link test. |
| `whole_reasoning.py` | The whole-reasoning release: manifests, the fluency canary, the shards, `plan` (JSON for the launcher), `budget`, `validate`, `prepare`, `run`. |
| `whole_reasoning_effects.py` | Replays and scores the finished rows (`extract`) and joins every release into `EFFECTS_WHOLE.json` (`combine`). |
| `release.py` | Seals a complement, relink, outcome, capture, sweep or continuation release: receipts plus a frozen copy of the code. A capture or sweep release records its job-name prefix (`oc`, `os`) and keeps its stores on FAST. |
| `plan_continuation.py` | Sizes the continuation of a complement release (read-only): its rows by saved state, its shard jobs by outcome, and the shards to run under a GPU-hour limit. |
| `stream.py`, `observer.py`, `operators.py` | The hooks that change the routing of the running model, and the operators: hard inclusion and exclusion, and the score override that follows SteerMoE. A row with `capture_mode: native_ids` also keeps the native top-k ids (uint8) and weights (float16) of every reasoning token. |
| `nominations.py`, `selection.py`, `allocation.py` | Which experts to edit for each state (the nominations), which layer of the network, and which question families go to which arm. |
| `runtime.py`, `session.py`, `penalties.py`, `fast_penalties.py` | Prompts and the durable per-request generation, one request lifecycle, and the repetition penalties applied to saved history. |

## Commands

A model `M` is a Hugging Face id (`openai/gpt-oss-20b`) or its short key (`gpt`); every module turns it into the short key (`models.roster.resolve`), and receipts and folders keep the short key. The study folder is the one `store.STUDY` names: `STUDY_DATA_ROOT`, default `report/final-routing-20261015-r1`.

```bash
python -m steering.pulsed_steering COMMAND [--model M] [--shard I/N] [--design main|strong]   # plan, plan-strong, run, check, status
python -m steering.pulsed_outcomes --model M [--design strong]
python -m steering.pulsed_effects --out DIR [--design strong] [--models M,M]                 # default models: those with a manifest in the study
python -m steering.lfm25_generation native|annotate|replay [--shard I/N]                      # LFM2.5 GPU stages
python -m steering.whole_reasoning COMMAND --release report/NAME [--model M] [--shard I/N]    # plan, budget, validate, prepare, canary, run, extract, combine
python -m steering.release KIND NAME --tests N [--models M,M] [--shards M:N,...] [--cutoff TIME]   # KIND: complement, relink, outcome, capture, sweep, continuation
python -m steering.plan_continuation report/PARENT [--max-gpu-hours H] [--shards M:N,...]     # plan of a continuation; seals nothing
```

GPU commands (`run`, `native`, `annotate`, `replay`, `canary`) refuse to start outside a Slurm GPU allocation. Use the launchers.

## Running a whole-reasoning release

A release is a folder `report/NAME`. Every stage takes `--release report/NAME` and reads its settings (models, strengths, shards, cut-off time, efficiency link) from the files in it.

```bash
scripts/steering/seal_release.sh complement NAME TESTS --cutoff 2026-10-24T06:00   # seal report/NAME
scripts/steering/submit_release_jobs.sh report/NAME                                # submit the jobs of a sealed release
DRY_RUN=1 scripts/steering/submit_release_jobs.sh report/NAME                      # print the sbatch commands only
```

The stages are `python -m steering.whole_reasoning plan | budget | validate | prepare | canary | run | extract |
combine`. Slurm is only in `scripts/steering/`: dependencies, resources and walltimes are in
`submit_release_jobs.sh`, which reads `plan` (JSON: the snapshot, the job prefix, the final command and the shards per
model). A shard asks for 4 hours of 2 GPUs. The release records its CPU and GPU ceilings, and
`submit_release_jobs.sh` submits nothing when the walltimes of its jobs add up to more (`budget`);
`scripts/slurm/billing_hours_used.sh whole-` adds up what was spent. `validate` runs the unit tests of the
snapshot and seals the result in `CPU_VALIDATION.json`; `prepare` and `combine` need it to have passed.
The fluency canary of a base release is run by hand (`python -m steering.whole_reasoning canary`), and its
receipt is what lets `run` start production.

A complement release whose shards could not all start or finish before its cutoff gets one continuation (`seal_release.sh continuation NAME TESTS --parent PARENT --cutoff TIME`; r21 continues r18). It runs the rows the parent never started and continues the ones it left PARTIAL, with the parent's UIDs, seeds and actions, after every parent shard job has ended (`steering.release` refuses while one is live). Print its plan first with `python -m steering.plan_continuation report/PARENT`: rows by state, the parent's shard jobs by outcome, and the shards chosen under 285 GPU-hours. A saved partial is copied into the continuation's own store and re-prefilled through the same hooks, and the parent's store is only read. The release names the eight-model efficiency link in `CONTINUATION.json`, extends the source list of `--chain` (default r20) with itself, and its combine joins the outcomes to the parent's rows by UID (`whole_reasoning_effects`). Pass the parent's extraction jobs, which the plan lists, to `submit_release_jobs.sh report/NAME JOB...` so the combine waits for them. The stores of a sweep, complement or continuation release are bound in `prepare`, under a lock, before any shard exists: shards of one model that start together used to fail with "nonempty unbound whole store".

Two rules apply to results: an outcome that is unknown (a continuation that reaches its allowance, or never finishes) keeps an interval and is never dropped, and nothing is redrawn after the fact. Token contrasts are `log(1+T_control) - log(1+T_treated)`, so a positive value means shorter reasoning than the control.

Tests: [`tests/steering`](../../tests/steering/README.md).
