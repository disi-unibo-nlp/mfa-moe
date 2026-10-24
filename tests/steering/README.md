# tests/steering

Tests of `src/steering`, with no model and no job: the routing operators and hooks (`test_science.py`, `test_stream_batch.py`), how a request is admitted, saved and resumed (`test_session.py`, `test_driver.py`, `test_whole_resume.py`), the pulsed design (`test_design.py`), the bounded effects with unknown outcomes (`test_effects.py`), the manifests, nominations and doses of the whole-reasoning releases with their frozen digests, the canary gate and the job graph that `scripts/steering/launch.sh` prints for a made-up release (`test_whole.py`, `test_whole_models.py`, `test_whole_launch.py`, `test_whole_complement.py`), release sealing (`test_release.py`), and the continuation of a complement release: its manifest, seeding, store binding and join to the parent (`test_whole_complement.py`), its plan and sealing (`test_followup.py`).

```bash
PYTHONPATH=src python -B -m unittest discover -s tests/steering -t tests/steering -v        # and tests/steering/outcome
```

The tests need the vLLM environment (torch, scipy, pandas). They cover the routing operators and
hooks, how a request is admitted, saved and resumed, the manifests and nominations of the
whole-reasoning releases (their frozen digests too), the canary gate, the bounded effects with
unknown outcomes, and the job graph that `scripts/steering/launch.sh` prints for a made-up release
(`DRY_RUN=1`). No test loads a model or submits a job. Every release runs them first in a Slurm job
(`python -m steering.whole validate`), which seals the result in `CPU_VALIDATION.json`.
In a fresh git worktree three tests error because they read sealed files from `report/` that git does not hold
(`test_driver.Selection` twice and `test_whole_launch.ActualNominationTests`). Set `TMPDIR` to a folder you own.
