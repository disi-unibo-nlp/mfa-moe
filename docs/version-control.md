# Version-controlled research artifacts

Commit authored source, regression tests, protocols, source seals, bounded audit
and launch receipts, manuscript source, plotting tables and paper figures.
Keep model weights, installed environments, captures, router banks, dense arrays,
Parquet tables, temporary extraction caches, databases, locks and console logs
in the existing private WORK locations. Ignore rules never delete these files.

The manuscript at `paper/6a510c6db0d9b08aa438c142/` is also an independent
Overleaf checkout. This repository mirrors its source and figure files as
ordinary files, so GitHub users can read them without access to nested Git
metadata. Commit and push manuscript edits to Overleaf as well as updating this
mirror. The compiled `main.pdf` stays local; source-bound figure PDFs are kept.

Campaign receipts bind exact local sources and runtime dependencies by hash.
Committing files does not change those bindings or qualify an experiment.
Historical project-owned source snapshots may be retained explicitly; installed
third-party runtime snapshots and generated scientific payloads stay local.
Live campaigns continue writing outputs independently of a Git snapshot.
