# F-0010: Simple live-only repo with one script per paper experiment
- status: active
- date: 2026-10-23
- request: "We need to finish the work of repo polishing. [...] only bash in scripts, and py in src. Everuthing separated in sufolders by topic. Each with a specific README. But with simple human terms. Also the code. Very simple and functional. Like for isntaince, to launch a vllm serve, no need to do infiiinte if-else, args etc. Just a simple script that run vllm serve model-name gen-configs its fine. If the file can be 6 rows then we can do that. We need to be on point with the experiments. So we have a clear script for each experiment." Amended: "envision also the use of the simplify and review skills". Answers 2026-10-23: remove old code (keep a git tag); full simplify with same numbers; port external code into the repo; branch now, merge after r18.
- purpose: A person can read the repo, find the script behind every paper result, and rerun it, without wading through 250k lines of finished campaigns.
- acceptance: (1) find scripts -name '*.py' and find src -name '*.sh' are empty; (2) every paper experiment has one script listed in the top README; (3) rerunning every CPU analysis and the paper exporters with the new code gives outputs identical to the golden baseline from commit 978f011 (mfa-moe/r/polish-golden/, provenance keys ignored); (4) all tests pass; (5) /simplify and /code-review high ran on every step with confirmed findings fixed.
- keywords: cleanup, layout, scripts, readme, simplify, legacy, archive, vllm serve, env.sh, release.sh
- paths: scripts, src/moe_exp, tests
- decisions: 

## Tasks

- [x] F-0010.1 Tag the last commit with all historical code and branch the work; evidence: tag archive/pre-cleanup-20261023 -> 978f011; branch polish/simple-repo
- [x] F-0010.2 Golden baseline: rerun every CPU analysis and the paper exporters with the current code; record pre-existing differences against sealed receipts; comparator script; evidence: mfa-moe/r/polish-golden/GOLDEN.md and GOLDEN.json (11 jobs, 27.2 core-hours; 7-model r3d readout/anticipation not rerun: boost CPU queue and 84 core-hours)
- [x] F-0010.3 Bring the external code that produced paper numbers into the repo unchanged (router capture, readout helpers dp/dynrt/rkin, benchmark table builder, release sealers); evidence: 1b2c8ab
- [x] F-0010.4 Copy the few functions live code takes from old campaigns into live modules, then remove old campaign code and dead code from the tree; evidence: a1a7438, 317f743, 9bf27e3; full suite job 61960382 334 passed (3 errors also on the tag)
- [x] F-0010.5 Reorganize live code into topic packages (generation, labeling, capture, routing, steering, paper) with matching scripts/ folders and tests/; evidence: e6b924e..66f0741, merge 12854a8 (F-0011 next_state), review fixes 477fad3
- [ ] F-0010.6 Bash: one shared Slurm env file, a ~6-line vLLM serve script per model and the judge, one script per paper experiment, one release script
- [ ] F-0010.7 Simplify the live Python topic by topic (drop unused designs, move Slurm plumbing out of Python), each topic identical on the golden comparison, /simplify and /code-review high applied
- [ ] F-0010.8 Plain-language README per folder and a top README mapping each paper experiment to its script and receipt; update CLAUDE.md, AGENTS.md, RUNBOOK, memory; retrace stale walkthroughs
- [ ] F-0010.9 End-to-end check: full golden rerun identical, paper export and build unchanged, all tests pass, final /code-review high
- [ ] F-0010.10 Merge into analysis/routing-dynamics after the r18/r20 exports have run; rerun the checks on the merged tree

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|

## Carry-over for F-0010.7 (from the code review of steps 2-4)

- Ported capture cannot run: `capture/bundle.py` refuses modules outside a frozen bundle. Replace with the release snapshot.
- `routing/readout.py` and `routing/anticipation.py` hash `OUT/code/{dp,dynrt,rkin}` as their code provenance but now run `moe_exp.routing.*`; make provenance name the code that ran, keep the frozen-input checks.
- `moe_exp.receipts` duplicates `moe_exp.io` (require, digest, load, publish, file_sha, check_files) with different normalisation; merge into io, keeping the campaign variant under an explicit name.
- `steering/pulse_effects.py` still writes 21 always-unknown `semantic_first_*` columns of the deleted rescue endpoint.
- `steering/pulse.code_hashes()` now hashes all of src; hash the code that ran or reuse `provenance.code_provenance()`.
- Lexical `readout-v1/fold-*.pkl` pickles name `moe_exp.study.readout_text.TextTransform`; `routing.lexical._Unpickler` maps only `__main__`, so loading them fails (before and after the flatten). Map the old module name in `find_class`.
- Launchers still to rework with the Python: the three steering seal-and-launch scripts (one shared block, Python builds the Slurm graph), one release mechanism instead of four, `VLLM_VERSION_RECORD` and the job deadline read in Python, hardcoded `tools/tmp` input paths.

## Layout change 2026-10-23 (user)

"If there's only one subfolder, do not create it [...] If there's only moe_exp, directly put moe_exp contents inside src. And remember the README inside each folder." Done in a6d44ad: packages sit directly in `src/`, `moe_exp/io.py` is `src/store.py`; full suite job 61971484 336 passed, same 3 pre-existing errors.

## Ports pending from analysis/routing-dynamics (peer session repo-a1, F-0012)

From 54591f8 on: AGENTS.md entry and F-0012 record; a new `study/` module for the Fig. 6 next-sentence labels and the Qwen3.6 sentence-table rebuild; a top-up path for r18's unfinished rows; strong-dose roster edits (8 models) in `scripts/paper/export_routing_results.py` and `verify_cl_tables.py`. Port into the new layout at merge (study/* -> routing/ or steering/, scripts/paper/*.py -> src/paper/).

## Golden finding to raise with the user (pre-existing, not a cleanup change)

Rerunning pulse outcomes with the tag code differs from the stored outcome tables used by `whole-reasoning-20261021-r7/corrected-main`: the current extractor takes a `reasoning_fallback` path for 2 GPT target_deactivate continuations (reasoning tokens 1348 vs 10087, 583 vs 4484; 38 GPT sentence rows fewer; qwen35 -1, lfm25 -3 rows), so 999 GPT/pooled values of the main effects differ from that receipt. The cleanup keeps today's behaviour; whether the receipt or the extractor is right is a science question.
- Peer session outcome-steering-r2: branch outcome/r2 (from 54591f8, old layout) changes outcome_steering/*, scripts/outcome_steering/*, tests/outcome_steering/* and study/{stream,driver,whole}.py; records D-0006, F-0013, F-0014. Port into steering/outcome, steering/{stream,pulse,whole} at merge.
