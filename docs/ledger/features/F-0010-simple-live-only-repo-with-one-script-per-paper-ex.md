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
- [ ] F-0010.2 Golden baseline: rerun every CPU analysis and the paper exporters with the current code; record pre-existing differences against sealed receipts; comparator script
- [ ] F-0010.3 Bring the external code that produced paper numbers into the repo unchanged (router capture, readout helpers dp/dynrt/rkin, benchmark table builder, release sealers)
- [ ] F-0010.4 Copy the few functions live code takes from old campaigns into live modules, then remove old campaign code and dead code from the tree
- [ ] F-0010.5 Reorganize live code into topic packages (generation, labeling, capture, routing, steering, paper) with matching scripts/ folders and tests/
- [ ] F-0010.6 Bash: one shared Slurm env file, a ~6-line vLLM serve script per model and the judge, one script per paper experiment, one release script
- [ ] F-0010.7 Simplify the live Python topic by topic (drop unused designs, move Slurm plumbing out of Python), each topic identical on the golden comparison, /simplify and /code-review high applied
- [ ] F-0010.8 Plain-language README per folder and a top README mapping each paper experiment to its script and receipt; update CLAUDE.md, AGENTS.md, RUNBOOK, memory; retrace stale walkthroughs
- [ ] F-0010.9 End-to-end check: full golden rerun identical, paper export and build unchanged, all tests pass, final /code-review high
- [ ] F-0010.10 Merge into analysis/routing-dynamics after the r18/r20 exports have run; rerun the checks on the merged tree

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|
