# F-0001: Audit steering results and prepare the reader correction
- status: abandoned
- date: 2026-10-20
- request: Check the status of the experiments. Check the lsat claude and deepseek-local-codex and code chats. Analyze all the things pending. Check what is missing to sbatch the run for the paper. We need the results from the stering operations. Knwing that the (de)Activate paper was able to change behvior od the model. So that is possible for sure. If we are not succeeding then it must be because we are measuring the wront thing. Also re-read the iclr submitted paper here in the repo we discussed many times about. Even if they do hiddens tate intervention, analyze the kind of xperiments and procedure they use which we can reuse. Deploy also subagents sol 6.1 xhigh to debate undependently and without bias to find possible bugs or issue. So that we can make the final code.
- purpose: Establish the actual experiment state, distinguish scientific evidence from pipeline failures, and repair demonstrated defects before another allocation.
- acceptance: Native Slurm accounting and saved artifacts reconcile; independent reviews cite code and papers; demonstrated corrections pass bounded regression checks; unfinished measurements and remaining budget are explicit.
- keywords: steering, rescue, reader, effects, routing, paper, measurement
- paths: src/moe_exp/study, tests/study
- decisions:

The user replaced remaining rescue work with Alignment v2 (F-0003) on 2026-10-20. Earlier evidence remains preserved.

## Tasks

- [x] F-0001.1 Reconcile live jobs, completed artifacts, failures and charged resources; evidence: report/final-routing-rescue-20261019-r1/OBSERVED_STATUS.json and GENERATION_PROVENANCE_AUDIT.json.
- [x] F-0001.2 Read recent relevant Claude, DeepSeek and Codex conversations and recover the governing instructions; evidence: report/final-routing-rescue-20261019-r1/README.md records verified thread IDs, providers, chronology and the approved rescue obligations.
- [x] F-0001.3 Independently review interventions, measurements and the two papers; exchange objections and resolve evidence-backed findings; evidence: report/final-routing-rescue-20261019-r1/README.md and AUDIT_VALIDATION.json; three GPT-6.1-sol xhigh reviewers.
- [x] F-0001.4 Preserve the completed generation source and repair demonstrated reader and interval defects with regression checks; evidence: report/final-routing-rescue-20261019-r1/AUDIT_VALIDATION-r1.json records 52 passing focused tests, immutable r8 preservation, execution/completion guards and corrected missing-outcome inference.
- [-] F-0001.5 Verify corrections on bounded saved inputs and report what remains before an allocated reader and corrected analysis can run. Actual campaign/program/source metadata checks pass; tokenizer replay exceeded the bounded development check and remains unverified.; dropped by user 2026-10-20: "stop the 512-token rescue and run the whole-reasoning efficiency experiment"; replacement: F-0003.
- [x] F-0001.6 Record the corrected run procedure, resource limits and remaining scientific results without treating missing outcomes as null effects; evidence: report/final-routing-rescue-20261019-r1/README.md. Rescue usage is 16.122222222222224 GPU-hours against the authorized 20 GPU-hours; no new jobs submitted.
- [-] F-0001.7 Complete release preparation: explicit cohort adjudication, 64-cell all-layer engineering selection, 512-context trajectory selection, reader/analysis integration, allocated qualification and measured revised budget. Preserve all failures and historical results; no fresh workload is authorized by this checklist.; dropped by user 2026-10-20: "stop the 512-token rescue and run the whole-reasoning efficiency experiment"; replacement: F-0003.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|
| Preserved source files | Rescue r8 generation | Current source matched all receipt entries | 15 files archived and verified | `report/final-routing-rescue-20261019-r1/SOURCE-r8.tar.gz`; SHA-256 `6f9e0a34b8bddaded7d14ef9302e1b89bb7fd04b7ebfa5fb7adbea90f16e4307` |
| Assignment provenance | Two-model rescue, all frozen assignments | Slurm reported both final generation jobs complete | 5,417 stored completions; 2,007 untouched; 5,380 meet r8 positive-source and 512-token allowance eligibility | `report/final-routing-rescue-20261019-r1/GENERATION_PROVENANCE_AUDIT.json` |
| Saved metadata verification | Actual GPT/Qwen rescue parent campaign, reader program and source tables | Campaign file and canonical receipt hashes are distinct | Corrected physical-file join and actual program-byte check pass for both models without model loading | `report/final-routing-rescue-20261019-r1/AUDIT_VALIDATION-r1.json` |
| Focused CPU regression tests | Reader, effects, generation identity/completion, actual selector reference | Demonstrated provenance, completion, signed-bound and missingness defects | 52 tests pass across four suites; full saved-token replay and allocated qualification remain incomplete | `report/final-routing-rescue-20261019-r1/AUDIT_VALIDATION-r1.json` |
