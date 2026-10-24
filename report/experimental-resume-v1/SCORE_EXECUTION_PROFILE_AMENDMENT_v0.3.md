# Score execution qualification amendment v0.3

Slurm 59196354 completed all 80 identical-prefix API assignments and failed the
prospective v3 numerical gate. Cross-API B mean/p99 differences were 0.000675 /
0.009022 nats; limits based on independent native repeats were 0.000468 /
0.006182. One prefix's first token dominates the error. All 16 fixture positions
had the same sampled token across the four APIs/repeats. This is not evidence
that designated-token indexing is wrong or that expert interventions fail.
The failed measurement result stays failed.

Test a concrete execution hypothesis before collecting expert-selection loss
data: serialize computation to one runnable request and disable CUDA graphs.
The original model, FP8 backend regime, exact populated pilot table, worker,
80 assignments, full-vocabulary reference, appended-token diagnostics and
1.25-repeat-plus-1e-6 mean/p99 criteria remain fixed. The new manifest binds
max_num_seqs=1 and enforce_eager=True; VLLM_BATCH_INVARIANT stays zero. This is
a different numerical execution profile and cannot be relabeled as evidence
for the previous batched configuration. A separate batch-invariant backend
experiment is running independently.

The same-base driver is immutable and a separately sealed wrapper changes only
engine construction. It reads all code/table hashes before load, records the
new profile in the qualification result and creates new manifest-derived UIDs
and output directory. A pass can qualify this score profile only. If it fails,
preserve its complete results and investigate a reference implementation or
different functional endpoint instead of relaxing the numeric threshold or
adding repeats until it passes.

Complete price: 45 minutes × two A100s = 1.5 GPU-hours maximum, comprising
1.5 times the measured v3 entire allocation (823s), 900s additional serial
processing reserve and 180s safety. The measured v3 warm stage was only 13s;
cold model setup dominated. The user's uncapped necessary-paper-workload
authorization covers this engineering recovery. No semantic sample or new
behavioral hypothesis is added.
