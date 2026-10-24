# Study job monitoring

Keep the inherited LEONARDO account rules and resource approvals in force.
After every authorized Slurm submission, verify the exact job ID and arm
`scripts/watch_slurm.py JOB_ID --max-seconds SECONDS` using `/usr/bin/python3.11 -B`
in the existing Codex tmux session. Run it as a bounded foreground tool session;
retain the tool session ID and verify its `ARMED` output. Choose a finite lifetime
covering queue wait and walltime (maximum 24 hours). Never start a daemon, cron,
server, new agent, or GPU allocation for monitoring. No watcher is needed when
there are no active jobs. If arming fails, report the monitoring gap immediately.

On `[SLURM_EVENT ...]`, inspect accounting, logs and actual artifacts, promptly
report to the user, record receipt with `scripts/watch_slurm.py JOB_ID --ack`,
and continue only within existing workload and budget authorization. A signal
never authorizes submission, resubmission, extra spending or cancellation.
`COMPLETED` alone does not establish experiment success. `WATCH_EXPIRED` means
monitoring stopped; inspect the job before deciding whether another bounded watch
is needed. Historical delivery is deduplicated per job ID.

The watcher requires this exact Codex process and tmux pane to remain alive. It
waits for an empty composer, never clears input, and saves undelivered events in
private HOME state. It cannot guarantee delivery through a changed UI, a closed
session, a login-node restart, or simultaneous human typing. State is at
`~/.local/state/mfa-moe-watches/JOB_ID.json`; distinguish `sent_to_tui` from
`acknowledged_at`. Do not store private session bindings in the repository.

# User communication preference

Use direct, plain statements. Do not use the formulation “we should/shouldn’t … rather …” or equivalent contrastive filler.
