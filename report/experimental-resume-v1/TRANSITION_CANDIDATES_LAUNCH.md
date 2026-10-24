# Prefix-only discovery candidate frame, 2026-10-01

`prepare_transition_candidates_unlabeled.py` reads the 48 frozen discovery
native traces, exact saved token offsets, complete sentence windows and the
versioned prefix-only detector. It reads no seven-class labels, correctness,
gold answer or intervention outputs. It retains every detector fire and up to
200 deterministic family-spread nonfires per fixed transition, over every
contiguous sentence pair. False fires in an unexpected starting class remain
in the frame. The later label-dependent audit is a distinct artifact.

Driver SHA-256:
`909380380193bc638768cc0750d6d3f3b5396ee79e6df6a989d9f63593e94a83`.
Output: `steering-v1/runs/routing-control-v1/dense-discovery/TRANSITION_AUDIT_CANDIDATES.json`.
One two-CPU, one-hour Slurm allocation costs at most two CPU core-hours.
This is selection and resource planning, not semantic rating or action discovery.
The resulting frame count will price blind LLM audits before any GPU submission.

Job `59120873` completed `0:0` in 65 seconds on two CPUs (0.03611 CPU
core-hour). Sealed output `f5b2e28c000ea276c5a9c60fa4956ccecba9451eb121ff00b756859bdcb8e17e`
contains 619 rating rows. Among 5,523 contiguous pairs per hypothesis, the
detector fired 15 times for candidate→verify across nine families, three times
for approach→commit across three families, and once for failed-check→revision
in one family. These are unvalidated trigger candidates. The semantic audit
will rate all fires and the fixed nonfire samples; low family coverage may
fail feasibility even if some candidate phrases are valid.
