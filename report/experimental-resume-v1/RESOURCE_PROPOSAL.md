# Resource proposal, resume v1

Historical proposal retained. The current approved ceilings, completed jobs and
failed recovery are in [RESOURCE_CHECKPOINT_v2.md](RESOURCE_CHECKPOINT_v2.md).

Approved on 2026-10-01 by the user's explicit reply: **“Authorize recovery and
separate study: +11.25 GPU-hours total.”** The approval authorizes the 0.75-hour
recovery and the separate 10.50-hour study. Qualification, eligibility and complete
stage-pricing gates still apply. The approval receipt is
`steering-v1/runs/resume-v1/RESOURCE_AUTHORIZATION.json` in the external run tree.

H1–H4 job **59090613** ended FAILED, exit **3:0**, after **865 seconds** on
two GPUs: **0.48056 GPU-hour**. All four verdicts say **not evaluated: deadline**.
The model finished startup near the deadline; no qualification batch was completed.
This is an operationally incomplete qualification, not evidence of a routing defect.
The raw verdicts and logs are preserved. This original failure remains preserved separately from the recovery below.

The original 0.5-hour line has only 0.01944 hour left, insufficient for another
model load. The proposed recovery is **one additional allocation of at most
0.75 GPU-hour**, using the same immutable test snapshot and frozen criteria:
two GPUs, 22-minute wall limit, no retry beyond that allocation. Its maximum
reservation is 0.73333 GPU-hour, including startup, batches and teardown. The
observed nearly 15-minute first attempt leaves about seven minutes of additional
walltime; this is a conservative execution envelope, not a guarantee of a pass.
Use a new output directory and job name, verify the job ID and all final artifacts.
Do not charge it to X2, X3 or the new study's qualification line.

The separate routing-action study retains the user's **10.50 GPU-hour authorization**:
qualification .75; discovery 2.00; mechanism 2.25; utility 4.75; reserve .75.
See `PROTOCOL_v0.1.md` for the maximum workloads and stopping rules. No part of
this new allocation has been spent. Complete context-matched stage pricing and
behavioral eligibility must pass before each stage; unknown costs hold submission.
In particular, the utility scout's 6,291,456-token maximum cannot be priced from
historical X1 throughput alone. A stage exceeding its ceiling requires a revised
proposal; no hidden cuts or transfers between lines.

Recovery **59092583** completed with exit **0:0**: all H1–H4 passed on 56 requests, including five deliberate preemptions. Its 838-second two-GPU allocation used **0.46556 GPU-hour**, leaving **0.28444** on the added recovery line. The original failure plus recovery cost **0.94611 GPU-hour**. Qualification artifacts and raw failure remain separate. X2 generation job **59093506** is authorized and submitted; its eight-hour line reserves six for generation, 1.5 for native NLL and .5 contingency.

The new study is authorized but held by its qualification, eligibility and complete pricing gates. `STUDY_GATES.json` records maximum request counts and explicit missing costs. With historical X1 timing and its registered .75 derating, the utility cap alone plus one observed model allocation exceeds **6.07 GPU-hours**, before prefill, grading and retries; this exceeds its **4.75** stage ceiling. This is a historical scenario, not a qualified controller price. A complete revised envelope requires the missing measurements. No study experiments have been submitted, and no allocation has been transferred between lines.

Legacy ceilings remain X2 eight GPU-hours including NLL and resumes, X3 twelve, M9 two, M10/M8 three. User approval for these named workloads and the additional 11.25-hour allocation is recorded; further permission is needed only for an actual increase to the approved envelope.
