# scripts/steering

Launchers for the routing interventions: edit the routers while a model reasons, then measure what changed. Python decides what runs (`src/steering`); these scripts decide where and in which order.

| Script | What it does |
| --- | --- |
| `pulse.sh SHARDS HOURS MODEL...` | GPU jobs of the pulsed design, one per shard of question families. |
| `pulse_analyze.sh OUT` | CPU jobs: the outcome tables, then the effects of the main and strong designs into `OUT`. |
| `whole.sh KIND NAME TESTS [OPTION...]` | Seals a complement, relink, outcome, capture, sweep or continuation release (`python -m steering.release`; `--cutoff` is required for every kind but a relink, which takes its parent's cutoff; a continuation also needs `--parent`, a sweep `--half`, `--capture` and `--models`), then runs `launch.sh`. `AFTER="JOB ..."` passes running jobs on to `launch.sh`, e.g. a relink's parent combine or a continuation's parent extractions. |
| `launch.sh NAME [JOB...]` | Submits the jobs of a sealed release: the unit tests (sealed in `CPU_VALIDATION.json`), then the manifests, the GPU shards, one extraction per model, the final analysis and the staged paper export. Job names start with the release's `job_prefix`: `whole` normally, `oc` for a capture, `os` for an outcome sweep (a capture release has no final analysis). Refuses to submit anything when the jobs' time limits exceed the release's CPU or GPU ceilings. A failed launch cancels what it queued and keeps their IDs in `JOBS.cancelled-TIME.txt`. `DRY_RUN=1` prints the sbatch lines only. |
| `outcome_usage.sh` | CPU job: how often each expert is used in every cohort-B attempt (input of an outcome release). |
| `outcome_derive.sh NAME MODEL...` | Queues, for a launched capture release, one CPU job per model after that model's extraction: class usage, expert sets, signal table and attempt selection (`steering.outcome derive`), run from the release snapshot. `DRY_RUN=1` prints the sbatch lines. |
| `outcome_pick.sh HALF_A MODEL NAME TESTS [OPTION...]` | CPU job, submitted behind half A's analysis: applies the frozen pick rule (`steering.outcome pick`) and, if it picks cells, seals and launches half B with `whole.sh sweep`. |
| `outcome_next.sh EFFECTS NAME TESTS [OPTION...]` | Applies the declared extension rule to an outcome analysis and, if it is met, seals and launches the next release. |
| `gpu.sh MODULE ARGS...` | The job script of a GPU stage: one Python module on 2 GPUs. |

## Pulsed steering

This gives the main and strong doses of the paper.

```bash
sbatch -J pulse-plan scripts/slurm/cpu.sh steering.pulse plan   # CPU: writes driver/MODEL/ASSIGNMENTS.json
scripts/steering/pulse.sh 2 12 gpt                            # 2 shards of 12 hours for GPT-OSS
sbatch -t 12:00:00 -J fr-gpt-0 scripts/steering/gpu.sh steering.pulse run --model gpt --shard 0/2 --design strong   # strong dose, after plan-strong
scripts/steering/pulse_analyze.sh /new/folder
```

## LFM2.5

LFM2.5 has no server script. Its stages run in this order, and `scripts/routing/lfm.sh` follows them:

1. `sbatch scripts/steering/gpu.sh steering.pulse native --shard I/N` generates its traces.
2. `sbatch scripts/slurm/cpu.sh routing.lfm materialize` picks and prepares the sentences to label (CPU).
3. `sbatch scripts/steering/gpu.sh steering.pulse annotate --shard I/N` labels them.
4. `sbatch scripts/steering/gpu.sh steering.pulse lfm-replay --shard I/N` records its routers.

## Whole-reasoning steering and the dose sweep

```bash
scripts/steering/whole.sh complement NAME TESTS --cutoff 2026-10-24T06:00   # seal report/NAME, then submit its jobs
DRY_RUN=1 scripts/steering/launch.sh NAME                                   # look at the jobs first
```

A complement whose shards could not all start before its cutoff gets a continuation that runs what is left. Read its plan first; `AFTER` names the parent's extraction jobs, which the plan lists:

```bash
python -m steering.followup report/whole-reasoning-20261021-r18          # plan of the follow-up of a complement: rows by state, shards, GPU ceiling
AFTER="JOB ..." scripts/steering/whole.sh continuation whole-reasoning-20261021-r21 TESTS --parent whole-reasoning-20261021-r18 --cutoff 2026-10-27T06:00
```

A release is the folder `report/NAME`: it fixes the models, strengths, shards, cut-off time and efficiency link of one run, and it holds a frozen copy of the code. Its jobs run from that copy in `mfa-moe/releases/<commit>`, not from the working tree. `whole.sh` needs a clean `src`, `tests` and `scripts` tree. `scripts/slurm/ledger.sh whole-` adds up what the releases cost.

## Outcome-selected nudge

This is not in the paper.

```bash
scripts/steering/outcome_usage.sh
scripts/steering/whole.sh outcome outcome-steering-20261022-r1 TESTS --shards 3 --cutoff 2026-10-23T23:00
sbatch scripts/steering/outcome_next.sh report/RELEASE/analysis/EFFECTS_OUTCOME.json NEXT_NAME TESTS --cutoff TIME --models gemma,glm
```

## Outcome steering r2 (class-conditioned)

Not in the paper. Ledger F-0013, F-0014, D-0006; code in [`src/steering/outcome`](../../src/steering/outcome/README.md).

```bash
scripts/steering/whole.sh capture NAME TESTS --shards gpt:3,glm:15 --cutoff 2026-10-30T23:00   # native attempts with routing, jobs oc-*
scripts/steering/outcome_derive.sh NAME gpt glm                                              # CPU chain per model after its extraction
scripts/steering/whole.sh sweep NAME_A TESTS --half A --capture NAME --models gpt --shards 15 --cutoff 2026-10-30T23:00   # jobs os-*
sbatch --dependency=afterok:ANALYSIS_JOB scripts/steering/outcome_pick.sh NAME_A gpt NAME_B TESTS --capture NAME --shards 2 --cutoff 2026-10-30T23:00
```

The capture and sweep stores go to FAST (`report/NAME/stores` is a link). `ANALYSIS_JOB` is the `os-analyze` line of `report/NAME_A/JOBS.txt`.

Paper use: pulsed steering gives figure `routing` and its tables, whole-reasoning steering gives `whole_states`, `whole_dose` and their tables. See [`src/steering`](../../src/steering/README.md).

<!-- ledger:begin -->
## Ledger

Decisions and features that govern this folder (generated by `ledger sync`):

- [D-0002](../../docs/ledger/decisions/D-0002-whole-reasoning-efficiency.md) [active] Whole-reasoning routing efficiency: chosen Alignment v3 control-feasible global sign-stable RD nomination with exact layer/quintile/role controls at both doses; original prompts, dual score override at epsilon 0.01, prompt and reasoning steering until closure, native and native_repeat. Continuous 64-slot admission with native and efficiency cells before other promote cells. Relative full-allowance canaries, two GPT and seven Qwen shards, immutable releases, fresh stores and six simultaneous bounded contrasts. Monitor deadlines/errors never stop production; actual CPU usage remains bounded separately.; scope: Whole-reasoning GPT/Qwen efficiency experiment and corrected main/strong CPU analyses.; not: Further 512-token rescue work, manuscript edits, pushes, outcome-driven tuning.
- [D-0003](../../docs/ledger/decisions/D-0003-outcome-selected-gentle-routing-nudge.md) [active] Outcome-selected gentle routing nudge: chosen Select experts by outcome on campaign-v3 cohort B (per-attempt usage per reasoning token, z-scored within question, mean wrong-minus-right over mixed-outcome questions, sign stable under leave-one-question-out). Exclude the shared core (top 5% native frequency per layer) and the most execution-specific experts (union over folds of the top 10% positive Implement/Analyze risk differences). Take stable positive pairs until 1.2% of native routing mass (5 to 40 pairs; fill to 5 by score with an unstable_fill flag if needed). Families whose question is in cohort B use the set selected without that question. Subtract delta in {0.4, 0.8, 1.6} from the selected experts' router logits at every prompt and reasoning token, keeping native top-k and renormalisation. Matched random controls with the same layers, frequency quintiles and exclusions. Reuse the base releases' native and native_repeat rows. Length-limit baseline: cut each family's native reasoning at 50/70/85% of its length, append the closing suffix and generate only the answer (allowance min(2048, remaining)).; scope: The outcome-steering experiment (F-0004) on gpt, qwen330b and nemotron; extension to the other models with cohort-B captures only if the three-model result is positive.; not: Changes to the finished whole-reasoning releases r7-r17 or their analysis; manuscript claims before a sealed receipt.
- [D-0006](../../docs/ledger/decisions/D-0006-class-conditioned-outcome-steering-design-r2.md) [active] Class-conditioned outcome steering design (r2): chosen Captures: the same 300 questions for every model (aime24 30, aime25 30, amc23 40, olympiad 100, minerva 50, math500 50; digest-ranked within benchmark, one question per family, no outcome read), 6 native attempts each, medium effort, original prompt tokens, 32,768-token completion cap on every row of r2, native top-k ids and weights of every reasoning token at every routed layer. Labels: lexical readout v1 (TF-IDF), out-of-fold by FOLDS.json; no judge. Score per class c: within-question z-scored usage on tokens labelled c, mean of wrong minus mean of right, over questions with both outcomes; capped or unknown attempts excluded. Sets: one per class (7) plus a class-blind set (GPT-OSS: Beyond the Trace's five pairs L12-E23, L12-E12, L12-E24, L16-E11, L20-E0; other models: the plain outcome set). Set rule from related work: candidates only in the three routed layers at relative depth 1/2, 2/3, 5/6 (n//2, 2n//3, 5n//6; 12/16/20 on GPT-OSS, Beyond the Trace); best positive pairs until they carry 1.2% of those layers' native gate weight per reasoning token, at most 10 pairs (the paper's own definition, confirmed by the ELM grounding check of 2026-10-23: "1.2% of the gated mass across the three recorded layers", gate weight accumulated over routed assignments; the slot share is reported beside it); committee experts (top 5% native use per layer, Illusion of Specialization) and the top-10% Implement/Analyze experts never nudged; logit shift with native top-k and renormalisation (SteerMoE-style operator). Each evaluation family uses sets selected without its own question. Arms: target nudge and two disjoint matched random draws (same layer, native-frequency quintile) per set and shift; shifts 0.4, 0.8, 1.2 at every prompt and reasoning token; unedited arm = the 6 capture attempts; stop-early baseline at 50/70/85% of a native attempt. Halves: the 300 questions split by a benchmark-stratified digest into A and B. Pick rule (frozen now): among half-A cells with accuracy against native no worse than -5 points (point estimate) take the most negative tokens-against-mean-random point estimate; if none qualifies, the best accuracy-against-random estimate. Half B confirms that cell and the class-blind set at its best shift. Primary reported contrast: tokens against the mean of the two random draws with accuracy against native noninferior within 5 points, on half B; every half-A cell is exploration; no success threshold gates anything. Attempt selection on the captures: routing consensus, outcome-profile and class-mix scores against majority vote, random, shortest and longest.; scope: The second outcome-steering campaign (F-0013, F-0014): native captures on all eight models, class-conditioned expert selection, the GPT-OSS nudge sweep on question half A, the automatic pick and its confirmation on half B, and the attempt-selection analysis. The other seven models' sweeps are decided jointly after the GPT-OSS half-B receipt.; not: Release r1 and its analysis (kept unchanged, reported as the broad-edit test); the whole-reasoning releases r7-r20; GPT-OSS low/high reasoning effort (a later GPT-only extension); manuscript claims before a sealed receipt.
- [D-0007](../../docs/ledger/decisions/D-0007-bash-holds-slurm-python-holds-the-science.md) [active] Bash holds Slurm, Python holds the science: chosen `scripts/<topic>/` holds only bash: one short launcher per experiment with positional arguments, `scripts/slurm/env.sh` sourced by every job, one `vllm serve` script per model. Job graphs (dependencies, resources, walltimes, budget checks, cancellation) live in bash (`scripts/steering/launch.sh`). `src/<topic>/` holds only Python, which seals releases (`python -m steering.release`), reads its configuration from the release files, and computes; it never submits, accounts or monitors Slurm jobs.; scope: Every launcher and job graph of the repository: model servers, generation, labeling, capture, routing analyses, steering releases, paper exports.; not: The computations themselves, sealed receipts and their formats, frozen pre-registered files.
- [F-0003](../../docs/ledger/features/F-0003-whole-reasoning-efficiency.md) [active 6/13] Execute the whole-reasoning efficiency experiment
- [F-0004](../../docs/ledger/features/F-0004-outcome-selected-gentle-steering-experiment.md) [active 6/6] Outcome-selected gentle steering experiment
- [F-0008](../../docs/ledger/features/F-0008-complete-the-whole-reasoning-per-state-heatmap-wit.md) [active 1/3] Complete the whole-reasoning per-state heatmap with the missing suppress cells
- [F-0009](../../docs/ledger/features/F-0009-lfm2-5-on-its-own-efficiency-link-signs.md) [active 5/7] LFM2.5 on its own efficiency-link signs
- [F-0012](../../docs/ledger/features/F-0012-every-paper-exhibit-on-all-eight-models.md) [active 1/8] Every paper exhibit on all eight models
- [F-0013](../../docs/ledger/features/F-0013-class-conditioned-outcome-steering-on-eight-models.md) [active 3/7] Class-conditioned outcome steering on eight models

<!-- ledger:end -->
