# src/steering

Code that pushes the routers of the eight models toward or away from chosen experts while a model reasons, and measures what that did to the length and the correctness of the reasoning. Launchers are in [`scripts/steering`](../../scripts/steering/README.md). Python decides what runs; the scripts decide where and in which order.

There are three experiments:

- Pulsed steering (`pulse.py`, `pulse_outcomes.py`, `pulse_effects.py`): a short push on the experts of one reasoning state at the start of a continuation. Activating a state forces its nominated experts into the selection; deactivating removes them. It runs at a central strength and, for four models, a stronger one. Each edit is compared with matched random experts at the same layers and with native routing. Its tables are the main and strong steering results of the paper (figure `routing`).
- Whole-reasoning steering (`whole.py`, `whole_effects.py`, `release.py`): the push is applied at every prompt and reasoning token until the reasoning closes, and the final answer is left to native routing. States are promoted or suppressed according to whether they precede shorter or longer remaining reasoning (the efficiency link), at quarter, half and full strength, with matched random experts as controls. The sealed result is `EFFECTS_WHOLE.json` (figures `whole_states`, `whole_dose`).
- Outcome-selected nudge (`outcome/`): not in the paper, the active next line. See [its README](outcome/README.md).

The same routing operators serve all three.

## Files

| File | What it does |
| --- | --- |
| `pulse.py` | The pulsed design: its manifests, one engine per model shard that admits requests while others finish, and the LFM2.5 stages (`native`, `annotate`, `lfm-replay`). |
| `pulse_outcomes.py` | Reads the saved continuations and writes one table of outcomes and one of sentences per model. |
| `pulse_effects.py` | The paired effects with bounds for unknown outcomes, the family bootstrap and the efficiency-link test. |
| `whole.py` | The whole-reasoning release: manifests, the fluency canary, the shards, `plan`, `prepare`, `run`. |
| `whole_effects.py` | Replays and scores the finished rows (`extract`) and joins every release into `EFFECTS_WHOLE.json` (`combine`). |
| `release.py` | Seals a complement, relink or outcome release: receipts plus a frozen copy of the code. |
| `stream.py`, `observer.py`, `operators.py` | The hooks that change the routing of the running model, and the operators: hard inclusion and exclusion, and the score override that follows SteerMoE. |
| `nominations.py`, `selection.py`, `allocation.py` | Which experts to edit for each state (the nominations), which layer of the network, and which question families go to which arm. |
| `runtime.py`, `session.py`, `penalties.py`, `fast_penalties.py` | Prompts and the durable per-request generation, one request lifecycle, and the repetition penalties applied to saved history. |

## Commands

```bash
python -m steering.pulse COMMAND [--model M] [--shard I/N] [--design main|strong]   # plan, plan-strong, run, native, annotate, lfm-replay, status
python -m steering.pulse_outcomes --model M [--design strong]
python -m steering.pulse_effects --out DIR [--design strong --models gpt,nemotron,qwen330b,gemma]
python -m steering.whole COMMAND --release report/NAME [--model M] [--shard I/N]     # plan, prepare, canary, run, extract, combine
python -m steering.release KIND NAME --tests N [--models ...] [--shards ...] [--cutoff TIME]   # KIND: complement, relink, outcome
```

GPU commands (`run`, `native`, `annotate`, `lfm-replay`, `canary`) refuse to start outside a Slurm GPU allocation. Use the launchers.

## Running a whole-reasoning release

A release is a folder `report/NAME`. Every stage takes `--release report/NAME` and reads its settings (models, strengths, shards, cut-off time, efficiency link) from the files in it.

```bash
scripts/steering/whole.sh complement NAME TESTS --cutoff 2026-10-24T06:00   # seal, then submit the jobs
scripts/steering/launch.sh NAME                          # submit the jobs of a sealed release
DRY_RUN=1 scripts/steering/launch.sh NAME                # print the sbatch commands only
```

The stages are `plan`, `prepare`, `canary`, `run`, `extract`, `combine`. Slurm is only in `scripts/steering/`: dependencies, resources and walltimes are in `launch.sh`. A shard asks for 4 hours of 2 GPUs, the release records its declared ceilings, and `scripts/slurm/ledger.sh whole-` adds up what was spent. The fluency canary of a base release runs by hand (`python -m steering.whole canary`); it checks that steered continuations do not break down (reach the limit or repeat themselves far more often than native ones), and its receipt is what lets `run` start production. A continuation still running at the end of an allocation is saved with its tokens and resumed later through the same hooks, so every position keeps its steering and nothing is redrawn.

Two rules apply to results: an outcome that is unknown (a continuation that reaches its allowance, or never finishes) keeps an interval and is never dropped, and nothing is redrawn after the fact. Token contrasts are `log(1+T_control) - log(1+T_treated)`, so a positive value means shorter reasoning than the control.

Tests: [`tests/steering`](../../tests/steering/README.md).

<!-- ledger:begin -->
## Ledger

Decisions and features that govern this folder (generated by `ledger sync`):

- [D-0002](../../../docs/ledger/decisions/D-0002-whole-reasoning-efficiency.md) [active] Whole-reasoning routing efficiency: chosen Alignment v3 control-feasible global sign-stable RD nomination with exact layer/quintile/role controls at both doses; original prompts, dual score override at epsilon 0.01, prompt and reasoning steering until closure, native and native_repeat. Continuous 64-slot admission with native and efficiency cells before other promote cells. Relative full-allowance canaries, two GPT and seven Qwen shards, immutable releases, fresh stores and six simultaneous bounded contrasts. Monitor deadlines/errors never stop production; actual CPU usage remains bounded separately.; scope: Whole-reasoning GPT/Qwen efficiency experiment and corrected main/strong CPU analyses.; not: Further 512-token rescue work, manuscript edits, pushes, outcome-driven tuning.
- [D-0003](../../../docs/ledger/decisions/D-0003-outcome-selected-gentle-routing-nudge.md) [active] Outcome-selected gentle routing nudge: chosen Select experts by outcome on campaign-v3 cohort B (per-attempt usage per reasoning token, z-scored within question, mean wrong-minus-right over mixed-outcome questions, sign stable under leave-one-question-out). Exclude the shared core (top 5% native frequency per layer) and the most execution-specific experts (union over folds of the top 10% positive Implement/Analyze risk differences). Take stable positive pairs until 1.2% of native routing mass (5 to 40 pairs; fill to 5 by score with an unstable_fill flag if needed). Families whose question is in cohort B use the set selected without that question. Subtract delta in {0.4, 0.8, 1.6} from the selected experts' router logits at every prompt and reasoning token, keeping native top-k and renormalisation. Matched random controls with the same layers, frequency quintiles and exclusions. Reuse the base releases' native and native_repeat rows. Length-limit baseline: cut each family's native reasoning at 50/70/85% of its length, append the closing suffix and generate only the answer (allowance min(2048, remaining)).; scope: The outcome-steering experiment (F-0004) on gpt, qwen330b and nemotron; extension to the other models with cohort-B captures only if the three-model result is positive.; not: Changes to the finished whole-reasoning releases r7-r17 or their analysis; manuscript claims before a sealed receipt.
- [F-0003](../../../docs/ledger/features/F-0003-whole-reasoning-efficiency.md) [active 6/13] Execute the whole-reasoning efficiency experiment
- [F-0004](../../../docs/ledger/features/F-0004-outcome-selected-gentle-steering-experiment.md) [active 6/6] Outcome-selected gentle steering experiment
- [F-0008](../../../docs/ledger/features/F-0008-complete-the-whole-reasoning-per-state-heatmap-wit.md) [active 1/3] Complete the whole-reasoning per-state heatmap with the missing suppress cells
- [F-0009](../../../docs/ledger/features/F-0009-lfm2-5-on-its-own-efficiency-link-signs.md) [active 4/6] LFM2.5 on its own efficiency-link signs
- [W-0001](../../../docs/ledger/walkthroughs/W-0001-causal-rescue-one-historical-gpt-assignment.md) [walkthrough @0084256] Causal rescue, one historical GPT assignment

<!-- ledger:end -->
