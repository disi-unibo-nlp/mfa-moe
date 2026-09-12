# MoE Routing Dynamics During Chain-of-Thought Reasoning

This repository currently contains four active workflows:

1. `gepaLLMAsJudge`: GEPA prompt optimization for the seven Schoenfeld
   reasoning-episode labels, using the served model as an informative judge.
2. `correlation_pipeline`: repeated benchmark generation, teacher-forced MoE
   router/hidden-state extraction, and cluster-aware correlation analysis.
3. `probeTest`: Qwen3.5 hidden-state probes trained on the released gold
   Schoenfeld traces.
4. `moe_guiding`: exploratory margin-triggered expert replacement during live
   top-2 Mixtral inference with vLLM.

The prioritized work agreed in the latest tutor meeting is tracked in
[`NEXT_STEPS.md`](NEXT_STEPS.md).

The earlier numbered experiments are archived under `src/moe_exp/old`. Their
launchers, results, documentation, and full report are retained under `old/`
and `results/old/`; they are not part of the current workflow.

## Setup

Requires Python 3.11 or newer. Install a CUDA-compatible PyTorch build first,
then install the project:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Install the renamed GEPA workflow and test dependencies when working on it:

```bash
pip install -e ".[dev,gepa-llm-as-judge]"
```

The Docker image supplies the supported Python and CUDA environment on hosts
where the system Python is older:

```bash
docker build -t moe-mfa-experiments:latest .
```

## gepaLLMAsJudge

The package is located at `src/moe_exp/gepaLLMAsJudge`. Its default output is
`results/gepaLLMAsJudge`.

```bash
python -m moe_exp.gepaLLMAsJudge.audit \
  --dataset-dir data/Schoenfeld_Reasoning \
  --output results/gepaLLMAsJudge/annotation_audit.json

python -m moe_exp.gepaLLMAsJudge.run \
  --dataset-dir data/Schoenfeld_Reasoning \
  --prompt-variant few-shot \
  --few-shot-examples 21 \
  --gepa-reward llm-judge \
  --gepa-auto heavy \
  --output-dir results/gepaLLMAsJudge/context-llmjudge-final
```

On SLURM, the launcher starts the llama.cpp server and GEPA containers:

```bash
sbatch run_gepaLLMAsJudge.sh --gepa-auto heavy
```

See [`src/moe_exp/gepaLLMAsJudge/README.md`](src/moe_exp/gepaLLMAsJudge/README.md)
for split isolation, metrics, safety selection, audit artifacts, and the locked
test policy.

## Correlation Pipeline

The current Qwen MoE benchmark workflow is self-contained under
`src/moe_exp/correlation_pipeline`. The default resumable run executes the
six benchmarks: MATH-500, AIME24, AIME25, OlympiadBench, AMC23 and Minerva. It
runs vLLM generation and frozen GEPA sentence tagging with eight concurrent
requests per stage, followed by forward replay and all
correctness, metric-pair, expert, class and position analyses:

```bash
src/moe_exp/correlation_pipeline/run_all.sh
```

Use `--skip-tagging` to omit the judge and class-based analyses while retaining
whole-reasoning, position and standard correlation/expert analyses. Untagged
outputs go to `reasoning-vllm-untagged-v1`. `--skip-annotate` instead reuses
existing labels and keeps class-based analyses.

GPQA-Diamond and MMLU-Pro remain available through explicit `--datasets` selection.
Use `--skip-generate` to reuse generations of the selected model. For the saved
GGUF generations, also pass
`--generation-model qwen3.5-35b-a3b-mtp-ud-q4-k-xl`. New generation uses
`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`; the judge uses
`unsloth/Qwen3.8-27B-NVFP4`. New annotations and analyses live under
`results/correlation_pipeline/reasoning-vllm-v1`, preserving older results. Generation, tagging, forward replay,
and analysis can also be run independently. See
[`src/moe_exp/correlation_pipeline/README.md`](src/moe_exp/correlation_pipeline/README.md)
for supported benchmarks, sampling settings, model pairing, output layout,
and Docker commands.

## probeTest

`probeTest` extracts pre-sentence Qwen3.5 hidden states for the 38 released
gold traces and trains layer-wise one-vs-rest probes. Results remain under
`results/probeTest`.

See [`src/moe_exp/probeTest/README.md`](src/moe_exp/probeTest/README.md) for the
protocol and launcher.

## MoE guiding

`moe_guiding` compares native top-2 routing with margin-triggered top-1 + least-1
routing, using either the selected experts' probabilities or the original
top-2 mixing coefficients. Start with the model-free arithmetic check:

```bash
python -m moe_exp.moe_guiding.run sanity
```

See [`src/moe_exp/moe_guiding/README.md`](src/moe_exp/moe_guiding/README.md) for
worker plugin setup, layer selection, generation commands, and validation limits.

## Reports and Archive

- Current report entry point: [`report/main.tex`](report/main.tex)
- Current experimental setup: [`report/experimental_setup.tex`](report/experimental_setup.tex)
- Current GEPA section: [`report/gepaLLMAsJudge.tex`](report/gepaLLMAsJudge.tex)
- Current boundary-probe section: [`report/probeTest.tex`](report/probeTest.tex)
- Current correlation section: [`report/correlation_pipeline.tex`](report/correlation_pipeline.tex)
- Current analysis and limitations: [`report/active_synthesis.tex`](report/active_synthesis.tex)
- Current reproducibility checklist: [`report/reproducibility.tex`](report/reproducibility.tex)
- Numbered source packages: [`src/moe_exp/old`](src/moe_exp/old)
- Historical README snapshot: [`old/README.md`](old/README.md)
- Historical full report: [`old/report/legacy_full_report.tex`](old/report/legacy_full_report.tex)
- Archived outputs: `results/old/exp1` through `results/old/exp5`

The archived pipeline remains runnable through
`old/scripts/run_pipeline.sh`; it writes to `results/old` by default.

## Active Layout

```text
src/moe_exp/
|-- correlation_pipeline/
|-- gepaLLMAsJudge/
|-- moe_guiding/
|-- probeTest/
|-- models/
|-- datasets/
|-- analysis/
`-- old/
```
