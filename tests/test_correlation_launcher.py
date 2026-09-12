"""Exercise the unified shell workflow without live Docker or model calls."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "src/moe_exp/correlation_pipeline/run_all.sh"
DEFAULT_DATASETS = ["math500", "aime24", "aime25", "olympiad", "amc23", "minerva"]


def run_script(tmp_path, *args, **env):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=tmp_path,
        env={**os.environ, "PHYS_DIR": str(tmp_path), **env},
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def commands(output):
    return [
        shlex.split(line)
        for line in output.splitlines()
        if line.startswith(("  env ", "  docker "))
    ]


def stage_commands(output):
    stages = {}
    for command in commands(output):
        for i, arg in enumerate(command):
            if arg.endswith("/run_docker.sh"):
                stages[command[i + 1]] = command[i + 2 :]
    return stages


def option(args, name):
    return args[args.index(name) + 1]


def test_default_plan_runs_all_stages_and_releases_each_server(tmp_path):
    # No launcher files or Docker executable are needed, and no output is written.
    result = run_script(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stderr
    stages = stage_commands(result.stdout)
    assert list(stages) == ["generate", "sample", "annotate", "forward", "analyze"]
    for args in stages.values():
        i = args.index("--datasets") + 1
        assert args[i : i + 6] == DEFAULT_DATASETS
        assert "gpqa_diamond" not in args and "mmlu_pro" not in args
    generation = "results/correlation_pipeline/generation"
    reasoning = "results/correlation_pipeline/reasoning-vllm-v1"
    assert option(stages["generate"], "--model") == "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    assert option(stages["generate"], "--target-model-id") == "unsloth/Qwen3.5-35B-A3B"
    for name in ("forward", "analyze"):
        assert option(stages[name], "--model-id") == "unsloth/Qwen3.5-35B-A3B"
    assert option(stages["generate"], "--output-dir") == generation
    assert option(stages["sample"], "--generation-dir") == generation
    assert option(stages["sample"], "--output-dir") == f"{reasoning}/sampling"
    assert option(stages["sample"], "--weight-by") == "error-rate"
    assert option(stages["sample"], "--max-sentences") == "100000"
    assert option(stages["sample"], "--sample-id") == "0"
    assert option(stages["sample"], "--seed") == "42"
    assert option(stages["generate"], "--max-tokens") == "32768"
    assert option(stages["generate"], "--workers") == "8"
    assert option(stages["annotate"], "--workers") == "8"
    assert option(stages["annotate"], "--output-dir") == f"{reasoning}/annotations"
    assert option(stages["annotate"], "--base-url") == "http://127.0.0.1:41800/v1"
    assert option(stages["forward"], "--annotation-dir") == f"{reasoning}/annotations"
    assert option(stages["forward"], "--output-dir") == f"{reasoning}/forward"
    assert option(stages["forward"], "--quantization") == "unsloth-4bit"
    assert option(stages["analyze"], "--forward-dir") == f"{reasoning}/forward"
    assert option(stages["analyze"], "--output-dir") == f"{reasoning}/analysis"
    for name in ("annotate", "forward"):
        assert option(stages[name], "--generation-dir") == f"{reasoning}/sampling/generation"
    for name in ("forward", "analyze"):
        i = stages[name].index("--views") + 1
        assert stages[name][i : i + 3] == ["full", "class", "position"]
    plan = commands(result.stdout)
    stops = [i for i, c in enumerate(plan) if c[:2] == ["docker", "stop"]]
    stage_indices = [i for i, c in enumerate(plan) if any(a.endswith("/run_docker.sh") for a in c)]
    starts = [i for i, c in enumerate(plan) if c[:2] == ["docker", "run"]]
    assert len(starts) == 2
    judge_start = starts[1]
    for index in starts:
        command = plan[index]
        assert option(command, "--max-num-seqs") == "8"
        assert option(command, "--max-num-batched-tokens") == "8192"
        assert option(command, "--port") == "41800"
        assert "--enable-prefix-caching" in command
        assert json.loads(option(command, "--speculative-config")) == {
            "method": "mtp",
            "num_speculative_tokens": 3,
        }
    assert option(plan[starts[0]], "--max-model-len") == "49152"
    assert option(plan[starts[1]], "--max-model-len") == "32768"
    assert (
        stage_indices[0] < stops[0] < stage_indices[1] < judge_start < stage_indices[2] < stops[1] < stage_indices[3]
    )
    assert stage_indices[3] < stage_indices[4]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "flags,generation_model",
    [
        (["--model", "org/custom-moe"], "org/custom-moe"),
        (
            ["--model", "org/custom-moe", "--generation-model", "org/custom-moe-int4"],
            "org/custom-moe-int4",
        ),
        (
            ["--generation-model", "org/custom-moe-int4", "--model", "org/custom-moe"],
            "org/custom-moe-int4",
        ),
    ],
)
def test_pipeline_model_keeps_server_and_stage_models_consistent(tmp_path, flags, generation_model):
    result = run_script(tmp_path, "--dry-run", *flags)
    assert result.returncode == 0, result.stderr
    stages = stage_commands(result.stdout)
    assert option(stages["generate"], "--model") == generation_model
    assert option(stages["generate"], "--draft-model-id") == generation_model
    assert option(stages["generate"], "--target-model-id") == "org/custom-moe"
    for name in ("annotate", "forward"):
        assert option(stages[name], "--generation-model") == generation_model
    for name in ("forward", "analyze"):
        assert option(stages[name], "--model-id") == "org/custom-moe"
    servers = [command for command in commands(result.stdout) if command[:2] == ["docker", "run"]]
    assert [option(command, "serve") for command in servers] == [
        generation_model,
        "unsloth/Qwen3.8-27B-NVFP4",
    ]
    assert option(servers[0], "--served-model-name") == generation_model
    assert option(stages["annotate"], "--judge-model") == "unsloth/Qwen3.8-27B-NVFP4"
    assert option(stages["annotate"], "--judge-program") == (
        "results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json"
    )
    assert not list(tmp_path.iterdir())


def test_pipeline_model_resumes_with_generation_environment_override(tmp_path):
    result = run_script(
        tmp_path,
        "--dry-run",
        "--model", "org/custom-moe",
        "--skip-generate",
        "--skip-annotate",
        GENERATION_MODEL="saved-custom-moe",
    )
    assert result.returncode == 0, result.stderr
    stages = stage_commands(result.stdout)
    assert list(stages) == ["forward", "analyze"]
    assert option(stages["forward"], "--generation-model") == "saved-custom-moe"
    for args in stages.values():
        assert option(args, "--model-id") == "org/custom-moe"


def test_resume_custom_paths_options_and_optional_benchmarks(tmp_path):
    result = run_script(
        tmp_path,
        "--dry-run",
        "--skip-generate",
        "--generation-dir",
        "results/saved generations",
        "--results-dir",
        "results/custom run",
        "--datasets",
        "gpqa_diamond",
        "mmlu_pro",
        "--limit",
        "2",
        "--position-bins",
        "5",
        "--judge-program",
        "results/frozen.json",
        "--judge-model",
        "custom-judge",
        "--generation-model",
        "qwen3.5-35b-a3b-mtp-ud-q4-k-xl",
        "--judge-workers",
        "4",
        "--quantization",
        "none",
        "--bootstrap-samples",
        "20",
    )
    assert result.returncode == 0, result.stderr
    stages = stage_commands(result.stdout)
    assert list(stages) == ["sample", "annotate", "forward", "analyze"]
    for args in stages.values():
        i = args.index("--datasets") + 1
        assert args[i : i + 2] == ["gpqa_diamond", "mmlu_pro"]
        if args is not stages["sample"]:
            assert option(args, "--limit") == "2"
    assert option(stages["annotate"], "--generation-dir") == "results/custom run/reasoning-vllm-v1/sampling/generation"
    assert option(stages["forward"], "--generation-dir") == "results/custom run/reasoning-vllm-v1/sampling/generation"
    assert option(stages["annotate"], "--judge-program") == "results/frozen.json"
    assert option(stages["annotate"], "--judge-model") == "custom-judge"
    assert option(stages["forward"], "--position-bins") == "5"
    assert option(stages["analyze"], "--bootstrap-samples") == "20"
    for name in ("annotate", "forward", "analyze"):
        output = "annotations" if name == "annotate" else "analysis" if name == "analyze" else name
        assert (
            option(stages[name], "--output-dir") == f"results/custom run/reasoning-vllm-v1/{output}"
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "flags,expected",
    [
        (["--skip-generate", "--skip-annotate", "--skip-forward"], ["analyze"]),
        (["--skip-generate", "--skip-annotate"], ["forward", "analyze"]),
        (["--skip-forward", "--skip-analyze"], ["generate", "sample", "annotate"]),
    ],
)
def test_skip_flags_only_run_requested_stages(tmp_path, flags, expected):
    result = run_script(tmp_path, "--dry-run", *flags)
    assert result.returncode == 0, result.stderr
    assert list(stage_commands(result.stdout)) == expected
    if "annotate" not in expected:
        assert "docker run" not in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "flags",
    [
        ["--limit", "0"],
        ["--max-sentences", "0"],
        ["--position-bins", "0"],
        ["--judge-model"],
        ["--model"],
        ["--model", ""],
        ["--model", "--dry-run"],
        ["--max-tokens", "32768", "--ctx-size", "16384"],
        ["--results-dir", "../outside"],
        ["--skip-generate", "--skip-annotate", "--skip-forward", "--skip-analyze"],
    ],
)
def test_invalid_options_fail_before_starting_work(tmp_path, flags):
    result = run_script(tmp_path, "--dry-run", *flags)
    assert result.returncode == 2
    assert not commands(result.stdout)
    assert not list(tmp_path.iterdir())


def fake_runtime(tmp_path):
    def executable(name, body):
        path = tmp_path / name
        path.write_text("#!/bin/bash\nset -eu\n" + body)
        path.chmod(0o755)
        return str(path)

    executable(
        "docker",
        r"""
case "$1" in
    image) exit 0 ;;
    container)
        name="${@: -1}"
        if [[ "$name" == *-generation-* ]]; then stage=generation; else stage=judge; fi
        test -f "$PHYS_DIR/$stage.running"
        ;;
    inspect) echo true ;;
    run)
        test ! -f "$PHYS_DIR/generation.running"
        test ! -f "$PHYS_DIR/judge.running"
        while [[ "$1" != --name ]]; do shift; done
        if [[ "$2" == *-generation-* ]]; then stage=generation; else stage=judge; fi
        touch "$PHYS_DIR/$stage.running"
        echo "${stage}_start" >> "$PHYS_DIR/events"
        ;;
    stop)
        name="${@: -1}"
        if [[ "$name" == *-generation-* ]]; then stage=generation; else stage=judge; fi
        rm "$PHYS_DIR/$stage.running"
        echo "${stage}_stop" >> "$PHYS_DIR/events"
        ;;
    rm) exit 0 ;;
    logs) echo fake_server_log ;;
    *) exit 91 ;;
esac
""",
    )
    executable("curl", "exit 0\n")
    stage = executable(
        "stage",
        r"""
case "$1" in
    generate) test -f "$PHYS_DIR/generation.running" ;;
    annotate)
        test -f "$PHYS_DIR/judge.running"
        test ! -f "$PHYS_DIR/generation.running"
        ;;
    *)
        test ! -f "$PHYS_DIR/generation.running"
        test ! -f "$PHYS_DIR/judge.running"
        ;;
esac
echo "$1" >> "$PHYS_DIR/events"
if [[ "$1" == "$FAIL_STAGE" ]]; then exit 17; fi
""",
    )
    (tmp_path / "program.json").write_text("{}")
    probes = tmp_path / "results/probeTest/qwen3.5-35b-a3b-gptq-int4/probes/results.json"
    probes.parent.mkdir(parents=True, exist_ok=True)
    probes.write_text(json.dumps({"best_by_target": {"Read": {"layer_idx": 34}}}))
    return {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "STAGE_LAUNCHER": stage,
        "JUDGE_PROGRAM": "program.json",
    }


@pytest.mark.parametrize("failure", ["none", "generate", "sample", "annotate", "forward", "analyze"])
def test_model_lifecycle_and_cleanup_on_stage_failure(tmp_path, failure):
    result = run_script(tmp_path, **fake_runtime(tmp_path), FAIL_STAGE=failure)
    assert result.returncode == (0 if failure == "none" else 17), result.stdout + result.stderr
    events = (tmp_path / "events").read_text().splitlines()
    expected = ["generation_start", "generate", "generation_stop"]
    if failure != "generate":
        expected += ["sample"]
    if failure not in ("generate", "sample"):
        expected += ["judge_start", "annotate", "judge_stop"]
        if failure != "annotate":
            expected += ["forward"]
            if failure != "forward":
                expected += ["analyze"]
    assert events == expected
    assert not list(tmp_path.glob("*.running"))
    if failure not in ("generate", "sample"):
        assert (
            "fake_server_log"
            in next((tmp_path / "slurm_logs").glob("correlation-vllm-judge-*.log")).read_text()
        )


@pytest.mark.parametrize("flags", [[], ["--skip-generate"], ["--skip-generate", "--skip-forward"]])
def test_skip_tagging_needs_no_judge_and_keeps_only_full_position_views(tmp_path, flags):
    runtime = fake_runtime(tmp_path)
    (tmp_path / "program.json").unlink()
    result = run_script(tmp_path, "--skip-tagging", *flags, **runtime, FAIL_STAGE="none")
    assert result.returncode == 0, result.stdout + result.stderr
    stages = stage_commands(result.stdout)
    assert "annotate" not in stages
    events = (tmp_path / "events").read_text().splitlines()
    assert "judge_start" not in events
    assert "judge_stop" not in events
    if "--skip-generate" not in flags:
        assert events[:3] == ["generation_start", "generate", "generation_stop"]
    for stage in ("forward", "analyze"):
        if stage not in stages:
            continue
        args = stages[stage]
        index = args.index("--views") + 1
        assert args[index : index + 2] == ["full", "position"]
        assert "class" not in args
        assert "--annotation-dir" not in args
        assert "reasoning-vllm-untagged-v1" in option(args, "--output-dir")
    if "forward" in stages:
        assert option(stages["forward"], "--output-dir") == option(
            stages["analyze"], "--forward-dir"
        )
    assert not list(tmp_path.glob("*.running"))


def test_skip_annotate_still_requires_saved_class_annotations(tmp_path):
    result = run_script(tmp_path, "--skip-annotate", "--dry-run")
    assert result.returncode == 0, result.stderr
    forward = stage_commands(result.stdout)["forward"]
    assert "class" in forward
    assert option(forward, "--annotation-dir").endswith("reasoning-vllm-v1/annotations")
    assert option(forward, "--generation-dir").endswith("reasoning-vllm-v1/sampling/generation")


def test_sampling_cap_override_and_presampled_inputs(tmp_path):
    result = run_script(tmp_path, "--dry-run", "--max-sentences", "1234")
    assert result.returncode == 0, result.stderr
    assert option(stage_commands(result.stdout)["sample"], "--max-sentences") == "1234"
    result = run_script(tmp_path, "--dry-run", "--skip-generate", "--skip-sampling",
                        "--generation-dir", "results/presampled/generation")
    assert result.returncode == 0, result.stderr
    stages = stage_commands(result.stdout)
    assert "sample" not in stages
    for name in ("annotate", "forward"):
        assert option(stages[name], "--generation-dir") == "results/presampled/generation"


@pytest.mark.parametrize("model,parser,mtp,context,quantization,language_only", [
    ("Qwen/Qwen3.6-35B-A3B", "qwen3", True, "49152", "unsloth-4bit", True),
    ("Qwen/Qwen3.5-35B-A3B", "qwen3", True, "49152", "unsloth-4bit", True),
    ("Qwen/Qwen3-30B-A3B", "qwen3", False, "40960", "none", False),
    ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nemotron_v3", True, "49152", "none", False),
    ("google/gemma-4-26B-A4B-it", "gemma4", False, "49152", "none", True),
    ("nvidia/Gemma-4-26B-A4B-NVFP4", "gemma4", False, "49152", "none", True),
    ("zai-org/GLM-4.7-Flash", "glm45", True, "49152", "none", False),
    ("openai/gpt-oss-20b", "openai_gptoss", False, "49152", "none", False),
])
def test_supported_model_profiles_preserve_judge_and_fixed_probe_defaults(
    tmp_path, model, parser, mtp, context, quantization, language_only,
):
    result = run_script(tmp_path, "--model", model, "--dry-run")
    assert result.returncode == 0, result.stderr
    servers = [c for c in commands(result.stdout) if c[:2] == ["docker", "run"]]
    generation, judge = servers
    assert option(generation, "serve") == model
    assert option(generation, "--reasoning-parser") == parser
    assert ("--speculative-config" in generation) is mtp
    assert ("--language-model-only" in generation) is language_only
    assert option(generation, "--max-model-len") == context
    assert option(judge, "serve") == "unsloth/Qwen3.8-27B-NVFP4"
    assert option(judge, "--reasoning-parser") == "qwen3"
    assert "--speculative-config" in judge
    assert "--language-model-only" in judge
    assert "--enforce-eager" in judge
    stages = stage_commands(result.stdout)
    assert "--save-token-ids" in stages["generate"]
    assert option(stages["generate"], "--draft-model-id") == (model if mtp else "")
    assert option(stages["forward"], "--quantization") == quantization
    assert "--all-router-layers" not in stages["forward"]
    assert "--probe-results" not in stages["forward"]
    assert "no stages executed" in result.stdout
    assert "Stage 4/4 complete" not in result.stdout


def test_generation_overrides_leave_judge_settings_unchanged(tmp_path):
    result = run_script(
        tmp_path, "--model", "Qwen/Qwen3-30B-A3B", "--dry-run",
        "--ctx-size", "38000", "--generation-speculation", "none",
        "--generation-quantization", "compressed-tensors", "--cpu-offload-gb", "40",
        "--tensor-parallel-size", "2", "--quantization", "bnb-4bit",
        "--all-router-layers", CUDA_VISIBLE_DEVICES="0,1",
    )
    assert result.returncode == 0, result.stderr
    generation, judge = [c for c in commands(result.stdout) if c[:2] == ["docker", "run"]]
    assert option(generation, "--max-model-len") == "38000"
    assert option(generation, "--quantization") == "compressed-tensors"
    assert option(generation, "--cpu-offload-gb") == "40"
    assert option(generation, "--tensor-parallel-size") == "2"
    assert option(generation, "--gpus") == '"device=0,1"'
    assert "--quantization" not in judge
    assert "--cpu-offload-gb" not in judge
    assert "--tensor-parallel-size" not in judge
    assert "--speculative-config" in judge
    stages = stage_commands(result.stdout)
    assert "--all-router-layers" in stages["forward"]
    assert option(stages["forward"], "--quantization") == "bnb-4bit"


def test_gpt_oss_fixed_probe_mismatch_fails_before_any_server_start(tmp_path):
    runtime = fake_runtime(tmp_path)
    result = run_script(tmp_path, "--model", "openai/gpt-oss-20b", **runtime, FAIL_STAGE="none")
    assert result.returncode == 2
    assert "--all-router-layers" in result.stderr
    assert "24 layers" in result.stderr
    assert not (tmp_path / "events").exists()
    result = run_script(
        tmp_path, "--model", "openai/gpt-oss-20b", "--all-router-layers",
        **runtime, FAIL_STAGE="none",
    )
    assert result.returncode == 0, result.stderr
