"""Check the Slurm spool entry point and tagging-only execution plan."""

import os
from pathlib import Path
import shutil
import subprocess

from test_correlation_launcher import commands, option, stage_commands


SOURCE = Path(__file__).parents[1] / "src/moe_exp/correlation_pipeline"


def workspace(tmp_path):
    launcher = tmp_path / "src/moe_exp/correlation_pipeline"
    launcher.mkdir(parents=True)
    for name in ("run_all.sh", "model_profiles.py"):
        shutil.copy2(SOURCE / name, launcher / name)
    traces = tmp_path / "saved/org--custom-model/math500/traces.jsonl"
    traces.parent.mkdir(parents=True)
    traces.write_text("{}\n")  # Dry-run validates presence without loading model data.
    spool = tmp_path / "slurm_script"
    shutil.copy2(SOURCE / "tag_slurm.sh", spool)
    return spool


def launch(tmp_path, spool, *args):
    env = {k: v for k, v in os.environ.items() if k not in (
        "PHYS_DIR", "DRY_RUN", "GENERATION_MODEL", "RESULTS_DIR", "JUDGE_PROGRAM",
    )}
    env.update(SLURM_SUBMIT_DIR=str(tmp_path), SLURM_JOB_ID="123",
               SLURM_JOB_GPUS="2", CUDA_VISIBLE_DEVICES="0")
    return subprocess.run(
        ["bash", str(spool), "--model", "org/custom-model",
         "--generation-dir", "saved", "--dry-run", *args],
        cwd="/tmp", env=env, capture_output=True, text=True, timeout=20,
    )


def test_spool_copy_tags_only_saved_model_on_allocated_gpu(tmp_path):
    result = launch(tmp_path, workspace(tmp_path))
    assert result.returncode == 0, result.stderr
    stages = stage_commands(result.stdout)
    assert list(stages) == ["annotate"]
    assert option(stages["annotate"], "--generation-model") == "org/custom-model"
    assert option(stages["annotate"], "--generation-dir") == "saved"
    assert option(stages["annotate"], "--datasets") == "math500"
    assert option(stages["annotate"], "--reasoning-effort") == "high"
    servers = [c for c in commands(result.stdout) if c[:2] == ["docker", "run"]]
    assert len(servers) == 1
    assert "vllm/vllm-openai:v0.29.0" in servers[0]
    assert option(servers[0], "--gpus") == '"device=2"'
    assert option(servers[0], "--tensor-parallel-size") == "1"
    assert "--speculative-config" not in servers[0]
    assert not (tmp_path / "results").exists()


def test_missing_dataset_fails_before_server_launch(tmp_path):
    result = launch(tmp_path, workspace(tmp_path), "--datasets", "aime24")
    assert result.returncode == 2
    assert "Missing or invalid dataset" in result.stderr
    assert "docker run" not in result.stdout


def test_cannot_enable_other_pipeline_stages(tmp_path):
    result = launch(tmp_path, workspace(tmp_path), "--skip-tagging")
    assert result.returncode == 2
    assert "Unknown option" in result.stderr


def test_output_and_worker_overrides_reach_annotation(tmp_path):
    result = launch(tmp_path, workspace(tmp_path), "--results-dir", "results/tag-test",
                    "--judge-workers", "4", "--limit", "2")
    assert result.returncode == 0, result.stderr
    annotate = stage_commands(result.stdout)["annotate"]
    assert option(annotate, "--output-dir") == "results/tag-test/reasoning-vllm-v1/annotations"
    assert option(annotate, "--workers") == "4"
    assert option(annotate, "--limit") == "2"


def test_judge_tensor_parallel_override_reaches_judge_server(tmp_path):
    result = launch(tmp_path, workspace(tmp_path),
                    "--judge-workers", "16", "--judge-tensor-parallel-size", "2")
    assert result.returncode == 0, result.stderr
    servers = [c for c in commands(result.stdout) if c[:2] == ["docker", "run"]]
    assert len(servers) == 1
    assert option(servers[0], "--tensor-parallel-size") == "2"
    assert option(servers[0], "--max-num-seqs") == "16"


def test_reasoning_effort_override_reaches_annotation(tmp_path):
    result = launch(
        tmp_path,
        workspace(tmp_path),
        "--judge-reasoning-effort",
        "medium",
    )
    assert result.returncode == 0, result.stderr
    annotate = stage_commands(result.stdout)["annotate"]
    assert option(annotate, "--reasoning-effort") == "medium"
