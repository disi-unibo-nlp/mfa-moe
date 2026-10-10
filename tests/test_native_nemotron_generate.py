"""Static contracts for the native Nemotron benchmark-generation launchers.

These tests exercise the real shell files; they never start a server, a GPU
process or a Slurm job.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
SERVER_HELPER = REPO / "sbatch" / "native_nemotron_server.sh"
PROBE = REPO / "sbatch" / "native_nemotron_probe.sbatch"
GENERATE = REPO / "sbatch" / "native_nemotron_generate.sbatch"
MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
MODEL_REVISION = "bee7596271d1495f6992ae224aefde4410e816b8"
DRAFT = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"
DRAFT_REVISION = "8a0177116d138011e63103110f136ec0ca09ebbf"


def server_argv(*extra: str) -> list[str]:
    result = subprocess.run(
        ["bash", str(SERVER_HELPER), *extra],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


@pytest.mark.parametrize("script", [SERVER_HELPER, PROBE, GENERATE])
def test_scripts_parse(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_server_argv_pins_a100_compatibility_settings() -> None:
    argv = server_argv(
        "--port", "41800", "--download-dir", "/cache/hub", "--speculation", "none"
    )

    assert argv[0] == "serve"
    assert argv[1] == MODEL
    assert argv[argv.index("--revision") + 1] == MODEL_REVISION
    assert argv[argv.index("--attention-config") + 1] == '{"backend":"TRITON_ATTN"}'
    assert argv[argv.index("--kv-cache-dtype") + 1] == "bfloat16"
    assert argv[argv.index("--linear-backend") + 1] == "marlin"
    assert argv[argv.index("--reasoning-parser") + 1] == "nemotron_v3"
    assert argv[argv.index("--max-model-len") + 1] == "49152"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--mamba-backend") + 1] == "triton"
    assert argv[argv.index("--served-model-name") + 1] == MODEL
    assert "--speculative-config" not in argv
    # The checkpoint is ModelOpt MIXED_PRECISION; vLLM must auto-detect it.
    assert "--quantization" not in argv
    assert not any("docker" in arg.lower() for arg in argv)


def test_server_argv_enables_dspark_draft() -> None:
    argv = server_argv("--port", "41800", "--speculation", "dspark")

    config = json.loads(argv[argv.index("--speculative-config") + 1])
    assert config["method"] == "dspark"
    assert config["model"] == DRAFT
    assert config["revision"] == DRAFT_REVISION
    assert config["num_speculative_tokens"] == 3


def test_server_argv_omits_ssu_algorithm_for_triton() -> None:
    argv = server_argv("--port", "41800", "--mamba-backend", "triton")

    assert argv[argv.index("--mamba-backend") + 1] == "triton"
    assert "--mamba-ssu-algorithm" not in argv


def test_server_argv_supports_flashinfer_with_ssu_algorithm() -> None:
    argv = server_argv("--port", "41800", "--mamba-backend", "flashinfer")

    assert argv[argv.index("--mamba-backend") + 1] == "flashinfer"
    assert argv[argv.index("--mamba-ssu-algorithm") + 1] == "simple"


def test_server_argv_allows_backend_and_parallel_overrides() -> None:
    argv = server_argv(
        "--port", "41901", "--speculation", "dspark", "--mamba-backend", "triton",
        "--tensor-parallel-size", "1", "--speculative-tokens", "5",
        "--draft-attention-backend", "FLEX_ATTENTION",
    )

    assert argv[argv.index("--mamba-backend") + 1] == "triton"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
    config = json.loads(argv[argv.index("--speculative-config") + 1])
    assert config["num_speculative_tokens"] == 5
    assert config["attention_backend"] == "FLEX_ATTENTION"


@pytest.mark.parametrize(
    "extra",
    [
        ["--speculation", "eagle"],
        ["--mamba-backend", "cuda"],
        ["--port", "not-a-port"],
        ["--tensor-parallel-size", "0"],
        ["--speculative-tokens", "0"],
    ],
)
def test_server_argv_rejects_unsupported_values(extra: list[str]) -> None:
    result = subprocess.run(
        ["bash", str(SERVER_HELPER), *extra], capture_output=True, text=True
    )

    assert result.returncode == 2


def test_probe_launcher_contract() -> None:
    text = PROBE.read_text()

    assert "#SBATCH --qos=boost_qos_dbg" in text
    assert "#SBATCH --time=00:30:00" in text
    assert "#SBATCH --gres=gpu:2" in text
    assert "native_nemotron_server.sh" in text
    assert "moe_exp.correlation_pipeline.generate" in text
    assert "--speculation" in text
    assert "SMOKE_OK" in text
    assert "runtime execution requires Slurm" in text
    assert "docker" not in text.lower()


def test_generate_launcher_contract() -> None:
    text = GENERATE.read_text()

    assert "#SBATCH --qos=normal" in text
    assert "#SBATCH --time=24:00:00" in text
    assert "#SBATCH --gres=gpu:2" in text
    assert "native_nemotron_server.sh" in text
    assert "moe_exp.correlation_pipeline.generate" in text
    assert "TEMPERATURE=1.0" in text
    assert "TOP_P=0.95" in text
    assert "MAX_TOKENS=32768" in text
    assert "--save-token-ids" in text
    assert "math500,aime24,aime25,olympiad,amc23,minerva" in text
    assert "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16" in text
    assert "server_manifest.json" in text
    assert "refusing to resume" in text
    assert "INTRINSIC_PROBLEMS" in text
    assert "--job-label" in text
    assert "runtime execution requires Slurm" in text
    assert "docker" not in text.lower()


@pytest.mark.parametrize("script", [PROBE, GENERATE])
def test_launchers_run_the_vllm_console_script(script: Path) -> None:
    text = script.read_text()

    assert 'srun --ntasks=1 --nodes=1 "$VLLM" "${SERVER_ARGV[@]}"' in text
    assert 'mapfile -t SERVER_ARGV' in text


def test_default_server_argv_is_the_verified_a100_config() -> None:
    """The flags below are the measured winning configuration (job 58122776)."""
    argv = server_argv("--port", "41800", "--download-dir", "/cache/hub")

    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--max-num-seqs") + 1] == "8"
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "8192"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.90"
    assert argv[argv.index("--mamba-backend") + 1] == "triton"
    assert argv[argv.index("--mamba-cache-mode") + 1] == "align"
    assert "--mamba-ssu-algorithm" not in argv
    assert argv[argv.index("--attention-config") + 1] == '{"backend":"TRITON_ATTN"}'
    assert argv[argv.index("--linear-backend") + 1] == "marlin"
    assert argv[argv.index("--kv-cache-dtype") + 1] == "bfloat16"
    config = json.loads(argv[argv.index("--speculative-config") + 1])
    assert config == {
        "method": "dspark",
        "model": DRAFT,
        "revision": DRAFT_REVISION,
        "num_speculative_tokens": 3,
    }


def test_generate_defaults_are_the_verified_run() -> None:
    result = subprocess.run(
        ["bash", str(GENERATE), "--plan-only"], capture_output=True, text=True, cwd=REPO
    )

    assert result.returncode == 0, result.stderr
    plan = result.stdout
    assert "speculation=dspark mamba_backend=triton port=41800 workers=8" in plan
    assert "nemotron-nvfp4-dspark" in plan
    assert "--temperature 1.0" in plan
    assert "--top-p 0.95" in plan
    assert "--max-tokens 32768" in plan
    assert "--draft-model-id " + DRAFT in plan
    assert "--mamba-backend triton" in plan
    assert "--mamba-backend flashinfer" not in plan
    assert "max_items=none samples=default" in plan


def test_job_label_isolates_parallel_shard_provenance() -> None:
    shard = subprocess.run(
        ["bash", str(GENERATE), "--plan-only", "--job-label", "aime24", "--datasets", "aime24"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )

    assert shard.returncode == 0, shard.stderr
    assert "job_label=aime24" in shard.stdout
    assert "parallel/aime24" in shard.stdout
    assert "datasets=aime24 " in shard.stdout
    assert "--datasets aime24" in shard.stdout

    default = subprocess.run(
        ["bash", str(GENERATE), "--plan-only"], capture_output=True, text=True, cwd=REPO
    )
    assert default.returncode == 0, default.stderr
    assert "job_label=none" in default.stdout
    assert "parallel/" not in default.stdout


def test_job_label_is_validated() -> None:
    result = subprocess.run(
        ["bash", str(GENERATE), "--plan-only", "--job-label", "bad label"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )

    assert result.returncode == 2


def test_probe_defaults_are_the_verified_run() -> None:
    result = subprocess.run(
        ["bash", str(PROBE), "--plan-only"], capture_output=True, text=True, cwd=REPO
    )

    assert result.returncode == 0, result.stderr
    plan = result.stdout
    assert "speculation=dspark mamba_backend=triton port=41800" in plan
    assert "nemotron-nvfp4-probe/dspark" in plan
    assert "--mamba-backend triton" in plan
    assert "flashinfer" not in plan


@pytest.mark.parametrize("script", [PROBE, GENERATE])
def test_plan_only_previews_commands_off_slurm(script: Path) -> None:
    result = subprocess.run(
        ["bash", str(script), "--plan-only"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )

    assert result.returncode == 0, result.stderr
    assert "server_command=" in result.stdout
    assert "generate_command=" in result.stdout
    assert MODEL in result.stdout


def test_generate_plan_only_carries_six_benchmarks_and_dspark() -> None:
    result = subprocess.run(
        ["bash", str(GENERATE), "--plan-only", "--speculation", "dspark"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )

    assert result.returncode == 0, result.stderr
    for dataset in ("math500", "aime24", "aime25", "olympiad", "amc23", "minerva"):
        assert dataset in result.stdout
    assert "dspark" in result.stdout
    assert "nemotron-nvfp4-dspark" in result.stdout


def test_probe_plan_only_none_has_no_dspark_config() -> None:
    result = subprocess.run(
        ["bash", str(PROBE), "--plan-only", "--speculation", "none"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )

    assert result.returncode == 0, result.stderr
    assert "--speculative-config" not in result.stdout
    assert "nemotron-nvfp4-probe/none" in result.stdout
