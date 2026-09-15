import re
from pathlib import Path


LAUNCHER = Path("sbatch/native_qwen_fullprecision_flash_attn_benchmark.sbatch")


def test_fullprecision_flash_attn_benchmark_launcher_contract() -> None:
    text = LAUNCHER.read_text()
    match = re.search(r"VLLM_COMMAND=\(\n(?P<body>.*?)\n\)", text, re.DOTALL)
    assert match is not None
    vllm_command = match.group("body")

    assert "#SBATCH --gres=gpu:2" in text
    assert "#SBATCH --mail-type=BEGIN,END,FAIL" in text
    assert "#SBATCH --mail-user=lorenzo.molfetta@unibo.it" in text
    assert "MODEL=Qwen/Qwen3.8-27B" in text
    assert "--tensor-parallel-size 2" in vllm_command
    assert "--attention-config '{\"backend\":\"FLASH_ATTN\"}'" in vllm_command
    assert "--kv-cache-dtype bfloat16" in vllm_command
    assert "--dtype bfloat16" in vllm_command
    assert vllm_command.count("--max-num-seqs 16") == 1
    assert "--workers 16" in text
    assert "--reasoning-effort high" in text
    assert "--enable-thinking" in text
    assert "--language-model-only" in vllm_command
    assert "--judge-program" in text
    assert "selected_program_20260827_173300.json" in text
    assert "math500 aime24 aime25 olympiad amc23 minerva" in text
    assert "SAMPLE_LIMIT=8" in text
    assert "SCRATCH_ROOT=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke" in text
    assert 'HF_HOME="$SCRATCH_ROOT/hf"' in text
    assert 'HF_CACHE_DIR="$HF_HOME/hub"' in text
    assert 'TMPDIR="$RUNTIME_ROOT/tmp"' in text
    assert "#SBATCH --output=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/logs/" in text
    assert "#SBATCH --error=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/logs/" in text
    assert '--download-dir "$HF_CACHE_DIR"' in vllm_command
    assert 'RUN_ROOT="$SCRATCH_ROOT/benchmarks/' in text
    assert 'HF_HOME="$P/cache/hf"' not in text
    assert 'TMPDIR="$P/cache/tmp"' not in text
    assert 'ENV_POINTER="$SCRATCH_ROOT/envs/qwen-fp-fa-dspy-current.path"' in text
    assert 'BASE_SITE="$P/envs/vllm-cu129/lib/python3.11/site-packages"' in text
    assert 'PY="$ENV_ROOT/bin/python"' in text
    assert 'VLLM="$P/envs/vllm-cu129/bin/vllm"' in text
    assert 'export PYTHONPATH="$ENV_ROOT/lib/python3.11/site-packages:$BASE_SITE:$P/repo/src"' in text
    assert "judge_env_ready" in text
    assert '"requests": units' in text
    assert '"responses": units' in text
    assert "SMOKE_OK" in text

    # The full-precision benchmark must not silently become the NVFP4 path.
    assert "--quantization" not in text
    assert "--linear-backend marlin" not in text
    assert "TRITON_ATTN" not in vllm_command
    assert "FLASHINFER" not in vllm_command

    # Preserve the operational constraints.
    assert "docker" not in text.lower()
    assert "StrictHostKeyChecking=no" not in text
    assert "pip install" not in text
