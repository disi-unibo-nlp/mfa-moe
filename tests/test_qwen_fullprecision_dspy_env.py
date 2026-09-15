from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_INPUT = ROOT / "requirements/qwen-fullprecision-smoke.in"
LOCK_FILE = ROOT / "requirements/qwen-fullprecision-smoke.lock"
BOOTSTRAP = ROOT / "scripts/bootstrap_qwen_fullprecision_env.sh"
LAUNCHER = ROOT / "sbatch/native_qwen_fullprecision_flash_attn_benchmark.sbatch"


def test_lock_input_declares_the_existing_project_constraints() -> None:
    text = LOCK_INPUT.read_text()
    assert "dspy>=3.0.0,<4" in text
    assert "gepa>=0.0.27" in text


def test_bootstrap_is_versioned_and_keeps_the_vllm_base_environment_read_only() -> None:
    text = BOOTSTRAP.read_text()
    assert "vllm-cu129/bin/python" in text
    assert "--system-site-packages" in text
    assert "--require-hashes" in text
    assert "qwen-fp-fa-dspy-" in text
    assert ".lock.sha256" in text
    assert "qwen-fp-fa-dspy-current.path" in text


def test_launcher_uses_the_overlay_only_for_the_annotation_python() -> None:
    text = LAUNCHER.read_text()
    command_block = text.split("VLLM_COMMAND=(", 1)[1].split(")", 1)[0]
    assert command_block.count("--max-num-seqs 16") == 1
    assert '--attention-config \'{"backend":"FLASH_ATTN"}\'' in command_block
    assert "--dtype bfloat16" in command_block
    assert "--kv-cache-dtype bfloat16" in command_block
    assert "dspy" in text
    assert "qwen-fp-fa-dspy-current.path" in text
    assert 'VLLM="$P/envs/vllm-cu129/bin/vllm"' in text
    assert "SMOKE_OK" in text
    assert "--mail-type=BEGIN,END,FAIL" in text
    assert "--mail-user=lorenzo.molfetta@unibo.it" in text
