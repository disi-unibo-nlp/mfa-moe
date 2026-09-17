from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "sbatch/native_qwen_annotate.sbatch"


def test_native_launcher_pins_the_production_annotation_contract():
    text = SCRIPT.read_text()
    for value in (
        "#SBATCH --account=iscrc_miosr",
        "#SBATCH --partition=boost_usr_prod",
        "#SBATCH --qos=normal",
        "#SBATCH --gres=gpu:2",
        "#SBATCH --time=24:00:00",
        "#SBATCH --no-requeue",
        "#SBATCH --mail-type=BEGIN,END,FAIL",
        "#SBATCH --mail-user=lorenzo.molfetta@unibo.it",
        "--max-model-len 49152",
        "--max-num-seqs 64",
        "--tensor-parallel-size 2",
        "--attention-config '{\"backend\":\"FLASH_ATTN\"}'",
        "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'",
        "--kv-cache-dtype bfloat16",
        "moe_exp.correlation_pipeline.annotation_batch",
        "--parts 4 --part-size 25000 --total 100000",
        "--batch-size 64 --max-tokens 16384",
        "--reasoning-effort low",
        "results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json",
    ):
        assert value in text
    assert "--kv-cache-dtype fp8" not in text
    assert "docker" not in text.lower()
    assert "#SBATCH --time=07:00:00" not in text
    assert "--max-model-len 32768" not in text


def test_native_launcher_stages_code_without_moving_data_paths():
    text = SCRIPT.read_text()
    for value in (
        'CODE_ROOT="${CODE_ROOT:-$REPO}"',
        'export PYTHONPATH="$CODE_ROOT/src:',
        'WATCHDOG_SCRIPT="${WATCHDOG_SCRIPT:-$REPO/sbatch/qwen_annotation_watchdog.sh}"',
        'test -s "$CODE_ROOT/src/moe_exp/correlation_pipeline/annotation_batch.py"',
        'results/correlation_pipeline/gpt-oss-20b/generation/openai--gpt-oss-20b',
        'results/correlation_pipeline/gemma-nvfp4-nf4/reasoning-vllm-v1/sampling/generation/nvidia--Gemma-4-26B-A4B-NVFP4',
    ):
        assert value in text


def test_native_launcher_uses_the_stalled_write_watchdog_without_requeue():
    text = SCRIPT.read_text()
    for value in (
        "MFA_FAULTHANDLER=1",
        '"$WATCHDOG_SCRIPT"',
        "--stall-seconds",
        "--poll-seconds",
        "--max-restarts",
        "--output-dir \"$OUTPUT_DIR\"",
        "--client-log \"$CLIENT_LOG\"",
        '"${client_command[@]}"',
    ):
        assert value in text


def test_native_launcher_refuses_login_node_execution():
    text = SCRIPT.read_text()
    assert "requires Slurm" in text
    assert "SLURM_JOB_ID" in text


def test_native_launcher_verifies_the_finished_part():
    text = SCRIPT.read_text()
    assert 'test -s "$OUTPUT_DIR/annotations.json"' in text
    assert 'test -s "$OUTPUT_DIR/summary.json"' in text
    assert '"completed"] == summary["expected"] == 25000' in text
