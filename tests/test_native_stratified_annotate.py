from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "sbatch/native_stratified_annotate.sbatch"


def test_stratified_launcher_pins_the_production_annotation_contract():
    text = SCRIPT.read_text()
    for value in (
        "#SBATCH --account=iscrc_miosr",
        "#SBATCH --partition=boost_usr_prod",
        "#SBATCH --qos=normal",
        "#SBATCH --gres=gpu:2",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=120G",
        "#SBATCH --time=12:00:00",
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
        "--part 0 --parts 1 --part-size 25000 --total 25000",
        "--batch-size 64 --max-tokens 16384 --temperature 1.0",
        "--top-p 0.95 --top-k 20 --min-p 0.0",
        "--reasoning-effort low",
        "results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json",
        "HF_HUB_OFFLINE=1",
    ):
        assert value in text
    assert "--kv-cache-dtype fp8" not in text
    assert "docker" not in text.lower()


def test_stratified_launcher_maps_each_source_to_its_own_roots_and_ports():
    text = SCRIPT.read_text()
    for value in (
        'STRAT_ROOT=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-stratified',
        "SOURCE_RESULTS=gpt-oss-20b",
        "SLUG=openai--gpt-oss-20b",
        "PORT_BASE=43000",
        "SOURCE_RESULTS=gemma-nvfp4-nf4",
        "SLUG=nvidia--Gemma-4-26B-A4B-NVFP4",
        "PORT_BASE=43100",
        "SOURCE_RESULTS=qwen36-35b-a3b-fp8",
        "SLUG=Qwen--Qwen3.6-35B-A3B-FP8",
        "PORT_BASE=43200",
        "SOURCE_RESULTS=nemotron-nvfp4-dspark",
        "SLUG=nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "PORT_BASE=43300",
        "SOURCE_RESULTS=glm-4.7-flash",
        "SLUG=zai-org--GLM-4.7-Flash",
        "PORT_BASE=43400",
        "SOURCE_RESULTS=qwen3-30b-a3b",
        "SLUG=Qwen--Qwen3-30B-A3B",
        "PORT_BASE=43500",
        'PARTS_ROOT="$REPO/results/correlation_pipeline/$SOURCE_RESULTS/reasoning-vllm-v1/$SAMPLING_DIR/parts"',
        'PRODUCTION_ROOT="$STRAT_ROOT/$SOURCE"',
        "EXPECTED_PORT=$((PORT_BASE + PART))",
        'TRACE_ROOT="$PARTS_ROOT/part-$(printf \'%02d\' "$PART")/generation/$SLUG"',
        'test -s "$PARTS_ROOT/part-$(printf \'%02d\' "$PART")/part_manifest.json"',
        "--source must be gpt, gemma, qwen36, nemotron, glm or qwen330b",
        "port mismatch",
        "output-dir must be",
        "qwen38-mtp3-stratified/logs/%x-%j.out",
    ):
        assert value in text
    for dataset in ("math500", "aime24", "aime25", "olympiad", "amc23", "minerva"):
        assert f'test -s "$TRACE_ROOT/{dataset}/traces.jsonl"' in text
    assert "qwen38-mtp3-production/" not in text


def test_stratified_launcher_stages_code_and_uses_the_watchdog():
    text = SCRIPT.read_text()
    for value in (
        'CODE_ROOT="${CODE_ROOT:-$REPO}"',
        'export PYTHONPATH="$CODE_ROOT/src:',
        'WATCHDOG_SCRIPT="${WATCHDOG_SCRIPT:-$REPO/sbatch/qwen_annotation_watchdog.sh}"',
        "MFA_FAULTHANDLER=1",
        '"$WATCHDOG_SCRIPT"',
        "--stall-seconds",
        "--max-restarts",
        '"${client_command[@]}"',
    ):
        assert value in text


def test_stratified_launcher_refuses_login_node_execution_and_verifies_the_part():
    text = SCRIPT.read_text()
    assert "requires Slurm" in text
    assert "SLURM_JOB_ID" in text
    assert 'test -s "$OUTPUT_DIR/annotations.json"' in text
    assert 'test -s "$OUTPUT_DIR/summary.json"' in text
    assert '"completed"] == summary["expected"] == 25000' in text


def test_stratified_launcher_resolves_the_sampling_dir_per_source():
    """gpt reads the v2 sample; the harmony path reports a constant reasoning_tokens=0."""
    text = SCRIPT.read_text()
    assert "SAMPLING_DIR=sampling-100k-stratified\n" in text
    assert "SAMPLING_DIR=sampling-100k-stratified-v2" in text
    assert 'reasoning-vllm-v1/$SAMPLING_DIR/parts' in text
