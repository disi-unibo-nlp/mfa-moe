from __future__ import annotations

import json
from pathlib import Path

from moe_exp.correlation_pipeline import native_qwen_smoke as smoke

REPO = Path(__file__).parents[1]
TRACE_FILE = REPO / "results/correlation_pipeline/gpt-oss-20b/generation/openai--gpt-oss-20b/math500/traces.jsonl"
PROGRAM_FILE = REPO / "results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json"
SBATCH_FILE = REPO / "sbatch" / "native_qwen_smoke.sbatch"


def test_load_cases_uses_real_traces_and_frozen_instructions():
    instructions, cases = smoke.load_smoke_cases(TRACE_FILE, PROGRAM_FILE)

    assert instructions
    assert len(cases) == 16
    assert all(case["dataset"] == "math500" for case in cases)
    assert all(case["sentence"] for case in cases)
    assert smoke.MODEL == "unsloth/Qwen3.8-27B-NVFP4"
    assert "gpt-oss" not in smoke.MODEL.lower()


def test_request_payload_is_qwen_high_effort_and_bounded():
    instructions, cases = smoke.load_smoke_cases(TRACE_FILE, PROGRAM_FILE)
    payload = smoke.build_payload(cases[0], instructions)

    assert payload["model"] == "unsloth/Qwen3.8-27B-NVFP4"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 4096
    assert payload["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "high",
        "preserve_thinking": False,
    }
    assert payload["messages"][0]["role"] == "user"
    assert cases[0]["sentence"] in payload["messages"][0]["content"]


def test_native_slurm_command_is_qwen_tp2_and_16_sequences():
    text = SBATCH_FILE.read_text()

    assert "unsloth/Qwen3.8-27B-NVFP4" in text
    assert "--tensor-parallel-size 2" in text
    assert "--linear-backend marlin" in text
    assert "--attention-config '{\"backend\":\"TRITON_ATTN\"}'" in text
    assert "--kv-cache-dtype bfloat16" in text
    assert "--max-num-seqs 16" in text
    assert "tag_slurm.sh" not in text
    assert "run_all.sh" not in text
    assert "run_docker.sh" not in text
    assert "docker" not in text.lower()


def test_result_contract_is_deterministic():
    result = smoke.result_payload(
        cases=[{"dataset": "math500", "problem_id": "p0"}] * 16,
        responses=[{"label": "Read", "content": "Read"}] * 16,
        elapsed_seconds=2.0,
    )

    assert result["status"] == "SMOKE_OK"
    assert result["model"] == "unsloth/Qwen3.8-27B-NVFP4"
    assert result["reasoning_effort"] == "high"
    assert result["workers"] == 16
    assert result["tensor_parallel_size"] == 2
    assert result["max_num_seqs"] == 16
    assert result["request_count"] == 16
    assert result["responses"] == 16
    assert result["requests_per_second"] == 8.0
    json.dumps(result)
