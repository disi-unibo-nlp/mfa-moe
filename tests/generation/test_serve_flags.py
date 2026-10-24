"""The `vllm serve` command lines built by generation/serve_flags.py are the ones the eight per-model scripts started."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from generation.model_profiles import API_KEY, CARD_PROFILES, MFA
from generation.serve_flags import serve_argv
from models.roster import HF_IDS, JUDGE_MODEL

SRC = Path(__file__).resolve().parents[2] / "src"
PORT = "12345"
# The flags every server shares (the old scripts/serve/args.sh), with the port of this test.
SHARED = ["--host", "127.0.0.1", "--port", PORT, "--api-key", "local-vllm-key", "--tensor-parallel-size", "2",
          "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16", "--generation-config", "vllm",
          "--enable-prefix-caching", "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.90"]
MTP3 = '{"method":"mtp","num_speculative_tokens":3}'
# (model name, what came before the shared flags, what came after them): the arguments after `vllm serve` in
# scripts/serve/{gpt,nemotron,qwen35,qwen36,gemma,glm,qwen330b,judge}.sh, as they were before serve_model.sh.
OLD_SCRIPTS = {
    "openai/gpt-oss-20b": (
        ["openai/gpt-oss-20b", "--revision", "6cee5e81ee83917806bbde320786a8fb61efebee"],
        ["--reasoning-parser", "openai_gptoss", "--attention-config", '{"backend":"TRITON_ATTN"}',
         "--max-model-len", "131072", "--max-num-seqs", "16"]),
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": (
        ["nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "--revision", "bee7596271d1495f6992ae224aefde4410e816b8"],
        ["--reasoning-parser", "nemotron_v3", "--attention-config", '{"backend":"TRITON_ATTN"}',
         "--max-model-len", "133120", "--max-num-seqs", "16", "--linear-backend", "marlin", "--mamba-backend", "triton",
         "--mamba-cache-mode", "align", "--speculative-config",
         '{"method":"dspark","model":"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",'
         '"revision":"8a0177116d138011e63103110f136ec0ca09ebbf","num_speculative_tokens":3}']),
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4": (
        [f"{MFA}/cache/native-replay/Qwen3.5-35B-A3B-GPTQ-Int4/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b",
         "--served-model-name", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"],
        ["--reasoning-parser", "qwen3", "--attention-config", '{"backend":"FLASH_ATTN"}', "--language-model-only",
         "--max-model-len", "133120", "--max-num-seqs", "16", "--speculative-config", MTP3]),
    "Qwen/Qwen3.6-35B-A3B-FP8": (
        ["Qwen/Qwen3.6-35B-A3B-FP8", "--revision", "95a723d08a9490559dae23d0cff1d9466213d989"],
        ["--reasoning-parser", "qwen3", "--attention-config", '{"backend":"FLASH_ATTN"}', "--language-model-only",
         "--max-model-len", "133120", "--max-num-seqs", "16", "--speculative-config", MTP3]),
    "nvidia/Gemma-4-26B-A4B-NVFP4": (
        [f"{MFA}/cache/native-replay/Gemma-4-26B-A4B-NVFP4/a19cfe00be84568a6867111c9a68c9c44fdcffe6",
         "--served-model-name", "nvidia/Gemma-4-26B-A4B-NVFP4"],
        ["--reasoning-parser", "gemma4", "--language-model-only", "--max-model-len", "133120", "--max-num-seqs", "8",
         "--linear-backend", "marlin"]),
    "zai-org/GLM-4.7-Flash": (
        ["zai-org/GLM-4.7-Flash", "--revision", "7dd20894a642a0aa287e9827cb1a1f7f91386b67"],
        ["--reasoning-parser", "glm45", "--attention-config", '{"backend":"TRITON_MLA"}',
         "--max-model-len", "133120", "--max-num-seqs", "8"]),
    "Qwen/Qwen3-30B-A3B": (
        ["Qwen/Qwen3-30B-A3B", "--revision", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"],
        ["--reasoning-parser", "qwen3", "--attention-config", '{"backend":"FLASH_ATTN"}',
         "--max-model-len", "40960", "--max-num-seqs", "16"]),
    JUDGE_MODEL: (
        [JUDGE_MODEL, "--revision", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"],
        ["--reasoning-parser", "qwen3", "--attention-config", '{"backend":"FLASH_ATTN"}', "--language-model-only",
         "--enforce-eager", "--max-model-len", "49152", "--max-num-seqs", "64", "--speculative-config", MTP3]),
}


@pytest.mark.parametrize("model", sorted(OLD_SCRIPTS))
def test_the_command_line_is_the_one_the_old_script_started(model):
    before, after = OLD_SCRIPTS[model]
    assert serve_argv(model, PORT) == before + SHARED + after


def test_every_study_model_with_a_server_and_the_judge_are_pinned():
    assert {HF_IDS[key] for key in CARD_PROFILES} | {JUDGE_MODEL} == set(OLD_SCRIPTS)


def test_a_model_can_be_named_by_its_short_key():
    assert serve_argv("gpt", PORT) == serve_argv("openai/gpt-oss-20b", PORT)


def test_lfm25_has_no_server_and_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="no vLLM server"):
        serve_argv(HF_IDS["lfm25"], PORT)
    with pytest.raises(ValueError, match="unknown model"):
        serve_argv("openai/gpt-oss-120b", PORT)


@pytest.mark.parametrize("model", sorted(OLD_SCRIPTS))
def test_the_flags_agree_with_the_recorded_generation_contract(model):
    """What the trace contract records (model_profiles.py) is what the server was started with."""
    argv = serve_argv(model, PORT)
    flags = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    assert flags["--kv-cache-dtype"] == "bfloat16" and flags["--tensor-parallel-size"] == "2"
    assert flags["--api-key"] == API_KEY and flags["--port"] == PORT
    key = next((key for key in CARD_PROFILES if HF_IDS[key] == model), None)
    if key is None:  # the judge
        return
    card = CARD_PROFILES[key]
    assert flags["--reasoning-parser"] == card["parser"]
    assert int(flags["--max-model-len"]) == card["max_model_len"] and int(flags["--max-num-seqs"]) == card["max_num_seqs"]
    assert (json.loads(flags["--attention-config"])["backend"] if "--attention-config" in flags else None) == card["attention"]
    assert (json.loads(flags["--speculative-config"]) if "--speculative-config" in flags else None) == card["speculation"]
    assert ("--language-model-only" in argv) == card["language_model_only"]
    if card["weights"]:
        assert argv[0] == card["weights"] and Path(card["weights"]).name == card["revision"]
        assert flags["--served-model-name"] == card["model"]
    else:
        assert argv[0] == card["model"] and flags["--revision"] == card["revision"]


def test_the_command_prints_one_argument_per_line():
    run = subprocess.run([sys.executable, "-m", "generation.serve_flags", "openai/gpt-oss-20b"], capture_output=True,
                         text=True, env={**os.environ, "PYTHONPATH": str(SRC), "PORT": PORT})
    assert run.returncode == 0, run.stderr
    assert run.stdout.splitlines() == serve_argv("openai/gpt-oss-20b", PORT)
    no_port = subprocess.run([sys.executable, "-m", "generation.serve_flags", "openai/gpt-oss-20b"], capture_output=True,
                             text=True, env={k: v for k, v in os.environ.items() if k != "PORT"} | {"PYTHONPATH": str(SRC)})
    assert no_port.returncode != 0 and "PORT is not set" in no_port.stderr
