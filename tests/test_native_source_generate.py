"""Contract tests for the parameterized BF16 source-model launchers.

Static only: every check either reads the script text or runs the argv helper /
--plan-only path, which never touch a GPU or Slurm.
"""
from __future__ import annotations

import subprocess
import os
from pathlib import Path

import pytest

SBATCH = Path(__file__).parents[1] / "sbatch"
SERVER = SBATCH / "native_source_server.sh"
GENERATE = SBATCH / "native_source_generate.sbatch"
PROBE = SBATCH / "native_source_probe.sbatch"
SOURCES = ("glm", "qwen330b")


def server_argv(*args: str) -> list[str]:
    result = subprocess.run(
        ["bash", str(SERVER), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()


def run_script(script: Path, *args: str) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=env)


def flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def test_scripts_parse() -> None:
    for script in (SERVER, GENERATE, PROBE):
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


@pytest.mark.parametrize(
    "source, model, revision, parser, backend, context",
    [
        ("glm", "zai-org/GLM-4.7-Flash", "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
         "glm45", "TRITON_MLA", "49152"),
        ("qwen330b", "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
         "qwen3", "FLASH_ATTN", "40960"),
    ],
)
def test_server_argv_pins_the_per_source_contract(
    source: str, model: str, revision: str, parser: str, backend: str, context: str
) -> None:
    argv = server_argv("--source", source, "--port", "41800")
    assert argv[0] == "serve" and argv[1] == model
    assert flag(argv, "--revision") == revision
    assert flag(argv, "--served-model-name") == model
    assert flag(argv, "--reasoning-parser") == parser
    assert flag(argv, "--attention-config") == '{"backend":"%s"}' % backend
    assert flag(argv, "--max-model-len") == context
    assert flag(argv, "--kv-cache-dtype") == "bfloat16"
    assert flag(argv, "--dtype") == "bfloat16"
    assert flag(argv, "--tensor-parallel-size") == "2"
    assert flag(argv, "--gpu-memory-utilization") == "0.90"
    # BF16 checkpoints: nothing to quantize, no Marlin layout, no hybrid SSM,
    # and neither model is multimodal.
    for absent in ("--quantization", "--linear-backend", "--mamba-backend",
                   "--language-model-only", "--speculative-config"):
        assert absent not in argv
    assert "docker" not in " ".join(argv).lower()


def test_glm_is_never_served_with_a_non_mla_backend_by_default() -> None:
    """vLLM routes glm4_moe_lite through MLA, where FLASH_ATTN is not valid."""
    argv = server_argv("--source", "glm", "--port", "41800")
    assert "FLASH_ATTN" not in " ".join(argv)


def test_empty_attention_backend_omits_the_flag() -> None:
    argv = server_argv("--source", "glm", "--port", "41800", "--attention-backend", "")
    assert "--attention-config" not in argv


def test_mtp_is_self_speculation_for_glm_and_refused_for_qwen() -> None:
    argv = server_argv("--source", "glm", "--port", "41800", "--speculation", "mtp")
    assert flag(argv, "--speculative-config") == '{"method":"mtp","num_speculative_tokens":3}'
    refused = run_script(SERVER, "--source", "qwen330b", "--port", "41800", "--speculation", "mtp")
    assert refused.returncode == 2 and "no MTP head" in refused.stderr


@pytest.mark.parametrize(
    "args, message",
    [
        (("--source", "bogus", "--port", "41800"), "--source must be"),
        (("--source", "glm", "--port", "0"), "--port must be"),
        (("--source", "glm", "--port", "41800", "--speculation", "dspark"), "--speculation must be"),
        (("--source", "qwen330b", "--port", "41800", "--max-model-len", "49152"), "native cap"),
    ],
)
def test_server_rejects_invalid_configurations(args: tuple[str, ...], message: str) -> None:
    result = run_script(SERVER, *args)
    assert result.returncode == 2 and message in result.stderr


def test_generate_launcher_contract() -> None:
    text = GENERATE.read_text()
    for value in (
        "#SBATCH --account=iscrc_miosr",
        "#SBATCH --partition=boost_usr_prod",
        "#SBATCH --qos=normal",
        "#SBATCH --gres=gpu:2",
        "#SBATCH --time=24:00:00",
        "#SBATCH --no-requeue",
        "native_source_server.sh",
        "mapfile -t SERVER_ARGV",
        'srun --ntasks=1 --nodes=1 "$VLLM" "${SERVER_ARGV[@]}"',
        "moe_exp.correlation_pipeline.generate",
        "--save-token-ids",
        "MAX_TOKENS=32768",
        "math500,aime24,aime25,olympiad,amc23,minerva",
        "server_manifest.json",
        "refusing to resume",
        "INTRINSIC_PROBLEMS",
        "--job-label",
        "runtime execution requires Slurm",
    ):
        assert value in text, value
    assert "docker" not in text.lower()
    # the shard's own provenance, not the shared generation root
    assert 'printf \'%s\\n\' "${SERVER_ARGV[@]}" >"$PROVENANCE_DIR/server_argv.txt"' in text
    assert '>"$GENERATION_DIR/server_argv.txt"' not in text
    # the verifier imports the attempt counts instead of duplicating them
    assert "from moe_exp.correlation_pipeline.benchmarks import BENCHMARKS" in text


def test_generate_launcher_keeps_the_existing_runs_untouched() -> None:
    text = GENERATE.read_text()
    for stale in ("nemotron-nvfp4", "dspark", "MAMBA_BACKEND"):
        assert stale not in text


@pytest.mark.parametrize("source", SOURCES)
def test_plan_only_previews_commands_off_slurm(source: str) -> None:
    result = run_script(GENERATE, "--source", source, "--plan-only")
    assert result.returncode == 0
    assert "server_command=" in result.stdout
    assert "generate_command=" in result.stdout
    assert f"source={source}" in result.stdout
    assert "--save-token-ids" in result.stdout


@pytest.mark.parametrize("source", SOURCES)
def test_plan_only_isolates_a_per_dataset_shard(source: str) -> None:
    result = run_script(GENERATE, "--source", source, "--job-label", "aime24", "--datasets", "aime24", "--plan-only")
    assert result.returncode == 0
    assert "job_label=aime24" in result.stdout
    assert f"parallel/aime24" in result.stdout


def test_generate_requires_a_known_source() -> None:
    assert run_script(GENERATE, "--plan-only").returncode == 2
    assert run_script(GENERATE, "--source", "nemotron", "--plan-only").returncode == 2


def test_probe_contract_gates_on_the_sampler_eligibility_predicate() -> None:
    text = PROBE.read_text()
    for value in (
        "#SBATCH --qos=boost_qos_dbg",
        "#SBATCH --time=00:30:00",
        "#SBATCH --gres=gpu:2",
        "native_source_server.sh",
        "SMOKE_OK",
        # the probe must use the sampler's own predicate, not a local copy
        "from moe_exp.correlation_pipeline.sample_stratified import reasoning_tokens",
        'assert verdict["length_metric_failures"] == 0',
        'assert verdict["min_reasoning_tokens"] >= 1',
        'assert verdict["distinct_reasoning_lengths"] >= 2',
        'assert verdict["reasoning_traces"] == verdict["traces"]',
    ):
        assert value in text, value


@pytest.mark.parametrize("source", SOURCES)
def test_probe_plan_only_has_no_speculative_config_by_default(source: str) -> None:
    result = run_script(PROBE, "--source", source, "--plan-only")
    assert result.returncode == 0
    assert "speculative-config" not in result.stdout


# Two 2-GPU jobs can be scheduled onto one 4-GPU node. Probe 58282929 failed
# exactly this way: it shared port 41800 with a concurrent GLM probe, passed
# /health against the sibling's server, and had every request rejected because
# the served model id did not match.


def test_each_source_probes_on_its_own_port() -> None:
    ports = {
        source: run_script(PROBE, "--source", source, "--plan-only").stdout
        for source in SOURCES
    }
    found = {source: [w for w in out.split() if w.startswith("port=")][0] for source, out in ports.items()}
    assert len(set(found.values())) == len(SOURCES), found
    assert "port=41800" not in found.values()


def test_parallel_shards_never_share_a_port() -> None:
    datasets = ("math500", "aime24", "aime25", "olympiad", "amc23", "minerva")
    seen: dict[str, tuple[str, str]] = {}
    for source in SOURCES:
        for dataset in datasets:
            out = run_script(
                GENERATE, "--source", source, "--job-label", dataset,
                "--datasets", dataset, "--plan-only",
            ).stdout
            url = next(w for w in out.split() if w.startswith("http://127.0.0.1:"))
            port = url.split(":")[2].split("/")[0]
            assert port not in seen, (port, seen[port], (source, dataset))
            seen[port] = (source, dataset)
    assert len(seen) == len(SOURCES) * len(datasets)


def test_shard_ports_are_stable_across_resubmission() -> None:
    """generate.py folds --base-url into generation_sha256, so a shard only
    resumes from its own cached traces if it is handed the same port again."""
    def port_of() -> str:
        out = run_script(
            GENERATE, "--source", "glm", "--job-label", "amc23",
            "--datasets", "amc23", "--plan-only",
        ).stdout
        return next(w for w in out.split() if w.startswith("http://127.0.0.1:"))
    assert port_of() == port_of()


@pytest.mark.parametrize("source", (*SOURCES, "qwen36"))
def test_forward_debug_launcher_plan_and_login_guard(source):
    launcher = SBATCH / "native_forward_debug.sbatch"
    result = run_script(launcher, source, "/unsubmitted/gate.json", "--plan-only")
    assert result.returncode == 0
    assert "srun" in result.stdout and "forward_debug run" in result.stdout
    assert "--cpu-gate" in result.stdout and source in result.stdout
    refused = run_script(launcher, source, "/unsubmitted/gate.json")
    assert refused.returncode == 2 and "Slurm" in refused.stderr
    assert run_script(SBATCH / "native_forward_cpu_tests.sbatch").returncode == 2


def test_forward_debug_launchers_have_explicit_budgets_and_parse():
    for name in ("native_forward_debug.sbatch", "native_forward_cpu_tests.sbatch"):
        path = SBATCH / name
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
        text = path.read_text()
        assert "#SBATCH --account=iscrc_miosr" in text
        assert "#SBATCH --qos=normal" in text
        assert "srun --ntasks=1" in text
        assert "HF_HUB_OFFLINE=1" in text
        assert "ulimit" not in text
    gpu = (SBATCH / "native_forward_debug.sbatch").read_text()
    assert "#SBATCH --gres=gpu:2" in gpu
    assert "#SBATCH --time=01:00:00" in gpu
    assert "HF_HUB_CACHE=" in gpu
    assert "#SBATCH --time=00:10:00" in (SBATCH / "native_forward_cpu_tests.sbatch").read_text()


def test_forward_panel_freezes_identity_and_replay_without_model_import(tmp_path):
    import json
    from moe_exp.correlation_pipeline.forward_debug import freeze_panel, SOURCES as specs, slug
    model = specs["qwen330b"][0]
    path = tmp_path / "source" / slug(model) / "math500/traces.jsonl"
    path.parent.mkdir(parents=True)
    rows = []
    for index in range(6):
        rows.append(dict(model_id=model, dataset="math500", problem_id=str(index),
                         sample_id=0, is_correct=bool(index % 2),
                         metadata={"finish_reason": "length" if index == 0 else "stop",
                                   "token_replay": {"schema_version": 1,
                                       "prompt_token_ids": [1], "completion_token_ids": [1, 2],
                                       "completion_offsets": [[0, 1], [1, 2]]}}))
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    path.with_name("manifest.json").write_text(json.dumps({"status": "complete", "target_model_id": model}))
    output = tmp_path / "panel"
    report = freeze_panel("qwen330b", tmp_path / "source", output)
    saved = [json.loads(line) for line in
             (output / "generation" / slug(model) / "math500/traces.jsonl").read_text().splitlines()]
    assert saved == rows[1:5]
    assert report["outcome_coverage"] == {"True": 2, "False": 2, "None": 0}
    assert freeze_panel("qwen330b", tmp_path / "source", output) == report
    rows[1]["problem_id"] = "changed"
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="differs"):
        freeze_panel("qwen330b", tmp_path / "source", output)


def test_forward_audit_numpy_hand_computed_oracle():
    import numpy as np
    from moe_exp.correlation_pipeline.forward_audit import numerical_oracle
    probabilities = np.array([[[.75,.25], [.75,.25], [.25,.75], [.25,.75]]])
    hidden = np.array([[[1.,0.], [1.,0.], [0.,1.], [0.,1.]]])
    experts = np.array([[[0], [0], [1], [1]]])
    metrics = numerical_oracle(np.log(probabilities), hidden, experts, [2], 4)
    expected = {"router_selected_mass": .75, "router_boundary_margin": .5,
                "router_switch_rate": 1/3, "router_topk_overlap": 2/3,
                "hidden_norm": 1., "hidden_step_distance": 1/3, "hidden_router_geometry": 1.}
    for name, value in expected.items():
        assert metrics[name+"_l02"] == pytest.approx(value)
    undefined = numerical_oracle(np.log(probabilities[:,:1]), hidden[:,:1], experts[:,:1], [2], 4)
    assert np.isnan(undefined["hidden_router_geometry_l02"])
    assert np.isnan(undefined["router_switch_rate_l02"])
