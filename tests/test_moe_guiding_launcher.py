import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "src/moe_exp/moe_guiding/run_rtx3070.sh"
DOCKER_LAUNCHER = LAUNCHER.with_name("run_docker.sh")


@pytest.mark.parametrize("condition", ["baseline", "selected", "transfer"])
def test_dry_run_uses_container_and_preserves_arguments(tmp_path, condition):
    workspace = tmp_path / "workspace with spaces"
    cache = tmp_path / "cache"
    env = dict(os.environ, DRY_RUN="true", PHYS_DIR=str(workspace), HF_CACHE_DIR=str(cache))
    result = subprocess.run(
        ["bash", str(LAUNCHER), condition, "--max-tokens", "8"],
        env=env, text=True, capture_output=True, check=True,
    )
    assert "--entrypoint python3" in result.stdout
    assert f"--condition {condition}" in result.stdout
    assert "--max-tokens 8" in result.stdout
    assert "VLLM_PLUGINS=moe_guiding" in result.stdout
    assert "--gpus" in result.stdout
    assert not workspace.exists() and not cache.exists()


def test_docker_launcher_maps_rootless_user_and_builds_missing_image(tmp_path):
    docker = tmp_path / "docker"
    log = tmp_path / "docker.log"
    docker.write_text(
        '#!/bin/bash\n'
        'printf "%s\\n" "$*" >> "$DOCKER_LOG"\n'
        'case "$1" in\n'
        '  info) echo \'["name=rootless"]\' ;;\n'
        '  image) exit 1 ;;\n'
        'esac\n'
    )
    docker.chmod(0o755)
    env = dict(
        os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
        DOCKER_LOG=str(log), PHYS_DIR=str(tmp_path / "workspace"),
        HF_CACHE_DIR=str(tmp_path / "cache"), DRY_RUN="false",
    )
    subprocess.run(["bash", str(DOCKER_LAUNCHER), "sanity"], env=env, check=True)
    calls = log.read_text().splitlines()
    assert sum(call.startswith("build ") for call in calls) == 1
    assert calls[-1].startswith("run --rm --user 0:0")
    assert "--gpus" not in calls[-1]
    assert calls[-1].endswith("-m moe_exp.moe_guiding.run sanity")


@pytest.mark.parametrize("arguments", [["--device", "cuda"], ["--device=cuda"]])
def test_cuda_sanity_requests_gpu(arguments):
    result = subprocess.run(
        ["bash", str(DOCKER_LAUNCHER), "sanity", *arguments],
        env=dict(os.environ, DRY_RUN="true"), text=True, capture_output=True, check=True,
    )
    assert "--gpus" in result.stdout
