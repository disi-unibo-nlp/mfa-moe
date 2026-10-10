"""Standalone stdlib assertions; run with the existing DSPy environment's python -B."""
from __future__ import annotations

import contextlib
import io
import json
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def test_smoke_payload() -> None:
    from moe_exp.correlation_pipeline import batch_judge, batch_predictor, mtp_probe_run

    launcher = REPO / "sbatch/qwen_mtp_probe.sbatch"
    text = launcher.read_text()
    original = subprocess.check_output(
        ["git", "show", "HEAD:sbatch/qwen_mtp_probe.sbatch"], cwd=REPO, text=True
    )
    expected_launcher = original.replace(
        '"num_speculative_tokens":1', '"num_speculative_tokens":3'
    ).replace(
        "--max-num-seqs 16", "--max-num-seqs 64"
    ).replace(
        "--batch-size 16", "--batch-size 64"
    ).replace(
        "    --max-tokens 4096 --temperature 0.0 \\\n",
        "    --max-tokens 4096 --temperature 1.0 \\\n"
        "    --top-p 0.95 --top-k 20 --min-p 0.0 \\\n"
        "    --presence-penalty 0.0 --repetition-penalty 1.0 \\\n",
    )
    assert text == expected_launcher, "unexpected launcher/server/runtime change"
    print("PASS: launcher differs only in requested sampling, MTP=3, and batch/server size=64")

    trace = REPO / (
        "results/correlation_pipeline/gpt-oss-20b/generation/"
        "openai--gpt-oss-20b/math500/traces.jsonl"
    )
    program = REPO / (
        "results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/"
        "selected_program_20260827_173300.json"
    )
    items = mtp_probe_run.build_probe_items(trace, 64)
    _, predict, adapter = batch_predictor.load_program_and_adapter(str(program))
    conversations = [adapter.format(predict.signature, predict.demos, item) for item in items]
    assert len(conversations) == 64
    sampling = {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0,
    }
    expected = {
        "model": "Qwen/Qwen3.8-27B", "messages": conversations, "max_tokens": 4096,
        **sampling,
        "chat_template_kwargs": {
            "enable_thinking": True, "reasoning_effort": "medium", "preserve_thinking": False,
        },
    }
    payload = batch_judge.build_batch_payload(
        conversations, model=expected["model"], max_tokens=4096,
        reasoning_effort="medium", **sampling,
    )
    assert payload == expected
    baseline = batch_judge.build_batch_payload(
        conversations, model=expected["model"], max_tokens=4096,
        temperature=0.0, reasoning_effort="medium",
    )
    assert baseline == {
        key: (0.0 if key == "temperature" else value)
        for key, value in expected.items() if key not in sampling or key == "temperature"
    }
    print("PASS: exact batch JSON, all six sampling fields including zeros, legacy omission")

    # Exercise the real CLI -> predictor -> payload -> urllib Request path.
    # Only the HTTP response is mocked; no inference or network access occurs.
    cli = text.split('"$PY" -m moe_exp.correlation_pipeline.mtp_probe_run', 1)[1]
    cli = cli.replace('"$TRACES"', shlex.quote(str(trace)))
    cli = cli.replace('"$PROGRAM_FILE"', shlex.quote(str(program)))
    cli = cli.replace('"$MODEL"', shlex.quote(expected["model"]))
    argv = ["mtp_probe_run", *shlex.split(cli.replace("\\\n", ""))]
    response = json.dumps({"choices": [
        {"index": i, "message": {"content": "[[ ## label ## ]]\nRead\n[[ ## completed ## ]]"}}
        for i in reversed(range(64))
    ]}).encode()
    output = io.StringIO()
    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
        with patch.object(batch_judge.urllib.request, "urlopen", return_value=io.BytesIO(response)) as post:
            mtp_probe_run.main()
    assert post.call_count == 1
    request = post.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:41800/v1/chat/completions/batch"
    assert request.method == "POST"
    assert post.call_args.kwargs == {"timeout": 600.0}
    assert json.loads(request.data) == expected
    summary = json.loads(output.getvalue().splitlines()[0])
    assert summary["batch_size"] == 64 and summary["labels"] == ["Read"] * 64
    assert summary["elapsed_seconds"] > 0
    assert summary["sentences_per_second"] == 64 / summary["elapsed_seconds"]
    assert "MTP_PROBE_OK batch_size=64 labels=64" in output.getvalue()
    print("PASS: launcher CLI to exact HTTP payload, one batch, real DSPy rendering/parsing")
    print("PASS: 64-label contract and instances/sec arithmetic (mock HTTP, not throughput)")


if __name__ == "__main__":
    # Match the launcher's existing overlay/base/source import order, without installation.
    scratch = Path("/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke")
    environment = Path((scratch / "envs/qwen-fp-fa-dspy-current.path").read_text().strip())
    sys.path[:0] = [
        str(environment / "lib/python3.11/site-packages"),
        str(REPO.parent / "envs/vllm-cu129/lib/python3.11/site-packages"),
        str(REPO / "src"),
    ]
    for relative in (
        "src/moe_exp/correlation_pipeline/batch_judge.py",
        "src/moe_exp/correlation_pipeline/batch_predictor.py",
        "src/moe_exp/correlation_pipeline/mtp_probe_run.py",
        "tests/test_batch_judge_client.py",
        "tests/test_mtp_probe_run.py",
    ):
        path = REPO / relative
        compile(path.read_text(), str(path), "exec")
    print("PASS: Python compile (in memory; no bytecode files)")
    test_smoke_payload()
