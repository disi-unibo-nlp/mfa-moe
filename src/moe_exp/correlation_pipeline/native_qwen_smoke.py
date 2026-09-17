"""Native CINECA smoke client for frozen Qwen sentence-label prompts."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MODEL = "unsloth/Qwen3.8-27B-NVFP4"
REASONING_EFFORT = "high"
WORKERS = 16
TENSOR_PARALLEL_SIZE = 2
MAX_NUM_SEQS = 16
LABEL_RE = re.compile(r"\b(Read|Analyze|Plan|Implement|Explore|Verify|Monitor)\b", re.IGNORECASE)


def _first_nonempty(values: Any) -> str:
    if isinstance(values, list):
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
    return ""


def _sentence(row: dict[str, Any]) -> str:
    sentence = _first_nonempty(row.get("steps"))
    if sentence:
        return sentence
    for line in str(row.get("cot_text") or "").splitlines():
        if line.strip():
            return line.strip()
    raise ValueError(f"trace {row.get('problem_id', '<unknown>')} has no smoke sentence")


def load_smoke_cases(
    trace_file: Path, judge_program: Path, *, limit: int = WORKERS
) -> tuple[str, list[dict[str, str]]]:
    program = json.loads(judge_program.read_text(encoding="utf-8"))
    instructions = str(
        program.get("classify", {}).get("signature", {}).get("instructions") or ""
    ).strip()
    if not instructions:
        raise ValueError(f"frozen GEPA program has no classifier instructions: {judge_program}")

    cases: list[dict[str, str]] = []
    with trace_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if len(cases) >= limit:
                break
            if not line.strip():
                raise ValueError(f"blank trace row at line {line_number}")
            row = json.loads(line)
            if row.get("dataset") != "math500":
                raise ValueError(f"expected math500 trace, got {row.get('dataset')!r}")
            problem_id = str(row.get("problem_id") or "").strip()
            problem = str(row.get("prompt") or "").strip()
            if not problem_id or not problem:
                raise ValueError(f"trace row {line_number} lacks problem_id or prompt")
            raw_steps = row.get("steps") if isinstance(row.get("steps"), list) else []
            steps = [str(value).strip() for value in raw_steps if str(value or "").strip()]
            sentence = steps[0] if steps else _sentence(row)
            previous = "<START OF RESPONSE>"
            next_sentence = steps[1] if len(steps) > 1 else "<END OF RESPONSE>"
            cases.append(
                {
                    "dataset": "math500",
                    "problem_id": problem_id,
                    "problem_statement": problem,
                    "previous_sentence": previous,
                    "sentence": sentence,
                    "next_sentence": next_sentence,
                }
            )
    if len(cases) != limit:
        raise ValueError(f"expected {limit} real traces, found {len(cases)} in {trace_file}")
    return instructions, cases


def build_payload(case: dict[str, str], instructions: str) -> dict[str, Any]:
    content = "\n\n".join(
        (
            instructions,
            f"PROBLEM STATEMENT:\n{case['problem_statement']}",
            f"PREVIOUS SENTENCE:\n{case['previous_sentence']}",
            f"TARGET SENTENCE:\n{case['sentence']}",
            f"NEXT SENTENCE:\n{case['next_sentence']}",
            "Return exactly one label from Read, Analyze, Plan, Implement, Explore, Verify, Monitor.",
        )
    )
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 4096,
        "stream": False,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": REASONING_EFFORT,
                "preserve_thinking": False,
            }
        },
    }


def _post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        if response.status != 200:
            raise RuntimeError(f"judge returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _response(case: dict[str, str], instructions: str, base_url: str, api_key: str) -> dict[str, Any]:
    payload = build_payload(case, instructions)
    response = _post_json(base_url, api_key, payload)
    try:
        content = str(response["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"judge response has no assistant content: {response!r}") from exc
    match = LABEL_RE.search(content)
    if not content or match is None:
        raise ValueError(f"judge response has no allowed label: {content!r}")
    return {"dataset": case["dataset"], "problem_id": case["problem_id"], "label": match.group(1), "content": content}


def result_payload(
    cases: list[dict[str, str]], responses: list[dict[str, Any]], elapsed_seconds: float
) -> dict[str, Any]:
    if len(cases) != WORKERS or len(responses) != WORKERS:
        raise ValueError(f"expected {WORKERS} cases and responses")
    if any(not item.get("content") for item in responses):
        raise ValueError("all smoke responses must be non-empty")
    elapsed = float(elapsed_seconds)
    if elapsed <= 0:
        raise ValueError("elapsed_seconds must be positive")
    return {
        "status": "SMOKE_OK",
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "workers": WORKERS,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "max_num_seqs": MAX_NUM_SEQS,
        "request_count": len(cases),
        "responses": len(responses),
        "elapsed_seconds": elapsed,
        "requests_per_second": len(responses) / elapsed,
        "problem_ids": [case["problem_id"] for case in cases],
        "labels": [item["label"] for item in responses],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--judge-program", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reasoning-effort", default=REASONING_EFFORT)
    parser.add_argument("--base-url", default="http://127.0.0.1:41800/v1")
    parser.add_argument("--api-key", default="local-vllm-key")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE)
    parser.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if (args.model, args.reasoning_effort, args.workers, args.tensor_parallel_size, args.max_num_seqs) != (
        MODEL, REASONING_EFFORT, WORKERS, TENSOR_PARALLEL_SIZE, MAX_NUM_SEQS
    ):
        raise ValueError("native smoke requires Qwen, high effort, TP=2, and 16 sequences/workers")
    instructions, cases = load_smoke_cases(args.trace_file, args.judge_program)
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_OK", "model": MODEL, "traces": len(cases), "workers": WORKERS, "tensor_parallel_size": TENSOR_PARALLEL_SIZE, "max_num_seqs": MAX_NUM_SEQS}))
        return
    if args.output is None:
        raise ValueError("--output is required unless --preflight is used")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_response, case, instructions, args.base_url, args.api_key) for case in cases]
        responses = [future.result() for future in futures]
    result = result_payload(cases, responses, time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"SMOKE_OK model={MODEL} requests={WORKERS} reasoning_effort={REASONING_EFFORT} "
        f"requests_per_second={result['requests_per_second']:.3f} output={args.output}"
    )


if __name__ == "__main__":
    main()
