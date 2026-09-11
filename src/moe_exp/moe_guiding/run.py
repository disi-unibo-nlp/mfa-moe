from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import entry_points, version
from pathlib import Path

from moe_exp.jsonl import iter_jsonl
from moe_exp.moe_guiding.config import ARCHITECTURE, CONDITIONS, RoutingConfig, parse_layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Margin-triggered top-1 + least-1 MoE routing.")
    commands = parser.add_subparsers(dest="command", required=True)
    sanity = commands.add_parser("sanity", help="Run the two routing examples without vLLM.")
    sanity.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    generate = commands.add_parser(
        "generate", help="Generate one experimental condition with vLLM."
    )
    generate.add_argument(
        "--model", required=True, help="Unquantized native top-2 Mixtral checkpoint."
    )
    generate.add_argument("--revision", default=None, help="Prefer an immutable model commit SHA.")
    generate.add_argument(
        "--prompts", type=Path, required=True, help="JSONL objects with id and prompt."
    )
    generate.add_argument("--output-dir", type=Path, default=None)
    generate.add_argument("--condition", choices=CONDITIONS, default="selected")
    generate.add_argument("--threshold", type=float, default=0.4)
    generate.add_argument(
        "--layers", type=parse_layers, default=(12,), help="Zero-based: 12, 0,12, or all."
    )
    generate.add_argument(
        "--chat", action="store_true", help="Apply the checkpoint's chat template."
    )
    generate.add_argument("--max-tokens", type=int, default=512)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    generate.add_argument("--tensor-parallel-size", type=int, default=1)
    generate.add_argument("--max-model-len", type=int, default=4096)
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    generate.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum concurrent sequences; use 1 on small GPUs.",
    )
    generate.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum tokens processed per engine step.",
    )
    return parser


def sanity_check(device: str = "cpu") -> list[dict]:
    import torch

    from moe_exp.moe_guiding.routing import MarginRouter

    probabilities = torch.tensor(
        [[0.50, 0.25, 0.15, 0.10], [0.40, 0.25, 0.20, 0.15]], device=device
    )
    hidden_states = torch.zeros(2, 1, device=device)
    rows = []
    for condition in CONDITIONS:
        router = MarginRouter(RoutingConfig(condition=condition))
        weights, ids = router(hidden_states, probabilities.log(), 2, True)
        expected_ids = [[0, 1], [0, 1]] if condition == "baseline" else [[0, 3], [0, 1]]
        assert ids.tolist() == expected_ids
        first = [5 / 6, 1 / 6] if condition == "selected" else [2 / 3, 1 / 3]
        torch.testing.assert_close(weights, torch.tensor([first, [8 / 13, 5 / 13]], device=device))
        rows.append(
            {"condition": condition, "expert_ids": ids.tolist(), "weights": weights.tolist()}
        )
    return rows


def load_prompts(path: Path) -> list[dict]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError("The prompt file must contain at least one record.")
    ids = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Prompt record {index} must be an object.")
        if not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError(f"Prompt record {index} needs a nonempty string id.")
        if row["id"] in ids:
            raise ValueError(f"Duplicate prompt id: {row['id']}.")
        ids.add(row["id"])
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Prompt record {index} needs a nonempty prompt string.")
    return rows


def validate_diagnostics(reports: list[dict], condition: str) -> None:
    if not reports:
        raise RuntimeError("No routing diagnostics were returned by the inference workers.")
    for rank, report in enumerate(reports):
        if report["condition"] != condition:
            raise RuntimeError(f"Worker {rank} used the wrong experimental condition.")
        if condition == "baseline":
            if report["layers"]:
                raise RuntimeError(
                    f"Worker {rank} installed custom routing in the native baseline."
                )
            continue
        expected = {str(layer) for layer in report["selected_layers"]}
        if not expected or set(report["layers"]) != expected:
            raise RuntimeError(f"Worker {rank} is missing selected routing callbacks.")
        if any(stats["token_evaluations"] == 0 for stats in report["layers"].values()):
            raise RuntimeError(f"Worker {rank} did not execute custom routing during generation.")


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate(args: argparse.Namespace) -> Path:
    config = RoutingConfig(args.condition, args.threshold, args.layers)
    rows = load_prompts(args.prompts)
    if args.max_tokens < 1 or args.tensor_parallel_size < 1 or args.max_model_len < 1:
        raise ValueError("Token limits and tensor parallel size must be positive.")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("GPU memory utilization must be in (0, 1].")
    for name in ("max_num_seqs", "max_num_batched_tokens"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive.")
    plugins = entry_points(group="vllm.general_plugins")
    if not any(plugin.name == "moe_guiding" for plugin in plugins):
        raise RuntimeError("Install the project in every worker environment: pip install -e .")
    allowed_plugins = os.environ.get("VLLM_PLUGINS")
    if allowed_plugins is not None and "moe_guiding" not in allowed_plugins.split(","):
        raise RuntimeError("VLLM_PLUGINS excludes moe_guiding; add it or unset the variable.")

    import torch
    from vllm import LLM, SamplingParams

    from moe_exp.moe_guiding.integration import routing_diagnostics

    engine_args = {
        "model": args.model,
        "revision": args.revision,
        "tokenizer_revision": args.revision,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "enforce_eager": True,
        "moe_backend": "triton",
        "enable_prefix_caching": False,
        "generation_config": "vllm",
        "hf_overrides": {"architectures": [ARCHITECTURE]},
        "additional_config": {"moe_guiding": config.to_dict()},
    }
    for name in ("max_num_seqs", "max_num_batched_tokens"):
        if (value := getattr(args, name)) is not None:
            engine_args[name] = value
    sampling_args = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    sampling = SamplingParams(**sampling_args)
    output_dir = args.output_dir or Path("results/moe_guiding") / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    # An exclusive manifest prevents mixing different conditions or overwriting a prior run.
    manifest = {
        "experiment": "moe_guiding",
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "routing": config.to_dict(),
        "token_scope": "prefill_and_decode",
        "engine_args": engine_args,
        "sampling_args": sampling_args,
        "chat_template": args.chat,
        "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "num_prompts": len(rows),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "vllm": version("vllm"),
            "transformers": version("transformers"),
        },
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    try:
        llm = LLM(**engine_args)
        prompts = [row["prompt"] for row in rows]
        engine_prompts = prompts
        if args.chat:
            tokenizer = llm.get_tokenizer()
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
            # Chat templates already contain BOS/EOS markers. Passing text back
            # through default tokenization can add a second BOS token.
            engine_prompts = [
                {"prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False)}
                for prompt in prompts
            ]
        # Discard model profiling/warmup calls; verify callbacks on actual generation.
        llm.collective_rpc(routing_diagnostics, kwargs={"reset": True})
        outputs = llm.generate(engine_prompts, sampling, use_tqdm=True)
        reports = llm.collective_rpc(routing_diagnostics)
        manifest["routing_diagnostics"] = reports
        validate_diagnostics(reports, config.condition)
        if len(outputs) != len(rows):
            raise RuntimeError("vLLM returned a different number of outputs than input prompts.")
        with (output_dir / "generations.jsonl").open("x", encoding="utf-8") as handle:
            for row, prompt, output in zip(rows, prompts, outputs, strict=True):
                completion = output.outputs[0]
                record = {
                    "id": row["id"],
                    "input": row,
                    "rendered_prompt": prompt,
                    "condition": config.condition,
                    "text": completion.text,
                    "prompt_token_ids": output.prompt_token_ids,
                    "generated_token_ids": list(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _write_json(manifest_path, manifest)
    return output_dir


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "sanity":
        print(json.dumps(sanity_check(args.device), indent=2))
        return
    try:
        output_dir = generate(args)
    except (ValueError, RuntimeError, FileExistsError) as error:
        parser.exit(1, f"moe_guiding: {error}\n")
    print(f"Saved generations and routing diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
