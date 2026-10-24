"""The `vllm serve` command line of every model server, built from one table.

A model's own flags (revision or local weights, reasoning parser, attention backend, context, speculation) are the
server settings of its card profile in model_profiles.py, the same values generate.py records in every trace's
contract; the judge's are below. The flags every server shares come from SHARED. scripts/serve/serve_model.sh runs
what this prints:

    PORT=20123 python -m generation.serve_flags openai/gpt-oss-20b     one argument per line

Why the A100 needs these settings is in scripts/serve/README.md.
"""
import argparse
import json
import os

from generation.model_profiles import API_KEY, CARD_PROFILES
from models.roster import JUDGE_MODEL, resolve
from store import require

# The judge: Qwen3.8-27B with 3-token MTP speculation, eager mode and 64 concurrent sequences.
JUDGE_SERVER = dict(
    model=JUDGE_MODEL, revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", weights=None, parser="qwen3",
    attention="FLASH_ATTN", speculation={"method": "mtp", "num_speculative_tokens": 3}, language_model_only=True,
    enforce_eager=True, max_model_len=49_152, max_num_seqs=64, extra_args=[])


def shared_flags(port: str) -> list[str]:
    """Two GPUs, bf16 weights and KV cache (the A100 has no fp8 KV cache), prefix caching, local address only."""
    return ["--host", "127.0.0.1", "--port", str(port), "--api-key", API_KEY, "--tensor-parallel-size", "2",
            "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16", "--generation-config", "vllm",
            "--enable-prefix-caching", "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.90"]


def compact(value: dict) -> str:
    return json.dumps(value, separators=(",", ":"))


def serve_argv(model: str, port: str) -> list[str]:
    """The arguments after `vllm serve` for a model given as its Hugging Face id (or short key) and the port."""
    if model == JUDGE_MODEL:
        server = JUDGE_SERVER
    else:
        key = resolve(model)
        require(key in CARD_PROFILES, f"{model} has no vLLM server: its engine is part of src/steering")
        server = CARD_PROFILES[key]
    # A local copy is named after its revision and served under the Hugging Face id.
    argv = [server["weights"], "--served-model-name", server["model"]] if server["weights"] else [
        server["model"], "--revision", server["revision"]]
    argv += shared_flags(port) + ["--reasoning-parser", server["parser"]]
    if server["attention"]:
        argv += ["--attention-config", compact({"backend": server["attention"]})]
    if server["language_model_only"]:
        argv.append("--language-model-only")
    if server.get("enforce_eager"):
        argv.append("--enforce-eager")
    argv += ["--max-model-len", str(server["max_model_len"]), "--max-num-seqs", str(server["max_num_seqs"]),
             *server["extra_args"]]
    if server["speculation"]:
        argv += ["--speculative-config", compact(server["speculation"])]
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(prog="generation.serve_flags", description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="Hugging Face id, for example openai/gpt-oss-20b")
    args = parser.parse_args()
    require("PORT" in os.environ, "PORT is not set: source scripts/slurm/env.sh inside a Slurm job")
    print("\n".join(serve_argv(args.model, os.environ["PORT"])))


if __name__ == "__main__":
    main()
