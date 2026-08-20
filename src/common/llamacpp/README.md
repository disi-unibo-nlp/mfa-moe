# llama.cpp handoff

Minimal local llama.cpp serving and generation handoff.

## 1. Build the Docker image

Run from this directory:

```bash
docker build -t llama.cpp:localcuda .
```

## 2. Launch one model server

For the local `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` under `/llms`:

```bash
./serve_llamacpp.sh
```

For the recommended correlation run, launch Unsloth Qwen3.5 with integrated
native MTP:

```bash
./serve_qwen3_5_35b_a3b_mtp.sh
```

This downloads `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` resumably from
`unsloth/Qwen3.5-35B-A3B-MTP-GGUF` into `/llms`, then uses llama.cpp's
`draft-mtp` backend. MTP is embedded in the single GGUF, so no separate draft
model is loaded. Current llama.cpp MTP support requires one parallel server
slot.

The earlier Qwen3.6 DFlash launcher remains available:

```bash
./serve_qwen3_6_35b_a3b_dflash.sh
```

This uses llama.cpp's native `draft-dflash` backend. If either GGUF is missing,
the launcher downloads it resumably into the same shared `/llms` folder used by
the other experiments: the target comes from
`unsloth/Qwen3.6-35B-A3B-GGUF`, and the Q8_0 0.4B draft comes from
`Alittlehammmer/Qwen3.6-35B-A3B-DFlash-GGUF-llama.cpp`. Interrupted downloads
remain as `.part` files and resume on the next run. DFlash is not standalone:
the target verifies every proposed token. The server advertises the stable API
alias `qwen3.6-35b-a3b-dflash`.

For a smaller non-MTP Qwen model downloaded through Hugging Face:

```bash
./serve_qwen3_4b_instruct_2507.sh
```

Launch only one server at a time on port `8080`. The 4B script is intentionally
the plain Qwen3-4B-Instruct-2507 non-thinking model, not an MTP example.

The scripts expose the OpenAI-compatible llama.cpp server at:

```text
http://127.0.0.1:8080/v1/chat/completions
```

The scripts use the default API key:

```text
local-llamacpp-key
```

## 3. Run a generation

In a second shell:

```bash
./generate_llamacpp.py \
  --question 'What is the main rule in the context?' \
  --context 'The server exposes an OpenAI-compatible llama.cpp endpoint on port 8080.' \
  --max-tokens 256
```

The client keeps the same general response-handling style as Vignali's dataset
code, but the task is plain QA:

- OpenAI-compatible chat completion call
- required JSON object with `answer` and `explanation`
- markdown fence stripping
- fallback extraction of the first `{...}` JSON object
- schema validation for a non-empty answer
- retry metadata: `parsed_ok`, `retries_used`, `max_retries`

The client explicitly sends Qwen-recommended decoding defaults:

```text
temperature=0.7
top_p=0.8
top_k=20
min_p=0
presence_penalty=0
repeat_penalty=1
stream=false
```

For a more deterministic smoke test, override them:

```bash
./generate_llamacpp.py \
  --question 'Answer with JSON: what is 2 + 2?' \
  --max-tokens 1024 \
  --temperature 0 \
  --top-p 1 \
  --top-k 0 \
  --min-p 0
```

The larger smoke-test limit leaves room for Qwen's `reasoning_content` before
the final JSON appears in `content`.

The script prints:

```json
{
  "raw_output": "...",
  "parsed_ok": true,
  "retries_used": 0,
  "max_retries": 4,
  "answer": "...",
  "explanation": "..."
}
```

Use `--context-file path/to/context.txt` instead of `--context` for a longer input.
Use `--raw` to print only raw output plus parse status.
