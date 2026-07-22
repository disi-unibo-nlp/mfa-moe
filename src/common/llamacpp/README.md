# llama.cpp handoff

Minimal local llama.cpp serving and generation handoff.

## 1. Build the Docker image

Run from this directory:

```bash
docker build -t llama.cpp:localcuda .
```

## 2. Launch one model server

For the existing local Qwen3.6 GGUF under `/llms`:

```bash
./serve_llamacpp.sh
```

For a smaller non-MTP Qwen model downloaded through Hugging Face:

```bash
./serve_qwen3_4b_instruct_2507.sh
```

Launch only one server at a time on port `8080`. The 4B script is intentionally
the plain Qwen3-4B-Instruct-2507 non-thinking model, not an MTP example.

Both scripts expose the OpenAI-compatible llama.cpp server at:

```text
http://127.0.0.1:8080/v1/chat/completions
```

Both scripts use the default API key:

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
  --max-tokens 64 \
  --temperature 0 \
  --top-p 1 \
  --top-k 0 \
  --min-p 0
```

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
