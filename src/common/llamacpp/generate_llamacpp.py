#!/usr/bin/env python3
import argparse
import json
import re
import sys
import urllib.error
import urllib.request


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question. "
    "Return only valid JSON with keys answer and explanation. "
    "Do not include markdown or extra text."
)

QA_TEMPLATE = """
Question:
{question}

Context:
{context}

Required JSON format:
{{
  "answer": "...",
  "explanation": "..."
}}
""".strip()


def post_json(url, api_key, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_json_object(text):
    text = text.strip()
    if not text:
        return None

    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def normalize_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_generation(payload):
    answer = normalize_text(payload.get("answer"))
    explanation = normalize_text(payload.get("explanation"))

    if not answer:
        return None
    return answer, explanation


def build_payload(args, user_prompt):
    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repeat_penalty": args.repetition_penalty,
        "stream": False,
    }


def load_prompt(args):
    if args.prompt:
        return args.prompt
    context = ""
    if args.context_file:
        context = args.context_file.read()
    elif args.context:
        context = args.context
    if args.question:
        return QA_TEMPLATE.format(question=args.question, context=context or "No additional context provided.")
    return None


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key", default="local-llamacpp-key")
    parser.add_argument("--model", default="local-llamacpp")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--context", default=None)
    parser.add_argument("--context-file", type=argparse.FileType("r"), default=None)
    parser.add_argument("--system", default=SYSTEM_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    user_prompt = load_prompt(args)
    if user_prompt is None:
        sys.stderr.write("Provide --question or --prompt.\n")
        return 2

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    raw_output = ""
    normalized = None
    retries_used = 0

    for attempt in range(args.max_retries + 1):
        payload = build_payload(args, user_prompt)
        try:
            response = post_json(url, args.api_key, payload)
        except urllib.error.HTTPError as exc:
            sys.stderr.write(exc.read().decode("utf-8", errors="replace") + "\n")
            return 1

        raw_output = response["choices"][0]["message"]["content"]
        parsed = parse_json_object(raw_output)
        normalized = normalize_generation(parsed) if parsed else None
        if normalized is not None:
            break
        if attempt < args.max_retries:
            retries_used += 1

    record = {
        "raw_output": raw_output,
        "parsed_ok": normalized is not None,
        "retries_used": retries_used,
        "max_retries": args.max_retries,
    }

    if args.raw:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0

    if normalized is None:
        record["text"] = strip_thinking(raw_output)
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 1

    answer, explanation = normalized
    record.update(
        {
            "answer": answer,
            "explanation": explanation,
        }
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
