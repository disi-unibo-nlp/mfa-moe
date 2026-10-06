"""Prefix-only native-model semantic veto prompt for discovery qualification.

The Qwen3.6 generation model would answer an unsteered side request. No
future sentence, correctness, class label, detector fire or intervention arm
enters the prompt. This module does not claim that the veto is qualified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
FRAME = BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json"
MODEL = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989")
VERSION = "native-prefix-semantic-veto-v0-discovery"
MAX_TOKENS = 64
INPUT_ALLOWLIST = frozenset({"problem", "emitted_prefix", "triggering_sentence"})
CRITERIA = {
    "candidate_to_verify": (
        "The current sentence states a complete, concrete candidate value, assignment, "
        "intermediate state, or proposed answer that has not already been substantively "
        "checked in the visible prefix. A bare step in an algebraic equation chain, a "
        "restatement of a given, an already checked conclusion, a plan to compute, or an "
        "unfinished expression is not an unchecked candidate."
    ),
    "approach_to_commit": (
        "The current sentence tentatively names a specific method, construction, case, "
        "formula, or representation and the operation it would perform. A vague wish, "
        "a bare 'try another way', or an already executed operation is not a tentative approach."
    ),
    "failed_check_to_revise": (
        "The current sentence visibly computes a contradiction, invalid value, or violation "
        "of an original condition, with enough mathematics or constraint detail to see what "
        "failed. An ordinary equation, unsupported doubt, or verification word is not a failed check."
    ),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed source: {path}")
    return value


def messages(row):
    inputs = row["reader_input"]
    if set(inputs) != INPUT_ALLOWLIST:
        raise ValueError("semantic veto input allowlist changed")
    problem, prefix, sentence = (inputs[key] for key in
                                 ("problem", "emitted_prefix", "triggering_sentence"))
    if not all(isinstance(value, str) for value in (problem, prefix, sentence)):
        raise ValueError("semantic veto inputs must be text")
    if not prefix.rstrip().endswith(sentence.rstrip()):
        raise ValueError("trigger is not at emitted prefix end")
    criterion = CRITERIA[row["transition"]]
    system = (
        "Judge only the visible starting condition in mathematical reasoning. Use the "
        "original problem and reasoning already emitted through the current sentence. "
        "Do not infer later text, correctness, hidden reasoning, or whether a future check occurs. "
        "When ambiguous, answer false. Return exactly one JSON object with one boolean key: "
        '{"start":true} or {"start":false}. No explanation.\nCriterion: ' + criterion
    )
    user = ("Original problem:\n" + problem +
            "\n\nAlready emitted reasoning prefix:\n" + prefix +
            "\n\nCurrent sentence (repeated for focus):\n" + sentence +
            "\n\nJSON:")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_rating(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and set(value) == {"start"} and type(value["start"]) is bool else None


def cache_receipt():
    cfg = MODEL / "config.json"
    index = MODEL / "model.safetensors.index.json"
    tokenizer = MODEL / "tokenizer.json"
    template = MODEL / "chat_template.jinja"
    for path in (cfg, index, tokenizer, template):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(cfg.read_text())
    if config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]:
        raise ValueError("unexpected native Qwen architecture")
    shards = {name: (MODEL / name).stat().st_size for name in
              set(json.loads(index.read_text())["weight_map"].values())}
    if not shards or any(size < 1 for size in shards.values()):
        raise ValueError("native Qwen weight cache incomplete")
    file_sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return {"snapshot": str(MODEL), "config_sha256": file_sha(cfg),
            "index_sha256": file_sha(index), "tokenizer_sha256": file_sha(tokenizer),
            "template_sha256": file_sha(template),
            "weight_shard_count": len(shards), "weight_bytes_total": sum(shards.values())}
