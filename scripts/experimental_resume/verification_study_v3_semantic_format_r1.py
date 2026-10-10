"""Declared CPU interpretation amendment; never rewrite a reader receipt.

Only a complete, sole lowercase JSON fence around the final post-think suffix
is removed. The frozen parser still supplies every JSON/schema decision. GPU
Reader.rate, its cached parser alias, prompts, seeds and caps remain unchanged.
The durable caller persists the pure ledger and binds it into derived artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
from types import MappingProxyType

from moe_exp.verification_study import semantics, readers as frozen_readers
from moe_exp.verification_study.contracts import (
    check_files, digest, file_sha, json_value, load, private_path, require, seal,
)
import verification_study_v3_reader_r1 as reader

SCRIPT = Path(__file__).resolve()
EXPECTED_CAMPAIGN_SHA256 = "57df4a026a7be4c909ece85ac1ebe3dc47c3f27bb53ffcab88e3aa9d82bfbb3b"
BINDING_SCHEMA = "verification-semantic-format-binding-v3"
LEDGER_SCHEMA = "verification-semantic-decision-ledger-v3"
FORMAT_POLICY = MappingProxyType({
    "post_think": "whole suffix after final </think>, stripped as in the frozen parser",
    "accepted_wrapper": "one complete lowercase ```json newline fence; no surrounding prose or other fences",
    "json_validation": "delegate unchanged frozen parse_rating; no object mining or schema relaxation",
    "effective_finish_reason": "stop only",
    "other_finish_reasons": "unknown; length never salvaged",
    "roles": ("eligibility", "outcome"),
    "draws": (0, 1),
    "generation": "unchanged prompts, sampler, seeds, caps, raw text and identities",
    "scope": "uniform retrospective and prospective CPU interpretation across all seven models",
})

_ORIGINAL_PARSER = semantics.parse_rating
require(_ORIGINAL_PARSER.__module__ == semantics.__name__ and
        _ORIGINAL_PARSER.__name__ == "parse_rating",
        "semantic format helper must capture the original frozen parser")
require(frozen_readers.parse_rating is _ORIGINAL_PARSER,
        "semantic format helper requires the untouched cached GPU reader parser")
# Opening and closing delimiters must each occupy their own entire line.
_JSON_FENCE = re.compile(r"```json[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```\Z")
_OTHER_FENCE_LINE = re.compile(r"(?m)^[ \t]*```")


def strict_suffix(text):
    """Return (whole normalized suffix, format status), without mining JSON.

    Rejected fences are returned unchanged and remain invalid to the original
    parser. Backticks occurring inside a valid JSON string are ordinary data.
    """
    if not isinstance(text, str):
        return text, "INVALID_TEXT"
    suffix = text.rsplit("</think>", 1)[-1].strip()
    matched = _JSON_FENCE.fullmatch(suffix)
    if matched and not _OTHER_FENCE_LINE.search(matched["body"]):
        return matched["body"].strip(), "JSON_FENCE"
    if "```" in suffix:
        return suffix, "REJECTED_FENCE"
    return suffix, "PLAIN_SUFFIX"


def parse_rating(text, role):
    """Original validation after the sole declared wrapper normalization."""
    normalized, _ = strict_suffix(text)
    return _ORIGINAL_PARSER(normalized, role)


@contextmanager
def installed_parser():
    """Scoped CPU adapter; reject nested/foreign installs and always restore."""
    require(semantics.parse_rating is _ORIGINAL_PARSER and
            frozen_readers.parse_rating is _ORIGINAL_PARSER,
            "semantic parser already installed or changed; nested/foreign install refused")
    semantics.parse_rating = parse_rating
    try:
        yield parse_rating
    finally:
        semantics.parse_rating = _ORIGINAL_PARSER


def binding_requirements(campaign_path):
    """Exact required annex fields, without reading or writing the annex."""
    campaign_path = private_path(campaign_path)
    campaign = load(campaign_path)
    require(campaign["sha256"] == EXPECTED_CAMPAIGN_SHA256 and
            len(campaign["code_files"]) == 35,
            "semantic format amendment requires the exact original 35-source campaign")
    check_files(campaign["code_files"])
    contract = load(campaign_path.parent / "READER.json", "verification-reader-contract-v3")
    require(contract["sha256"] == campaign["reader_sha256"] and
            contract["draws"] == [0, 1] and contract["output_cap"] == 1024,
            "semantic format amendment reader contract changed")
    check_files(contract["config_files"])
    executor = reader.binding(campaign)
    check_files(executor["code_files"])
    return {"campaign_sha256": campaign["sha256"],
            "scientific_code_files": campaign["code_files"],
            "helper_file_sha256": file_sha(SCRIPT),
            "reader_contract_sha256": contract["sha256"],
            "reader_executor_sha256": executor["sha256"],
            "parser_policy": json_value(dict(FORMAT_POLICY))}


def load_binding(campaign_path, binding_path=None):
    """Validate the new annex without adding it to its own source closure."""
    campaign_path = private_path(campaign_path)
    path = private_path(binding_path) if binding_path is not None else (
        campaign_path.parent / "SEMANTIC_FORMAT_BINDING-r1.json")
    required = binding_requirements(campaign_path)
    binding = load(path, BINDING_SCHEMA)
    require(all(binding.get(key) == value for key, value in required.items()),
            "semantic format binding differs from the exact campaign/parser/deployment")
    if "helper_path" in binding:
        require(Path(binding["helper_path"]).resolve() == SCRIPT,
                "semantic format binding helper path differs")
    # Optional additional source seals may bind tests, but cannot make an annex
    # depend on its own hash. The mandatory helper hash is already checked above.
    if "code_files" in binding:
        require(str(path.resolve()) not in {str(Path(p).resolve()) for p in binding["code_files"]},
                "semantic format binding cannot seal itself")
        check_files(binding["code_files"])
    return binding


def decision_ledger(campaign_path, frame_path, ratings_dir, role, *, binding_path=None):
    """Seal every original full-frame draw's interpretation; perform no writes.

    The original finite reader validator checks deployment, frame, UID, draw,
    invocation and runtime before interpretation. Both draws for every frame
    record are mandatory, including truncated and otherwise unknown responses.
    raw_text_json_sha256 uses contracts.digest(text), the canonical JSON-string
    digest, rather than a file-byte or UTF-8-text hash.
    """
    require(role in FORMAT_POLICY["roles"], "unknown semantic ledger role")
    campaign_path = private_path(campaign_path)
    frame_path = private_path(frame_path); ratings_dir = private_path(ratings_dir)
    binding = load_binding(campaign_path, binding_path)
    campaign, contract, frame, executor = reader.deployment(campaign_path, frame_path)
    require(executor["sha256"] == binding["reader_executor_sha256"],
            "semantic ledger reader executor differs")
    reader.validate_directory(ratings_dir, campaign, contract, frame, executor, role)
    rows, sources = [], {}
    for record in frame["records"]:
        for draw in (0, 1):
            path = ratings_dir / (record["uid"] + f".reader-{draw}.json")
            require(path.is_file(), "semantic ledger requires both fixed draws for the complete original frame")
            receipt = reader.validate_rating(path, record, draw, role,
                                             campaign, contract, frame, executor)
            text = receipt["text"]
            require(isinstance(text, str), "reader raw text is not a string")
            old_parsed = _ORIGINAL_PARSER(text, role)
            require("parsed" in receipt and receipt["parsed"] == old_parsed,
                    "reader diagnostic differs from the original frozen parser")
            normalized, format_status = strict_suffix(text)
            finish = receipt.get("finish_reason")
            effective = parse_rating(text, role) if finish == "stop" else None
            status = ("STOP_PARSED" if effective is not None else "STOP_UNPARSEABLE") if finish == "stop" else (
                "LENGTH_UNKNOWN" if finish == "length" else "NONSTOP_UNKNOWN")
            sources[str(path)] = receipt["sha256"]
            rows.append({"uid": record["uid"], "reader": draw, "role": role,
                "input_sha256": receipt["input_sha256"], "full_frame_sha256": frame["sha256"],
                "raw_receipt_path": str(path), "raw_receipt_sha256": receipt["sha256"],
                "raw_text_json_sha256": digest(text), "normalized_text_json_sha256": digest(normalized),
                "receipt_parsed": receipt["parsed"], "old_parsed": old_parsed,
                "effective_parsed": effective, "finish_reason": finish,
                "format_status": format_status, "status": status,
                "recovered_stop": finish == "stop" and old_parsed is None and effective is not None,
                "job_id": receipt["job_id"],
                "invocation_runtime_path": receipt["invocation_runtime_path"],
                "invocation_runtime_sha256": receipt["invocation_runtime_sha256"]})
    return seal({"schema": LEDGER_SCHEMA, "campaign_sha256": campaign["sha256"],
        "semantic_format_binding_sha256": binding["sha256"],
        "helper_file_sha256": file_sha(SCRIPT),
        "scientific_executor_sha256": digest(campaign["code_files"]),
        "reader_contract_sha256": contract["sha256"], "reader_executor_sha256": executor["sha256"],
        "frame_path": str(frame_path), "full_frame_sha256": frame["sha256"],
        "ratings_directory": str(ratings_dir), "role": role, "draws": [0, 1],
        "raw_text_digest_method": "contracts.digest(raw text): SHA256 of canonical JSON string",
        "source_receipts": sources, "records": rows,
        "interpretation": "declared format amendment; original length and disagreement rules remain unknown"})
