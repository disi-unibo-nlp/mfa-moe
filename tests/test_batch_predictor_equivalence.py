from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = (
    ROOT
    / "results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3"
    / "selected_program_20260827_173300.json"
)

dspy = pytest.importorskip("dspy")

from moe_exp.correlation_pipeline.batch_predictor import (  # noqa: E402
    load_program_and_adapter,
    render_conversation,
)

SAMPLE = {
    "problem_statement": "What is 2+2?",
    "previous_sentence": "<START OF RESPONSE>",
    "sentence": "We add two and two.",
    "next_sentence": "<END OF RESPONSE>",
}


@pytest.mark.skipif(not PROGRAM.is_file(), reason="saved GEPA program not present")
def test_batch_prompt_is_identical_to_the_single_request_prompt() -> None:
    """The batched path must send exactly what DSPy's single-request path would send."""
    _, predict, adapter = load_program_and_adapter(str(PROGRAM))

    batched = render_conversation(adapter, predict, SAMPLE)
    control = adapter.format(predict.signature, predict.demos, SAMPLE)

    assert batched == control
    assert isinstance(batched, list) and batched
    assert all("role" in message and "content" in message for message in batched)


@pytest.mark.skipif(not PROGRAM.is_file(), reason="saved GEPA program not present")
def test_optimized_instructions_survive_into_the_rendered_prompt() -> None:
    _, predict, adapter = load_program_and_adapter(str(PROGRAM))
    rendered = "\n".join(m["content"] for m in render_conversation(adapter, predict, SAMPLE))
    assert "Schoenfeld" in rendered
    for label in ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor"):
        assert label in rendered


@pytest.mark.skipif(not PROGRAM.is_file(), reason="saved GEPA program not present")
def test_missing_input_field_is_rejected() -> None:
    _, predict, adapter = load_program_and_adapter(str(PROGRAM))
    with pytest.raises(ValueError):
        render_conversation(adapter, predict, {"problem_statement": "x"})
