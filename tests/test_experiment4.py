import numpy as np

from moe_exp.experiment4.run import _evaluate_source, _labels_for_prefix
from moe_exp.schemas import StepLabels, TraceRecord


def _trace(*, correct: bool, first_error_step: int | None) -> TraceRecord:
    return TraceRecord(
        dataset="synthetic",
        problem_id="0",
        prompt="p",
        gold_answer="a",
        model_id="m",
        model_answer="a",
        is_correct=correct,
        cot_text="s0\ns1",
        steps=["s0", "s1"],
        step_labels=StepLabels(first_error_step=first_error_step),
    )


def test_prefix_labels_do_not_include_past_errors() -> None:
    failed = _trace(correct=False, first_error_step=1)
    assert _labels_for_prefix(failed, prefix_end=4, first_error_token=7) == {
        "final_incorrect": 1,
        "future_first_error": 1,
    }
    assert _labels_for_prefix(failed, prefix_end=8, first_error_token=7) == {
        "final_incorrect": 1,
        "future_first_error": None,
    }
    correct = _trace(correct=True, first_error_step=None)
    assert _labels_for_prefix(correct, prefix_end=8, first_error_token=None) == {
        "final_incorrect": 0,
        "future_first_error": 0,
    }


def test_layerwise_probe_returns_each_layer() -> None:
    rng = np.random.default_rng(7)
    y = np.asarray([0, 1] * 10, dtype=np.int8)
    x = rng.normal(size=(20, 3, 5)).astype(np.float32)
    x[:, :, 0] += y[:, None]
    result = _evaluate_source(
        x,
        y,
        source="hidden",
        folds=2,
        seed=7,
        bootstrap_samples=10,
    )
    assert result["status"] == "complete"
    assert [row["layer"] for row in result["layers"]] == [0, 1, 2]
