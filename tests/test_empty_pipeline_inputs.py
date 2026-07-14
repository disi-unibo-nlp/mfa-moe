from pathlib import Path

import pytest

from moe_exp.experiment2 import run as experiment2
from moe_exp.experiment3 import run as experiment3
from moe_exp.utils import write_jsonl


@pytest.mark.parametrize("module", [experiment2, experiment3])
def test_gpu_stages_reject_empty_input_before_loading_model(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("", encoding="utf-8")

    def unexpected_model_load(*args, **kwargs):
        raise AssertionError("the model must not be loaded for an empty input")

    monkeypatch.setattr(module, "load_model_and_tokenizer", unexpected_model_load)

    kwargs = {
        "input_path": input_path,
        "model_id": "unused/model",
        "output_path": tmp_path / "output.json",
    }
    with pytest.raises(RuntimeError, match="zero traces"):
        module.process_file(**kwargs)


def test_write_jsonl_replaces_output_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    output_path = tmp_path / "traces.jsonl"
    output_path.write_text("stale\n", encoding="utf-8")

    write_jsonl([{"problem_id": "one"}], output_path)

    assert output_path.read_text(encoding="utf-8").strip() == '{"problem_id": "one"}'
    assert not (tmp_path / ".traces.jsonl.tmp").exists()
