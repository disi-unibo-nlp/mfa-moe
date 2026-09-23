import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from moe_exp.moe_identity_guiding.execution import (
    execute, matching_baseline, read_records, run_lock,
)
from moe_exp.moe_identity_guiding.routing import IdentityBias
from moe_exp.moe_margin_guiding.routing import MarginGuide
from moe_exp.moe_identity_guiding.run import check_reports, compare


def fixture_data(tmp_path, condition="baseline"):
    rows = [dict(id=str(i), dataset="test", source_problem_id=str(i),
                 prompt="question", gold_answer="B", answer_type="choice")
            for i in range(3)]
    requests = [dict(seed=100+i, temperature=.6) for i in range(3)]
    manifest = dict(experiment="moe_identity_guiding", status="running", condition=condition,
                    strength=1., policy={"layers": {"1": {}}}, policy_sha256="policy",
                    prompts_sha256="prompts", sampling_args={"temperature": .6},
                    versions={"vllm": "test"}, scoring_contract="test",
                    seed_strategy="original", token_scope="prefill_and_decode",
                    calibration_overlap=[], engine_args=dict(model="model-a",
                    worker_extension_cls="moe_exp.moe_identity_guiding.routing.IdentityWorkerExtension"))
    args = SimpleNamespace(output_dir=tmp_path / "run", condition=condition, strength=1.,
                           resume=True, baseline_search_root=[tmp_path / "sources"],
                           diagnostics="minimal", no_reuse_baseline=False)
    return args, manifest, rows, requests


def fake_vllm(monkeypatch, condition="baseline", fail_after=None):
    submitted = []
    class LLM:
        def __init__(self, **kwargs):
            self.llm_engine = self
            self.pending = []
            self.steps = 0
        def get_tokenizer(self):
            return SimpleNamespace(apply_chat_template=lambda *a, **kw: "rendered",
                                   encode=lambda *a, **kw: [1, 2])
        def collective_rpc(self, name, **kwargs):
            return [dict(condition=condition, layers={} if condition == "baseline" else
                         {"1": {"token_evaluations": 10}})]
        def add_request(self, request_id, prompt, params):
            submitted.append((request_id, params))
            self.pending.append(request_id)
        def has_unfinished_requests(self):
            return bool(self.pending)
        def step(self):
            if fail_after is not None and self.steps == fail_after:
                raise RuntimeError("interrupted")
            self.steps += 1
            # Deliberately finish out of order.
            return [SimpleNamespace(request_id=self.pending.pop(), finished=True,
                prompt_token_ids=[1, 2], outputs=[SimpleNamespace(
                    text=r"\boxed{B}", token_ids=[3, 4], finish_reason="stop")])]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw))
    monkeypatch.setattr("moe_exp.moe_identity_guiding.execution.load_tokenizer",
                        lambda engine: LLM.__new__(LLM).get_tokenizer())
    monkeypatch.setitem(sys.modules, "vllm.sampling_params",
                        SimpleNamespace(RequestOutputKind=SimpleNamespace(FINAL_ONLY="final")))
    return submitted


@pytest.mark.parametrize("condition", ["baseline", "guided"])
def test_incremental_failure_resume_and_completed_skip(tmp_path, monkeypatch, capsys, condition):
    args, manifest, rows, requests = fixture_data(tmp_path, condition)
    fake_vllm(monkeypatch, condition, fail_after=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        execute(args, copy.deepcopy(manifest), rows, requests, "identity", check_reports)
    output = args.output_dir / "generations.jsonl"
    first = json.loads(output.read_text())
    assert first["id"] == "2" and first["sampling_args"]["seed"] == 102
    assert json.loads((args.output_dir / "manifest.json").read_text())["status"] == "failed"
    with output.open("ab") as handle:
        handle.write(b'{"id": "unfinished')
    submitted = fake_vllm(monkeypatch, condition)
    execute(args, copy.deepcopy(manifest), rows, requests, "identity", check_reports)
    assert [key for key, _ in submitted] == ["0", "1"]
    assert len(read_records(output, rows, requests, condition)) == 3
    display = capsys.readouterr().err
    assert f"Identity {condition}" in display
    assert "1/3" in display and "3/3" in display and "100%" in display
    assert json.loads((args.output_dir / "manifest.json").read_text())["status"] == "complete"
    # A completed run must be validated and skipped without even importing vLLM.
    monkeypatch.setitem(sys.modules, "vllm", None)
    execute(args, copy.deepcopy(manifest), rows, requests, "identity", check_reports)


def test_cross_experiment_baseline_reuse_and_comparison(tmp_path, monkeypatch):
    args, manifest, rows, requests = fixture_data(tmp_path)
    args.output_dir = tmp_path / "sources" / "identity" / "baseline"
    fake_vllm(monkeypatch)
    execute(args, copy.deepcopy(manifest), rows, requests, "identity", check_reports)
    source_bytes = (args.output_dir / "generations.jsonl").read_bytes()
    args.output_dir = tmp_path / "margin" / "baseline"
    target = copy.deepcopy(manifest)
    target.update(experiment="moe_margin_guiding", policy_sha256="different-policy")
    target["engine_args"]["worker_extension_cls"] = "moe_exp.moe_margin_guiding.routing.MarginWorkerExtension"
    monkeypatch.setitem(sys.modules, "vllm", None)
    execute(args, copy.deepcopy(target), rows, requests, "margin", check_reports)
    result = json.loads((args.output_dir / "manifest.json").read_text())
    assert result["reused_baseline"]["manifest"]["policy_sha256"] == "policy"
    assert result["policy_sha256"] == "different-policy"
    assert (args.output_dir / "generations.jsonl").read_bytes() == source_bytes
    baseline = args.output_dir
    args.output_dir = tmp_path / "margin" / "guided"
    args.condition = target["condition"] = "guided"
    fake_vllm(monkeypatch, "guided")
    execute(args, target, rows, requests, "margin", check_reports)
    assert compare(baseline, args.output_dir)["accuracy_delta"] == 0


@pytest.mark.parametrize("field,value", [
    ("engine_args", {"model": "other"}), ("sampling_args", {"temperature": 0}),
    ("versions", {"vllm": "other"}), ("prompts_sha256", "other"),
    ("status", "running"), ("condition", "guided"),
    ("routing_diagnostics", [{"condition": "baseline", "layers": {"1": {}}}]),
])
def test_reuse_rejects_incompatible_baselines(tmp_path, field, value):
    _, target, _, _ = fixture_data(tmp_path)
    source = copy.deepcopy(target)
    source.update(status="complete", routing_diagnostics=[{"condition": "baseline", "layers": {}}])
    assert matching_baseline(source, target)
    source[field] = value
    assert not matching_baseline(source, target)


def test_resume_rejects_changes_and_live_lock(tmp_path, monkeypatch):
    args, manifest, rows, requests = fixture_data(tmp_path)
    fake_vllm(monkeypatch, fail_after=1)
    with pytest.raises(RuntimeError):
        execute(args, copy.deepcopy(manifest), rows, requests, "identity", check_reports)
    with run_lock(args.output_dir):
        with pytest.raises(RuntimeError, match="Another process"):
            execute(args, copy.deepcopy(manifest), rows, requests, "identity", check_reports)
    changed = copy.deepcopy(manifest)
    changed["engine_args"]["model"] = "other"
    with pytest.raises(ValueError, match="engine_args differs"):
        execute(args, changed, rows, requests, "identity", check_reports)
    requests[2]["seed"] = 999
    with pytest.raises(ValueError, match="sampling differ"):
        execute(args, manifest, rows, requests, "identity", check_reports)


def test_resume_rejects_duplicate_records(tmp_path, monkeypatch):
    args, manifest, rows, requests = fixture_data(tmp_path)
    fake_vllm(monkeypatch, fail_after=1)
    with pytest.raises(RuntimeError):
        execute(args, copy.deepcopy(manifest), rows, requests, "identity", check_reports)
    p = args.output_dir / "generations.jsonl"
    p.write_text(p.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        execute(args, manifest, rows, requests, "identity", check_reports)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["identity", "margin"])
def test_minimal_diagnostics_preserve_intervention_exactly(dtype, kind):
    torch.manual_seed(5)
    logits = torch.randn(30, 8).to(dtype)
    if kind == "identity":
        factory = lambda mode: IdentityBias([0, 1, 0, 0, .3, 0, 0, 0], 1., 2, diagnostics=mode)
    else:
        factory = lambda mode: MarginGuide({"lower": .02, "upper": .04}, 1., 2, 8, diagnostics=mode)
    minimal, full = factory("minimal"), factory("full")
    assert torch.equal(minimal(None, (), logits), full(None, (), logits))
    assert minimal.snapshot()["token_evaluations"] == 30
    assert "changed_topk_sets" not in minimal.snapshot()


def test_finalize_after_last_saved_attempt_without_loading_model(tmp_path, monkeypatch):
    args, manifest, rows, requests = fixture_data(tmp_path)
    fake_vllm(monkeypatch)
    execute(args, copy.deepcopy(manifest), rows, requests, "identity", check_reports)
    p = args.output_dir / "manifest.json"
    saved_manifest = json.loads(p.read_text())
    saved_manifest["status"] = "running"
    p.write_text(json.dumps(saved_manifest))
    monkeypatch.setitem(sys.modules, "vllm", None)
    execute(args, manifest, rows, requests, "identity", check_reports)
    assert json.loads(p.read_text())["status"] == "complete"


def test_legacy_incomplete_run_is_not_resumed(tmp_path, monkeypatch):
    fake_vllm(monkeypatch)
    args, manifest, rows, requests = fixture_data(tmp_path)
    args.output_dir.mkdir()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="legacy incomplete"):
        execute(args, manifest, rows, requests, "identity", check_reports)


def test_render_freezes_template_clock_and_retains_options():
    from moe_exp.moe_identity_guiding.execution import render_inputs
    from jinja2 import Environment, StrictUndefined
    template = 'Current date: {{ strftime_now("%Y-%m-%d") }} {{ enable_thinking }}'
    class Tokenizer:
        chat_template = template
        def apply_chat_template(self, messages, **kwargs):
            return Environment(undefined=StrictUndefined).from_string(kwargs['chat_template']).render(**kwargs)
        def encode(self, text, **kwargs):
            return list(text.encode())
    rows = [dict(prompt='question', original_generation_config={'chat_template_kwargs': {'enable_thinking': True}})]
    text, tokens = render_inputs(Tokenizer(), rows, '2026-09-22')
    assert text == ['Current date: 2026-09-22 True']
    assert tokens[0]['prompt_token_ids'] == list(text[0].encode())
    assert Tokenizer.chat_template == template


def test_mismatched_rendered_baseline_is_rejected_before_gpu(tmp_path, monkeypatch):
    args, manifest, rows, requests = fixture_data(tmp_path, 'guided')
    fake_vllm(monkeypatch, 'guided')
    args.output_dir = tmp_path / 'pair' / 'guided'
    baseline = args.output_dir.parent / 'baseline'
    baseline.mkdir(parents=True)
    (baseline / 'generations.jsonl').write_text(json.dumps(dict(
        id='0', rendered_prompt='Current date: 2026-09-21', prompt_token_ids=[999]))+'\n')
    monkeypatch.setitem(sys.modules, 'vllm', None)
    with pytest.raises(ValueError, match='Rendered prompts differ'):
        execute(args, manifest, rows, requests, 'identity', check_reports)
    assert not (args.output_dir / 'manifest.json').exists()


def test_reuse_rejects_changed_rendered_prompt(tmp_path, monkeypatch):
    from moe_exp.moe_identity_guiding.execution import reuse_baseline
    args, manifest, rows, requests = fixture_data(tmp_path)
    args.output_dir = tmp_path / 'sources' / 'original'
    fake_vllm(monkeypatch)
    execute(args, copy.deepcopy(manifest), rows, requests, 'identity', check_reports)
    args.output_dir = tmp_path / 'new'
    args.output_dir.mkdir()
    assert not reuse_baseline(args, manifest, rows, requests,
                              ['different date'] * 3, [{'prompt_token_ids': [1,2]}] * 3)
    assert not (args.output_dir / 'generations.jsonl').exists()
