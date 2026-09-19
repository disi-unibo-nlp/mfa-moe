import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from moe_exp.moe_margin_guiding.calibration import fit, problem_key, validate_policy
from moe_exp.moe_margin_guiding.prepare import prepare_global
from moe_exp.moe_margin_guiding.routing import MarginGuide, MarginWorkerExtension, install
from moe_exp.moe_margin_guiding.run import check_reports, generate


def row(name, correct, margin, layer=7):
    return dict(dataset="math", problem_id=name, model_id="test", is_correct=correct,
                metadata={"correlation_features": dict(
                    num_experts=4, top_k=2, num_layers=1, layer_indices=[layer],
                    values={f"router_boundary_margin_l{layer:02d}": margin,
                            f"router_margin_l{layer:02d}": margin})})


def policy():
    return fit([row("good", True, .1), row("bad", False, .3)], model="test",
               num_experts=4, top_k=2, min_support=1, bins=2)


def gap(logits, rank=2):
    p = logits.float().softmax(-1).topk(rank + 1, dim=-1).values
    return p[:, rank - 1] - p[:, rank]


@pytest.mark.parametrize("metric,rank", [("router_boundary_margin", 2), ("router_margin", 1)])
@pytest.mark.parametrize("strength", [0, .5, 1])
def test_guides_both_directions_to_range_and_preserves_rank(metric, rank, strength):
    logits = torch.tensor([[.55, .30, .10, .05], [.28, .26, .24, .22]]).log()
    before = gap(logits, rank)
    hook = MarginGuide(dict(lower=.08, upper=.12), strength, 2, 4, metric)
    output = hook(None, (), (logits, "aux"))
    after = gap(output[0], rank)
    torch.testing.assert_close(after, before + strength * (before.clamp(.08, .12) - before))
    assert output[1] == "aux" and output[0].dtype == logits.dtype
    assert torch.equal(output[0].argsort(-1), logits.argsort(-1))
    stats = hook.snapshot()
    assert stats["changed_topk_sets"] == 0
    assert stats["mean_distance_after"] <= stats["mean_distance_before"] + 1e-7
    if strength == 0:
        assert output[0] is logits


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_dtype_empty_rows_noop_and_extreme_logits(dtype):
    hook = MarginGuide(dict(lower=.05, upper=.1), 1, 2, 4)
    logits = torch.tensor([[1000, 0, -1000, -2000]], dtype=dtype)
    result = hook(None, (), logits)
    assert torch.isfinite(result).all() and result.dtype == dtype
    assert gap(result).item() == pytest.approx(.05, abs=.002)
    assert hook(None, (), logits[:0]).shape == (0, 4)
    hook.reset()
    assert hook.snapshot()["token_evaluations"] == 0
    noop = MarginGuide(dict(lower=0, upper=.5), 1, 2, 4)
    assert torch.equal(noop(None, (), logits), logits)


def test_changes_native_renormalized_mixture_weights():
    logits = torch.tensor([[.55, .3, .1, .05]]).log()
    guided = MarginGuide(dict(lower=.05, upper=.05), 1, 2, 4)(None, (), logits)
    assert not torch.allclose(logits.topk(2).values.softmax(-1),
                              guided.topk(2).values.softmax(-1))


def test_fit_learns_supported_range_without_attempt_or_length_bias():
    p = policy()
    assert p["layers"]["7"]["lower"] == .1
    assert p["layers"]["7"]["accuracy"] == 1
    validate_policy(p, "test")
    records = [row("good", True, .1)] * 20 + [row("bad", False, .3)]
    repeated = fit(records, model="test", num_experts=4, top_k=2, min_support=1, bins=2)
    assert repeated["baseline_accuracy"] == pytest.approx(.5)
    assert repeated["layers"]["7"]["problem_support"] == 1
    assert repeated["layers"]["7"]["lower"] == .1
    with pytest.raises(ValueError, match="No supported"):
        fit(records, model="test", num_experts=4, top_k=2, min_support=2)


def test_raw_logits_and_cached_features_agree(tmp_path):
    tensors = [torch.tensor([[[.4, .3, .2, .1]]]).log(),
               torch.tensor([[[.5, .4, .08, .02]]]).log()]
    records = []
    for i, tensor in enumerate(tensors):
        path = tmp_path / f"{i}.pt"
        torch.save(tensor, path)
        records.append(dict(id=str(i), is_correct=not i, layer_indices=[7],
                            router_logits=str(path)))
    records[0]["is_correct"] = True
    records[1]["is_correct"] = False
    p = fit(records, model="test", num_experts=4, top_k=2, bins=2, min_support=1)
    assert p["layers"]["7"]["lower"] == pytest.approx(.1)


@pytest.mark.parametrize("issue", ["nan", "negative", "counts", "layers", "unscored", "metric"])
def test_bad_calibration_rejected(issue):
    records = [row("good", True, .1), row("bad", False, .3)]
    cached = records[0]["metadata"]["correlation_features"]
    if issue in ("nan", "negative"):
        cached["values"]["router_boundary_margin_l07"] = float("nan") if issue == "nan" else -.1
    elif issue == "counts":
        cached["num_experts"] = 5
    elif issue == "layers":
        records[1] = row("bad", False, .3, layer=8)
    elif issue == "unscored":
        records[0]["is_correct"] = None
    with pytest.raises(ValueError):
        fit(records, model="test", num_experts=4, top_k=2, min_support=1,
            metric="bad" if issue == "metric" else "router_boundary_margin")


class Gate(torch.nn.Module):
    def forward(self, value):
        return value


@pytest.mark.parametrize("family", ["qwen", "oss", "gemma"])
def test_native_hook_locations_and_bypass_detection(family):
    m = torch.nn.Module()
    m.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(8)])
    for layer in m.layers:
        if family == "gemma":
            layer.router = Gate()
        else:
            layer.mlp = torch.nn.Module()
            setattr(layer.mlp, "gate" if family == "qwen" else "router", Gate())
    p = policy()
    hooks, handles = install(m, p, 1, family)
    report = dict(condition="guided", layers={k: h.snapshot() for k, h in hooks.items()})
    with pytest.raises(RuntimeError, match="bypassed"):
        check_reports([report], p, "guided")
    gate = m.layers[7].router if family == "gemma" else getattr(
        m.layers[7].mlp, "gate" if family == "qwen" else "router")
    logits = torch.tensor([[.4, .3, .25, .05]]).log()
    assert gap(gate(logits)).item() == pytest.approx(.1)
    check_reports([dict(condition="guided", layers={k: h.snapshot() for k, h in hooks.items()})],
                  p, "guided")
    for handle in handles:
        handle.remove()
    assert gate(logits) is logits


def test_worker_baseline_and_runtime_safety():
    worker = MarginWorkerExtension()
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True, hf_text_config=SimpleNamespace(
            model_type="qwen3_5_moe_text", num_experts=4, num_experts_per_tok=2)),
        parallel_config=SimpleNamespace(enable_expert_parallel=False, enable_eplb=False,
                                        pipeline_parallel_size=1, data_parallel_size=1),
        speculative_config=None)
    assert worker.margin_configure(policy(), 1, "baseline") == dict(condition="baseline", layers={})
    with pytest.raises(RuntimeError, match="already configured"):
        worker.margin_configure(policy(), 1, "guided")


def test_global_split_retains_features_and_holds_out_whole_problems(tmp_path):
    generations, traces = [], []
    for i in range(12):
        for sample in range(2):
            record = row(str(i), sample == 0, .1 if sample == 0 else .3)
            record.update(sample_id=sample, prompt="Question", gold_answer="1")
            generations.append(record)
            traces.append(record)
    paths = [tmp_path / "generations.jsonl", tmp_path / "traces.jsonl"]
    for path, records in zip(paths, (generations, traces)):
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    out = tmp_path / "split"
    prepare_global(paths[:1], paths[1:], out)
    calibration = [json.loads(s) for s in (out / "calibration.jsonl").read_text().splitlines()]
    prompts = [json.loads(s) for s in (out / "prompts.full.jsonl").read_text().splitlines()]
    assert all(r["metadata"]["correlation_features"] for r in calibration)
    assert not {problem_key(r) for r in calibration} & {problem_key(r) for r in prompts}
    fit(calibration, model="test", num_experts=4, top_k=2)


def test_generation_rejects_calibration_overlap_before_loading_vllm(tmp_path):
    p = policy()
    policy_path, prompts_path = tmp_path / "policy.json", tmp_path / "prompts.jsonl"
    policy_path.write_text(json.dumps(p))
    prompts_path.write_text(json.dumps(dict(id="x", prompt="Question", dataset="math", problem_id="good")))
    args = SimpleNamespace(policy=policy_path, model="test", prompts=prompts_path,
                           strength=1, max_tokens=8, max_model_len=16, max_num_seqs=1,
                           tensor_parallel_size=1, gpu_memory_utilization=.9,
                           allow_calibration_overlap=False)
    with pytest.raises(ValueError, match="overlaps"):
        generate(args)


@pytest.mark.parametrize("profile", ["qwen", "oss", "gemma"])
def test_launcher_preview_uses_new_experiment(profile, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", "src/moe_exp/moe_margin_guiding/run_global.sh", "all"],
                            cwd=root, env={**os.environ, "DRY_RUN": "true", "MODEL_PROFILE": profile,
                                           "OUTPUT_ROOT": str(tmp_path / "run")},
                            text=True, capture_output=True, check=True)
    assert "moe_exp.moe_margin_guiding.run" in result.stdout
    assert "--metric router_boundary_margin" in result.stdout
    assert str(tmp_path / "run") in result.stdout
    assert "prompts.full.sampling.jsonl" in result.stdout
    assert "--temperature 0.6 --top-p 0.95 --top-k 0 --seed 0" in result.stdout
    assert "--require-original-sampling" in result.stdout
    assert "/sampling/full/" in result.stdout


def test_policy_validation():
    p = copy.deepcopy(policy())
    p["layers"]["7"]["upper"] = 1
    with pytest.raises(ValueError, match="range"):
        validate_policy(p)


def test_generate_preserves_original_request_sampling(tmp_path, monkeypatch):
    import sys
    from moe_exp.moe_margin_guiding import run
    template_calls = []
    class LLM:
        def __init__(self, **kwargs):
            pass
        def get_tokenizer(self):
            def render(messages, **kwargs):
                template_calls.append(kwargs)
                return 'rendered'
            return SimpleNamespace(apply_chat_template=render, encode=lambda *a, **k: [1])
        def collective_rpc(self, name, **kwargs):
            return [{'condition': 'baseline', 'layers': {}}]
        def generate(self, prompts, sampling):
            assert len(prompts) == len(sampling) == 32
            assert [p['seed'] for p in sampling] == list(range(7000, 7032))
            assert all(p['temperature'] == .6 and p['top_p'] == .95 and
                       p['top_k'] == -1 and p['max_tokens'] == 32768 for p in sampling)
            return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                text='B', token_ids=[2], finish_reason='stop')]) for _ in prompts]
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=LLM, SamplingParams=lambda **k: k))
    monkeypatch.setattr(run, 'version', lambda name: 'test')
    p = tmp_path / 'policy.json'
    p.write_text(json.dumps(policy()))
    prompts = tmp_path / 'prompts.jsonl'
    prompts.write_text(''.join(json.dumps(dict(
        id=str(i), dataset='math', problem_id='heldout', prompt='Question',
        gold_answer='B', answer_type='choice', original_model_id='test',
        original_generation_config=dict(seed=7000+i, temperature=.6, top_p=.95,
            top_k=0, max_tokens=32768, chat_template_kwargs={'enable_thinking': True})))+'\n'
        for i in range(32)))
    args = SimpleNamespace(policy=p, model='test', prompts=prompts, strength=1.,
        allow_calibration_overlap=False, revision=None, dtype='auto', tensor_parallel_size=1,
        max_model_len=49152, max_tokens=32768, max_num_seqs=16, gpu_memory_utilization=.9,
        seed=0, temperature=.6, top_p=.95, top_k=0, require_original_sampling=True,
        output_dir=tmp_path/'baseline', condition='baseline')
    run.generate(args)
    assert all(c['enable_thinking'] for c in template_calls)
    records = [json.loads(line) for line in (args.output_dir/'generations.jsonl').read_text().splitlines()]
    assert [r['sampling_args']['seed'] for r in records] == list(range(7000,7032))
