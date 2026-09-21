import json
from types import SimpleNamespace

import pytest
import torch

from moe_exp.moe_identity_guiding.calibration import fit, problem_key, validate_policy
from moe_exp.moe_identity_guiding.routing import IdentityBias, install, IdentityWorkerExtension
from moe_exp.moe_identity_guiding.run import check_reports, compare


def rows():
    return [dict(id="good", is_correct=True, layer_indices=[7, 11],
                 selected_experts=torch.tensor([[[0, 2], [0, 2]], [[0, 1], [0, 1]]])),
            dict(id="bad", is_correct=False, layer_indices=[7, 11],
                 selected_experts=torch.tensor([[[0, 1]], [[0, 2]]]))]


def policy():
    return fit(rows(), model="qwen-test", num_experts=3, top_k=2,
               min_support=1, max_experts=1)


def test_identity_is_any_topk_and_layer_specific():
    p = policy()
    assert p["layers"]["7"]["scores"] == [0, 0, 1]
    assert p["layers"]["11"]["scores"] == [0, 1, 0]
    assert p["baseline_accuracy"] == .5
    validate_policy(p, "qwen-test")
    with pytest.raises(ValueError, match="checkpoint-specific"):
        validate_policy(p, "another-model")


def test_length_and_repeated_attempts_do_not_dominate():
    r = rows()
    r[0]["selected_experts"] = r[0]["selected_experts"].repeat(1, 100, 1)
    r.append(dict(r[0]))
    p = fit(r, model="qwen-test", num_experts=3, top_k=2, min_support=1, max_experts=1)
    assert p["baseline_accuracy"] == .5
    assert p["layers"]["7"]["experts"][0]["accuracy"] == .5
    assert p["layers"]["7"]["experts"][2]["problem_support"] == 1


@pytest.mark.parametrize("issue", ["unscored", "duplicate_pool", "wrong_k", "layers", "support"])
def test_bad_calibration(issue):
    r = rows()
    if issue == "unscored":
        r[0]["is_correct"] = None
    if issue == "duplicate_pool":
        r[0]["selected_experts"][0, 0] = torch.tensor([0, 0])
    if issue == "wrong_k":
        r[0]["selected_experts"] = r[0]["selected_experts"][:, :, :1]
    if issue == "layers":
        r[0]["layer_indices"] = [7, 7]
    with pytest.raises(ValueError):
        fit(r, model="qwen-test", num_experts=3, top_k=2,
            min_support=10 if issue == "support" else 1, max_experts=1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_bias_promotes_identity_and_preserves_native_topk(dtype):
    logits = torch.tensor([[3., 2., 1., 0.]], dtype=dtype)
    hook = IdentityBias([0, 0, 0, 1], 4, 2)
    output = hook(None, (), (logits, None))
    assert output[0].dtype == dtype and output[1] is None
    assert output[0].topk(2).indices.tolist() == [[3, 0]]
    assert hook.snapshot()["changed_topk_sets"] == 1
    assert hook.snapshot()["expert_counts_before"] == [1, 1, 0, 0]
    assert hook.snapshot()["expert_counts_after"] == [1, 0, 0, 1]
    hook.reset()
    assert hook.snapshot()["token_evaluations"] == 0
    zero = IdentityBias([0, 0, 0, 1], 0, 2)
    assert zero(None, (), logits) is logits


def test_qwen_top8_empty_batches_and_weights():
    hook = IdentityBias([0.] * 255 + [1.], 3, 8)
    logits = torch.arange(256, 0, -1).float()[None] / 256
    guided = hook(None, (), logits)
    assert 255 in guided.topk(8).indices[0]
    assert guided.softmax(-1)[0, 255] > logits.softmax(-1)[0, 255]
    assert hook(None, (), logits[:0]).shape == (0, 256)


class Gate(torch.nn.Module):
    def forward(self, value):
        return value, None


def model():
    m = torch.nn.Module()
    m.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(12)])
    for layer in m.layers:
        layer.mlp = torch.nn.Module()
        layer.mlp.gate = Gate()
        layer.mlp.shared_expert_gate = Gate()
    return m


def test_install_and_worker_rpc_with_native_gate():
    m = model()
    hooks, handles = install(m, policy(), 4)
    logits = torch.tensor([[3., 2., 1.]])
    assert m.layers[7].mlp.gate(logits)[0].topk(2).indices.tolist() == [[2, 0]]
    assert m.layers[8].mlp.gate(logits)[0] is logits
    assert m.layers[7].mlp.shared_expert_gate(logits)[0] is logits
    m.layers[11].mlp.gate(logits)
    check_reports([dict(condition="guided", layers={k: h.snapshot() for k,h in hooks.items()})],
                  policy(), "guided")
    for h in handles:
        h.remove()
    assert m.layers[7].mlp.gate(logits)[0] is logits
    worker = IdentityWorkerExtension()
    worker.model_runner = SimpleNamespace(get_model=lambda: m)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True, hf_text_config=SimpleNamespace(
            model_type="qwen3_5_moe_text", num_experts=3, num_experts_per_tok=2)),
        parallel_config=SimpleNamespace(enable_expert_parallel=False, enable_eplb=False,
                                        pipeline_parallel_size=1, data_parallel_size=1),
        speculative_config=None)
    report = worker.identity_configure(policy(), 1, "baseline")
    assert report == {"condition": "baseline", "layers": {}}
    with pytest.raises(RuntimeError, match="already configured"):
        worker.identity_configure(policy(), 1, "guided")


def test_fail_if_hook_bypassed_or_fused():
    with pytest.raises(RuntimeError, match="bypassed"):
        check_reports([dict(condition="guided", layers={
            k: {"token_evaluations": 0} for k in policy()["layers"]})], policy(), "guided")
    m = model()
    m.layers[7].mlp.experts = SimpleNamespace(_fse_fuse_gate=True)
    with pytest.raises(RuntimeError, match="bypasses"):
        install(m, policy(), 1)


def test_matched_accuracy_and_token_comparison(tmp_path):
    for name, correct, count in (("baseline", False, 20), ("guided", True, 12)):
        p = tmp_path / name
        p.mkdir()
        (p / "manifest.json").write_text(json.dumps(dict(
            status="complete", condition=name, prompts_sha256="p", policy_sha256="q",
            engine_args={}, sampling_args={}, versions={}, scoring_contract="test", calibration_overlap=[])))
        (p / "generations.jsonl").write_text(json.dumps(dict(
            id="a", input={"prompt": "x"}, prompt_token_ids=[1], is_correct=correct,
            scoring_method="exact", generated_token_count=count, finish_reason="stop")) + "\n")
    result = compare(tmp_path / "baseline", tmp_path / "guided")
    assert result["accuracy_delta"] == 1
    assert result["mean_generated_tokens_delta"] == -8
    assert result["wrong_to_right"] == 1
    path = tmp_path / "guided" / "generations.jsonl"
    path.write_text(path.read_text().replace('"a"', '"b"'))
    with pytest.raises(ValueError, match="same IDs"):
        compare(tmp_path / "baseline", tmp_path / "guided")


def test_problem_overlap_uses_source_not_attempt_id():
    a = {"dataset": "math", "problem_id": "p__sample_00", "source_problem_id": "p"}
    b = dict(a, problem_id="p__sample_01")
    assert problem_key(a) == problem_key(b)


def test_prepare_excludes_unscored_and_splits_by_problem(tmp_path):
    from moe_exp.moe_identity_guiding.prepare import prepare
    traces = []
    for i in range(5):
        for sample in range(2):
            traces.append(dict(dataset="math", problem_id=f"p{i}_{sample}", source_problem_id=f"p{i}",
                               sample_id=sample, is_correct=bool(i % 2), prompt="Question",
                               gold_answer="1", model_logs={"selected_experts": "saved.pt"}))
    traces.append(dict(traces[0], is_correct=None))
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in traces))
    out = tmp_path / "split"
    info = prepare([path], out)
    assert info["excluded"]["unscored"] == 1
    calibration = [json.loads(s) for s in (out / "calibration.jsonl").read_text().splitlines()]
    evaluation = [json.loads(s) for s in (out / "prompts.jsonl").read_text().splitlines()]
    assert not {problem_key(r) for r in calibration} & {problem_key(r) for r in evaluation}
    assert len(calibration) + len(evaluation) == 10


@pytest.mark.parametrize("strength", [-1., 1., 2.])
def test_generate_worker_configuration_and_saved_scoring(tmp_path, monkeypatch, strength):
    import sys
    from moe_exp.moe_identity_guiding import run
    calls = []
    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.llm_engine = self
            self.pending = []
        def get_tokenizer(self):
            return SimpleNamespace(apply_chat_template=lambda messages, **kw: "rendered",
                                   encode=lambda prompt, **kw: [1, 2])
        def collective_rpc(self, name, **kwargs):
            calls.append(name)
            return [{"condition": "guided", "layers": {
                k: {"token_evaluations": 2} for k in policy()["layers"]}}]
        def add_request(self, request_id, prompt, params):
            assert prompt == {"prompt_token_ids": [1, 2]}
            self.pending.append(request_id)
        def has_unfinished_requests(self):
            return bool(self.pending)
        def step(self):
            return [SimpleNamespace(request_id=self.pending.pop(), finished=True,
                prompt_token_ids=[1, 2], outputs=[SimpleNamespace(
                    text=r"\boxed{B}", token_ids=[3, 4], finish_reason="stop")])]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params",
                        SimpleNamespace(RequestOutputKind=SimpleNamespace(FINAL_ONLY="final")))
    monkeypatch.setattr(run, "version", lambda name: "test")
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy()))
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps(dict(id="heldout", prompt="question", gold_answer="B",
                                      answer_type="choice")) + "\n")
    args = SimpleNamespace(policy=p, model="qwen-test", prompts=prompts, strength=strength,
                           allow_calibration_overlap=False, revision=None, dtype="auto",
                           tensor_parallel_size=1, max_model_len=128, max_tokens=16,
                           max_num_seqs=1, gpu_memory_utilization=.9, seed=42, temperature=0.,
                           top_p=1., output_dir=tmp_path / "guided", condition="guided")
    run.generate(args)
    assert calls[0]["enforce_eager"] and not calls[0]["enable_prefix_caching"]
    assert calls[1:] == ["identity_configure", "identity_diagnostics"]
    saved = json.loads((args.output_dir / "generations.jsonl").read_text())
    assert saved["is_correct"] is True and saved["generated_token_count"] == 2
    assert json.loads((args.output_dir / "manifest.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        run.generate(args)


def test_all_correct_calibration_cannot_identify_accuracy_lift():
    r = rows()
    r[1]["is_correct"] = True
    with pytest.raises(ValueError, match="both correct and incorrect"):
        fit(r, model="qwen-test", num_experts=3, top_k=2, min_support=1, max_experts=1)


def test_global_population_and_attempt_modes(tmp_path):
    from moe_exp.moe_identity_guiding.prepare import prepare_global
    generations, traces = [], []
    tensor = tmp_path / 'experts.pt'
    torch.save(torch.tensor([[[0, 1]]]), tensor)
    for dataset in ('a', 'b'):
        for i in range(10):
            for sample in range(3):
                row = dict(dataset=dataset, problem_id=f'p{i}_{sample}', source_problem_id=f'p{i}',
                           sample_id=sample, model_id='qwen', prompt=f'question {i}', gold_answer='1',
                           is_correct=None if sample == 2 else bool(sample),
                           model_logs={'selected_experts': str(tensor), 'layer_indices': [7]})
                generations.append(row)
                traces.append(row)
    gen = tmp_path / 'generations.jsonl'
    trace = tmp_path / 'traces.jsonl'
    gen.write_text('\n'.join(json.dumps(r) for r in generations))
    trace.write_text('\n'.join(json.dumps(r) for r in traces))
    out = tmp_path / 'global'
    m = prepare_global([gen], [trace], out)
    single = [json.loads(s) for s in (out/'prompts.single.jsonl').read_text().splitlines()]
    full = [json.loads(s) for s in (out/'prompts.full.jsonl').read_text().splitlines()]
    calibration = [json.loads(s) for s in (out/'calibration.jsonl').read_text().splitlines()]
    assert m['evaluation_problems'] == len(single) == 6
    assert m['evaluation_attempts'] == len(full) == 18
    assert all(r['sample_id'] == 0 for r in single)
    assert {problem_key(r) for r in single} == {problem_key(r) for r in full}
    assert not {problem_key(r) for r in calibration} & {problem_key(r) for r in full}
    assert len(calibration) == 28  # all scored calibration attempts, not just sample zero
    assert m['datasets']['a']['calibration_problems'] == 7
    assert m['excluded']['unscored_calibration'] == 14
    # Unscored evaluation attempts are included; evaluation does not select by correctness.
    assert len([r for r in full if r['sample_id'] == 2]) == 6
    with pytest.raises(FileExistsError):
        prepare_global([gen], [trace], out)
    generations[1]['prompt'] = 'different question'
    gen.write_text('\n'.join(json.dumps(r) for r in generations))
    with pytest.raises(ValueError, match='disagree'):
        prepare_global([gen], [trace], tmp_path/'bad')


@pytest.mark.parametrize('family,n,k', [('oss', 32, 4), ('gemma', 128, 8)])
def test_model_specific_native_router_hooks(family, n, k):
    from moe_exp.moe_identity_guiding.routing import routing_spec
    # OSS returns logits directly; Gemma returns normalized/projected logits
    # before its existing top-k + per-expert-scale dispatch function.
    class NativeRouter(torch.nn.Module):
        def forward(self, x):
            return x
    m = torch.nn.Module()
    m.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for layer in m.layers:
        layer.mlp = torch.nn.Module()
        if family == 'oss':
            layer.mlp.router = NativeRouter()
        else:
            layer.router = NativeRouter()
            layer.moe = torch.nn.Module()
            layer.moe.per_expert_scale = torch.nn.Parameter(torch.linspace(.5, 2., n))
    target = m.layers[1].mlp.router if family == 'oss' else m.layers[1].router
    untouched = m.layers[0].mlp.router if family == 'oss' else m.layers[0].router
    p = dict(schema_version=1, model='test', num_experts=n, top_k=k,
             layers={'1': {'scores': [0.] * (n-1) + [1.]}})
    config = (SimpleNamespace(model_type='gpt_oss', num_local_experts=n, num_experts_per_tok=k)
              if family == 'oss' else SimpleNamespace(model_type='gemma4_text',
                                                      num_experts=n, top_k_experts=k))
    assert routing_spec(config) == (family, n, k)
    logits = torch.linspace(1., 0., n)[None]
    hooks, handles = install(m, p, 3., family)
    guided = target(logits)
    assert guided.topk(k).indices[0, 0] == n-1
    assert guided.topk(k).indices.shape == (1,k)
    assert untouched(logits) is logits
    if family == 'gemma':
        # Native expert scales are still applied after selected-weight normalization.
        ids = guided.topk(k).indices
        weights = guided.softmax(-1).gather(1, ids)
        weights = weights / weights.sum(-1, keepdim=True)
        scaled = weights * m.layers[1].moe.per_expert_scale[ids]
        assert scaled[0,0] == weights[0,0] * 2
        torch.testing.assert_close(m.layers[1].moe.per_expert_scale,
                                   torch.linspace(.5, 2., n))
    assert hooks['1'].snapshot()['token_evaluations'] == 1
    for handle in handles:
        handle.remove()
    assert target(logits) is logits


@pytest.mark.parametrize('checkpoint,expected', [
    ('Qwen/Qwen3.5-35B-A3B-GPTQ-Int4', (256,8)),
    ('openai/gpt-oss-20b', (32,4)),
    ('nvidia/Gemma-4-26B-A4B-NVFP4', (128,8)),
])
def test_fit_expert_presets(checkpoint, expected):
    from moe_exp.moe_identity_guiding.profiles import expert_defaults
    assert expert_defaults(checkpoint) == expected


@pytest.mark.parametrize('family', ['oss', 'gemma'])
def test_worker_native_dimensions_and_baseline(family, monkeypatch):
    import sys
    from moe_exp.moe_identity_guiding.routing import routing_spec
    monkeypatch.setitem(sys.modules, 'vllm.platforms', SimpleNamespace(
        current_platform=SimpleNamespace(is_rocm=lambda: False)))
    text = (SimpleNamespace(model_type='gpt_oss', num_local_experts=3, num_experts_per_tok=2)
            if family == 'oss' else SimpleNamespace(model_type='gemma4_text',
                                                    num_experts=3, top_k_experts=2))
    worker = IdentityWorkerExtension()
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True, hf_text_config=text),
        parallel_config=SimpleNamespace(enable_expert_parallel=False, enable_eplb=False,
                                        pipeline_parallel_size=1, data_parallel_size=1),
        speculative_config=None)
    assert worker.identity_configure(policy(), 1, 'baseline')['layers'] == {}
    text.model_type = 'unsupported'
    with pytest.raises(ValueError, match='supports'):
        routing_spec(text)


@pytest.mark.parametrize('profile,model,root', [
    ('qwen', 'Qwen/Qwen3.5-35B-A3B-GPTQ-Int4', 'qwen_global'),
    ('oss', 'openai/gpt-oss-20b', 'oss_global'),
    ('gemma', 'nvidia/Gemma-4-26B-A4B-NVFP4', 'gemma_global'),
])
def test_global_launcher_profiles(profile, model, root, tmp_path):
    import os
    import subprocess
    env = dict(os.environ, MODEL_PROFILE=profile, DRY_RUN='true')
    for key in ('MODEL', 'OUTPUT_ROOT', 'GENERATION_ROOT', 'ROUTING_ROOT'):
        env.pop(key, None)
    env['OUTPUT_ROOT'] = str(tmp_path / root)
    result = subprocess.run(['bash', 'src/moe_exp/moe_identity_guiding/run_global.sh', 'all'],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.count('--model '+model) == 3  # fit, baseline, guided
    assert root in result.stdout
    assert 'prompts.full.sampling.jsonl' in result.stdout
    assert '--temperature 0.6 --top-p 0.95 --top-k 0 --seed 0' in result.stdout
    assert '/sampling/full/' in result.stdout


def test_original_sampling_seeds_and_template_metadata(tmp_path):
    from moe_exp.moe_identity_guiding.sampling import prepare_sampling, request_sampling
    rows = []
    for sample in range(32):
        rows.append(dict(dataset='aime24', problem_id=f'p_{sample}', source_problem_id='p',
                         sample_id=sample, model_id='gemma', prompt='question', gold_answer='1',
                         metadata={'generation_config': dict(seed=7000+sample, temperature=.6,
                           top_p=.95, top_k=0, max_tokens=32768,
                           chat_template_kwargs={'enable_thinking': True})}))
    generation = tmp_path/'traces.jsonl'
    generation.write_text('\n'.join(json.dumps(r) for r in rows))
    prompts = tmp_path/'prompts.jsonl'
    prompts.write_text('\n'.join(json.dumps(dict(
        id=json.dumps([r['dataset'],r['problem_id'],r['sample_id']]),
        **{k:r[k] for k in ('dataset','problem_id','source_problem_id','prompt','gold_answer')}
    )) for r in rows))
    out = tmp_path/'sampling.jsonl'
    assert prepare_sampling(prompts, [generation], out)['prompts'] == 32
    prepared = [json.loads(s) for s in out.read_text().splitlines()]
    sampling = dict(seed=0, temperature=.6, top_p=.95, top_k=-1, max_tokens=32768)
    baseline = request_sampling(prepared, sampling, require_original=True, model='gemma')
    guided = request_sampling(prepared, sampling, require_original=True, model='gemma')
    assert baseline == guided
    assert [p['seed'] for p in baseline] == list(range(7000,7032))
    assert prepared[0]['original_generation_config']['chat_template_kwargs'] == {'enable_thinking': True}
    assert prepare_sampling(prompts,[generation],out)['prompts'] == 32  # safe repeated preparation
    with pytest.raises(ValueError, match='temperature'):
        request_sampling(prepared, dict(sampling, temperature=0.), require_original=True, model='gemma')
    with pytest.raises(ValueError, match='Missing original'):
        request_sampling([{'id':'x'}], sampling, require_original=True)
    with pytest.raises(ValueError, match='model differs'):
        request_sampling(prepared, sampling, require_original=True, model='other')


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagnostics", ["minimal", "full"])
def test_negative_strength_penalizes_favored_expert(dtype, diagnostics):
    logits = torch.tensor([[3., 2.5, 1., 0.]], dtype=dtype)
    hook = IdentityBias([1., 0., 0., 0.], -1., 1, diagnostics=diagnostics)
    guided, extra = hook(None, (), (logits, "native"))
    assert extra == "native" and guided.dtype == dtype
    assert torch.equal(guided, torch.tensor([[2., 2.5, 1., 0.]], dtype=dtype))
    assert logits.argmax(-1).item() == 0
    assert guided.argmax(-1).item() == 1
    assert hook.snapshot()["token_evaluations"] == 1


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -float("inf")])
def test_identity_strength_must_still_be_finite(strength):
    with pytest.raises(ValueError, match="finite"):
        IdentityBias([1., 0.], strength, 1)
