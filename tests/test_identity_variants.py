"""Negative-association selection and paper-style expert intervention."""
import os
import subprocess

import pytest
import torch

from moe_exp.moe_identity_guiding.calibration import fit, validate_policy
from moe_exp.moe_identity_guiding.routing import IdentityBias, install


def test_negative_policy_selects_failure_expert_not_neutral():
    rows = [
        dict(id="a", is_correct=True, selected_experts=torch.tensor([[[0, 1]]])),
        dict(id="b", is_correct=False, selected_experts=torch.tensor([[[0, 2]]])),
    ]
    kwargs = dict(model="test", num_experts=3, top_k=2, min_support=1, max_experts=1)
    positive = fit(rows, **kwargs)
    negative = fit(rows, **kwargs, expert_polarity="negative", guiding_method="paper")
    assert positive["layers"]["0"]["scores"] == [0, 1, 0]
    assert negative["layers"]["0"]["scores"] == [0, 0, 1]
    assert negative["layers"]["0"]["experts"][2]["lift"] < 0
    validate_policy(negative)
    logits = torch.tensor([[2., 1., -4.]])
    hook = IdentityBias(negative["layers"]["0"]["scores"], 1, 2, guiding_method="paper")
    assert 2 in hook(None, (), logits).topk(2).indices[0]
    # Older policies retain fixed, positive semantics.
    for key in ("expert_polarity", "guiding_method", "paper_epsilon"):
        positive.pop(key)
    validate_policy(positive)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strength", [-1, 1])
def test_paper_extrema_and_native_dtype(dtype, strength):
    logits = torch.tensor([[3., 2., 1., -2.], [-5., 1., 4., 2.]], dtype=dtype)
    hook = IdentityBias([0, 1, 0, 0.5], strength, 2, guiding_method="paper")
    result, marker = hook(None, (), (logits, "aux"))
    scores = logits.float().log_softmax(-1)
    # Unselected log-probabilities remain unchanged; selected values reach
    # epsilon beyond the original extreme (or one representable step beyond).
    assert torch.equal(result[:, [0, 2]], scores[:, [0, 2]].to(dtype))
    extreme = scores.max(-1).values if strength == 1 else scores.min(-1).values
    target = extreme + strength * .01
    tolerance = 2 * torch.finfo(dtype).eps * extreme.abs().clamp_min(1)
    assert ((result[:, 1].float() - target).abs() <= tolerance).all()
    if strength > 0:
        assert (result[:, 1] > extreme.to(dtype)).all()
    else:
        assert (result[:, 1] < extreme.to(dtype)).all()
    assert marker == "aux" and result.dtype == dtype
    assert result.topk(2).indices.sort().values.tolist() == (
        [[1, 3], [1, 3]] if strength == 1 else [[0, 2], [0, 2]])
    # Selected experts have equal target values, irrespective of calibrated magnitude.
    assert torch.equal(result[:, 1], result[:, 3])


def test_paper_noops_and_empty_batch():
    logits = torch.randn(3, 4)
    assert IdentityBias([0, 1, 0, 0], 0, 2, guiding_method="paper")(None, (), logits) is logits
    hook = IdentityBias([0, 0, 0, 0], 1, 2, guiding_method="paper")
    assert torch.equal(hook(None, (), logits), logits)
    assert hook(None, (), logits[:0]).shape == (0, 4)


@pytest.mark.parametrize("kwargs", [
    dict(paper_epsilon=0), dict(paper_epsilon=float("nan")),
    dict(guiding_method="unknown"), dict(guiding_method="paper", strength=2),
])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        IdentityBias([0, 1, 0], top_k=2, **({"strength": 1} | kwargs))


def test_install_reads_frozen_paper_method():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.layers[0].mlp = torch.nn.Module()
            self.layers[0].mlp.gate = torch.nn.Identity()
    policy = dict(schema_version=1, model="test", num_experts=3, top_k=2,
                  guiding_method="paper", expert_polarity="negative", paper_epsilon=.02,
                  layers={"0": {"scores": [0, 0, 1]}})
    model = Model()
    hooks, handles = install(model, policy, 1)
    assert hooks["0"].guiding_method == "paper"
    assert hooks["0"].paper_epsilon == .02
    assert model.layers[0].mlp.gate(torch.tensor([[3., 2., -8.]])).argmax().item() == 2
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("polarity,method", [
    ("negative", "fixed"), ("negative", "paper"), ("positive", "paper"),
])
def test_launcher_isolates_variants_even_with_custom_root(tmp_path, polarity, method):
    env = dict(os.environ, DRY_RUN="true", OUTPUT_ROOT=str(tmp_path / "runs"),
               EXPERT_POLARITY=polarity, GUIDING_METHOD=method, PAPER_EPSILON="0.01", STRENGTH="1")
    result = subprocess.run(["bash", "src/moe_exp/moe_identity_guiding/run_global.sh", "all"],
                            env=env, capture_output=True, text=True, check=True)
    assert f"--expert-polarity {polarity}" in result.stdout
    assert f"--guiding-method {method}" in result.stdout
    folder = f"variants/{polarity}_{method}"
    if method == "paper":
        folder += "_eps_0.01"
    assert f"{folder}/strength_1/sampling/full/guided" in result.stdout
    assert not (tmp_path / "runs").exists()
