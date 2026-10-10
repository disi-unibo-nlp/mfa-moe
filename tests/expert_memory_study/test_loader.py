from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from moe_exp.expert_memory_study.loader import RouterMask, mapping, tensor_action, weight_iterator
from moe_exp.expert_memory_study.worker import numerical_error, storage_inventory


@pytest.mark.parametrize("suffix", ["weight", "weight_scale", "weight_scale_2", "weight_scale_inv", "qweight", "qzeros", "scales", "g_idx", "input_scale"])
def test_all_individual_packed_metadata_uses_same_map(suffix):
    masks = {"3": [1, 3, 5]}
    name = "model.layers.3.mlp.experts.3.down_proj." + suffix
    assert tensor_action(name, [8, 2], masks, 8) == ("model.layers.3.mlp.experts.1.down_proj." + suffix, None)
    assert tensor_action(name.replace("experts.3.", "experts.2."), [], masks, 8)[0] is None


def test_fused_gpt_scales_bias_and_tensor_slices(tmp_path):
    import json
    root = tmp_path / "weights"
    root.mkdir()
    tensors = {"model.layers.0.mlp.experts.gate_up_proj_scales": torch.arange(24).reshape(8, 3),
               "model.layers.0.mlp.experts.down_proj_bias": torch.arange(8).float(),
               "model.layers.0.mlp.experts.2.down_proj.qzeros": torch.arange(4),
               "model.layers.0.mlp.experts.1.down_proj.qzeros": torch.arange(4),
               "model.layers.0.mlp.router.weight": torch.ones(8, 3)}
    save_file(tensors, root / "weights.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "weights.safetensors" for k in tensors}}))
    ledger = {}
    values = dict(weight_iterator(root, {"0": [1, 3, 5]}, 8, ledger=ledger))
    assert torch.equal(values["model.layers.0.mlp.experts.gate_up_proj_scales"], tensors["model.layers.0.mlp.experts.gate_up_proj_scales"][[1, 3, 5]])
    assert values["model.layers.0.mlp.experts.down_proj_bias"].tolist() == [1, 3, 5]
    assert "model.layers.0.mlp.experts.0.down_proj.qzeros" in values
    assert ledger["model.layers.0.mlp.experts.2.down_proj.qzeros"]["status"] == "OMITTED"
    assert values["model.layers.0.mlp.router.weight"].shape[0] == 8


def test_ambiguous_fused_metadata_and_nonbijective_map_fail_closed():
    with pytest.raises(ValueError, match="unclassified"):
        tensor_action("model.layers.0.mlp.experts.scales", [1], {"0": [1, 3]}, 8)
    with pytest.raises(ValueError, match="bijective"):
        mapping([1, 1, 3], 8)


class Selector:
    def __init__(self, *, sigmoid=False):
        self.top_k = 2
        self.e_score_correction_bias = torch.tensor([1000., 0., 1000., 0.]) if sigmoid else None
        self.sigmoid = sigmoid

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        scores = router_logits.sigmoid() if self.sigmoid else router_logits.softmax(-1)
        selection = scores if self.e_score_correction_bias is None else scores + self.e_score_correction_bias
        ids = selection.topk(2, dim=-1).indices
        weights = scores.gather(-1, ids)
        return weights / weights.sum(-1, keepdim=True), ids.to(indices_type)


@pytest.mark.parametrize("sigmoid", [False, True])
def test_zero_removed_dispatch_original_router_width_and_bias_restoration(sigmoid):
    router = Selector(sigmoid=sigmoid)
    prior = router.e_score_correction_bias
    patch = RouterMask(router, [1, 3], 4, compact=True)
    logits = torch.tensor([[10., 2., 10., 1.]])
    weights, compact = router._compute_routing(torch.ones(1, 2), logits, torch.int32)
    assert sorted(compact[0].tolist()) == [0, 1]
    assert weights.sum().item() == pytest.approx(1.)
    assert router.e_score_correction_bias is prior and patch.removed_dispatch == 0
    assert logits.tolist() == [[10., 2., 10., 1.]]
    patch.restore()
    assert sorted(router._compute_routing(torch.ones(1, 2), logits, torch.int32)[1][0].tolist()) == [0, 2]


def test_reference_all_keep_fidelity_at_same_state():
    router = Selector()
    logits = torch.tensor([[1., 2., 3., 4.]])
    native = router._compute_routing(torch.ones(1, 2), logits, torch.int32)
    patch = RouterMask(router, list(range(4)), 4, compact=True)
    actual = patch.compute(torch.ones(1, 2), logits, torch.int32)
    assert torch.equal(native[1], actual[1])
    assert numerical_error(native[0], actual[0], atol=1e-6, rtol=1e-5)["pass"]


def test_unique_storage_counts_aliases_once():
    module = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.ones(3, 4))
    module.register_parameter("weights", parameter)
    module.register_buffer("alias", parameter.detach())
    assert storage_inventory(module)["unique_bytes"] == 48
