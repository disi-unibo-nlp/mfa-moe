from types import SimpleNamespace
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from measurement_study.capture_buffers import (
    CaptureBuffers, capture_memory_budget, verified_modular_backend,
    compare_backend_captures, MAX_WORKER_BYTES,
)


def signals(n=2):
    import torch
    return dict(ids=torch.tensor([[0, 1]] * n, dtype=torch.int32),
                weights=torch.tensor([[.712345, .287655]] * n),
                router=torch.tensor([[2., 1., 0.]] * n),
                selection=torch.tensor([[2., 1., 0.]] * n),
                hidden=torch.tensor([[.2, .3, .4, .5]] * n, dtype=torch.bfloat16))


def test_out_of_order_chunks_preallocate_and_preserve_native_precision():
    import torch
    buffers = CaptureBuffers([1, 3], 4)
    values = signals()
    for layer in [3, 1]:
        buffers.write(layer, [2, 0], values)
        buffers.write(layer, [3, 1], values)
    arrays = buffers.require_complete()
    assert arrays['ids'].shape == (2, 4, 2)
    assert arrays['weights'].dtype == torch.float32
    pointer = arrays['hidden'].data_ptr()
    buffers.write(1, [2, 0], values)
    assert arrays['hidden'].data_ptr() == pointer
    stored = buffers.for_storage()
    assert stored['hidden'] is arrays['hidden']
    assert stored['weights'].dtype == torch.float16
    assert arrays['weights'].dtype == torch.float32
    assert not torch.equal(stored['weights'].float(), arrays['weights'])


def test_missing_duplicate_changed_and_out_of_range_tokens_fail():
    buffers = CaptureBuffers([2], 4)
    buffers.write(2, [0, 1], signals())
    with pytest.raises(RuntimeError, match='Incomplete'):
        buffers.require_complete()
    with pytest.raises(ValueError, match='Duplicate'):
        buffers.write(2, [1, 1], signals())
    with pytest.raises(ValueError, match='outside'):
        buffers.write(2, [1, 4], signals())
    changed = signals(); changed['weights'][0, 0] += 1e-7
    with pytest.raises(RuntimeError, match='Recomputation'):
        buffers.write(2, [0, 1], changed)


def test_memory_cap_checked_before_allocating():
    buffers = CaptureBuffers([1, 2], 32768, max_bytes=1024)
    with pytest.raises(ValueError, match='before allocation'):
        buffers.write(1, [0, 1], signals())
    assert buffers.allocated_bytes == 0 and buffers.arrays == {}


def test_scaled_cap_supports_long_traces_and_rejects_unbounded_allocations():
    short = capture_memory_budget(32768, 40, 2048, 256, 8)
    long = capture_memory_budget(131072, 40, 2048, 256, 8)
    assert long['transient_bytes'] > 15 * 1024**3
    assert short['worker_bytes'] < long['worker_bytes'] < MAX_WORKER_BYTES
    with pytest.raises(ValueError, match='envelope'):
        capture_memory_budget(1310720, 40, 2048, 256, 8)


def test_dtype_or_shape_drift_and_nonfinite_rejected():
    import torch
    buffers = CaptureBuffers([0], 2)
    buffers.write(0, [0, 1], signals())
    with pytest.raises(ValueError, match='dtype or width'):
        buffers.write(0, [0, 1], {'weights': signals()['weights'].double()})
    with pytest.raises(ValueError, match='Nonfinite'):
        buffers.write(0, [0, 1], {'hidden': torch.full((2, 4), float('nan'))})


def test_nonzero_rank_excludes_hidden_and_storage_weight_overflow_rejected():
    values = signals(); values.pop('hidden')
    buffers = CaptureBuffers([0], 2, rank=1)
    buffers.write(0, [0, 1], values)
    assert 'hidden' not in buffers.require_complete()
    with pytest.raises(ValueError, match='Unexpected'):
        buffers.write(0, [0, 1], {'hidden': signals()['hidden']})
    other = CaptureBuffers([0], 2)
    values = signals(); values['weights'] *= 1e6
    other.write(0, [0, 1], values)
    with pytest.raises(ValueError, match='finite nonnegative fp16'):
        other.for_storage()


@pytest.mark.parametrize('attribute,name', [
    ('wna16_backend', 'MARLIN'), ('wna16_moe_backend', 'MARLIN'),
    ('mxfp4_backend', 'MARLIN'),
    ('mxfp4_backend', 'TRITON_UNFUSED'), ('unquantized_backend', 'TRITON'),
    ('fp8_backend', 'MARLIN'), ('nvfp4_backend', 'MARLIN'),
])
def test_allowlisted_native_modular_backends(attribute, name):
    quant = SimpleNamespace(is_monolithic=False, **{attribute: SimpleNamespace(name=name)})
    assert verified_modular_backend(quant) == attribute + '/' + name
    quant.is_monolithic = True
    with pytest.raises(RuntimeError, match='monolithic'):
        verified_modular_backend(quant)


def test_unknown_or_ambiguous_backend_fails_closed():
    with pytest.raises(RuntimeError, match='Unverified'):
        verified_modular_backend(SimpleNamespace(is_monolithic=False, wna16_backend=SimpleNamespace(name='UNKNOWN')))
    with pytest.raises(RuntimeError, match='one verified'):
        verified_modular_backend(SimpleNamespace(is_monolithic=False))


def differential_pair():
    data = {name: value.unsqueeze(0) for name, value in signals().items()}
    reference = {name: value.clone() for name, value in data.items()}
    reference['weights'] *= 1.8
    reference['audit'] = {'native_dispatch': True, 'layers': [1]}
    data['weights'] = data['weights'].half()
    data['audit'] = {'native_dispatch': True, 'layers': [1], 'routed_output_scaling': {'1': 1.8}}
    return reference, data


def test_glm_differential_respects_audited_output_scaling_and_fp16_rounding():
    a, b = differential_pair()
    receipt = compare_backend_captures(a, b, total_input_tokens=5, layers=[1])
    assert receipt['status'] == 'passed' and receipt['vllm_weight_scale'] == 1.8
    assert receipt['identical_kernels'] is False and receipt['flips'] == 0
    assert receipt['per_layer'][0]['weight_relative_l2'] is not None
    b['audit']['routed_output_scaling']['1'] = 1.0
    with pytest.raises(ValueError, match='effective weights'):
        compare_backend_captures(a, b, total_input_tokens=5, layers=[1])


def test_differential_rejects_true_mismatch_long_trace_and_missing_provenance():
    import torch
    a, b = differential_pair()
    b['ids'][0, 0, 0] = 2
    b['selection'][0, 0] = torch.tensor([0., 1., 2.])  # native side stays internally consistent
    with pytest.raises(ValueError, match='sets differ'):
        compare_backend_captures(a, b, total_input_tokens=5, layers=[1], identical_kernels=True)
    # a true mismatch is rejected by the selection-score noise gate
    with pytest.raises(ValueError, match='selection scores relative-L2|beyond the observed score noise'):
        compare_backend_captures(a, b, total_input_tokens=5, layers=[1])
    a, b = differential_pair()
    with pytest.raises(ValueError, match='32768'):
        compare_backend_captures(a, b, total_input_tokens=32769, layers=[1])
    del b['audit']['routed_output_scaling']
    with pytest.raises(ValueError, match='scales'):
        compare_backend_captures(a, b, total_input_tokens=5, layers=[1])


def near_tie_pair(tokens=2000, flipped=1):
    """HF and vLLM pick different experts only where the reference gap < eps."""
    import torch
    ids=torch.tensor([[[0,1]]*tokens],dtype=torch.int32)
    native_ids=ids.clone()
    for token in range(flipped):
        native_ids[0,token,1]=2
    reference_scores=torch.zeros(1,tokens,3)
    reference_scores[...,0]=1.0;reference_scores[...,1]=0.9999;reference_scores[...,2]=0.99985
    native_scores=reference_scores.clone()
    for token in range(flipped):
        native_scores[0,token,1]=0.99985;native_scores[0,token,2]=0.9999
    common=dict(weights=torch.full((1,tokens,2),.5),router=torch.ones(1,tokens,3),
                hidden=torch.zeros(1,tokens,4,dtype=torch.bfloat16))
    reference=dict(ids=ids,selection=reference_scores,audit={'native_dispatch':True},**common)
    native=dict(ids=native_ids,selection=native_scores,audit={'native_dispatch':True},
                weights=common['weights'].clone(),router=common['router'].clone(),hidden=common['hidden'].clone())
    return reference,native


def test_score_noise_gate_and_per_layer_weight_bound():
    import torch
    from measurement_study.capture_buffers import compare_hf_vllm_capture
    reference,native=near_tie_pair(flipped=1)
    receipt=compare_hf_vllm_capture(reference,native,total_input_tokens=2000,layers=[0])
    assert receipt['status']=='passed'
    assert receipt['near_ties']==2000 and receipt['near_tie_flips']==1 and receipt['unexplained_flips']==0
    assert receipt['flip_fraction']==1/2000 and receipt['selection_relative_l2']<5e-3
    assert receipt['native_selected_sets_identical'] is False
    assert receipt['per_layer'][0]['near_tie_flips']==1 and receipt['per_layer'][0]['score_shift_max']>0
    # Flips explained by a small score shift pass whatever their count (reported, not gated),
    # so a trace with few near-ties cannot fail on a single numerical flip.
    reference,native=near_tie_pair(flipped=40)
    receipt=compare_hf_vllm_capture(reference,native,total_input_tokens=2000,layers=[0])
    assert receipt['flips']==40 and receipt['unexplained_flips']==0
    # A large score disagreement fails on the selection-score bound, flip or not.
    reference,native=near_tie_pair(flipped=1)
    reference['selection'][0,0]=torch.tensor([1.0,0.9,0.1])
    native['selection'][0,0]=torch.tensor([1.0,0.1,0.9])
    native['selection'][0,1:]=native['selection'][0,1:]*1.02
    with pytest.raises(ValueError,match='selection scores relative-L2'):
        compare_hf_vllm_capture(reference,native,total_input_tokens=2000,layers=[0])
    # Per-layer effective-weight drift beyond the 0.5% bound is rejected.
    reference,native=near_tie_pair(flipped=0)
    native['weights']=native['weights']*1.1
    with pytest.raises(ValueError,match='effective weights relative-L2'):
        compare_hf_vllm_capture(reference,native,total_input_tokens=2000,layers=[0])
    # Identical kernels still require exact expert sets.
    reference,native=near_tie_pair(flipped=1)
    with pytest.raises(ValueError,match='sets differ'):
        compare_hf_vllm_capture(reference,native,total_input_tokens=2000,layers=[0],identical_kernels=True)


def test_unexplained_flip_is_rejected_even_when_scores_agree():
    from measurement_study import capture_buffers, native_selection
    reference,native=near_tie_pair(flipped=0)
    native['ids']=native['ids'].clone();native['ids'][0,0,1]=2
    # bypass the native boundary check to reach the noise gate directly
    original=native_selection.require_native_boundary
    native_selection.require_native_boundary=lambda selection, ids: None
    try:
        with pytest.raises(ValueError,match='beyond the observed score noise'):
            capture_buffers.compare_hf_vllm_capture(reference,native,total_input_tokens=2000,layers=[0])
    finally:
        native_selection.require_native_boundary=original


def test_snapshot_path_confined_to_new_campaign_roots():
    from measurement_study.capture_buffers import approved_capture_path
    roots = [
        '/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/campaign-v3',
        '/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3',
        '/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3',
        '/leonardo/home/userexternal/lmolfett/measurements/campaign-v3',
    ]
    for root in roots:
        assert str(approved_capture_path(root + '/capture/model/transient.pt')).endswith('/transient.pt')
        for outside in (root, root + '-other/transient.pt', root + '/../historical/transient.pt'):
            with pytest.raises(ValueError, match='Unapproved'):
                approved_capture_path(outside)


def test_backend_differential_rejects_overflowed_norm():
    import torch
    from measurement_study.capture_buffers import compare_hf_vllm_capture
    ids=torch.tensor([[[0]]])
    base={'ids':ids,'weights':torch.ones(1,1,1),'selection':torch.tensor([[[2.,1.]]]),
          'router':torch.tensor([[[2.,1.]]]),'hidden':torch.full((1,1,4),3e38),
          'audit':{'native_dispatch':True}}
    altered={**base,'hidden':torch.full((1,1,4),-3e38)}
    with pytest.raises(ValueError,match='differential error'):
        compare_hf_vllm_capture(base,altered,total_input_tokens=2,layers=[0])


def stacked_pair(layers=3, tokens=50):
    import torch
    reference, native = near_tie_pair(tokens=tokens, flipped=0)
    for data in (reference, native):
        for key in ('ids', 'selection', 'weights', 'router'):
            data[key] = torch.cat([data[key]] * layers, 0)
        data['hidden'] = torch.ones(layers, tokens, 4)
    return reference, native


def test_router_and_hidden_drift_is_judged_per_layer():
    import torch
    from measurement_study.capture_buffers import compare_hf_vllm_capture, DifferentialFailed
    layers = list(range(8))
    reference, native = stacked_pair(layers=8)
    # bf16 drift that grows smoothly with depth: 0.1% at the first routed layer, ~2.7% at the last
    native['router'] = reference['router'] * (1 + 0.001 * 1.6 ** torch.arange(8.))[:, None, None]
    receipt = compare_hf_vllm_capture(reference, native, total_input_tokens=50, layers=layers)
    assert receipt['status'] == 'passed' and receipt['router_relative_l2'] > .005
    per_layer = [row['router_relative_l2'] for row in receipt['per_layer']]
    assert per_layer == sorted(per_layer) and per_layer[0] < .002
    # a semantic error already shows at the first routed layer
    native['router'] = reference['router'] * 1.02
    with pytest.raises(ValueError, match='first routed layer 0: router='):
        compare_hf_vllm_capture(reference, native, total_input_tokens=50, layers=layers)
    # a single-layer spike (mapping bug) is a discontinuity even when the rest agrees
    spike = torch.ones(8); spike[5] = 1.3
    native['router'] = reference['router'] * spike[:, None, None]
    with pytest.raises(ValueError, match='router error discontinuity at layers \\[5\\]'):
        compare_hf_vllm_capture(reference, native, total_input_tokens=50, layers=layers)
    # gross drift fails globally even when the first layer agrees, and the report is kept
    native['router'] = reference['router'] * torch.tensor([1.0] + [1.5] * 7)[:, None, None]
    with pytest.raises(DifferentialFailed) as failed:
        compare_hf_vllm_capture(reference, native, total_input_tokens=50, layers=layers)
    report = failed.value.report
    assert report['status'] == 'failed' and report['per_layer'][1]['router_relative_l2'] > .4
    assert any('differential error: router=' in failure for failure in report['failures'])


def test_effective_weights_are_gated_at_the_first_routed_layer_and_reported_deeper():
    import torch
    from measurement_study.capture_buffers import compare_hf_vllm_capture
    reference, native = stacked_pair()
    native['weights'] = reference['weights'] * torch.tensor([1.0, 1.02, 1.03])[:, None, None]
    receipt = compare_hf_vllm_capture(reference, native, total_input_tokens=50, layers=[0, 1, 2])
    assert receipt['status'] == 'passed' and receipt['per_layer'][2]['weight_relative_l2'] > .02
    native['weights'] = reference['weights'] * torch.tensor([1.02, 1.0, 1.0])[:, None, None]
    with pytest.raises(ValueError, match='layer 0 effective weights relative-L2'):
        compare_hf_vllm_capture(reference, native, total_input_tokens=50, layers=[0, 1, 2])


def test_dynamics_agreement_and_discontinuities():
    import numpy as np
    from measurement_study.capture_buffers import dynamics_agreement, discontinuities
    rng = np.random.default_rng(0)
    # routing that drifts over the trace: a slowly moving preferred expert band
    layers, tokens, experts = 3, 2048, 32
    base = (np.arange(tokens)[None, :, None] // 97 + np.arange(4)[None, None, :] * 3) % experts
    ids = np.repeat(base, layers, 0) + rng.integers(0, 2, size=(layers, tokens, 4))
    ids = np.minimum(ids, experts - 1)
    same = dynamics_agreement(ids, ids, experts)
    assert same['windows'] == tokens // 32 and all(rho > .999 for rho in same['spearman'].values())
    scrambled = rng.integers(0, experts, size=ids.shape)
    assert min(dynamics_agreement(ids, scrambled, experts)['spearman'].values()) < .5
    drift = [0.002 * 1.08 ** i for i in range(40)]
    assert discontinuities(drift) == []
    spike = list(drift); spike[20] = 0.5
    assert discontinuities(spike) == [20]


def test_first_layer_weights_gate_on_the_elementwise_tail():
    import torch
    from measurement_study.capture_buffers import compare_hf_vllm_capture
    layers = list(range(8))
    reference, native = stacked_pair(layers=8, tokens=400)
    noise = 1 + 0.004 * torch.randn(native['weights'].shape, generator=torch.Generator().manual_seed(0))
    native['weights'] = reference['weights'] * noise
    receipt = compare_hf_vllm_capture(reference, native, total_input_tokens=400, layers=layers)
    assert receipt['status'] == 'passed' and receipt['first_layer_weight_elementwise']['p99'] < .02
    # a bug touching one expert's weights (2% of elements off by 20%) fails on the tail
    native['weights'] = reference['weights'].clone()
    native['weights'][0, :16, 0] *= 1.2
    with pytest.raises(ValueError, match='elementwise p99'):
        compare_hf_vllm_capture(reference, native, total_input_tokens=400, layers=layers)
