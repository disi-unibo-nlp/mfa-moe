"""Scoped native pre-reduction contributions; installed inference code is untouched."""
from contextlib import contextmanager
import torch
from .contracts import require


class MethodLimitation(RuntimeError):
    campaign_failure_scope='method'


@contextmanager
def measurement_kernel(runner):
    method=getattr(runner.routed_experts,'quant_method',None)
    kernel=getattr(method,'moe_kernel',None)
    if kernel is None:
        raise MethodLimitation('native backend exposes no modular pre-reduction interface')
    original=kernel
    experts=kernel.fused_experts
    # GPT exposes the same quantization/activation through an unfused path.
    if type(experts).__name__=='OAITritonExperts':
        from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import UnfusedOAITritonExperts
        from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
        experts=UnfusedOAITritonExperts(experts.moe_config,experts.quant_config)
        kernel=FusedMoEKernel(kernel.prepare_finalize,experts)
        method.moe_kernel=kernel
    if not callable(getattr(experts,'moe_sum',None)):
        method.moe_kernel=original
        raise MethodLimitation('native backend does not expose selected-slot output before reduction')
    try:yield experts
    finally:method.moe_kernel=original


def contributions(runner,x,weights,ids):
    """Run unchanged IDs/geometry, capture and repeat the SAME native reduction."""
    with measurement_kernel(runner) as experts:
        original=experts.moe_sum
        own='moe_sum' in vars(experts);prior=vars(experts).get('moe_sum')
        captures=[]
        def reduce(input,output,*args,**kwargs):
            require(input.ndim==3 and tuple(input.shape[:2])==tuple(ids.shape),
                    'native contribution slot geometry differs')
            saved=input.detach().clone()
            reconstructed=torch.empty_like(output)
            original(saved,reconstructed,*args,**kwargs)
            captures.append((saved,reconstructed))
            return original(input,output,*args,**kwargs)
        experts.moe_sum=reduce
        try:output=runner.routed_experts.forward_modular(x.clone(),weights,ids)
        finally:
            if own:experts.moe_sum=prior
            else:del experts.moe_sum
        if len(captures)!=1:
            raise MethodLimitation('native backend pre-reduction capture was not uniquely observable')
        selected,reconstruction=captures[0]
        if runner.moe_config.tp_size>1:
            from vllm.distributed import tensor_model_parallel_all_reduce
            selected=tensor_model_parallel_all_reduce(selected)
            reconstruction=tensor_model_parallel_all_reduce(reconstruction)
        if not runner._fused_output_is_reduced and runner.moe_config.tp_size>1:
            output=tensor_model_parallel_all_reduce(output)
        width=x.shape[-1]
        return output[...,:width],selected[...,:width],reconstruction[...,:width]


def measure(runner,x,native_weights,ids):
    # Compare the measurement adapter to the ordinary native operator at the
    # same states, preserving actual routing weights and reduction arithmetic.
    ordinary=runner.routed_experts.forward_modular(x.clone(),native_weights,ids)
    if not runner._fused_output_is_reduced and runner.moe_config.tp_size>1:
        from vllm.distributed import tensor_model_parallel_all_reduce
        ordinary=tensor_model_parallel_all_reduce(ordinary)
    ordinary=ordinary[...,:x.shape[-1]]
    adapted,weighted,reconstructed=contributions(runner,x,native_weights,ids)
    supported=bool(torch.allclose(ordinary.float(),adapted.float(),atol=.02,rtol=.02) and
                   torch.allclose(ordinary.float(),reconstructed.float(),atol=.02,rtol=.02))
    error=max(float((ordinary.float()-adapted.float()).abs().max()),float((ordinary.float()-reconstructed.float()).abs().max()))
    # This invocation changes weights only; selected IDs, native top-k geometry,
    # biases, quantized weights, activation and TP semantics remain identical.
    _,unweighted,_=contributions(runner,x,torch.ones_like(native_weights),ids)
    return unweighted.float().norm(dim=-1),supported,error
