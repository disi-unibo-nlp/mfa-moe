"""Native boundary scores for already captured HF GLM routing.

The legacy probability gaps remain unchanged. New temporal boundaries use the
actual sigmoid affinity plus the router's correction bias, with one group only.
"""
def glm_reducer(model):
    from moe_exp.models.router_adapters import decoder_layers,available_router_layers
    from moe_exp.correlation_pipeline.dynamics.routing import reduce_arrays,probabilities
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.dynamics.common import DEFAULT_CONFIG,digest
    from moe_exp.correlation_pipeline.dynamics import SCHEMA_VERSION
    from moe_exp.correlation_pipeline.spans import trace_digest
    from moe_exp.correlation_pipeline.provenance import code_provenance
    config=model.config
    if config.model_type!='glm4_moe_lite' or config.n_group!=1 or config.topk_group!=1:
        raise ValueError('Only verified single-group GLM selection is supported')
    biases={l:decoder_layers(model)[l].mlp.gate.e_score_correction_bias.detach().float().cpu().numpy().copy() for l in available_router_layers(model)}
    def reduce(trace,router,selected,weights,layers,semantics):
        layout=saved_layout(trace)
        if layout['token_count']!=router.shape[1] or not semantics.get('native_selection'):raise ValueError('Native exact-token capture required')
        class FullProbabilities:
            shape=router.shape
            def __getitem__(self,index):return probabilities(router[index].detach().float().cpu().numpy())
        class NativeScores:
            shape=router.shape
            def __getitem__(self,index):
                import numpy as np
                layer,tokens=index
                return np.exp(router[index].detach().float().cpu().numpy())+biases[layers[layer]]
        semantics={**semantics,'deterministic_topk':True,'selection_scores':'native_sigmoid_plus_bias','num_experts':router.shape[-1],'selection_groups':1}
        rows=reduce_arrays(selected,layers=layers,model=trace.model_id,probabilities_array=FullProbabilities(),weights=weights,selection_scores=NativeScores(),semantics=semantics,token_indices=layout['reasoning_tokens'])
        return {'schema_version':SCHEMA_VERSION,'config':dict(DEFAULT_CONFIG),'trace_sha256':trace_digest(trace),'annotation_sha256':digest(trace.metadata.get('reasoning_annotation')),'forward_provenance':trace.metadata.get('forward_provenance'),'code':code_provenance(),'semantics':semantics,'windows':rows}
    return reduce
