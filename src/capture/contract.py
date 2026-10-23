"""Strict completeness contract for the cross-model measurement campaign.

This contract is independent of the earlier partial, exploratory CPU summaries.
No metric or layer may disappear silently from a comparative result.
"""
from __future__ import annotations
import math

SCHEMA_VERSION = 1
DATASETS = ('math500','aime24','aime25','amc23','minerva','olympiad')
LEGACY = ('router_entropy','router_confidence','router_margin','router_selected_mass',
          'router_boundary_margin','router_topk_non_topk_gap','router_switch_rate',
          'router_topk_overlap','hidden_norm','hidden_step_distance','hidden_router_geometry')
TEMPORAL = ('selection_entropy','boundary_gap_mean','weak_boundary_fraction',
    'full_local_entropy','full_marginal_entropy','full_entropy_difference','full_jsd',
    'full_shuffled_jsd','full_jsd_excess','mixture_local_entropy',
    'mixture_marginal_entropy','mixture_entropy_difference','mixture_jsd',
    'mixture_shuffled_jsd','mixture_jsd_excess','mixture_effective_experts','set_turnover',
    *tuple(f'{prefix}_lag{lag}' for prefix in ('overlap','overlap_corrected','overlap_shuffled','overlap_excess')
           for lag in (1,4,16,64)))
CLASSES = ('Read','Analyze','Plan','Implement','Explore','Verify','Monitor')
CLASS_METRICS = ('sentence_occupancy','token_occupancy','next_sentence_transitions',
 'conditional_class_change','previous_conditioned_transitions','transition_entropy',
 'stay_rate','exit_rate','observed_run_sentences','observed_run_tokens',
 'return_sentences','return_tokens','motifs_2','motifs_3','cause_specific_exit_hazards',
 'class_routing_contrasts','expert_class_selection_rate_difference',
 'expert_class_npmi','backtracking_steps','contradiction_steps','self_correction_steps')
SIGNALS = ('exact_prompt_ids','exact_completion_ids','completion_offsets',
 'reasoning_segments','native_selected_ids','native_executed_weights',
 'full_router_distribution','selection_rule_and_scores','decoder_input_hidden_summaries',
 'layer_qualified_expert_counts','sentence_labels','answer_correctness','termination')


def routed_layers(config):
    """Absolute decoder depths; hybrid dense/attention-only layers are excluded."""
    c=config.get('text_config',config);family=c['model_type']
    if family=='nemotron_h':
        kinds=c.get('layers_block_type')
        if kinds is not None:return [i for i,k in enumerate(kinds) if k=='moe']
        pattern=c.get('hybrid_override_pattern')
        if pattern is None:raise ValueError('Missing hybrid topology')
        return [i for i,k in enumerate(pattern) if k=='E']
    n=c['num_hidden_layers']
    if family=='glm4_moe_lite':return list(range(c['first_k_dense_replace'],n))
    if family in ('gemma4','gemma4_text'):
        if not c.get('enable_moe_block'):raise ValueError('Gemma configuration is not MoE')
        return list(range(n))
    if family=='qwen3_moe':
        step=c.get('decoder_sparse_step',1);dense=c.get('mlp_only_layers',[])
        return [i for i in range(n) if i not in dense and (i+1)%step==0]
    if family in ('qwen3_5_moe','qwen3_5_moe_text','gpt_oss'):return list(range(n))
    raise ValueError('Unverified routing topology: '+family)


def extraction_options(spec, generation, annotations, output, datasets):
    """Common replay flags. No probe subset or router-only fast path is allowed."""
    if not spec.get('revision'):raise ValueError('Pinned checkpoint revision required')
    return ['--model-id',spec['forward_model'],'--generation-model',spec['generation_model'],
        '--revision',spec['revision'],'--local-files-only','--quantization',spec['quantization'],
        '--generation-dir',str(generation),'--annotation-dir',str(annotations),
        '--output-dir',str(output),'--datasets',*datasets,'--all-router-layers',
        '--dynamics','--views','full','class','position','--max-geometry-tokens','128']


def check_capture(evidence, expected):
    """Reject partial capture or changed execution before running production."""
    problems=[]
    if evidence.get('schema_version')!=SCHEMA_VERSION:problems.append('capture_schema')
    if evidence.get('layers')!=expected['layers']:problems.append('layer_coverage')
    for key in ('checkpoint','revision','precision','backend','hidden_observation_point'):
        if evidence.get(key)!=expected.get(key):problems.append(key)
    for key in ('native_dispatch_verified','exact_token_alignment','shared_branch_preserved'):
        if evidence.get(key) is not True:problems.append(key)
    for signal in SIGNALS:
        if evidence.get('signals',{}).get(signal)!='verified':problems.append('signal:'+signal)
    if problems:raise ValueError('Incomplete measurement capture: '+', '.join(problems))


def check_result(result, expected):
    """A manifest listing metric names is insufficient: require measured coverage."""
    check_capture(result.get('capture',{}),expected)
    if result.get('planned_attempts')!=result.get('completed_attempts') or not result.get('planned_attempts'):
        raise ValueError('Incomplete attempt coverage')
    if result.get('question_grouped') is not True or result.get('caps_retained') is not True:
        raise ValueError('Invalid statistical denominator')
    missing=[]
    for layer in expected['layers']:
        for metric in LEGACY:
            cell=result.get('legacy',{}).get(str(layer),{}).get(metric,{})
            if cell.get('status')!='measured' or cell.get('n',0)<1:missing.append(f'{layer}/{metric}')
        for width in (64,256):
            for metric in TEMPORAL:
                cell=result.get('temporal',{}).get(str(layer),{}).get(str(width),{}).get(metric,{})
                if cell.get('status')!='measured' or cell.get('n',0)<1:missing.append(f'{layer}/{width}/{metric}')
    for metric in CLASS_METRICS:
        if metric not in result.get('class_metrics',{}):missing.append(metric)
    # Structurally undefined quantities need explicit reviewed applicability records;
    # they do not pass this strict common-metric contract by being renamed or zeroed.
    if missing:raise ValueError('Unfinished metrics: '+', '.join(missing[:20]))
    thresholds=result.get('weak_boundary_calibration',{})
    train=set(thresholds.get('question_ids',[]));evaluation=set(result.get('evaluation_question_ids',[]))
    if not train or train & evaluation or thresholds.get('deduplicated_tokens') is not True:
        raise ValueError('Weak-boundary thresholds require disjoint calibration and unique tokens')
    return True
