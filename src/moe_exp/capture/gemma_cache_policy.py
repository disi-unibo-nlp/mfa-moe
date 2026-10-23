"""Keep an explicitly requested cache sensitivity replay outside primary results."""
from pathlib import Path
ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study')
SENSITIVITY_ROOT=ROOT/'results/measurement-study/gemma-bf16-kv-sensitivity'
VARIANT='nvfp4-weights-bf16-kv-sensitivity'

def cache_options(spec, output):
    cache=spec.get('kv_cache_dtype')
    if cache is None:return {}
    if spec.get('capture_version')==2 and spec.get('corpus')=='card-v3':
        from moe_exp.capture.campaign_v3_plan import validate_storage_root
        validate_storage_root(output)
        if cache!='bfloat16' or not spec['generation_model'].startswith('nvidia/Gemma') or spec.get('precision')!='nvfp4':
            raise ValueError('Primary Gemma requires native NVFP4 with BF16 KV')
        return {'kv_cache_dtype':'bfloat16'}
    if not spec['generation_model'].startswith('nvidia/Gemma'):raise ValueError('Gemma-only cache override')
    if cache!='bfloat16' or spec.get('precision')!='nvfp4' or spec.get('replay_variant')!=VARIANT:
        raise ValueError('Explicit NVFP4/BF16 cache sensitivity label required')
    if not Path(output).resolve().is_relative_to(SENSITIVITY_ROOT.resolve()):
        raise ValueError('Cache sensitivity must use isolated output storage')
    return {'kv_cache_dtype':'bfloat16'}
