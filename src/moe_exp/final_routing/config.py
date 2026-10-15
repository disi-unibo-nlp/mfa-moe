"""Human-readable configuration, eight-model registry and source sealing."""
from pathlib import Path
from copy import deepcopy
from .common import REPO, MODELS, CLASSES, check_files, digest, file_sha, load, private, publish, require


def read_config(path):
    import yaml
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    require(tuple(value['models']) == MODELS, 'the final study requires all eight models in fixed order')
    require(tuple(value['design']['classes']) == CLASSES, 'reasoning classes changed')
    for key in ('root', 'source_campaign', 'bulk_root', 'environment'):
        p = Path(value[key])
        value[key] = str(private(p if p.is_absolute() else REPO/p))
    value['config_path'] = str(path)
    value['config_file_sha256'] = file_sha(path)
    return value


def registry(config):
    parent = load(Path(config['source_campaign'])/'CAMPAIGN.json')
    profiles = deepcopy(parent['models'])
    lfm = config['lfm25']
    folder = Path(lfm['local_path'])
    card = {'model': lfm['repository'], 'revision': lfm['revision'],
            'max_model_len': lfm['max_model_len'], 'max_tokens': None,
            'sampler': {k: lfm[k] for k in ('temperature', 'top_k', 'top_p', 'min_p',
                                           'repetition_penalty', 'presence_penalty', 'frequency_penalty')},
            'template_kwargs': {'enable_thinking': True}, 'parser': 'lfm2',
            'reasoning_effort': None, 'attention': None, 'language_model_only': True, 'extra_args': []}
    files = {str(folder/name): file_sha(folder/name) for name in
             ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors.index.json')
             if (folder/name).is_file()}
    profiles['lfm25'] = {'key': 'lfm25', 'checkpoint': lfm['repository'], 'revision': lfm['revision'],
        'local_path': str(folder), 'layers': list(range(2, 40)), 'expected_routed_layers': 38,
        'experts': 64, 'native_k': 4, 'native_context_limit': lfm['max_model_len'],
        'card': card, 'sampler': card['sampler'], 'template_kwargs': card['template_kwargs'],
        'reasoning_parser': 'lfm2', 'reasoning_effort': None, 'config_files': files,
        'attention': None, 'extra_args': [], 'language_model_only': True,
        'kv_cache_memory_bytes': 8*1024**3, 'shared_paths': 'NATIVE_UNCHANGED',
        'routed_module_pattern': 'layers.N.feed_forward.experts'}
    for key, profile in profiles.items():
        profile['key'] = key
        profile['profile_sha256'] = digest({k: v for k, v in profile.items() if k != 'profile_sha256'})
    require(set(profiles) == set(MODELS), 'model registry mismatch')
    return profiles


def sources(config):
    paths = list((REPO/'src/moe_exp/final_routing').glob('*.py'))
    paths += list((REPO/'tests/final_routing').glob('*.py'))
    paths += [REPO/'scripts/final_routing.py', REPO/'scripts/slurm/final_routing.sbatch',
              Path(config['config_path'])]
    # Bind scientific dependencies we reuse without editing their source closure.
    parent = load(Path(config['source_campaign'])/'CAMPAIGN.json')
    bindings = {str(p): file_sha(p) for p in paths if p.is_file()}
    bindings.update(parent['code_files'])
    bindings.update(parent.get('runtime_files', {}))
    bindings.update(parent.get('reader_runtime_files', {}))
    bindings.update(parent.get('renderer', {}).get('files', {}))
    from .readouts import binding_files
    bindings.update(binding_files())
    site=Path(config['environment'])/'lib/python3.11/site-packages'
    for name in ('lfm2_moe.py','lfm2.py'):
        p=site/'vllm/model_executor/models'/name
        require(p.is_file(),'installed LFM implementation missing: '+str(p))
        bindings[str(p)]=file_sha(p)
    return bindings


def initialize(config):
    root = Path(config['root'])
    path = root/'BINDING.json'
    if path.exists():
        from .common import verified
        binding = verified(path)
        require(binding['config_file_sha256'] == config['config_file_sha256'], 'use the frozen config')
        check_files(binding['code_files'])
        return binding
    from .budget import preparation
    parent = Path(config['source_campaign'])/'CAMPAIGN.json'
    binding = {'schema': config['schema'], 'root': str(root), 'config': config,
               'config_file_sha256': config['config_file_sha256'], 'models': registry(config),
               'source_campaign': config['source_campaign'], 'source_campaign_sha256': file_sha(parent),
               'code_files': sources(config), 'preparation': preparation(config),
               'authorization': 'User: PLEASE IMPLEMENT THIS PLAN',
               'scientific_scope': 'eight-model count-selected policy interventions',
               'canonical_manuscript_edits': False, 'remote_pushes': False}
    from .readouts import binding_files
    binding['readout_files'] = binding_files()
    binding['recovered_authorization_sha256'] = file_sha(root/'RECOVERED_AUTHORIZATION.json')
    return publish(path, binding)
