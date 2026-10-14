"""Small installed-interface checks before spending the single validation allocation."""
import ast
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace
import unittest

from moe_exp.research_campaign_v3_r6.contracts import REPO, file_sha, publish, require
from .policy import check_binding


def installed_abort_method():
    path = Path(distribution('vllm').locate_file('vllm/v1/engine/output_processor.py'))
    tree = ast.parse(path.read_text())
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'abort_requests')
    # Future annotations avoid importing GPU packages; the exact installed method
    # body is executed against its real external/internal ID mapping interface.
    code = 'from __future__ import annotations\n' + ast.unparse(node)
    namespace = {}
    exec(compile(code, str(path), 'exec'), namespace)
    return namespace['abort_requests'], str(path), file_sha(path)


def installed_abort_check():
    method, path, sha = installed_abort_method()
    class Processor:
        def __init__(self):
            self.request_states = {'original-randomid': SimpleNamespace(parent_req=None, queue=None, lora_name=None)}
            self.external_req_ids = {'original': ['original-randomid']}
            self.parent_requests = {}
            self.lora_states = SimpleNamespace(request_finished=lambda *args: None)
        def _get_internal_request_ids(self, ids, internal):
            if internal:
                return ids
            return [uid for external in ids for uid in self.external_req_ids.get(external, [])]
    wrong, correct = Processor(), Processor()
    wrong_result = method(wrong, ['original'], internal=True)
    result = method(correct, ['original'], internal=False)
    require(not wrong_result and len(wrong.request_states) == 1 and result == ['original-randomid'] and
            not correct.request_states and not correct.external_req_ids, 'Installed abort namespace behavior differs')
    return {'path': path, 'sha256_file': sha, 'wrong_namespace_left_pending': True,
            'external_namespace_removed_request': True}


def interfaces():
    from vllm import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor
    validator = object.__new__(InputProcessor)
    validator.model_config = SimpleNamespace(max_model_len=64, runner_type='generate', get_vocab_size=lambda: 8)
    validator.renderer = SimpleNamespace(tokenizer=SimpleNamespace(max_token_id=7))
    validator.skip_prompt_length_check = False
    validator.supports_mm_inputs = False
    validator.mm_encoder_cache_size = 0
    validator._validate_model_input({'type': 'token', 'prompt_token_ids': [1, 2, 3]}, 'decoder')
    try:
        validator._validate_model_input({'type': 'token', 'prompt_token_ids': {'input_ids': [1, 2, 3]}}, 'decoder')
    except TypeError:
        reproduced = True
    else:
        raise ValueError('Original dictionary input unexpectedly passed')
    sample = SamplingParams(temperature=1., top_p=.95, top_k=20, max_tokens=16, seed=20261008)
    require(sample.max_tokens == 16, 'Installed sampling API differs')
    return {'renderer_backed_tokenizer_passed': True, 'dictionary_failure_reproduced': reproduced,
            'sampling_api_passed': True, 'gpu_loaded': False, 'scientific_draws': 0,
            'abort_mapping': installed_abort_check()}


def run(campaign):
    binding = check_binding(campaign)
    suite = unittest.defaultTestLoader.discover(str(REPO / 'tests/research_campaign_v3_r6_ops_r1'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    require(result.wasSuccessful(), 'Operations preflight tests failed')
    boundary = interfaces()
    check_binding(campaign)
    return publish(Path(campaign['root']) / 'OPS_R1_PREFLIGHT.json', {
        'passed': True, 'tests_run': result.testsRun, 'binding_sha256': binding['sha256'],
        'installed_interfaces': boundary, 'job_submissions': 0, 'full_validation_executed': False,
    })
