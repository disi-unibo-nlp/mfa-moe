"""The two ways to replay a saved trace through a model and record its routers.

HFBackend runs the Hugging Face model (gpt, glm reference, qwen330b); VLLMBackend runs the vLLM engine on two GPUs with the
hooks of `capture.vllm_capture` (glm, qwen36, nemotron, gemma, qwen35). Both return the same dict for one trace:
router signal, hidden states, selected expert ids, their weights, selection scores, semantics and an audit.
The token ids come from the saved trace, never from re-tokenising the text.
"""
import math
from pathlib import Path

from capture.manifest import atomic
from store import file_sha

GPT = 'openai/gpt-oss-20b'
GLM = 'zai-org/GLM-4.7-Flash'


class HFBackend:
    def __init__(self, spec, out):
        from models.loader import load_model_and_tokenizer
        from models.router_adapters import available_router_layers
        from provenance import forward_provenance
        self.spec = spec
        self.model, self.tokenizer = load_model_and_tokenizer(
            spec['checkpoint'], quantization='none', revision=spec['revision'], local_files_only=True,
            offload_folder=str(out / 'offload'))
        if 'disk' in getattr(self.model, 'hf_device_map', {}).values():
            raise ValueError('Unbudgeted disk weight offload')
        if available_router_layers(self.model) != spec['layers']:
            raise ValueError('Changed native layer topology')
        quantizer = getattr(self.model, 'hf_quantizer', None)
        if spec['precision'] == 'mxfp4':
            if quantizer is None or quantizer.quantization_config.dequantize:
                raise ValueError('Native MXFP4 required')
        elif quantizer is not None:
            raise ValueError('Unexpected quantization')
        self.provenance = forward_provenance(
            self.model, self.tokenizer, revision=spec['revision'], local_files_only=True)
        if spec['generation_model'] == GPT:
            from capture.gpt_attention import install_gpt_query_chunking
            self.provenance.update(install_gpt_query_chunking(self.model, 128))

    def validate_attention(self, trace, out):
        """gpt-oss: the query-chunked attention must give the native attention's experts and signals."""
        import torch
        from transformers.models.gpt_oss import modeling_gpt_oss as implementation
        chunked = implementation.eager_attention_forward
        try:
            implementation.eager_attention_forward = chunked._measurement_native
            native = self.capture_single(trace, 'native-differential')
        finally:
            implementation.eager_attention_forward = chunked
        bounded = self.capture_single(trace, 'bounded-differential')
        if not torch.equal(native['ids'].sort(-1).values, bounded['ids'].sort(-1).values):
            raise ValueError('Query chunking changed native selected expert sets on validation trace')
        evidence = {}
        for name in ('router', 'hidden'):
            x = native[name].float()
            y = bounded[name].float()
            error = float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(x).clamp_min(1e-12))
            if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not math.isfinite(error) or error > 0.005:
                raise ValueError('Query chunking differential error: ' + name)
            evidence[name + '_relative_l2'] = error
        native_weights = native['weights'].gather(-1, native['ids'].argsort(-1))
        bounded_weights = bounded['weights'].gather(-1, bounded['ids'].argsort(-1))
        if (not torch.isfinite(native_weights).all() or not torch.isfinite(bounded_weights).all()
                or not torch.allclose(native_weights, bounded_weights, rtol=0.01, atol=0.001)):
            raise ValueError('Query chunking changed native weights beyond rounding tolerance')
        replay = trace.metadata['token_replay']
        atomic(out / 'attention-differential.json',
               dict(status='passed', native_selected_sets_identical=True,
                    completion_tokens=len(replay['completion_token_ids']), **evidence))

    def capture(self, trace, key):
        replay = trace.metadata['token_replay']
        total = len(replay['prompt_token_ids']) + len(replay['completion_token_ids'])
        if self.spec['generation_model'] == GPT and (self.spec.get('force_chunked') or total > 32768):
            from capture.hf_chunked_capture import capture_chunked
            return capture_chunked(self.model, self.tokenizer, trace, self.spec['layers'],
                                   self.spec.get('prefill_chunk_size', 512))
        return self.capture_single(trace, key)

    def capture_single(self, trace, key):
        import torch
        from models.inference import extract_logs_single_pass
        from models.router_adapters import decoder_layers
        details = {}
        router, hidden = extract_logs_single_pass(
            model=self.model, tokenizer=self.tokenizer, problem=trace.prompt, cot_text=trace.cot_text,
            system_prompt=trace.system_prompt, messages=trace.generation_messages, extract_hidden_states=True,
            layer_indices=self.spec['layers'], token_replay=trace.metadata['token_replay'],
            routing_details=details)
        semantics = details['semantics']
        ids = details['selected_experts']
        weights = details['expert_weights']
        if not semantics.get('native_selection') or semantics.get('executed_weights') != 'native_expert_input_weights':
            raise ValueError('Native dispatch capture required')
        selection = router
        if self.spec['generation_model'] == GLM:
            config = self.model.config
            if config.n_group != 1 or config.topk_group != 1:
                raise ValueError('Unsupported grouped selection')
            bias = torch.stack([
                decoder_layers(self.model)[layer].mlp.gate.e_score_correction_bias.detach().float().cpu()
                for layer in self.spec['layers']])
            selection = router.float().exp() + bias[:, None, :]
            semantics = {**semantics, 'selection_scores': 'native_sigmoid_plus_bias', 'deterministic_topk': True}
        return dict(router=router, hidden=hidden, ids=ids, weights=weights, selection=selection,
                    semantics=semantics, audit={'native_dispatch': True, 'backend': 'hf', 'layers': self.spec['layers']})


class VLLMBackend:
    def __init__(self, spec, out, transient_cap_bytes=15 * 1024 ** 3):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.spec, self.out = spec, out
        self.transient_cap_bytes = transient_cap_bytes
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec['checkpoint'], revision=spec['revision'], local_files_only=True)
        nemotron = 'Nemotron' in spec['generation_model']
        kwargs = dict(
            model=spec['checkpoint'], revision=spec['revision'], tokenizer_revision=spec['revision'],
            dtype='bfloat16', tensor_parallel_size=2, max_num_seqs=1, enforce_eager=True,
            enable_prefix_caching=False, enable_expert_parallel=False, enable_eplb=False,
            speculative_config=None, async_scheduling=False,
            max_model_len=int(spec.get('max_input_tokens', 49151)) + 1, max_num_batched_tokens=1024,
            gpu_memory_utilization=0.8, generation_config='vllm', enable_chunked_prefill=True,
            kernel_config={'moe_backend': 'marlin', 'linear_backend': 'marlin'},
            worker_extension_cls='capture.vllm_capture.MeasurementWorkerExtension')
        if spec.get('source') == 'glm' or spec['generation_model'] == GLM:
            kwargs['kernel_config'] = {'moe_backend': 'triton'}
            kwargs['kv_cache_dtype'] = 'bfloat16'
        if spec.get('source') == 'qwen35' or 'GPTQ' in spec['generation_model']:
            kwargs['quantization'] = 'gptq_marlin'
        if nemotron:
            kwargs.update(mamba_backend='triton', mamba_cache_mode='align', kv_cache_dtype='bfloat16',
                          attention_config={'backend': 'TRITON_ATTN'})
        else:
            kwargs['language_model_only'] = True
        if spec.get('kv_cache_dtype') is not None:
            # Only the primary Gemma replay (native NVFP4 weights, bf16 KV cache) sets this.
            if (spec['kv_cache_dtype'] != 'bfloat16' or not spec['generation_model'].startswith('nvidia/Gemma')
                    or spec.get('precision') != 'nvfp4'):
                raise ValueError('Primary Gemma requires native NVFP4 with BF16 KV')
            kwargs['kv_cache_dtype'] = 'bfloat16'
        self.llm = LLM(**kwargs)
        if spec['generation_model'].startswith('nvidia/Gemma'):
            from vllm import SamplingParams
            prompt = self.tokenizer.encode('Compute 2 + 2.', add_special_tokens=True)
            generated = self.llm.generate([{'prompt_token_ids': prompt}],
                                          SamplingParams(max_tokens=8, temperature=0), use_tqdm=False)
            atomic(out / 'short-generation.json',
                   dict(status='complete', token_ids=list(generated[0].outputs[0].token_ids),
                        text=generated[0].outputs[0].text))
            if not generated[0].outputs[0].token_ids:
                raise RuntimeError('Gemma short generation empty')
        self.configured = self.llm.collective_rpc('measurement_configure', kwargs={'spec': spec})
        atomic(out / 'backend.json', {'workers': self.configured, 'engine': kwargs})
        self.provenance = dict(
            checkpoint=spec['generation_model'], revision=spec['revision'], precision=spec['precision'],
            kv_cache_dtype=kwargs.get('kv_cache_dtype', 'auto'), replay_variant=spec.get('replay_variant', 'primary'),
            historical_kv_cache_dtype_status=spec.get('historical_kv_cache_dtype_status', 'unverified'),
            backend='vllm-native-' + ('triton' if spec['generation_model'] == GLM else 'marlin'),
            hidden_observation_point=spec.get('hidden_observation_point', 'decoder_layer_input_reconstructed_residual'),
            capture_code_sha256=file_sha(Path(__file__).with_name('vllm_capture.py')))

    def capture(self, trace, key):
        import torch
        from vllm import SamplingParams
        from models.token_replay import validate_token_replay
        from capture.buffers import capture_memory_budget
        replay = trace.metadata['token_replay']
        prompt, completion = validate_token_replay(replay, self.tokenizer, trace.cot_text)
        tokens = prompt + completion
        budget = capture_memory_budget(
            len(completion), len(self.spec['layers']), int(self.spec['hidden_size']),
            int(self.spec.get('num_experts', self.spec['experts'])), int(self.spec['top_k']))
        if budget['transient_bytes'] > self.transient_cap_bytes:
            raise ValueError('Manifest transient budget is too small for trace')
        self.llm.collective_rpc('measurement_register',
                                kwargs={'key': key, 'tokens': tokens, 'prompt_length': len(prompt),
                                        'max_buffer_bytes': budget['worker_bytes']})
        engine = self.llm.llm_engine
        internal = engine.add_request(key, {'prompt_token_ids': tokens},
                                      SamplingParams(max_tokens=1, temperature=0, ignore_eos=True, detokenize=False))
        if internal != key:
            raise RuntimeError('Request ID randomization is active')
        while engine.has_unfinished_requests():
            engine.step()
        raw = self.out / 'transient.pt'
        reports = self.llm.collective_rpc('measurement_snapshot',
                                          kwargs={'key': key, 'output': str(raw), 'max_bytes': self.transient_cap_bytes})
        # Both tensor-parallel ranks must have seen identical routing.
        if len(reports) != 2 or {r['rank'] for r in reports} != {0, 1} or reports[0]['hashes'] != reports[1]['hashes']:
            raise RuntimeError('TP capture mismatch')
        if any(r['tokens'] != len(completion) or r['layers'] != self.spec['layers'] for r in reports):
            raise RuntimeError('Missing token/layer coverage')
        atomic(self.out / (key + '.native-audit.json'), {'workers': reports})
        self.llm.collective_rpc('measurement_release', kwargs={'key': key})
        if raw.stat().st_size > self.transient_cap_bytes:
            raise RuntimeError('Transient storage cap exceeded')
        data = torch.load(raw, map_location='cpu', weights_only=True)
        raw.unlink()
        sigmoid = 'Nemotron' in self.spec['generation_model'] or self.spec['generation_model'] == GLM
        data['semantics'] = dict(
            native_selection=True, executed_weights='native_expert_input_weights',
            full_distribution='normalized_sigmoid_affinity' if sigmoid else 'softmax_probability',
            deterministic_topk=True, selection_scores='native_sigmoid_plus_bias' if sigmoid else 'router_signal',
            num_experts=data['router'].shape[-1])
        data['audit'] = dict(native_dispatch=True, layers=self.spec['layers'], workers=reports,
                             routed_output_scaling=self.configured[0].get('routed_output_scaling', {}))
        return data
