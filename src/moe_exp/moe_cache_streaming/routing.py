"""Single-sequence eager vLLM worker; hook-selected IDs, not native dispatch logs."""
import re

from moe_exp.moe_identity_guiding.routing import IdentityWorkerExtension, routing_spec


class DecodeRecorder:
    def __init__(self, experts, top_k):
        self.experts, self.top_k = experts, top_k
        self.active = False

    def begin(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens
        self.prefill_remaining = prompt_tokens
        self.calls = 0
        self.rows = []
        self.active = True

    def __call__(self, module, args, output):
        if not self.active:
            return
        logits = output[0] if isinstance(output, tuple) else output
        if logits.ndim != 2 or logits.shape[1] != self.experts:
            raise RuntimeError('Unexpected router shape')
        if self.prefill_remaining:
            if not 0 < logits.shape[0] <= self.prefill_remaining:
                raise RuntimeError('Prefill chunk exceeds remaining real prompt tokens')
            self.prefill_remaining -= logits.shape[0]
        else:
            if logits.shape[0] != 1:
                raise RuntimeError('Decode must contain exactly one real token')
            self.rows.append(logits.float().topk(self.top_k, dim=-1).indices[0].cpu().tolist())
        self.calls += 1


class CacheWorkerExtension(IdentityWorkerExtension):
    def cache_configure(self, policy, strength, condition):
        config = self.vllm_config
        if config.parallel_config.tensor_parallel_size != 1:
            raise ValueError('Use tensor parallel size 1')
        if config.scheduler_config.async_scheduling:
            raise ValueError('Ordered capture requires synchronous scheduling; async can execute past EOS')
        if config.scheduler_config.max_num_seqs != 1:
            raise ValueError('Use one sequence')
        # Bias hooks are installed first, so recorders see their returned logits.
        self.identity_configure(policy, strength, condition, diagnostics='minimal')
        text = getattr(config.model_config, 'hf_text_config', None)
        if text is None:
            text = config.model_config.hf_config
        family, experts, top_k = routing_spec(text)
        self.cache_recorders, self.cache_handles = {}, []
        for name, module in self.model_runner.get_model().named_modules():
            suffix = '' if family == 'gemma' else r'\.mlp'
            match = re.search(r'(?:^|\.)layers\.(\d+)' + suffix + '$', name)
            if not match:
                continue
            gate = getattr(module, 'gate' if family == 'qwen' else 'router', None)
            if gate is None:
                continue
            if getattr(getattr(module, 'experts', None), '_fse_fuse_gate', False):
                raise ValueError('Fused gates bypass routing capture')
            layer = match[1]
            if layer in self.cache_recorders:
                raise ValueError('Duplicate layer')
            recorder = DecodeRecorder(experts, top_k)
            self.cache_recorders[layer] = recorder
            self.cache_handles.append(gate.register_forward_hook(recorder))
        if len(self.cache_recorders) != text.num_hidden_layers:
            raise ValueError('Expected one routed gate per model layer; hybrid dense models unsupported')
        return dict(layers=sorted(self.cache_recorders, key=int), experts=experts, top_k=top_k)

    def cache_begin(self, prompt_tokens):
        for recorder in self.cache_recorders.values():
            recorder.begin(prompt_tokens)

    def cache_finish(self, generated_tokens):
        result = {}
        for layer, recorder in self.cache_recorders.items():
            recorder.active = False
            if recorder.prefill_remaining != 0 or len(recorder.rows) != generated_tokens - 1:
                raise RuntimeError(
                    f'Layer {layer}: {len(recorder.rows)} decode steps, '
                    f'{generated_tokens} returned tokens, '
                    f'{recorder.prefill_remaining} prefill tokens remaining; '
                    'expected generated_tokens - 1 decode steps')
            result[layer] = recorder.rows
        return result
