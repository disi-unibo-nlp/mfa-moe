"""Explicit single-request vLLM instrumentation for GPT-OSS and Qwen3 MoE.

Patch the *existing* native router instance, after backend construction. A fused
path bypassing it fails the observed-call gate. This is not a compatibility claim
for a backend until its separate no-op/output qualification has passed.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import re

from .native import NativePolicy, TimedNativeRouter, TokenContext


class RequestClock:
    def __init__(self, request_id, prompt_ids, prefix_ids, *, closing_sequences):
        if not request_id or not prompt_ids or not closing_sequences:
            raise ValueError("Request, exact prompt, and reasoning-close token sequences required")
        self.request_id = request_id
        self.prompt_ids, self.prefix_ids = list(prompt_ids), list(prefix_ids)
        self.saved_ids = self.prompt_ids + self.prefix_ids
        self.closing_sequences = [tuple(s) for s in closing_sequences]
        if any(not s for s in self.closing_sequences):
            raise ValueError("Empty phase marker")
        self.closed = False
        self.tail = []
        self.last_position = -1
        self.transitions = []

    def observe(self, input_ids, positions):
        ids, positions = input_ids.detach().cpu().tolist(), positions.detach().cpu().tolist()
        if len(ids) != len(positions) or any(type(p) is not int for p in positions):
            raise ValueError("One-dimensional absolute positions required")
        contexts = []
        for token, position in zip(ids, positions):
            if position != self.last_position + 1:
                raise ValueError("Request positions are not contiguous: batching, cache reuse or replay")
            self.last_position = position
            if position < len(self.saved_ids) and token != self.saved_ids[position]:
                raise ValueError("Runtime input differs from exact branch prefix")
            output_token = position - len(self.prompt_ids) + 1
            if position >= len(self.prompt_ids):
                self.tail.append(token)
                self.tail = self.tail[-max(map(len, self.closing_sequences)):]
                if not self.closed and any(tuple(self.tail[-len(s):]) == s for s in self.closing_sequences):
                    self.closed = True
                    self.transitions.append(dict(request_id=self.request_id,
                        output_token=output_token, phase="reasoning_terminated"))
            contexts.append(TokenContext(self.request_id, output_token,
                                         reasoning=output_token >= 0 and not self.closed))
        return contexts


def validate_config(config):
    m, p = config.model_config, config.parallel_config
    if m.hf_config.model_type not in ("gpt_oss", "qwen3_moe"):
        raise ValueError("Native adapter qualification starts with GPT and Qwen3-30B")
    if not m.enforce_eager or config.speculative_config is not None:
        raise ValueError("Eager execution without speculation required for exact timing")
    if (p.enable_expert_parallel or p.enable_eplb or p.pipeline_parallel_size != 1
            or p.data_parallel_size != 1 or getattr(p, "enable_dbo", False)
            or getattr(p, "use_sequence_parallel_moe", False)):
        raise ValueError("Only tensor parallelism is supported by this request adapter")
    if config.scheduler_config.max_num_seqs != 1 or config.cache_config.enable_prefix_caching:
        raise ValueError("Single request and disabled prefix caching required")


@contextmanager
def instrument(model, policy, clock, sink):
    """Attach to exactly one native layer; restore bindings even after failure."""
    found = []
    for name, module in model.named_modules():
        match = re.search(r"(?:^|\.)layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts$", name)
        router = getattr(module, "router", None)
        if match and int(match[1]) == policy.layer and hasattr(router, "_compute_routing"):
            quant = getattr(getattr(module, "routed_experts", None), "quant_method", None)
            if quant is not None and getattr(quant, "is_monolithic", False):
                raise ValueError("Monolithic MoE kernel bypasses routing hooks")
            found.append(router)
    if len(found) != 1:
        raise ValueError("Exactly one native router instance must be reachable at the selected layer")
    router = found[0]
    if getattr(router, "eplb_state", None) is not None:
        raise ValueError("Expert load balancing invalidates native expert identity")
    original = router._compute_routing
    signature = inspect.signature(original)
    if "router_logits" not in signature.parameters:
        raise ValueError("Native router does not expose pre-selection logits")
    timed = TimedNativeRouter(policy, top_k=router.top_k, sink=sink)
    contexts = []

    def before(module, args, kwargs):
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        ids, positions = bound.arguments.get("input_ids"), bound.arguments.get("positions")
        if ids is None or positions is None:
            raise ValueError("Model forward omitted exact input token attribution")
        contexts[:] = clock.observe(ids, positions)

    def compute(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        logits = bound.arguments["router_logits"]
        if logits.shape[-1] != router.global_num_experts:
            raise ValueError("Hook received hidden states instead of native expert logits")
        def select(changed):
            bound.arguments["router_logits"] = changed
            return original(*bound.args, **bound.kwargs)
        return timed(logits, select, contexts)

    handle = model.register_forward_pre_hook(before, with_kwargs=True)
    router._compute_routing = compute
    try:
        yield timed
        if not timed.rows_seen:
            raise RuntimeError("Backend bypassed native router instrumentation; qualification failed")
    finally:
        router._compute_routing = original
        handle.remove()


class NativeWorkerExtension:
    """vLLM control RPCs; one synchronous branch at a time, finite lifetime."""
    def native_begin(self, *, policy, request_id, prompt_ids, prefix_ids, closing_sequences):
        if getattr(self, "_native_context", None) is not None:
            raise RuntimeError("A request is already active")
        validate_config(self.vllm_config)
        self._native_events = []
        self._native_clock = RequestClock(request_id, prompt_ids, prefix_ids,
                                           closing_sequences=closing_sequences)
        context = instrument(self.model_runner.get_model(), NativePolicy(**policy),
                             self._native_clock, self._native_events.append)
        self._native_router = context.__enter__()
        self._native_context = context
        return dict(status="attached", request_id=request_id)

    def native_end(self):
        context = getattr(self, "_native_context", None)
        if context is None:
            raise RuntimeError("No instrumented request")
        self._native_context = None
        try:
            context.__exit__(None, None, None)
            return dict(events=self._native_events, phase_transitions=self._native_clock.transitions,
                        token_rows=self._native_router.rows_seen,
                        controller_seconds=self._native_router.controller_seconds)
        finally:
            self._native_events = []
