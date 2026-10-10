"""Pinned profiles, full native contexts and remaining cumulative allowances."""
from __future__ import annotations

from pathlib import Path
import time
from .contracts import allocated, check_files, digest, publish, require
from .adapters import validate_action

LOGITS_PROCESSORS=("moe_exp.research_campaign_v1.penalties:PrefixPenalties","routing_local_logits_v2:EngineeringRepeat")

def remaining_allowance(profile, prompt_tokens, prefix_tokens):
    context = profile["max_model_len"]
    native_cap = profile["max_tokens"]
    cap = min(context - prompt_tokens - prefix_tokens,
              native_cap - prefix_tokens if native_cap is not None else context)
    require(cap > 0, "no cumulative continuation allowance remains")
    return cap


class Backend:
    def __init__(self, campaign, model, *, mask=None, deadline):
        allocated(gpu=True)
        check_files(campaign["code_files"])
        check_files(campaign["runtime_files"])
        self.campaign, self.contract, self.deadline = campaign, campaign["models"][model], deadline
        from .loader import register_loader
        register_loader()
        from moe_exp.verification_study.registry import engine_kwargs
        from vllm import LLM
        native = self.contract
        E, layers = native["experts"], native["layers"]
        masks = mask if mask is not None else {str(l): list(range(E)) for l in layers}
        require(set(masks) == set(map(str, layers)), "physical layer mask roster mismatch")
        self.binding = {"model": model, "checkpoint": native["local_path"], "experts": E,
                        "masks": masks, "compact": mask is not None,
                        "native_layer_limit": native["native_layer_limit"],
                        "model_files": native["config_files"], "runtime_files": campaign["runtime_files"]}
        binding_path = Path(campaign["root"]) / "loaders" / (digest(self.binding) + ".json")
        publish(binding_path, self.binding)
        self.kwargs = engine_kwargs(native, context=native["card"]["max_model_len"])
        self.kwargs.update(worker_cls="moe_exp.research_campaign_v1.loader.ResearchWorker",
            load_format="research_memory_v1", model_loader_extra_config={"binding_path": str(binding_path)},
            kv_cache_memory_bytes=native["kv_cache_memory_bytes"], enable_chunked_prefill=True,
            logits_processors=list(LOGITS_PROCESSORS))
        self.start = time.monotonic()
        self.llm = LLM(**self.kwargs)
        self.load_seconds = time.monotonic() - self.start
        self.tokenizer = self.llm.get_tokenizer()
        require(type(self.llm.llm_engine.engine_core).__name__ == "InprocClient", "unqualified asynchronous executor")
        from moe_exp.verification_study.backend import closure_sequences
        self.closure = closure_sequences(self.tokenizer, model)

    def generate(self, row, prompt, prefix=(), *, engineering_tokens=None, audit=False):
        from vllm import SamplingParams
        validate_action(row.get("action"), self.contract["experts"], self.contract["layers"])
        cap = remaining_allowance(self.contract["card"], len(prompt), len(prefix))
        if engineering_tokens is None:
            require(row["cap"] == cap, "assignment allowance differs from pinned native cap")
        else:
            require(0 < engineering_tokens <= cap, "invalid qualification cap")
            cap = engineering_tokens
        sampler = {k: v for k, v in self.contract["sampler"].items()}
        if sampler.get("top_k") == 0:
            sampler["top_k"] = -1
        extra = {}
        if prefix:
            extra["research_prefix_penalties"] = {"prefix": list(prefix),
                "presence": sampler.get("presence_penalty", 0.), "frequency": sampler.get("frequency_penalty", 0.),
                "repetition": sampler.get("repetition_penalty", 1.)}
            sampler.update(presence_penalty=0., frequency_penalty=0., repetition_penalty=1.)
        if engineering_tokens:
            extra["routing_engineering_repeat"] = {"token_id": self.tokenizer.encode(" test", add_special_tokens=False)[0],
                                                  "cap": cap, "overrides": {}}
        params = SamplingParams(**sampler, seed=row["seed"], max_tokens=cap, extra_args=extra or None,
                                ignore_eos=bool(engineering_tokens))
        engine = self.llm.llm_engine
        require(not engine.has_unfinished_requests(), "neighboring request still active")
        capture_layers = row.get("capture_layers", [])
        if capture_layers:
            self.llm.collective_rpc("campaign_begin", kwargs={"contract": self.contract, "prompt": prompt,
                "prefix": list(prefix), "closure": self.closure, "action": row.get("action"),
                "layers": capture_layers, "window": row["window"], "audit": audit})
        before = self.llm.collective_rpc("memory_status", kwargs={"reset": True})
        start, final, capture = time.monotonic(), None, None
        self.last_partial={}
        self.last_capture=None
        engine.add_request(row["uid"], {"prompt_token_ids": list(prompt) + list(prefix)}, params)
        try:
            while engine.has_unfinished_requests():
                require(time.time() < self.deadline, "allocation deadline during assigned generation")
                for response in engine.step():
                    require(response.request_id == row["uid"] and len(response.outputs) == 1, "request attribution mismatch")
                    if response.finished:
                        final = response.outputs[0]
                    partial=response.outputs[0]
                    self.last_partial={"token_ids":partial.token_ids,"text":partial.text,
                        "generated_tokens":len(partial.token_ids),"finish_reason":"interrupted","seconds":time.monotonic()-start,
                        "load_seconds":self.load_seconds,"cap":cap,"native_context":self.kwargs["max_model_len"]}
            require(final is not None, "missing terminal native result")
            after = self.llm.collective_rpc("memory_status")
            require(all(a["cache_sha256"] == b["cache_sha256"] for a,b in zip(before,after)), "qualified KV capacity drift")
            return {"status": "complete", "text": self.tokenizer.decode(final.token_ids, skip_special_tokens=False),
                    "token_ids": list(final.token_ids), "generated_tokens": len(final.token_ids),
                    "finish_reason": final.finish_reason, "stop_reason": final.stop_reason,
                    "seconds": time.monotonic()-start, "load_seconds": self.load_seconds,
                    "memory_before": before, "memory_after": after, "native_context": self.kwargs["max_model_len"],
                    "cap": cap, "prefix_penalty_history": digest(list(prefix)), "engineering_only": bool(engineering_tokens)}
        finally:
            if engine.has_unfinished_requests():
                engine.abort_request([row["uid"]], internal=True)
            if capture_layers:
                reports = self.llm.collective_rpc("campaign_end")
                require(len(reports) == 2 and reports[0]["capture_sha256"] == reports[1]["capture_sha256"], "TP route capture disagreement")
                # The caller receives captures through this scoped property even
                # when the generation returns a terminal receipt above.
                self.last_capture = reports[0]
                if final is not None:
                    # A native EOS can be predicted without appearing in returned
                    # token IDs. Keep its routing receipt separately so the
                    # dynamics vectors align exactly with generated tokens.
                    terminal={}
                    origin=len(prefix)
                    for layer,record in self.last_capture["capture"].items():
                        kept=[i for i,p in enumerate(record["positions"]) if p<origin+len(final.token_ids)]
                        ended=[i for i,p in enumerate(record["positions"]) if p>=origin+len(final.token_ids)]
                        terminal[layer]={k:[v[i] for i in ended] for k,v in record.items()}
                        self.last_capture["capture"][layer]={k:[v[i] for i in kept] for k,v in record.items()}
                    self.last_capture.update(terminal_prediction=terminal,capture_sha256=digest(self.last_capture["capture"]),
                        alignment="absolute completion prediction index; emitted tokens and terminal prediction separated")

    def forward(self, source, trace):
        from moe_exp.schemas import TraceRecord
        from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
        replay = trace["metadata"]["token_replay"]
        prompt, completion = replay["prompt_token_ids"], replay["completion_token_ids"]
        layout = saved_layout(TraceRecord(**trace))
        reasoning = list(layout["reasoning_tokens"])
        require(reasoning and len(prompt) + len(completion) <= self.kwargs["max_model_len"], "saved sequence exceeds preserved native context")
        # Deterministic rank quantiles cover the entire saved reasoning progress.
        import numpy as np
        bank = sorted({reasoning[i] for i in np.linspace(0, len(reasoning)-1, min(64,len(reasoning)), dtype=int)})
        excluded=int(len(prompt)+len(completion)==self.kwargs["max_model_len"])
        saved=completion[:-1] if excluded else completion
        self.llm.collective_rpc("measurement_begin", kwargs={"contract": self.contract, "prompt": prompt,
            "completion": saved, "closure": self.closure,
            "reasoning_positions": [p for p in reasoning if p < len(saved)],
            "bank_positions": [p for p in bank if p < len(saved)]})
        engine = self.llm.llm_engine
        from vllm import SamplingParams
        fixture = self.tokenizer.encode(" test", add_special_tokens=False)[0]
        params = SamplingParams(temperature=0., max_tokens=1, ignore_eos=True,
            extra_args={"routing_engineering_repeat": {"token_id": fixture, "cap": 1, "overrides": {}}})
        uid = digest(["forward", source["attempt_id"], self.contract["profile_sha256"]])
        started = time.monotonic()
        engine.add_request(uid, {"prompt_token_ids": prompt + saved}, params)
        try:
            while engine.has_unfinished_requests():
                require(time.time() < self.deadline, "deadline during saved-sequence measurement")
                engine.step()
        finally:
            if engine.has_unfinished_requests():
                engine.abort_request([uid], internal=True)
            outputs = self.llm.collective_rpc("measurement_end")
        require(outputs[0]["measurement_sha256"] == outputs[1]["measurement_sha256"], "TP measurement disagreement")
        return {**outputs[0], "seconds": time.monotonic()-started, "source": source,
                "load_seconds": self.load_seconds, "saved_tokens": len(saved),
                "last_source_token_excluded": bool(excluded),
                "reasoning_endpoint_unmeasured":bool(excluded and len(completion)-1 in reasoning),
                "forced_discarded_engineering_token": fixture}

    def close(self):
        self.llm.llm_engine.engine_core.shutdown()
