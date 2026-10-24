"""Finite generation and within-execution qualification on two allocated GPUs."""
from __future__ import annotations

import os
from pathlib import Path
import time

from .contracts import allocated, check_files, digest, load, require, save, seal
from .protocol import CONTEXT, OUTPUT_CAP


def compact_memory(status):
    return [{"total_expert_bytes": s["total_expert_bytes"],
             "total_model_tensor_bytes": s["total_model_storage"]["unique_bytes"],
             "layer_expert_bytes": {l: v["unique_bytes"] for l, v in s["expert_storage"].items()},
             "memory": s["memory"], "load_memory": s["load_memory"], "cache_sha256": s["cache_sha256"],
             "routes": {l: {k: v for k, v in r.items() if k != "original_to_compact"} for l, r in s["routes"].items()}}
            for s in status]


def loader_binding(campaign, model, selection, arm, output):
    require(selection["campaign_sha256"] == campaign["sha256"] and selection["model"] == model,
            "foreign selection")
    contract = campaign["models"][model]
    layers, E = selection["layers"], selection["experts"]
    mask = [list(range(E)) for _ in layers] if arm == "native" else selection["final_masks"][arm]
    return save(output, {"schema": "expert-memory-loader-binding-v1", "model": model, "arm": arm,
                         "checkpoint": contract["local_path"], "experts": E,
                         "masks": {str(l): ids for l, ids in zip(layers, mask)}, "compact": arm != "native",
                         "native_layer_limit": contract["native_layer_limit"],
                         "model_files": contract["config_files"], "runtime_files": campaign["runtime_files"],
                         "selection_sha256": selection["sha256"], "campaign_sha256": campaign["sha256"]})


class Backend:
    def __init__(self, campaign, model, binding_path, *, deadline):
        allocated(gpu=True)
        check_files(campaign["code_files"])
        check_files(campaign["runtime_files"])
        contract = campaign["models"][model]
        check_files(contract["config_files"])
        from .loader import register_loader
        register_loader()
        from moe_exp.verification_study.registry import engine_kwargs
        from vllm import LLM
        self.contract, self.deadline = contract, deadline
        self.kwargs = engine_kwargs(contract, context=CONTEXT)
        self.kwargs.update(worker_cls="moe_exp.expert_memory_study.worker.ExpertMemoryWorker",
                           load_format="expert_memory_v1", model_loader_extra_config={"binding_path": str(binding_path)},
                           kv_cache_memory_bytes=contract["kv_cache_memory_bytes"],
                           enable_chunked_prefill=True)
        started = time.monotonic()
        self.llm = LLM(**self.kwargs)
        self.load_seconds = time.monotonic() - started
        self.tokenizer = self.llm.get_tokenizer()
        require(type(self.llm.llm_engine.engine_core).__name__ == "InprocClient", "asynchronous engine unsupported")

    def prompt_ids(self, problem):
        from moe_exp.verification_study.readers import normalize_prompt_ids
        kwargs = dict(self.contract["template_kwargs"])
        if self.contract["reasoning_effort"] is not None:
            kwargs["reasoning_effort"] = self.contract["reasoning_effort"]
        return normalize_prompt_ids(self.tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}], tokenize=True, add_generation_prompt=True, **kwargs))

    def generate(self, assignment, problem, *, engineering=False):
        from vllm import SamplingParams
        require(time.time() < self.deadline, "allocation deadline before assignment")
        ids = self.prompt_ids(problem)
        cap = 2 if engineering else OUTPUT_CAP
        require(len(ids) + cap <= CONTEXT, "full output cap does not fit; input truncation forbidden")
        sampler = dict(self.contract["sampler"])
        if sampler.get("top_k") == 0:
            sampler["top_k"] = -1
        params = SamplingParams(**sampler, seed=assignment["seed"], max_tokens=OUTPUT_CAP)
        if engineering:
            # Context feasibility is exercised without a repeat-only full-cap draw.
            params.max_tokens = 2
            params.min_tokens = 2
        engine = self.llm.llm_engine
        require(not engine.has_unfinished_requests(), "neighboring request active")
        before = self.llm.collective_rpc("memory_status", kwargs={"reset": True})
        start = time.monotonic()
        first, final, prefill_memory, phase_overhead = None, None, None, 0.
        uid = assignment["uid"]
        engine.add_request(uid, {"prompt_token_ids": ids}, params)
        try:
            while engine.has_unfinished_requests():
                require(time.time() < self.deadline, "allocation deadline during assignment")
                for response in engine.step():
                    require(response.request_id == uid and len(response.outputs) == 1,
                            "neighboring or multi-sample request")
                    if first is None and response.outputs[0].token_ids:
                        first = time.monotonic()
                        phase_start = time.monotonic()
                        prefill_memory = self.llm.collective_rpc("memory_phase", kwargs={"reset": True})
                        phase_overhead = time.monotonic() - phase_start
                    if response.finished:
                        final = response.outputs[0]
            end = time.monotonic()
            require(final is not None and first is not None, "request has no terminal output")
            after = self.llm.collective_rpc("memory_status")
            require(len(before) == len(after) == 2, "two TP ranks required")
            for a, b in zip(before, after):
                require(a["cache_sha256"] == b["cache_sha256"] and
                        all(v["removed_dispatch"] == 0 and v["calls"] > 0 for v in b["routes"].values()),
                        "cache drift or missing routing-invariant observations")
            return {"execution_complete": True, "text": final.text, "token_ids": list(final.token_ids),
                    "generated_tokens": len(final.token_ids), "finish_reason": final.finish_reason,
                    "stop_reason": final.stop_reason, "prompt_tokens": len(ids), "prompt_sha256": digest(ids),
                    "problem_sha256": digest(problem), "context": CONTEXT, "output_cap": cap,
                    "generation_seconds": end - start, "prefill_seconds": first - start,
                    "decode_seconds": max(0., end - first - phase_overhead), "load_seconds": self.load_seconds,
                    "prefill_memory": prefill_memory, "memory_phase_telemetry_seconds": phase_overhead,
                    "memory_before": compact_memory(before), "memory_after": compact_memory(after),
                    "engineering_only": engineering, "job_id": os.environ["SLURM_JOB_ID"]}
        finally:
            if engine.has_unfinished_requests():
                engine.abort_request(uid)

    def close(self):
        self.llm.llm_engine.engine_core.shutdown()


def qualify(campaign, model, selection, output, *, deadline):
    """Each frozen policy is checked; equal policies are explicitly aliased."""
    allocated(gpu=True)
    root = Path(output)
    cases, failures, backend, seen = [], [], None, {}
    start = time.monotonic()
    for arm in ("native", "state_balanced", "labelled_frequency", "random"):
        signature = "native" if arm == "native" else digest(selection["final_masks"][arm])
        if signature in seen:
            cases.append({"arm": arm, "alias_of": seen[signature], "status": "PASS"})
            continue
        try:
            binding = loader_binding(campaign, model, selection, arm, root / (arm + "-loader.json"))
            backend = Backend(campaign, model, root / (arm + "-loader.json"), deadline=deadline)
            status = backend.llm.collective_rpc("memory_status")
            require(all(s["memory"]["nvml_process_residence"] for s in status), "NVML residence unmeasured")
            qualification = None if arm == "native" else backend.llm.collective_rpc("memory_qualify")
            if qualification is not None:
                require(all(q["status"] == "PASS" and len(q["cases"]) == 2 * len(selection["layers"])
                            for q in qualification), "layer/reference qualification incomplete")
            # Real prefill/decode at full context from ordinary fixed engineering input.
            short = backend.tokenizer.encode(" test", add_special_tokens=False)
            require(short, "engineering context fixture token missing")
            original_prompt = backend.prompt_ids
            backend.prompt_ids = lambda problem: (short * (CONTEXT // len(short) + 1))[:CONTEXT - 2]
            try:
                fixture = backend.generate({"seed": 20261008, "uid": digest([model, arm, "context-fixture"])},
                                           "engineering fixture", engineering=True)
            finally:
                backend.prompt_ids = original_prompt
            case = {"arm": arm, "status": "PASS", "binding_sha256": binding["sha256"],
                    "loader": status, "within_execution": qualification, "context_fixture": fixture,
                    "load_seconds": backend.load_seconds}
            cases.append(case)
            seen[signature] = arm
        except Exception as exc:
            failures.append({"arm": arm, "type": type(exc).__name__, "message": str(exc)[:2000]})
            break
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    failures.append({"arm": arm, "type": type(exc).__name__, "message": "cleanup: " + str(exc)[:500]})
                backend = None
    if len(cases) == 4 and not failures:
        cache = {digest([r["cache_spec"] for r in c["loader"]]) for c in cases if "loader" in c}
        backends = {digest([r["backend"] for r in c["loader"]]) for c in cases if "loader" in c}
        if len(cache) != 1 or len(backends) != 1:
            failures.append({"type": "InvariantFailure", "message": "cache or native backend differs across arms"})
        compact_sizes = {digest([r["total_expert_bytes"] for r in c["loader"]])
                         for c in cases if "loader" in c and c["arm"] != "native"}
        if len(compact_sizes) != 1:
            failures.append({"type": "InvariantFailure", "message": "compact arms have unequal realized expert storage"})
    return save(root / "QUALIFICATION.json", {"schema": "expert-memory-qualification-v1",
                "status": "PASS" if len(cases) == 4 and not failures else "FAIL", "cases": cases,
                "failures": failures, "model": model, "selection_sha256": selection["sha256"],
                "campaign_sha256": campaign["sha256"], "elapsed_seconds": time.monotonic() - start,
                "job_id": os.environ["SLURM_JOB_ID"], "scientific_accuracy": "NOT_MEASURED"})
