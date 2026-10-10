"""Bounded native generation and exact per-request action timing in Slurm."""
from __future__ import annotations

import getpass
import math
import os
import socket
import time

from .contracts import digest, require, seal, verify
from .registry import engine_kwargs
from .semantics import eligible_proposal


def validate_raw_action(contract, action):
    """Reject malformed client actions before sampler imports or worker RPCs."""
    if action is None:
        return
    require(isinstance(action, dict) and set(action) == {"layer", "expert", "bias"} and
            type(action["layer"]) is int and action["layer"] >= 0 and
            type(action["expert"]) is int and 0 <= action["expert"] < contract["experts"] and
            type(action["bias"]) in (int, float) and math.isfinite(action["bias"]) and action["bias"] >= 0,
            "invalid singleton client logit action")
    # Hybrid checkpoints use physical layer IDs, not dense routed-layer indices.


def require_compute():
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID") and
            getpass.getuser() == "lmolfett" and socket.gethostname().startswith("lrdn") and
            os.environ.get("SLURM_JOB_ACCOUNT") == "iscrc_miosr" and
            os.environ.get("SLURM_JOB_PARTITION") == "boost_usr_prod", "inference requires LEONARDO GPU Slurm step")
    require(len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) == 2,
            "native generator requires exactly two allocated GPUs")


def closure_sequences(tokenizer, key):
    text = "<|channel|>final<|message|>" if key == "gpt" else "<channel|>" if key == "gemma" else "</think>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    require(ids and tokenizer.decode(ids, skip_special_tokens=False) == text,
            "pinned tokenizer cannot reproduce native closure marker")
    return [ids]


class Backend:
    def __init__(self, model, *, deadline, engineering=False):
        require_compute(); verify(model)
        from vllm import LLM
        self.contract, self.deadline, self.engineering = model, deadline, engineering
        self.kwargs = engine_kwargs(model)
        self.kwargs.update(worker_extension_cls="moe_exp.verification_study.adapters.VerificationWorker")
        if engineering:
            self.kwargs["logits_processors"] = ["routing_local_logits_v2:EngineeringRepeat"]
        started = time.monotonic()
        self.llm = LLM(**self.kwargs)
        self.load_seconds = time.monotonic() - started
        require(type(self.llm.llm_engine.engine_core).__name__ == "InprocClient", "engine can run ahead of controller")
        self.tokenizer = self.llm.get_tokenizer()
        self.closure = closure_sequences(self.tokenizer, model["key"])
        from .prepare import runtime_binding, code_binding
        self.runtime = runtime_binding()
        self.executor_sha256 = digest(code_binding())

    def prompt_ids(self, problem):
        from moe_exp.verification_study.readers import normalize_prompt_ids
        kwargs = dict(self.contract["template_kwargs"])
        if self.contract["reasoning_effort"] is not None:
            kwargs["reasoning_effort"] = self.contract["reasoning_effort"]
        return normalize_prompt_ids(self.tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}], tokenize=True, add_generation_prompt=True, **kwargs))

    def generate(self, assignment, problem, *, action, recognition=None, capture_all=False, repeat=None, policy=None, bank=False):
        validate_raw_action(self.contract, action)
        from vllm import SamplingParams
        require(time.time() < self.deadline, "allocation deadline reached before request")
        require(repeat is None or self.engineering, "forced fixture is engineering-only")
        if not self.engineering:
            from .contracts import identity
            require(assignment["model_sha256"] == self.contract["sha256"] and
                    assignment["uid"] == identity({k: v for k, v in assignment.items() if k != "uid"}),
                    "foreign or changed scientific assignment")
            from .protocol import validate_assignment
            validate_assignment(assignment,calibration=bank)
            if assignment["stage"] in ("conditional", "utility"):
                from .policy import action_for
                require(policy is not None and action == action_for(policy, assignment), "action differs from frozen policy")
            else:
                require(action is None or action["bias"] == 0., "native source received intervention")
        engine = self.llm.llm_engine
        require(not engine.has_unfinished_requests(), "neighboring request is active")
        prefix = assignment.get("prefix") or {}
        prefix_ids = list(prefix.get("prefix_token_ids", []))
        prompt = self.prompt_ids(problem)
        require(len(prompt) + len(prefix_ids) + assignment["cap"] <= self.kwargs["max_model_len"],
                "input exceeds model context; truncation forbidden")
        if prefix_ids:
            require(digest(prefix_ids) == prefix["prefix_ids_sha256"], "frozen prefix tokens changed")
            if not self.engineering:
                require(prefix["original_prompt_ids_sha256"] == digest(prompt) and
                        prefix["candidate"]["problem_sha256"] == digest(problem), "foreign source-native prompt/candidate")
        if recognition is not None:
            verify(recognition)
            require(recognition["status"] == "PASS" and recognition["model_sha256"] == self.contract["sha256"],
                    "online recognizer unqualified")
            require(recognition.get("method", "cheap") == "cheap", "semantic veto requires isolated executor qualification")
        start_action = None if action is None else {**action, "start": len(prefix_ids),
            "bias": action["bias"] if assignment["stage"] == "conditional" and assignment["arm"] != "native" else 0.}
        sampler = {k: v for k, v in self.contract["sampler"].items()}
        if sampler.get("top_k") == 0:
            sampler["top_k"] = -1
        extra = {"routing_engineering_repeat": repeat} if repeat else None
        params = SamplingParams(**sampler, seed=assignment["seed"], max_tokens=assignment["cap"],
                                extra_args=extra, ignore_eos=bool(repeat))
        uid = assignment["uid"]
        emitted, decisions, opportunities = [], [], []
        fired, closed, completion = False, False, None
        start = time.monotonic(); controller_seconds = 0.; controller_input_tokens = 0; controller_calls = 0
        prefill_seconds, capture_seconds = None, None
        try:
            attach = self.llm.collective_rpc("verification_begin", kwargs={"contract": self.contract,
                "prompt": prompt, "prefix": prefix_ids, "closure": self.closure,
                "action": start_action, "capture_all": capture_all,
                "audit_noop": self.engineering and capture_all and assignment["cap"] <= 300})
            require(len(attach) == 2 and attach[0]["geometry"] == attach[1]["geometry"], "TP geometry disagreement")
            engine.add_request(uid, {"prompt_token_ids": prompt + prefix_ids}, params)
            while engine.has_unfinished_requests():
                require(time.time() < self.deadline, "allocation deadline during assignment")
                for result in engine.step():
                    require(result.request_id == uid and len(result.outputs) == 1, "neighboring request or multiple samples")
                    item = result.outputs[0]
                    require(list(item.token_ids[:len(emitted)]) == emitted and len(item.token_ids) <= len(emitted) + 1,
                            "engine skipped or rewrote an emitted token")
                    if len(item.token_ids) == len(emitted):
                        continue
                    emitted = list(item.token_ids)
                    if prefill_seconds is None:
                        prefill_seconds = time.monotonic()-start
                    completion = item
                    all_ids = prefix_ids + emitted
                    closed = closed or any(tuple(all_ids[-len(s):]) == tuple(s) for s in self.closure)
                    if recognition and not closed and assignment["stage"] != "conditional" and not fired:
                        before = time.monotonic()
                        controller_calls += 1
                        controller_input_tokens += len(prompt) + len(all_ids)
                        text = self.tokenizer.decode(all_ids, skip_special_tokens=False)
                        proposal = eligible_proposal(problem, text)
                        controller_seconds += time.monotonic() - before
                        if proposal:
                            opportunity = {"position": len(all_ids), "candidate": proposal,
                                "prefix_token_ids": list(all_ids), "prefix_ids_sha256": digest(all_ids),
                                "prefix_only": True, "recognizer_sha256": recognition["sha256"]}
                            # Final enrollment must bind to the qualified detector, not a future reader.
                            opportunities.append(opportunity)
                            if assignment["arm"] != "native" and action:
                                live = {**action, "start": len(all_ids)}
                                ack = self.llm.collective_rpc("verification_action", kwargs={"action": live})
                                require(len(ack) == 2 and all(r["action"] == live for r in ack), "TP action acknowledgment differs")
                                decisions.append({"status": "FIRED", "position": len(all_ids), "action": live})
                            fired = True
            require(completion is not None, "no generated result")
            if bank:
                require(not self.engineering and assignment["stage"] == "discovery" and action["bias"] == 0.,
                        "calibration bank can only contain native discovery predictions")
                self.llm.collective_rpc("verification_bank", kwargs={"uid": uid, "family": assignment["family"],
                    "source_sha256": digest(assignment)})
        finally:
            try:
                if engine.has_unfinished_requests():
                    engine.abort_request([uid], internal=True)
            finally:
                capture_started = time.monotonic()
                reports = self.llm.collective_rpc("verification_end")
                capture_seconds = time.monotonic()-capture_started
        require(len(reports) == 2 and reports[0]["capture_sha256"] == reports[1]["capture_sha256"],
                "TP executed routing disagreement")
        require(all(len(v["positions"]) == len(emitted) for v in reports[0]["capture"].values()),
                "capture does not account for every emitted token")
        pulse_positions=[p for values in reports[0]["capture"].values()
                         for p,a in zip(values["positions"],values["active"]) if a]
        elapsed = time.monotonic() - start
        return seal({"schema": "verification-generation-receipt-v3", "assignment_uid": uid,
            "model_sha256": self.contract["sha256"], "job_id": os.environ["SLURM_JOB_ID"],
            "prompt_input_tokens":len(prompt),"prefix_input_tokens":len(prefix_ids),
            "assignment": assignment, "action":action, "original_prompt_ids_sha256": digest(prompt),
            "prefix_ids_sha256": digest(prefix_ids), "emitted_token_ids": emitted,
            "emitted_text": self.tokenizer.decode(emitted, skip_special_tokens=False),
            "finish_reason": completion.finish_reason, "generated_tokens": len(emitted),
            "controller_tokens": 0, "controller_input_tokens": controller_input_tokens,
            "controller_calls": controller_calls,
            "controller_seconds": controller_seconds, "generation_seconds": elapsed,
            "prefill_seconds": prefill_seconds, "capture_closeout_seconds": capture_seconds,
            "decode_inclusive_seconds": elapsed-(prefill_seconds or 0)-(capture_seconds or 0),
            "load_seconds": self.load_seconds, "capture": reports[0],
            "runtime_sha256": self.runtime["sha256"], "executor_sha256": self.executor_sha256,
            "engine_kwargs": self.kwargs,
            "recognizer_sha256": recognition["sha256"] if recognition else None,
            "qualified_opportunities": opportunities, "decisions": decisions,
            "pulse_fired":bool(pulse_positions),"pulse_active_predictions":len(pulse_positions),
            "pulse_start":min(pulse_positions) if pulse_positions else None,
            "pulse_last_prediction":max(pulse_positions) if pulse_positions else None,
            "closed": closed, "engineering_only": self.engineering,
            "closure_token_sequences": self.closure,
            "execution_complete": True, "reader_truth": "NOT_YET_MEASURED"})

    def close(self):
        self.llm.llm_engine.engine_core.shutdown()
