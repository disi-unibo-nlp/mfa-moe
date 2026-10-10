"""Prefix-only, first-opportunity controller with auditable no-fires/resume.

This CPU reference is also the boundary for a GPU adapter. Per-token observe
calls are mandatory after the initial frozen prefix. Model RNG/KV resume and
execution-local capture require separate GPU evidence; a seed restart cannot
claim sampler continuity.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from .context_v2 import transform_context
from .prefix import PrefixInput
from .state_analysis_v1 import trailing_features, validate_capture
from .study_v1 import STATES, digest, require
from .study_v2 import Action, CAP, TRANSITIONS, seal, verify

THINK_END_ID = 248069


def freeze_context_gate(model, telemetry, *, accepted_contexts, position_edges=(0, 1024, 4096, 16385), window=64):
    """Freeze skip probabilities from discovery's first opportunities only."""
    verify(model)
    require(model.get("population") == "discovery" and model.get("status") in ("FALLBACK_K1", "MULTIPLE_CONTEXTS_SUPPORTED") and
            window in (64, 256) and len(position_edges) >= 2 and position_edges[0] == 0 and
            all(type(p) is int for p in position_edges) and list(position_edges) == sorted(set(position_edges)),
            "unqualified context gate model or strata")
    require(accepted_contexts and set(accepted_contexts) <= set(TRANSITIONS) and
            all(values and set(values) <= set(range(model["selected_K"])) for values in accepted_contexts.values()),
            "invalid discovery context acceptance rule")
    counts, seen = {}, set()
    for row in telemetry:
        require(row.get("population") == "discovery" and row.get("first_qualified_opportunity") is True and
                row["family"] in model["discovery_families"] and
                row.get("model_sha256") == model["sha256"] and row["transition"] in accepted_contexts and
                row["source_state"] in STATES and row["prefix_status"] in
                ("intermediate", "complete_unchecked", "complete_checked", "unresolved") and
                type(row["position"]) is int and type(row["context"]) is int and
                0 <= row["context"] < model["selected_K"], "invalid discovery opportunity telemetry")
        identity = row["family"], row["transition"], row["seed"]
        require(identity not in seen and row["seed"] in (0, 1), "duplicate first opportunity")
        seen.add(identity)
        bin_index = position_bin(row["position"], position_edges)
        key = digest([row["transition"], row["source_state"], row["prefix_status"], bin_index])
        item = counts.setdefault(key, {"opportunities": 0, "accepted": 0})
        item["opportunities"] += 1
        item["accepted"] += row["context"] in accepted_contexts[row["transition"]]
    require(counts, "discovery random-skip exposure telemetry unavailable")
    for value in counts.values():
        value["accept_probability"] = value["accepted"] / value["opportunities"]
    return seal({"schema": "routing-first-opportunity-gate-v2", "status": "PASS", "population": "discovery",
        "context_model_sha256": model["sha256"], "accepted_contexts": accepted_contexts, "window": window,
        "position_edges": list(position_edges), "strata": counts, "telemetry_sha256": digest(telemetry),
        "decline": "NO_FIRE_FOR_TRANSITION", "unseen_stratum": "ABSTAIN", "evaluates": "FIRST_QUALIFIED_OPPORTUNITY"})


def position_bin(position, edges):
    require(edges[0] <= position < edges[-1], "position outside frozen opportunity strata")
    return next(i for i in range(len(edges) - 1) if edges[i] <= position < edges[i + 1])


class PrefixPolicyController:
    def __init__(self, actions, recognizer, recognition_audit, *, uid, seed, mode="state_only",
                 context_model=None, gate=None, ordered=False, origin_position=0):
        verify(recognition_audit)
        require(recognition_audit.get("status") == "PASS" and recognition_audit.get("prefix_only") is True and
                recognition_audit.get("recognizes_substantive_acts") is True, "passing prefix semantic audit required")
        require(1 <= len(actions) <= 2 and all(isinstance(a, Action) for a in actions) and
                len({a.action_id for a in actions}) == len(actions) and
                len({a.transition for a in actions}) == len(actions) and
                all(a.transition in recognition_audit["supported_transitions"] for a in actions),
                "at most one supported action per transition and two per request")
        require(uid and seed in (0, 1) and mode in ("state_only", "routing_context", "random_skip") and
                type(origin_position) is int and origin_position >= 0, "invalid request controller")
        if mode != "state_only":
            verify(context_model)
            verify(gate)
            require(gate.get("status") == "PASS" and gate.get("population") == "discovery" and
                    gate.get("context_model_sha256") == context_model["sha256"] and
                    all(a.transition in gate["accepted_contexts"] for a in actions), "unbound context gate")
        self.actions, self.recognizer, self.audit = tuple(actions), recognizer, deepcopy(recognition_audit)
        self.uid, self.seed, self.mode = uid, seed, mode
        self.model, self.gate, self.ordered = deepcopy(context_model), deepcopy(gate), ordered
        self.origin = origin_position
        self.previous, self.closure_position = None, None
        self.opportunities, self.decisions, self.calls = {}, {}, []
        self.finished = False

    def _gate_decision(self, action, response, position, ids, weights):
        if self.mode == "state_only":
            return True, {"mode": self.mode}
        state, status = response.get("source_state"), response.get("candidate_status")
        if state not in STATES or status not in ("intermediate", "complete_unchecked", "complete_checked", "unresolved"):
            return False, {"mode": self.mode, "status": "AMBIGUOUS_STRATUM"}
        try:
            bin_index = position_bin(position, self.gate["position_edges"])
        except ValueError:
            return False, {"mode": self.mode, "status": "UNSEEN_STRATUM"}
        key = digest([action.transition, state, status, bin_index])
        stratum = self.gate["strata"].get(key)
        if stratum is None:
            return False, {"mode": self.mode, "status": "UNSEEN_STRATUM", "stratum": key}
        if self.mode == "random_skip":
            # Counter-based, request-local randomness is preserved by resume.
            uniform = int(digest(["random-skip-v2", self.gate["sha256"], self.uid, self.seed, action.transition])[:13], 16) / 16**13
            probability = stratum["accept_probability"]
            return uniform < probability, {"mode": self.mode, "stratum": key, "uniform": uniform,
                                            "accept_probability": probability}
        if ids is None:
            return False, {"mode": self.mode, "status": "MISSING_PREFIX_ROUTING"}
        # Reject a caller's future capture; every available row is already emitted.
        validated, selected = validate_capture(self.previous.emitted_token_ids, ids, weights)
        feature = trailing_features(validated, position, weights=selected, window=self.gate["window"])
        if feature["status"] != "AVAILABLE":
            return False, {"mode": self.mode, "status": feature["status"]}
        contexts, _ = transform_context(self.model["model"], feature["selection"][None])
        context = int(contexts[0])
        return context in self.gate["accepted_contexts"][action.transition], {
            "mode": self.mode, "stratum": key, "context": context,
            "prefix_capture_sha256": digest(validated.tolist()), "window": self.gate["window"]}

    def observe(self, record, decode, *, routing_ids=None, selected_weights=None, terminal=False):
        require(not self.finished, "controller assignment already finished")
        require(type(terminal) is bool, "invalid terminal prefix receipt")
        # Future text, grades, answers and offline labels are never inspected.
        prefix = PrefixInput.from_record({"problem": record["problem"], "emitted_token_ids": record["emitted_token_ids"],
                                          "emitted_text": decode(record["emitted_token_ids"])})
        position = len(prefix.emitted_token_ids)
        require(self.origin <= position <= self.origin + CAP, "16k continuation cap exceeded")
        if self.previous is not None:
            require(prefix.problem == self.previous.problem and
                    prefix.emitted_token_ids[:len(self.previous.emitted_token_ids)] == self.previous.emitted_token_ids,
                    "emitted prefix changed its history")
            if prefix.emitted_token_ids == self.previous.emitted_token_ids:
                require(prefix.emitted_text == self.previous.emitted_text, "decoder changed identical prefix")
                return []
            require(position == len(self.previous.emitted_token_ids) + 1, "executor skipped emitted-token observation")
        else:
            require(position == self.origin, "first observation must be the frozen initial prefix")
        self.previous = prefix
        if THINK_END_ID in prefix.emitted_token_ids:
            self.closure_position = prefix.emitted_token_ids.index(THINK_END_ID)
        if self.closure_position is not None:
            for action in self.actions:
                if action.action_id not in self.decisions:
                    self.decisions[action.action_id] = {"status": "NO_FIRE_CLOSED", "position": position}
            return []
        if terminal:
            return []  # Final emitted token is accounted, with no later pulse.
        response = self.recognizer(prefix)
        require(isinstance(response, dict), "recognizer must account for failed and ambiguous calls")
        for key in ("controller_tokens", "controller_input_tokens", "controller_seconds"):
            value = response.get(key)
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value >= 0 and
                    (key.endswith("seconds") or type(value) is int), "missing controller expenditure")
        require(response.get("prefix_sha256") == digest(asdict(prefix)), "recognizer answered another prefix")
        self.calls.append({"position": position, **{k: response[k] for k in
            ("controller_tokens", "controller_input_tokens", "controller_seconds", "prefix_sha256")}})
        acts = response.get("substantive_acts")
        if type(response.get("boundary")) is not bool or not isinstance(acts, list) or any(a not in TRANSITIONS for a in acts):
            for action in self.actions:
                opportunity = self.opportunities.get(action.action_id)
                if opportunity and action.action_id not in self.decisions and position >= opportunity["position"] + action.delay:
                    self.decisions[action.action_id] = {"status": "NO_FIRE_AMBIGUOUS_DELAY", "position": position}
            return []  # Ambiguous recognition abstains, preserving its charge.
        fired = []
        for index, action in enumerate(self.actions):
            key = action.action_id
            if key in self.decisions:
                continue
            if action.transition in acts:
                self.decisions[key] = {"status": "NO_FIRE_DESIRED_ACT_OCCURRED", "position": position}
                continue
            eligible = response["boundary"] is True and (
                action.transition == "candidate_to_verify" and response.get("candidate_status") == "complete_unchecked" or
                action.transition == "checked_candidate_to_close" and response.get("candidate_status") == "complete_checked" or
                action.transition == "approach_to_commit" and response.get("approach_identified") is True)
            if self.ordered and any(self.decisions.get(a.action_id, {}).get("status") != "FIRED" for a in self.actions[:index]):
                if eligible:
                    self.opportunities[key] = {"position": position, "prefix_sha256": response["prefix_sha256"]}
                    self.decisions[key] = {"status": "NO_FIRE_ORDER_UNAVAILABLE", "position": position}
                continue
            if eligible and key not in self.opportunities:
                accept, context = self._gate_decision(action, response, position, routing_ids, selected_weights)
                self.opportunities[key] = {"position": position, "prefix_sha256": response["prefix_sha256"], "gate": context}
                if not accept:
                    self.decisions[key] = {"status": "NO_FIRE_CONTEXT_DECLINED", "position": position}
                    continue
            if key not in self.opportunities:
                continue
            start = self.opportunities[key]["position"] + action.delay
            if position < start:
                continue
            require(position == start, "executor skipped delayed boundary")
            if any(d.get("status") == "FIRED" and d["end"] > position for d in self.decisions.values()):
                # A first opportunity during another pulse is retained as a
                # no-fire. It cannot be postponed to a more favorable context.
                self.decisions[key] = {"status": "NO_FIRE_PULSE_OVERLAP", "position": position}
                continue
            decision = {"status": "FIRED", "position": position, "end": position + action.pulse_tokens,
                        "action": action.worker_action(), "transition": action.transition}
            self.decisions[key] = decision
            fired.append(deepcopy(decision))
        return fired

    def routing_rows(self, positions):
        positions = np.asarray(positions)
        require(positions.ndim == 1 and positions.dtype.kind in "iu", "invalid isolated request prediction rows")
        names = np.full(len(positions), "native", object)
        for action in self.actions:
            decision = self.decisions.get(action.action_id, {})
            if decision.get("status") != "FIRED":
                continue
            take = (positions >= decision["position"]) & (positions < decision["end"])
            if self.closure_position is not None:
                take &= positions <= self.closure_position  # closure-producing row stays captured/routed
            require(np.all(names[take] == "native"), "composition pulses overlap")
            names[take] = action.action_id
        return names

    def ledger(self):
        return deepcopy({"uid": self.uid, "seed": self.seed, "mode": self.mode, "opportunities": self.opportunities,
            "decisions": self.decisions, "controller_calls": self.calls,
            "controller_tokens": sum(c["controller_tokens"] for c in self.calls),
            "controller_input_tokens": sum(c["controller_input_tokens"] for c in self.calls),
            "controller_seconds": sum(c["controller_seconds"] for c in self.calls),
            "recognizer_sha256": self.audit["sha256"], "context_gate_sha256": self.gate["sha256"] if self.gate else None,
            "closure_position": self.closure_position, "continuation_cap": CAP, "origin_position": self.origin,
            "top_k": 8, "shared_expert": "unchanged"})

    def finish(self, *, hit_cap=False, error=False):
        if hit_cap:
            require(self.previous and len(self.previous.emitted_token_ids) == self.origin + CAP, "cap finish before 16k")
        for action in self.actions:
            if action.action_id not in self.decisions:
                self.decisions[action.action_id] = {"status": "NO_FIRE_EXECUTION_ERROR" if error else
                    "NO_FIRE_CAP" if hit_cap else "NO_ELIGIBLE_OPPORTUNITY", "position":
                    len(self.previous.emitted_token_ids) if self.previous else self.origin}
        self.finished = True
        return self.ledger()

    def snapshot(self, sampler_state, *, manifest_sha256, capture_sha256):
        verify(sampler_state)
        position = len(self.previous.emitted_token_ids) if self.previous else self.origin
        require(sampler_state.get("assignment_uid") == self.uid and sampler_state.get("next_prediction_index") == position and
                sampler_state.get("rng_state_sha256") and sampler_state.get("engine_state_sha256") and
                manifest_sha256 and capture_sha256, "resume requires exact engine/RNG/capture state")
        return seal({"schema": "routing-controller-resume-v2", "manifest_sha256": manifest_sha256,
            "capture_sha256": capture_sha256, "sampler_state": deepcopy(sampler_state), "ledger": self.ledger(),
            "actions": [asdict(a) for a in self.actions], "ordered": self.ordered, "finished": self.finished,
            "previous": asdict(self.previous) if self.previous else None})

    def restore(self, snapshot, *, sampler_state, manifest_sha256, capture_sha256):
        verify(snapshot)
        verify(sampler_state)
        require(self.previous is None and not self.calls and not self.decisions and
                snapshot["manifest_sha256"] == manifest_sha256 and snapshot["capture_sha256"] == capture_sha256 and
                snapshot["sampler_state"] == sampler_state and digest(snapshot["actions"]) == digest([asdict(a) for a in self.actions]) and
                snapshot["ordered"] == self.ordered, "resume binding or fresh controller differs")
        ledger = snapshot["ledger"]
        require(ledger["uid"] == self.uid and ledger["seed"] == self.seed and ledger["mode"] == self.mode and
                ledger["recognizer_sha256"] == self.audit["sha256"] and ledger["origin_position"] == self.origin and
                ledger["context_gate_sha256"] == (self.gate["sha256"] if self.gate else None), "resume request or policy differs")
        self.previous = PrefixInput.from_record(snapshot["previous"]) if snapshot["previous"] else None
        position = len(self.previous.emitted_token_ids) if self.previous else self.origin
        require(sampler_state["assignment_uid"] == self.uid and sampler_state["next_prediction_index"] == position,
                "sampler output cursor differs")
        self.opportunities, self.decisions, self.calls = deepcopy(ledger["opportunities"]), deepcopy(ledger["decisions"]), deepcopy(ledger["controller_calls"])
        self.closure_position, self.finished = ledger["closure_position"], snapshot["finished"]
        require(self.ledger() == ledger, "resume cost ledger differs")
