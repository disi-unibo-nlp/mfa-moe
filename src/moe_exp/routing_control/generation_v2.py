"""Executable request loop for a qualified synchronous routing backend.

The backend keeps the live engine sampler, returns exactly one newly emitted
token and all 40 layers' executed IDs/weights, and records native-at-the-same-
hidden-state IDs/weights for fidelity/dose. It must be GPU-qualified against
the execution contract before use; this loop imports no inference runtime.
"""
from __future__ import annotations

import time

import numpy as np

from .policy_stages_v2 import THINK_END_ID
from .state_analysis_v1 import validate_capture
from .study_v1 import digest, require
from .study_v2 import CAP, EARLY_WINDOW, _execution, seal, verify
from .capture_v2 import array_binding, CaptureStore


def execute_request(assignment, execution, backend, decode, *, dispatch, manifest, controller=None, prefix_ids=(), prefix_routes=None, capture_dir=None):
    """One isolated assignment; never restart generation by seed at a pulse.

    Backend API: start(assignment, prefix_ids, max_new_tokens), step(policy_name,
    worker_action, prediction_position), finish(). `step` receipts must contain
    token_id, row_position, policy_name, executed_ids, selected_weights,
    native_ids, native_weights, finished, finish_reason, and processed_input_tokens.
    Backend qualification binds code, engine profile, capture and sampler.
    """
    _execution(execution)
    verify(dispatch)
    verify(manifest)
    require(dispatch.get("status") == "AUTHORIZED" and dispatch.get("manifest_sha256") == manifest["sha256"] and
            manifest["execution"]["sha256"] == execution["sha256"] and
            any(r == assignment for r in manifest["assignments"]), "assignment lacks a matching authorized dispatch")
    qualification = verify(backend.qualification)
    require(qualification.get("execution_contract_sha256") == execution["sha256"] and qualification.get("status") == "PASS" and
            qualification.get("synchronous_one_token_steps") is True and qualification.get("preserves_live_sampler") is True,
            "backend must qualify token stepping and live sampler")
    require(assignment.get("continuation_cap") == CAP and assignment.get("early_endpoint_window") == EARLY_WINDOW and
            assignment.get("execution_contract_sha256") == execution["sha256"] and assignment.get("uid"), "assignment contract differs")
    return _execute_loop(assignment, backend, decode, controller=controller, prefix_ids=prefix_ids,
                         prefix_routes=prefix_routes, capture_dir=capture_dir, cap=CAP, execution=execution)


def execute_engineering_request(assignment, backend, decode, *, manifest, authorization, controller=None,
                                prefix_ids=(), capture_dir=None, observer=None):
    """Bootstrap path; receipts cannot authorize scientific generation."""
    verify(manifest)
    verify(authorization)
    require(manifest.get('schema') == 'routing-engineering-manifest-v2' and
            authorization.get('status') == 'AUTHORIZED_ENGINEERING_ONLY' and
            authorization.get('manifest_sha256') == manifest['sha256'] and assignment.get('engineering_only') is True and
            assignment in manifest['assignments'], 'engineering assignment lacks bounded authorization')
    cap = assignment['continuation_cap']
    require(type(cap) is int and 1 <= cap <= CAP, 'invalid engineering cap')
    return _execute_loop(assignment, backend, decode, controller=controller, prefix_ids=prefix_ids,
                         prefix_routes=None, capture_dir=capture_dir, cap=cap, execution=None, observer=observer)


def _execute_loop(assignment, backend, decode, *, controller, prefix_ids, prefix_routes, capture_dir, cap,
                  execution, observer=None):
    prefix = list(prefix_ids)
    require(all(type(t) is int and t >= 0 for t in prefix), "invalid frozen prefix tokens")
    if prefix:
        require(assignment.get("prefix_ids_sha256") == digest(prefix), "frozen prefix token IDs differ")
    require(controller is None or controller.uid == assignment["uid"] and controller.origin == len(prefix), "controller request or prefix differs")
    require(assignment.get('arm') != 'native' or controller is None, 'native assignment cannot fire interventions')
    allowed = assignment.get('actions') or ([assignment['action']] if assignment.get('action') else [])
    require(bool(allowed) == (controller is not None), 'assignment action requires its matching controller')
    if controller is not None:
        from dataclasses import asdict
        require(allowed and digest([asdict(a) for a in controller.actions]) == digest(allowed), 'controller actions differ from assignment arm')
    history_ids, history_weights = None, None
    if prefix_routes is not None:
        history_ids, history_weights = validate_capture(prefix, prefix_routes[0], prefix_routes[1])
    all_ids = np.empty((len(prefix) + cap, 40, 8), np.uint16)
    all_weights = np.empty((len(prefix) + cap, 40, 8), np.float32)
    if history_ids is not None:
        all_ids[:len(prefix)] = history_ids
    if history_weights is not None:
        all_weights[:len(prefix)] = history_weights
    history_available = not prefix or history_ids is not None
    history_weights_available = not prefix or history_weights is not None
    tokens, routes, weights = [], [], []
    dose_array = np.empty((cap, 40, 2), np.float32)
    store = CaptureStore(capture_dir, cap) if capture_dir is not None else None
    inactive_checks, processed_input, finish = 0, 0, None
    started = time.monotonic()
    try:
        backend.start(assignment, prefix, max_new_tokens=cap)
        for offset in range(cap):
            position = len(prefix) + offset
            if controller is not None:
                current_routes = all_ids[:position] if history_available else None
                current_weights = all_weights[:position] if history_weights_available else None
                controller.observe({"problem": assignment["problem"], "emitted_token_ids": prefix + tokens}, decode,
                                   routing_ids=current_routes, selected_weights=current_weights)
                name = controller.routing_rows(np.array([position], dtype=np.int64))[0]
                action = next((d["action"] for d in controller.decisions.values() if d.get("status") == "FIRED" and
                               d["action"]["name"] == name), None)
            else:
                name, action = "native", None
            if observer:
                observer(backend, controller, offset, name)
            receipt = backend.step(policy_name=name, worker_action=action, prediction_position=position)
            token = receipt["token_id"]
            require(type(token) is int and token >= 0 and receipt["row_position"] == position and receipt["policy_name"] == name,
                    "backend skipped/duplicated prediction row or used a different pulse")
            require(np.asarray(receipt["selected_weights"]).dtype == np.float32 and
                    np.asarray(receipt["native_weights"]).dtype == np.float32,
                    "capture requires float32 executed selected weights without lossy conversion")
            actual, selected = validate_capture([token], np.asarray(receipt["executed_ids"])[None], np.asarray(receipt["selected_weights"])[None])
            native, native_weight = validate_capture([token], np.asarray(receipt["native_ids"])[None], np.asarray(receipt["native_weights"])[None])
            require(type(receipt["finished"]) is bool and type(receipt["processed_input_tokens"]) is int and
                    receipt["processed_input_tokens"] >= 0, "backend finish/input accounting missing")
            if name == "native":
                require(np.array_equal(actual, native) and np.array_equal(selected, native_weight), "inactive-path fidelity failed within execution")
                inactive_checks += 1
            else:
                target_layer = action["experts"][0][0]
                untouched = [layer for layer in range(40) if layer != target_layer]
                require(np.array_equal(actual[:, untouched], native[:, untouched]) and
                        np.array_equal(selected[:, untouched], native_weight[:, untouched]), "action changed an untargeted layer")
            # ID-aligned sparse execution-weight L1, separated from inclusion.
            sparse_actual = np.zeros((40, 256))
            sparse_native = np.zeros((40, 256))
            np.put_along_axis(sparse_actual, actual[0], selected[0], axis=1)
            np.put_along_axis(sparse_native, native[0], native_weight[0], axis=1)
            dose_array[offset, :, 0] = 8 - np.any(actual[0, :, :, None] == native[0, :, None, :], axis=-1).sum(axis=-1)
            dose_array[offset, :, 1] = np.abs(sparse_actual - sparse_native).sum(axis=1) / 2
            if store:
                store.append(receipt, dose_array[offset, :, 0], dose_array[offset, :, 1])
            tokens.append(token)
            routes.append(actual[0])
            weights.append(selected[0])
            all_ids[position] = actual[0]
            all_weights[position] = selected[0]
            processed_input += receipt["processed_input_tokens"]
            if controller is not None and token == THINK_END_ID:
                # Observe closure immediately; the producing row above remains
                # in capture and subsequent answer rows receive native routing.
                controller.observe({"problem": assignment["problem"], "emitted_token_ids": prefix + tokens}, decode)
            if receipt["finished"]:
                require(receipt["finish_reason"] in ("stop", "length") and
                        (receipt["finish_reason"] != "length" or len(tokens) == cap), "invalid natural/cap finish")
                finish = receipt["finish_reason"]
                break
        if finish is None:
            finish = "length"
        if controller:
            controller.observe({"problem": assignment["problem"], "emitted_token_ids": prefix + tokens}, decode, terminal=True)
        ledger = controller.finish(hit_cap=finish == "length" and cap == CAP) if controller else {"controller_tokens": 0, "controller_calls": []}
        validate_capture(tokens, np.asarray(routes), np.asarray(weights))
        captured_ids = all_ids[len(prefix):len(prefix) + len(tokens)]
        captured_weights = all_weights[len(prefix):len(prefix) + len(tokens)]
        body = {"schema": "routing-study-generation-receipt-v2", "assignment_uid": assignment["uid"],
            "execution_contract_sha256": execution["sha256"] if execution else None, "continuation_cap": cap, "early_endpoint_window": EARLY_WINDOW,
            'engineering_only': execution is None,
            "generated_tokens": len(tokens), "emitted_token_ids": tokens, "finish_reason": finish,
            "controller": ledger, "total_policy_tokens": len(tokens) + ledger["controller_tokens"],
            "processed_input_tokens": processed_input, "inactive_rows_verified": inactive_checks,
            "dose": array_binding(dose_array[:len(tokens)]),
            "capture_binding": store.close() if store else None,
            "generation_wall_seconds": time.monotonic() - started, "captures_sha256": digest({
                "ids": array_binding(captured_ids), "selected_weights": array_binding(captured_weights)})}
        store = None
        return seal(body), captured_ids, captured_weights
    except BaseException as error:
        error.routing_partial = {"assignment_uid": assignment["uid"], "emitted_token_ids": tokens,
            "status": "EXPLICIT_EXECUTION_FAILURE", "controller": controller.finish(error=True) if controller else None,
            "generated_tokens": len(tokens), "processed_input_tokens": processed_input,
            "generation_wall_seconds": time.monotonic() - started, 'capture_binding': store.close() if store else None}
        store = None
        raise
    finally:
        backend.finish()
