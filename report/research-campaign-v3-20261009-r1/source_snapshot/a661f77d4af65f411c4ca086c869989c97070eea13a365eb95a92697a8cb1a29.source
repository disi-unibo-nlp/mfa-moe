"""Keep the first failure and record every subsequent cleanup failure."""
import os
import sys
import traceback


def failure_scope(exc):
    if hasattr(exc,"campaign_failure_scope"):return exc.campaign_failure_scope
    return "model" if any(marker in str(exc).lower() for marker in (
        "out of memory", "cuda error", "checkpoint", "saved sequence exceeds",
        "native selector fidelity fails", "compact/full-resident masked equivalence failed",
        "kv capacity differs", "actual process residence unavailable")) else "shared"


def exception_record(exc, phase, assignment=None):
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "driver"))
    if "vllm.distributed" in sys.modules:
        try:
            rank = sys.modules["vllm.distributed"].get_tensor_model_parallel_rank()
        except Exception:
            pass
    return {"error_type": type(exc).__name__, "error": str(exc)[:4000],
            "phase": getattr(exc,"campaign_phase",phase), "rank": rank,
            "assignment": assignment, "traceback": "".join(traceback.format_exception(exc))[-12000:],
            "cleanup_errors": getattr(exc, "campaign_cleanup_errors", []),
            "worker_failures": getattr(exc, "campaign_worker_failures", [])}


def guard(observer, operation, phase):
    def call(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except BaseException as exc:
            if getattr(observer, "failure", None) is None:
                observer.failure = exception_record(exc, phase)
            raise
    return call


def detach_failed(llm, method, primary):
    reports = llm.collective_rpc(method)
    primary.campaign_worker_failures = [r["failure"] for r in reports if r.get("failure")]
    primary.campaign_cleanup_errors = getattr(primary, "campaign_cleanup_errors", []) + [e for r in reports for e in r.get("cleanup_errors", [])]


def cleanup(actions, primary=None):
    errors = []
    for phase, action in actions:
        try:
            action()
        except BaseException as exc:
            errors.append(exception_record(exc, phase))
    if primary is not None:
        primary.campaign_cleanup_errors = getattr(primary, "campaign_cleanup_errors", []) + errors
    elif errors:
        failure = RuntimeError("cleanup failed: " + errors[0]["error"])
        failure.campaign_cleanup_errors = errors
        raise failure
    return errors


def restore(observer):
    actions = []
    if observer.handle is not None:
        handle, observer.handle = observer.handle, None
        actions.append(("remove_forward_hook", handle.remove))
    for router, original in reversed(observer.originals):
        actions.append(("restore_router", lambda r=router, o=original: setattr(r, "_compute_routing", o)))
    observer.originals = []
    return cleanup(actions, sys.exception())


def close_preserving(backend):
    return cleanup([("backend_shutdown", backend.close)], sys.exception())
