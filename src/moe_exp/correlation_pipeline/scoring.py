from __future__ import annotations

import multiprocessing
import os
import re
import threading
from typing import Any

from moe_exp.utils import answers_match, extract_model_answer

_CHOICE_RE = re.compile(r"\\boxed\s*\{\s*([A-J])\s*\}", re.IGNORECASE)
# math_verify parses the submitted answer text; the final answer sits at its end,
# so a bounded tail keeps pathological long outputs from dominating scoring time.
MAX_SCORING_CHARS = 20_000
# Outer wall-clock bound for one math_verify call executed in the helper process.
# math_verify's own signal timeouts (5 s parse, 5 s verify) still apply inside it.
MATH_VERIFY_TIMEOUT_S = 30.0

_POOL_LOCK = threading.Lock()
_POOL: Any = None


def extract_choice(text: str) -> str:
    matches = _CHOICE_RE.findall(text)
    return matches[-1].upper() if matches else ""


def extract_answer(text: str, answer_type: str) -> str:
    return extract_choice(text) if answer_type == "choice" else extract_model_answer(text)


def _math_verify_direct(model_text: str, gold_answer: str) -> tuple[bool | None, str]:
    """Run math_verify with its signal-based timeouts; only valid in a main thread."""
    try:
        from math_verify import parse, verify

        gold = parse(f"${gold_answer}$")
        prediction = parse(model_text)
        if not gold:
            return None, "gold_parse_failed"
        if not prediction:
            return None, "prediction_parse_failed"
        return bool(verify(gold, prediction)), "ok"
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as error:  # noqa: BLE001 - TimeoutException subclasses BaseException
        return None, f"error:{type(error).__name__}"


def _math_verify_worker(model_text: str, gold_answer: str) -> tuple[bool | None, str]:
    delay = float(os.environ.get("MOE_EXP_SCORING_TEST_DELAY_S", "0") or 0)
    if delay:  # test hook for the outer timeout
        import time
        time.sleep(delay)
    return _math_verify_direct(model_text, gold_answer)


def _pool():
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            size = int(os.environ.get("MOE_EXP_SCORING_PROCESSES", "0") or 0) or max(
                2, min(8, (os.cpu_count() or 4) // 2))
            _POOL = multiprocessing.get_context("spawn").Pool(processes=size)
        return _POOL


def _reset_pool(stale: Any) -> None:
    global _POOL
    with _POOL_LOCK:
        if _POOL is stale and _POOL is not None:
            _POOL.terminate()
            _POOL = None


def shutdown_scoring_pool() -> None:
    _reset_pool(_POOL)


def _math_verify(model_text: str, gold_answer: str) -> tuple[bool | None, str]:
    """Thread-safe math_verify: direct in the main thread, else in a helper process.

    math_verify's timeouts use signal.alarm, which raises in non-main threads; the
    previous implementation therefore silently fell back on every threaded call.
    """
    if threading.current_thread() is threading.main_thread():
        return _math_verify_direct(model_text, gold_answer)
    for attempt in range(2):
        pool = _pool()
        try:
            result = pool.apply_async(_math_verify_worker, (model_text, gold_answer))
            return result.get(timeout=MATH_VERIFY_TIMEOUT_S)
        except multiprocessing.TimeoutError:
            _reset_pool(pool)
            return None, "timeout"
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as error:  # noqa: BLE001 - pool torn down by another thread
            _reset_pool(pool)
            if attempt == 1:
                return None, f"error:{type(error).__name__}"
    return None, "error:unreachable"


def score_completion_detailed(
    example: dict[str, Any],
    *,
    answer_type: str,
    model_text: str,
) -> dict[str, Any]:
    gold = str(example.get("gold_answer") or "").strip()
    model_answer = extract_answer(model_text, answer_type)
    if not gold:
        return dict(model_answer=model_answer, is_correct=None,
                    method="unscored_no_gold_answer", math_verify_status="not_run")
    if answer_type == "choice":
        return dict(model_answer=model_answer, is_correct=model_answer == gold.upper(),
                    method="exact_multiple_choice", math_verify_status="not_run")
    verified, status = _math_verify(model_text[-MAX_SCORING_CHARS:], gold)
    if verified is not None:
        return dict(model_answer=model_answer, is_correct=verified, method="math_verify",
                    math_verify_status=status)
    return dict(model_answer=model_answer, is_correct=answers_match(model_answer, gold),
                method="normalized_exact_numeric_fallback", math_verify_status=status)


def score_completion(
    example: dict[str, Any],
    *,
    answer_type: str,
    model_text: str,
) -> tuple[str, bool | None, str]:
    result = score_completion_detailed(example, answer_type=answer_type, model_text=model_text)
    return result["model_answer"], result["is_correct"], result["method"]
