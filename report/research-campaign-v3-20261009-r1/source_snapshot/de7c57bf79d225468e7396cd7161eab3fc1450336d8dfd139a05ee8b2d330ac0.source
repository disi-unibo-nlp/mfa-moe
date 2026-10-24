"""No blind retries after interrupted execution or ambiguous submission."""
from __future__ import annotations

import fcntl
from contextlib import contextmanager
import time

from .contracts import digest, load, private_path, require, save


class Journal:
    def __init__(self, root):
        self.root = private_path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    @contextmanager
    def lock(self, uid):
        require(len(uid) == 64 and all(c in "0123456789abcdef" for c in uid), "invalid journal identity")
        with (self.root / (uid + ".lock")).open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def begin(self, assignment, *, retry_receipt=None, maximum_retries=0):
        uid = assignment["uid"]
        require(uid == digest({k: v for k, v in assignment.items() if k != "uid"}), "assignment changed")
        with self.lock(uid):
            if (self.root / (uid + ".result.json")).exists():
                return {"status": "ALREADY_COMMITTED", "result": load(self.root / (uid + ".result.json"))}
            attempts = sorted(self.root.glob(uid + ".start-*.json"))
            if attempts:
                require(maximum_retries == 1 and len(attempts) == 1 and retry_receipt is not None,
                        "unresolved assignment; explicit card-approved reconciliation required")
                require(retry_receipt.get("uid") == uid and retry_receipt.get("terminal") is True and
                        retry_receipt.get("retry_authorized") is True and retry_receipt.get("draw_committed") is False,
                        "retry reconciliation incomplete")
            return save(self.root / f"{uid}.start-{len(attempts)}.json",
                        {"status": "STARTED", "assignment": assignment, "attempt": len(attempts),
                         "retry_reconciliation": retry_receipt})

    def commit(self, assignment, result):
        with self.lock(assignment["uid"]):
            require(bool(list(self.root.glob(assignment["uid"] + ".start-*.json"))), "assignment not started")
            return save(self.root / (assignment["uid"] + ".result.json"),
                        {"schema": "expert-memory-result-v1", "assignment": assignment, **result})

    def submission_intent(self, card):
        uid = digest(["submission", card["sha256"]])
        request = card.get("request") or {}
        units = set(request.get("assignment_uids", []))
        with self.lock(digest(["all-submission-intents"])):
            require(not (self.root / (uid + ".intent.json")).exists(),
                    "existing submission intent must be reconciled before another submission")
            for p in self.root.glob("*.intent.json"):
                existing = load(p)
                terminal_path = p.with_name(existing["uid"] + ".terminal.json")
                terminal = load(terminal_path) if terminal_path.exists() else None
                if terminal is not None:
                    require(terminal["intent_sha256"] == existing["sha256"] and terminal["terminal"] is True,
                            "foreign terminal submission reconciliation")
                    continue
                require(not (units & set(existing.get("assignment_uids", []))) and
                        not (request.get("output") and request["output"] == existing.get("output")),
                        "overlapping unresolved submission must be reconciled")
            return save(self.root / (uid + ".intent.json"), {"uid": uid, "card": card["sha256"],
                        "status": "UNRESOLVED", "assignment_uids": sorted(units),
                        "output": request.get("output"), "created_epoch": time.time()})
