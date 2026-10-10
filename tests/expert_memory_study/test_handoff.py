"""Submission ownership, protection ordering and prospective stage gates."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from moe_exp.expert_memory_study.contracts import digest, load, save, seal
import expert_memory_study_v1_handoff_r1 as handoff


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "ROOT", tmp_path)
    save(tmp_path / "CAMPAIGN.json", {"engineering_order": ["m"], "models": {"m": {
        "local_path": "checkpoint", "template_kwargs": {}, "reasoning_effort": None}}})
    save(tmp_path / "WORKLOAD_AUTHORIZATION-r1.json", {"user_quote": "Go on and submit everything we need"})
    return tmp_path


def test_scheduler_environment_is_cleaned_only_for_submission_children(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SBATCH_MEM_PER_NODE", "99G")
    calls = []
    monkeypatch.setattr(handoff.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    with handoff.transport():
        handoff.subprocess.run(["sbatch", "child"])
        handoff.subprocess.run(["squeue"])
    assert "SLURM_JOB_ID" not in calls[0][1]["env"]
    assert "SBATCH_MEM_PER_NODE" not in calls[0][1]["env"]
    assert "env" not in calls[1][1]
    assert handoff.os.environ["SLURM_JOB_ID"] == "123"


def test_primary_and_all_ancestor_guards_are_accepted_before_release(root, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    binding = save(root / "CONTINUATION_READINESS-r1.json", {"code_files": {}})
    wave = save(root / "wave.json", {"jobs": [{"job_id": "50"}, {"job_id": "51"}]})
    events, cards = [], []

    def accept(path):
        card = load(path)
        cards.append(card)
        job = str(100 + len(cards))
        events.append(("accepted", job))
        return seal({"job_id": job})

    def command(args, **kwargs):
        if args[0] == "squeue":
            return "101|PENDING\n102|PENDING\n103|PENDING"
        if args[1:3] == ["show", "job"]:
            return "JobState=PENDING Reason=JobHeldUser"
        events.append(("release", args[-1]))
        return ""

    monkeypatch.setattr(handoff, "controller_accept", accept)
    monkeypatch.setattr(handoff, "command", command)
    proof = handoff.install("pricing", root / "wave.json")
    assert [event[0] for event in events[:3]] == ["accepted"] * 3
    assert cards[0]["dependency"] == "afterany:50:51:99"
    assert cards[1]["dependency"] == "afternotok:101"
    assert cards[2]["dependency"] == "afternotok:101:102"
    assert cards[2]["failure_ancestor_job_ids"] == ["101", "102"]
    assert cards[0]["readiness_sha256"] == binding["sha256"]
    assert proof["all_failure_guards_accepted_before_release"]


def test_ambiguous_controller_intent_is_never_resubmitted(root, monkeypatch):
    card = save(root / "controller.json", {"phase": "pricing"})
    save(root / "continuation/submissions" / (card["sha256"] + ".intent.json"),
         {"created_epoch": 1.})
    calls = []
    monkeypatch.setattr(handoff, "command", lambda args, **kwargs: calls.append(args) or "")
    with pytest.raises(ValueError, match="ambiguous"):
        handoff.controller_accept(root / "controller.json")
    assert all(args[0] != "sbatch" for args in calls)


def test_qualification_failure_preserves_scope_and_does_not_dispatch(root, monkeypatch):
    campaign = load(root / "CAMPAIGN.json")
    selection = seal({"model": "m"})
    save(root / "FROZEN_MASKS.json", {"models": {"m": selection}})
    save(root / "m/qualification/QUALIFICATION.json", {
        "campaign_sha256": campaign["sha256"], "selection_sha256": selection["sha256"],
        "status": "FAIL", "failures": [{"type": "FidelityError"}]})
    monkeypatch.setattr(handoff, "MODELS", ("m",))
    monkeypatch.setattr(handoff, "wave_dispatch", lambda *args: pytest.fail("failed gate dispatched"))
    result = handoff.qualification_phase([{"scheduler_success": True}])
    assert result["status"] == "QUALIFICATION_LIMIT"
    assert result["evidence"]["accuracy_unmeasured"]
    assert result["evidence"]["no_model_scope_reduction"]


def test_live_review_subtracts_pending_reservations_and_rejects_extra_workload(monkeypatch):
    queue = "1|em1-qualification|PENDING|16|02:00:00\n2|cv3-run-qwen35|RUNNING|16|1-00:00:00"

    def command(args, **kwargs):
        if args[0] == "lfs":
            return ("20149688 P /leonardo_work/IscrC_MIOSR" if args[1] == "project" else
                    "/leonardo_work 975315228 1073741824 0 - 1491503 0 0 -")
        return {"id": "lmolfett", "hostname": "login01.leonardo.local",
                "saldo": "IscrC_MIOSR 20260504 20270204 68000 25950 25950 38.2 7391 5983",
                "cinQuota": "/leonardo_work/IscrC_MIOSR 930G 1T - 1000", "squeue": queue}[args[0]]

    monkeypatch.setattr(handoff, "command", command)
    review = handoff.live_review(100, 1024)
    assert review["remaining_core_hours"] == 42050 - 32 - 384
    assert review["storage_bytes_available"] == (1073741824 - 975315228) * 1024
    queue += "\n3|new-unreviewed-test|RUNNING|16|02:00:00"
    with pytest.raises(ValueError, match="exposure"):
        handoff.live_review()


def test_native_largest_prompt_selection_and_aliases_use_frozen_inputs(root, monkeypatch):
    campaign = load(root / "CAMPAIGN.json")
    selection = seal({"model": "m"})
    save(root / "FROZEN_MASKS.json", {"models": {"m": selection}})
    save(root / "m/qualification/QUALIFICATION.json", {
        "campaign_sha256": campaign["sha256"], "selection_sha256": selection["sha256"], "status": "PASS"})
    rows = [{"model": "m", "arm": arm, "alias_of_arm": "state_balanced" if arm == "labelled_frequency" else arm,
             "family": f"f{i}", "question_id": f"q{i}", "uid": arm + str(i)}
            for arm in handoff.ARMS for i in range(2)]
    save(root / "PILOT_ASSIGNMENTS.json", {"records": rows})
    save(root / "POPULATION.json", {"pilot": [{"question_id": "q0", "problem": "short"},
                                               {"question_id": "q1", "problem": "long"}]})
    monkeypatch.setattr(handoff, "MODELS", ("m",))
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs:
                                [1] if messages[0]["content"] == "short" else [1, 2])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: tokenizer)))
    monkeypatch.setattr(handoff, "live_review", lambda *args: {})
    cards = []
    monkeypatch.setattr(handoff, "make_card", lambda *args, **kwargs: cards.append(kwargs) or root / str(len(cards)))
    monkeypatch.setattr(handoff, "wave_dispatch", lambda *args: seal({"dispatch": True}))
    handoff.qualification_phase([{"scheduler_success": True}])
    inputs = load(root / "PRICING_INPUTS-r1.json")
    assert len(cards) == 3
    assert all(c["request"]["assignment_uids"][0].endswith("1") for c in cards)
    assert inputs["retained_uids"]["m"]["labelled_frequency"] == "labelled_frequency1"


def test_predecessor_accounting_must_bind_exact_accepted_job(root, monkeypatch):
    card = save(root / "card.json", {"stage": "qualification", "request": {}})
    uid = digest(["submission", card["sha256"]])
    accepted = save(root / "submissions" / (uid + ".accepted.json"), {"job_id": "50"})
    wave = {"jobs": [{"job_id": "51", "card_path": str(root / "card.json"),
                       "card_sha256": card["sha256"], "accepted_sha256": accepted["sha256"]}]}
    monkeypatch.setattr(handoff, "command", lambda *args, **kwargs: pytest.fail("foreign accounting query"))
    with pytest.raises(ValueError, match="receipt"):
        handoff.terminal_jobs(wave)


def test_stage_submit_resumes_exact_intent_without_new_sbatch(root, monkeypatch):
    from moe_exp.expert_memory_study import cards
    card = save(root / "card.json", {"stage": "pricing", "request": {}})
    uid = digest(["submission", card["sha256"]])
    save(root / "submissions" / (uid + ".intent.json"), {"uid": uid})
    save(root / "submissions" / (uid + ".accepted.json"), {"job_id": "50"})
    monkeypatch.setattr(cards, "reconcile_submission", lambda *args: {"status": "ACCEPTED"})
    monkeypatch.setattr(cards, "submit", lambda *args: pytest.fail("duplicate submission"))
    assert handoff.submit_card(root / "card.json")["job_id"] == "50"


def test_direct_controller_execution_requires_allocation(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm"):
        handoff.run("unread-card.json")
