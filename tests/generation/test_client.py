"""Request fields of a model-card sampler, and the job deadline that stops new requests."""
import os
import subprocess
from datetime import datetime
from types import SimpleNamespace

import pytest

from generation import client


def test_sampling_fields_send_what_the_card_sets():
    card = {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": None, "presence_penalty": 1.5}
    assert client._sampling_fields(card, 7) == {
        "temperature": 1.0, "top_p": 0.95, "top_k": -1, "presence_penalty": 1.5, "seed": 7, "stream": False}


def test_an_explicit_none_top_k_is_sent_as_null():
    fields = client._sampling_fields({"temperature": 0.6, "top_p": 0.95, "top_k": None}, 1)
    assert fields["top_k"] is None


def test_deadline_is_outside_slurm_none_and_inside_the_job_end_minus_the_margin(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert client.slurm_deadline(3.0) is None
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1000000")
    monkeypatch.setattr(client.subprocess, "run", lambda *a, **k: pytest.fail("squeue must not be needed"))
    assert client.slurm_deadline(2.0) == 1000000 - 2 * 3600


def test_deadline_falls_back_to_squeue_without_the_end_time_variable(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    monkeypatch.setattr(client.subprocess, "run",
                        lambda command, **kwargs: SimpleNamespace(stdout="2026-10-24T12:00:00\n"))
    assert client.slurm_deadline(3.0) == datetime(2026, 10, 24, 9, 0).timestamp()


@pytest.mark.skipif("SLURM_JOB_END_TIME" not in os.environ, reason="only inside a Slurm job")
def test_job_end_time_variable_agrees_with_squeue():
    reply = subprocess.run(["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%e"],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert float(os.environ["SLURM_JOB_END_TIME"]) == datetime.fromisoformat(reply).timestamp()
