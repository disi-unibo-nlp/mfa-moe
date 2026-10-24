"""models/roster.py: one registry of the study models, their Hugging Face ids and the named lists."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from generation.model_profiles import CARD_PROFILES
from models.roster import CAPTURED, HF_IDS, JUDGE_MODEL, READOUT_ORDER, resolve
from store import MODELS

SRC = Path(__file__).resolve().parents[2] / "src"


def run_roster(*flags):
    run = subprocess.run([sys.executable, "-m", "models.roster", *flags], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(SRC)})
    assert run.returncode == 0, run.stderr
    return run.stdout.splitlines()


def test_the_roster_keeps_the_order_of_store_models():
    assert tuple(HF_IDS) == MODELS == ("gpt", "nemotron", "qwen35", "qwen36", "gemma", "glm", "qwen330b", "lfm25")


def test_ids_come_from_the_card_profiles_and_lfm25():
    assert {key: HF_IDS[key] for key in CARD_PROFILES} == {key: card["model"] for key, card in CARD_PROFILES.items()}
    assert HF_IDS["lfm25"] == "LiquidAI/LFM2.5-24B-A2B"
    assert JUDGE_MODEL == "Qwen/Qwen3.8-27B" and JUDGE_MODEL not in HF_IDS.values()


@pytest.mark.parametrize("key", MODELS)
def test_resolve_takes_a_key_or_a_hugging_face_id(key):
    assert resolve(key) == resolve(HF_IDS[key]) == key


def test_resolve_refuses_what_is_not_a_study_model():
    for name in (JUDGE_MODEL, "openai/gpt-oss-120b", "judge", ""):
        with pytest.raises(ValueError, match="unknown model"):
            resolve(name)


def test_the_captured_models_are_all_but_lfm25_and_the_readout_order_is_a_permutation():
    assert CAPTURED == MODELS[:-1] and "lfm25" not in CAPTURED
    assert READOUT_ORDER == ("gpt", "qwen36", "gemma", "glm", "nemotron", "qwen330b", "qwen35")  # frozen with the readout
    assert sorted(READOUT_ORDER) == sorted(CAPTURED)


def test_the_command_lists_hugging_face_ids():
    assert run_roster() == run_roster("--all") == list(HF_IDS.values())
    assert run_roster("--captured") == [HF_IDS[key] for key in CAPTURED]
    assert run_roster("--judge") == [JUDGE_MODEL]
