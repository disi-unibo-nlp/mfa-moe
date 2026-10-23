"""The server settings recorded in model_profiles.py must match what scripts/serve/MODEL.sh starts."""
import json
import shlex
from pathlib import Path

import pytest

from generation.answer_equivalence import JUDGE_MODEL
from generation.model_profiles import API_KEY, CARD_PROFILES, MFA

SERVE = Path(__file__).resolve().parents[2] / "scripts" / "serve"
# The only flags a model script may add to the shared ones; anything else (say --kv-cache-dtype fp8) would change
# how the traces are generated without the contract recording it.
MODEL_FLAGS = {"--served-model-name", "--revision", "--reasoning-parser", "--max-model-len", "--max-num-seqs",
               "--attention-config", "--speculative-config", "--language-model-only",
               "--linear-backend", "--mamba-backend", "--mamba-cache-mode"}


def read_serve_script(name):
    """(model argument, {flag: value or True}) of the `vllm serve` line of scripts/serve/NAME.sh."""
    text = (SERVE / f"{name}.sh").read_text().replace("\\\n", " ").replace("$MFA", MFA).replace("$SERVE_ARGS", "")
    model, *rest = shlex.split(next(line for line in text.splitlines() if line.startswith("exec vllm serve")))[3:]
    rest.append("--")  # so that the last flag has a successor
    return model, {flag: True if value.startswith("--") else value
                   for flag, value in zip(rest, rest[1:]) if flag.startswith("--")}


@pytest.mark.parametrize("name", sorted(CARD_PROFILES))
def test_profile_matches_its_serve_script(name):
    card = CARD_PROFILES[name]
    model, flags = read_serve_script(name)
    assert set(flags) <= MODEL_FLAGS
    if card["weights"]:  # local copy, named after its revision
        assert model == card["weights"] and Path(card["weights"]).name == card["revision"]
        assert flags["--served-model-name"] == card["model"]
    else:
        assert model == card["model"] and flags["--revision"] == card["revision"]
    assert flags["--reasoning-parser"] == card["parser"]
    assert int(flags["--max-model-len"]) == card["max_model_len"]
    assert int(flags["--max-num-seqs"]) == card["max_num_seqs"]
    attention = json.loads(flags["--attention-config"])["backend"] if "--attention-config" in flags else None
    assert attention == card["attention"]
    speculation = json.loads(flags["--speculative-config"]) if "--speculative-config" in flags else None
    assert speculation == card["speculation"]
    assert ("--language-model-only" in flags) == card["language_model_only"]
    extra = [item for flag in ("--linear-backend", "--mamba-backend", "--mamba-cache-mode")
             if flag in flags for item in (flag, flags[flag])]
    assert extra == card["extra_args"]


def test_shared_flags_match_the_recorded_contract():
    shared = " ".join((SERVE / "args.sh").read_text().split())
    assert "--kv-cache-dtype bfloat16" in shared  # "kv_cache_dtype" of the generation contract
    assert "--tensor-parallel-size 2" in shared and f"--api-key {API_KEY}" in shared


def test_judge_serves_the_model_the_answer_checker_asks_for():
    model, flags = read_serve_script("judge")
    assert model == JUDGE_MODEL
