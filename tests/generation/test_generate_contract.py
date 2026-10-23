"""The generation contract of the saved traces must not drift, and the job deadline is read from Slurm."""
import pytest

from generation import generate

# generation_sha256 stored in the first MATH-500 attempt (seed 20337925 = 20260925 + 77 * 1000) of each model
# in results/correlation_pipeline/card-v3/<model>/generation. A new hash here would make every saved shard stale.
STORED_HASHES = {
    "gpt": "94e9382d5d2ff79538415c429ca2534e58fbdcd9b9659c5abeeeb505a7737c99",
    "qwen36": "b21209bfa77a0c577287f87792893f8c883067fc8737d7c454f4f58909189f70",
    "gemma": "2e3c3242d4897390402d527abd052aaddfaace239b947e017750bd9632ca77b4",
    "glm": "632cff0d547251a4b216189d3390cc2cf9dc79fc6d698c8c8c1f1bb0c26c1ba4",
    "nemotron": "ad6d549b09ea8217f240431180ecd1d99f5f9595f05eea2f771f1a283fc57cbe",
    "qwen35": "bf6e4049c3fe12962a4f07edbe9487ba840406b36b040b5cacfa00b370f10a70",
    "qwen330b": "0fd3c2ca2905b3a1de816ba5bcad52b64ae2e45ffad84d6085c30b6c1d8b5688",
}


@pytest.mark.parametrize("profile", sorted(STORED_HASHES))
def test_contract_hash_equals_the_hash_stored_in_the_traces(profile, monkeypatch):
    monkeypatch.setattr(generate, "vllm_version", lambda: "0.29.0")
    args = generate.build_parser().parse_args(["--profile", profile])
    example = {"problem_id": "math500_test/algebra/1004.json", "prompt": "question", "gold_answer": "1"}
    contract = generate._job_contract("math500", example, 77, 0, args)
    assert contract["generation_config"]["seed"] == 20337925
    assert contract["generation_sha256"] == STORED_HASHES[profile]


def test_vllm_version_drops_the_build_tag_and_is_read_once(monkeypatch):
    generate.vllm_version.cache_clear()
    calls = []
    monkeypatch.setattr(generate.importlib.metadata, "version", lambda name: calls.append(name) or "0.29.0+cu129")
    assert generate.vllm_version() == generate.vllm_version() == "0.29.0"
    assert calls == ["vllm"]
    generate.vllm_version.cache_clear()


def test_contract_copies_the_profile_objects(monkeypatch):
    monkeypatch.setattr(generate, "vllm_version", lambda: "0.29.0")
    args = generate.build_parser().parse_args(["--profile", "nemotron"])
    example = {"problem_id": "p", "prompt": "question", "gold_answer": "1"}
    config = generate._job_contract("math500", example, 0, 0, args)["generation_config"]
    card = generate.CARD_PROFILES["nemotron"]
    for key, field in (("sampler", "sampler"), ("chat_template_kwargs", "template_kwargs"),
                       ("speculation", "speculation"), ("server_extra_args", "extra_args")):
        assert config[key] == card[field] and config[key] is not card[field]
