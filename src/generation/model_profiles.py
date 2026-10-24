"""What the client sends to each model server, plus the server settings recorded with every trace.

Request (sent with every call): sampler, max_tokens, template_kwargs, reasoning_effort. They come from the
model card; `card` names the line of the card (or generation_config.json) that gives them.

Server (nothing here starts a server; scripts/serve/serve_model.sh does that, with the flags that serve_flags.py builds
from these values). generate.py records revision, weights, max_model_len, attention, speculation,
language_model_only and extra_args in every trace's generation contract, so changing one forces fresh generation
instead of silently mixing traces. parser (used by the continuation scorer) and max_num_seqs (how many requests the
client keeps in flight) are not in the contract. tests/generation/test_serve_flags.py pins the flags built from them.
"""
from functools import lru_cache


MFA = "/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe"
API_KEY = "local-vllm-key"  # the key serve_flags.py starts every server with
CARD_OUTPUT_TOKENS = 131_072
CARD_CONTEXT = 133_120  # output cap plus prompt room; every chosen checkpoint supports it

# Each model's own recommended thinking-mode sampler, and a 131,072-token output cap wherever the
# native context allows. `card` cites the line of the local model card or generation_config.json
# that gives the sampler.
CARD_PROFILES: dict[str, dict] = {
    "qwen36": dict(
        model="Qwen/Qwen3.6-35B-A3B-FP8", revision="95a723d08a9490559dae23d0cff1d9466213d989",
        weights=None, parser="qwen3",
        sampler=dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,
                     repetition_penalty=1.0),
        max_tokens=CARD_OUTPUT_TOKENS, max_model_len=CARD_CONTEXT, template_kwargs={},
        reasoning_effort=None, speculation={"method": "mtp", "num_speculative_tokens": 3},
        attention="FLASH_ATTN", language_model_only=True, max_num_seqs=16, extra_args=[],
        card="Qwen3.6-35B-A3B-FP8 README.md:670 (thinking, general tasks)"),
    "qwen35": dict(
        model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", revision="3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b",
        weights=f"{MFA}/cache/native-replay/Qwen3.5-35B-A3B-GPTQ-Int4/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b",
        parser="qwen3",
        sampler=dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,
                     repetition_penalty=1.0),
        max_tokens=CARD_OUTPUT_TOKENS, max_model_len=CARD_CONTEXT, template_kwargs={},
        reasoning_effort=None, speculation={"method": "mtp", "num_speculative_tokens": 3},
        attention="FLASH_ATTN", language_model_only=True, max_num_seqs=16, extra_args=[],
        card="Qwen3.5-35B-A3B-GPTQ-Int4 README.md:1008 (thinking, general tasks)"),
    "glm": dict(
        model="zai-org/GLM-4.7-Flash", revision="7dd20894a642a0aa287e9827cb1a1f7f91386b67",
        weights=None, parser="glm45",
        sampler=dict(temperature=1.0, top_p=0.95, top_k=0, min_p=None, presence_penalty=None,
                     repetition_penalty=None),
        max_tokens=CARD_OUTPUT_TOKENS, max_model_len=CARD_CONTEXT, template_kwargs={},
        reasoning_effort=None, speculation=None, attention="TRITON_MLA",
        language_model_only=False, max_num_seqs=8, extra_args=[],
        card="GLM-4.7-Flash README.md:48-50 (T 1.0, top-p 0.95, max new tokens 131072)"),
    "nemotron": dict(
        model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        revision="bee7596271d1495f6992ae224aefde4410e816b8", weights=None, parser="nemotron_v3",
        sampler=dict(temperature=1.0, top_p=0.95, top_k=0, min_p=None, presence_penalty=None,
                     repetition_penalty=None),
        max_tokens=CARD_OUTPUT_TOKENS, max_model_len=CARD_CONTEXT, template_kwargs={},
        reasoning_effort=None,
        speculation={"method": "dspark",
                     "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
                     "revision": "8a0177116d138011e63103110f136ec0ca09ebbf",
                     "num_speculative_tokens": 3},
        attention="TRITON_ATTN", language_model_only=False, max_num_seqs=16,
        extra_args=["--linear-backend", "marlin", "--mamba-backend", "triton",
                    "--mamba-cache-mode", "align"],
        card="Nemotron-3.5-Lightning NVFP4 README.md:54,606 (T 1.0, top-p 0.95)"),
    "gemma": dict(
        model="nvidia/Gemma-4-26B-A4B-NVFP4", revision="a19cfe00be84568a6867111c9a68c9c44fdcffe6",
        weights=f"{MFA}/cache/native-replay/Gemma-4-26B-A4B-NVFP4/a19cfe00be84568a6867111c9a68c9c44fdcffe6",
        parser="gemma4",
        sampler=dict(temperature=1.0, top_p=0.95, top_k=64, min_p=None, presence_penalty=None,
                     repetition_penalty=None),
        max_tokens=CARD_OUTPUT_TOKENS, max_model_len=CARD_CONTEXT,
        template_kwargs={"enable_thinking": True}, reasoning_effort=None, speculation=None,
        attention=None, language_model_only=True, max_num_seqs=8,
        extra_args=["--linear-backend", "marlin"],
        card="Gemma-4-26B-A4B-NVFP4 generation_config.json:11-13 (T 1.0, top-k 64, top-p 0.95); "
             "BF16 KV cache because the FP8 KV cache is unsupported on A100"),
    "qwen330b": dict(
        model="Qwen/Qwen3-30B-A3B", revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        weights=None, parser="qwen3",
        sampler=dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=None,
                     repetition_penalty=None),
        max_tokens=38_912, max_model_len=40_960, template_kwargs={}, reasoning_effort=None,
        speculation=None, attention="FLASH_ATTN", language_model_only=False, max_num_seqs=16,
        extra_args=[],
        card="Qwen3-30B-A3B README.md:131,329 (thinking T 0.6, top-p 0.95, top-k 20, min-p 0); "
             "38,912 competition output length; native 40,960 context (no YaRN)"),
    "gpt": dict(
        model="openai/gpt-oss-20b", revision="6cee5e81ee83917806bbde320786a8fb61efebee",
        weights=None, parser="openai_gptoss",
        sampler=dict(temperature=1.0, top_p=1.0, top_k=0, min_p=None, presence_penalty=None,
                     repetition_penalty=None),
        max_tokens=None, max_model_len=131_072, template_kwargs={}, reasoning_effort="medium",
        speculation=None, attention="TRITON_ATTN", language_model_only=False, max_num_seqs=16,
        extra_args=[],
        card="gpt-oss-20b card gives no sampler: checkpoint defaults (T 1, top-p 1) and the "
             "default medium reasoning level (README.md:147-155); output bounded by the "
             "131,072-token context"),
}


@lru_cache(maxsize=None)
def load_tokenizer(profile: str):
    """The tokenizer that decodes the server's token IDs (a local copy has no revision to pin)."""
    from transformers import AutoTokenizer
    card = CARD_PROFILES[profile]
    if card["weights"]:
        return AutoTokenizer.from_pretrained(card["weights"])
    return AutoTokenizer.from_pretrained(card["model"], revision=card["revision"])
