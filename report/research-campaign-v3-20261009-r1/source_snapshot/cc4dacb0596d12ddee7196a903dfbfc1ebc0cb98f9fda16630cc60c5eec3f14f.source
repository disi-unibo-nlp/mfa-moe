"""Frozen reader deployment and serialization, loaded only on GPU compute."""
from __future__ import annotations

from collections.abc import Mapping
import time

from .contracts import digest, require, seal
from .semantics import reader_messages, parse_rating


def normalize_prompt_ids(value):
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)) and len(value) == 1 and isinstance(value[0], (tuple, list)):
        value = value[0]
    require(isinstance(value, (list, tuple)) and value and all(type(t) is int and t >= 0 for t in value),
            "chat template returned non-token or multiple-sequence values")
    return list(value)


class Reader:
    def __init__(self, checkpoint):
        from .backend import require_compute
        require_compute()
        from vllm import LLM
        started = time.monotonic()
        self.llm = LLM(model=checkpoint, tokenizer=checkpoint, tensor_parallel_size=2,
            max_model_len=49152, max_num_seqs=8, enforce_eager=True, gpu_memory_utilization=.80,
            language_model_only=True, enable_prefix_caching=False, dtype="bfloat16", kv_cache_dtype="bfloat16",
            attention_config={"backend":"FLASH_ATTN"}, generation_config="vllm", async_scheduling=False)
        self.load_seconds = time.monotonic() - started
        self.tokenizer = self.llm.get_tokenizer()
        self.checkpoint = checkpoint

    def rate(self, record, role, reader):
        from vllm import SamplingParams
        prompt = normalize_prompt_ids(self.tokenizer.apply_chat_template(reader_messages(record, role=role),
            tokenize=True, add_generation_prompt=True, enable_thinking=True, reasoning_effort="low", return_dict=False))
        require(len(prompt) + 1024 <= 49152, "reader input exceeds pinned context; truncation forbidden")
        started = time.monotonic()
        seed = int(digest(["verification-reader-v3", digest(record), reader])[:8], 16) % 2000000000
        output = self.llm.generate([{"prompt_token_ids": prompt}],
            SamplingParams(temperature=.2, top_p=.95, seed=seed, max_tokens=1024), use_tqdm=False)[0].outputs[0]
        return seal({"schema": "verification-reader-receipt-v3", "uid": record["uid"], "reader": reader,
            "role": role, "input_sha256": digest(record), "text": output.text,
            "parsed": parse_rating(output.text, role), "finish_reason": output.finish_reason,
            "input_tokens": len(prompt), "output_tokens": len(output.token_ids),
            "seconds": time.monotonic() - started, "checkpoint": self.checkpoint})

    def close(self):
        self.llm.llm_engine.engine_core.shutdown()

    def grade(self, item):
        """Offline gold-visible J1; preserve the existing prompt/sampler/3 votes."""
        from collections import Counter
        from vllm import SamplingParams
        from moe_exp.correlation_pipeline.answer_equivalence import SYSTEM,JUDGE_SAMPLER,parse_verdict
        visible={k:item[k] for k in ("problem","gold","candidate")}
        content=f"Problem:\n{visible['problem']}\n\nReference answer:\n{visible['gold']}\n\nCandidate:\n{visible['candidate']}"
        prompt=normalize_prompt_ids(self.tokenizer.apply_chat_template(
            [{"role":"system","content":SYSTEM},{"role":"user","content":content}],tokenize=True,
            add_generation_prompt=True,enable_thinking=True,return_dict=False))
        require(len(prompt)+8192<=49152,"J1 context exceeds contract")
        votes,draws=[],[]
        seed=int(digest(["verification-J1",item["uid"]])[:8],16)%2000000000
        for vote in range(3):
            started=time.monotonic()
            output=self.llm.generate([{"prompt_token_ids":prompt}],SamplingParams(**JUDGE_SAMPLER,seed=seed+vote,
                max_tokens=8192),use_tqdm=False)[0].outputs[0]
            decision=parse_verdict(output.text.rsplit("</think>",1)[-1]) if output.finish_reason=="stop" else "TRUNCATED"
            votes.append(decision)
            draws.append({"vote":vote,"seed":seed+vote,"finish_reason":output.finish_reason,"input_tokens":len(prompt),
                          "output_tokens":len(output.token_ids),"seconds":time.monotonic()-started,"text":output.text})
        top,n=Counter(votes).most_common(1)[0]
        return seal({"schema":"verification-j1-verdict-v3","uid":item["uid"],"input_sha256":digest(item),
            "grade_sha256":item.get("grade_sha256"),"votes":votes,"draws":draws,
            "verdict":top if n>=2 and top in ("EQUIVALENT","NOT_EQUIVALENT") else "UNCERTAIN",
            "checkpoint":self.checkpoint,"prompt_version":1,"parser_version":2})
