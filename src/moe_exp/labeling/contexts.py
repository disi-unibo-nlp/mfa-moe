"""Reasoning/answer split of a decoded continuation and the judge contexts of its sentences.

`context_id`, `scientific_config`, `parser_parts` and `sentence_contexts` are vendored verbatim from
`research_campaign_v3_r6.annotation`.
"""
from moe_exp.receipts import digest, require


def parts(tokenizer, profile, text):
    if profile['key'] == 'lfm25':
        reasoning, marker, final = text.partition('</think>')
        return reasoning.removeprefix('<think>'), final if marker else None
    if profile['key'] == 'gpt':
        # vLLM 0.29's GptOssReasoningParser no longer extracts text; split Harmony channels directly
        # (analysis bodies joined as vLLM did, final channel up to <|return|>/<|end|>).
        import re
        from moe_exp.generation.continuation import final_answer
        bodies = re.findall(r'<\|channel\|>analysis<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|call\|>|\Z)',
                            text, flags=re.DOTALL)
        return ('\n'.join(bodies).strip() or None), final_answer(text, 'openai_gptoss')
    return parser_parts(tokenizer, profile, text)


def context_id(inputs, annotator):
    return digest({"inputs": inputs, "program_sha256": annotator["config"]["program_sha256"],
                   "reader_revision": annotator["revision"], "config": scientific_config(annotator["config"])})


def scientific_config(config):
    return {k:v for k,v in config.items() if k not in ("base_url","source_plan_sha256","batch_size")}


def parser_parts(tokenizer, contract, text):
    from vllm.reasoning import ReasoningParserManager
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    name = contract["reasoning_parser"]
    if not name:
        return None,text
    parser = ReasoningParserManager.get_reasoning_parser(name)(tokenizer)
    request = ChatCompletionRequest(model=contract["checkpoint"],messages=[{"role":"user","content":""}],
        reasoning_effort=contract["reasoning_effort"])
    return parser.extract_reasoning(text,request)


def sentence_contexts(trace, replay, row, result):
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    layout=saved_layout(trace)
    units = layout["units"]
    offsets = replay["completion_offsets"]
    prefix = row["source"]["prefix_tokens"]
    selected = []
    generated = []
    for i,u in enumerate(units):
        owned = layout["unit_tokens"][i] or []
        if not owned or max(owned)<prefix:
            continue
        # A prefix must be at the end of its saved sentence. An overlapping
        # boundary is a compatibility failure, never outcome-based replacement.
        require(min(owned)>=prefix,"selected prefix bisects a reasoning sentence")
        end = max(owned)+1
        complete = i<len(units)-1 or result["finish_reason"]=="stop" or u["text"].rstrip().endswith((".","!","?"))
        generated.append((i,u,min(owned),end,complete))
    first = next((i for i,u,a,b,complete in generated if complete),None)
    for i,u,start,end,complete in generated:
        if start-prefix>=row["window"] and i!=first:
            continue
        from moe_exp.labeling.spans import reasoning_ranges
        segment=next(j for j,(a,b) in enumerate(reasoning_ranges(trace)) if a<=u["start"]<b)
        selected.append({"sentence_index":i,"segment":segment,"token_start":start-prefix,"token_end":end-prefix,
            "n_tokens":end-start,"complete":complete,"first_complete":i==first,
            "inputs":{"problem_statement":trace.prompt,
                "previous_sentence":units[i-1]["text"] if i else "",
                "sentence":u["text"],"next_sentence":units[i+1]["text"] if i+1<len(units) else ""}})
    return selected
