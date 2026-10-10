"""Original seven-class rendering, complete-context reuse and fixed native draws."""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3
import time
import subprocess
import sys
from types import SimpleNamespace

from .contracts import CLASSES, SEED, Units, allocated, check_files, digest, load, publish, require
from .inputs import BASE, read_source


class Renderer:
    def __init__(self,campaign):
        from .contracts import REPO
        check_files(campaign["renderer"]["files"])
        self.child=subprocess.Popen([sys.executable,"-B",str(REPO/"scripts/research_campaign_v1/renderer.py"),
            campaign["annotator"]["program"]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,
            env={**os.environ,"DSPY_CACHEDIR":os.environ["TMPDIR"]+"/dspy","LITELLM_LOCAL_MODEL_COST_MAP":"True"})

    def request(self,operation,**value):
        import select
        self.child.stdin.write(json.dumps({"operation":operation,**value},ensure_ascii=False)+"\n")
        self.child.stdin.flush()
        require(select.select([self.child.stdout],[],[],45)[0],"original renderer child response timed out")
        line=self.child.stdout.readline()
        require(line,"original renderer child exited")
        return json.loads(line)

    def close(self):
        self.child.stdin.close()
        self.child.wait(timeout=30)


def context_id(inputs, annotator):
    return digest({"inputs": inputs, "program_sha256": annotator["config"]["program_sha256"],
                   "reader_revision": annotator["revision"], "config": scientific_config(annotator["config"])})


def scientific_config(config):
    return {k:v for k,v in config.items() if k not in ("base_url","source_plan_sha256","batch_size")}


def label_index(campaign, model):
    """The index contains context digests and receipts, never copied prompts."""
    root = Path(campaign["root"])
    path = root/model/"LABEL_INDEX.sqlite"
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if (root/model/"LABEL_INDEX.json").exists():
        check_files({str(path): load(root/model/"LABEL_INDEX.json")["sha256_file"]})
        return path
    temporary = path.with_suffix(".pending.sqlite")
    if temporary.exists():
        temporary.unlink()  # An uncommitted derived index is safe to rebuild.
    db = sqlite3.connect(temporary)
    db.execute("create table labels (context text primary key, label text, source text, row_index integer)")
    db.execute("create table receipts (context text, label text, source text, row_index integer)")
    records = 0
    ambiguous = set()
    try:
        for source in sorted((BASE/"campaign-v3/labels"/model).glob("part-*/annotations.json")):
            checkpoints = sorted((source.parent/"checkpoints").glob("batch-*.json"))
            require(checkpoints, "annotation source has no original binding")
            for checkpoint in checkpoints:
                config = json.loads(checkpoint.read_text())["binding"]["config"]
                require(scientific_config(config) == scientific_config(campaign["annotator"]["config"]), "original annotation configuration differs")
            for i, r in enumerate(json.loads(source.read_text())):
                key = context_id(r["inputs"], campaign["annotator"])
                label = r.get("label")
                label = label if label in CLASSES else None
                existing = db.execute("select label from labels where context=?", (key,)).fetchone()
                db.execute("insert or ignore into labels values (?,?,?,?)", (key,label,str(source),i))
                db.execute("insert into receipts values (?,?,?,?)", (key,label,str(source),i))
                if existing is not None and existing[0] != label:
                    ambiguous.add(key)
                if key in ambiguous:
                    db.execute("update labels set label=NULL where context=?",(key,))
                records += 1
        db.commit()
    finally:
        db.close()
    os.chmod(temporary,0o600)
    try:
        os.link(temporary,path)
    except FileExistsError:
        from .contracts import file_sha
        require(file_sha(path)==file_sha(temporary),"interrupted annotation index commit differs")
    temporary.unlink()
    from .contracts import file_sha
    publish(root/model/"LABEL_INDEX.json", {"records":records,"sha256_file":file_sha(path),
        "complete_neighboring_context":True,"invalid_original_labels_are_terminal_unknown":True,
        "conflicting_original_contexts":len(ambiguous),"conflict_rule":"retain every original receipt; terminal unknown without redraw"})
    return path


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


def rebuilt(campaign, model, row, result, tokenizer):
    """Transient reconstruction; the stored execution remains a prefix reference."""
    from moe_exp.models.token_replay import make_token_replay, validate_token_replay
    source = read_source(row["source"]["location"])
    original = source["metadata"]["token_replay"]
    validate_token_replay(original,tokenizer,source["cot_text"])
    if result.get("reused"):
        return SimpleNamespace(**source),original
    prefix = original["completion_token_ids"][:row["source"]["prefix_tokens"]]
    require(digest(prefix)==row["source"]["prefix_sha256"],"saved prefix changed")
    tokens = prefix + result["token_ids"]
    require(tokens,"empty generation has no sentence context")
    text,replay = make_token_replay(tokenizer,original["prompt_token_ids"],tokens)
    reasoning,content = parser_parts(tokenizer,campaign["models"][model],text)
    metadata = {**source["metadata"],"token_replay":replay,"reasoning_content":reasoning,
                "assistant_content":content,"finish_reason":result["finish_reason"]}
    return SimpleNamespace(**{**source,"cot_text":text,"metadata":metadata}), replay


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
        from moe_exp.correlation_pipeline.spans import reasoning_ranges
        segment=next(j for j,(a,b) in enumerate(reasoning_ranges(trace)) if a<=u["start"]<b)
        selected.append({"sentence_index":i,"segment":segment,"token_start":start-prefix,"token_end":end-prefix,
            "n_tokens":end-start,"complete":complete,"first_complete":i==first,
            "inputs":{"problem_statement":trace.prompt,
                "previous_sentence":units[i-1]["text"] if i else "",
                "sentence":u["text"],"next_sentence":units[i+1]["text"] if i+1<len(units) else ""}})
    return selected


def prepare_contexts(campaign, model, phase="production"):
    from transformers import AutoTokenizer
    root=Path(campaign["root"])
    index=label_index(campaign,model)
    db=sqlite3.connect(f"file:{index}?mode=ro",uri=True)
    tokenizer=AutoTokenizer.from_pretrained(campaign["models"][model]["local_path"],local_files_only=True)
    path=root/model/("NATIVE_TASKS.json" if phase=="native" else "TASKS.json")
    if not path.exists():path=root/model/"NATIVE_TASKS.json"
    table=load(path)
    units=Units(root/model/"executions")
    annotations=Units(root/model/"annotations")
    missing, contexts=[],[]
    try:
        for row in table["records"]:
            result=units.state(row["uid"])
            if not row["source"]["prefix_tokens"] or (result or {}).get("status")!="complete":
                continue
            if not result["generated_tokens"]:
                contexts.append({"uid":row["uid"],"sentences":[],"status":"no_generated_sentence"})
                continue
            trace,replay=rebuilt(campaign,model,row,result,tokenizer)
            sentences=sentence_contexts(trace,replay,row,result)
            for s in sentences:
                key=context_id(s["inputs"],campaign["annotator"])
                task={"uid":key,"inputs":s.pop("inputs"),"model":model}
                s["annotation_uid"]=key
                reused=db.execute("select label,source,row_index from labels where context=?",(key,)).fetchone()
                if reused is not None and annotations.state(key) is None:
                    annotations.begin(task,"original-context-reuse")
                    annotations.commit(task,{"status":"reused","label":reused[0],"unknown":reused[0] is None,
                        "original_source":reused[1],"original_row_index":reused[2],
                        "original_receipts":[{"label":v[0],"source":v[1],"row_index":v[2]} for v in
                            db.execute("select label,source,row_index from receipts where context=?",(key,))]})
                elif reused is None:
                    missing.append(task)
            contexts.append({"uid":row["uid"],"sentences":sentences,"status":"complete"})
    finally:
        db.close()
    unique={t["uid"]:t for t in missing}
    tasks=sorted(unique.values(),key=lambda t:t["uid"])
    for i,t in enumerate(tasks):
        t["task_index"]=i
    suffix="" if phase=="production" else "-"+phase
    publish(root/model/("CONTEXTS"+suffix+".json"),{"records":contexts,"largest_window_only":True,
        "first_complete_sentence_retained":True,"prefixes_copied":False})
    return publish(root/model/("ANNOTATION_TASKS"+suffix+".json"),{"records":tasks,"original_program_and_parser":True})


def annotate(campaign, model, rows, deadline):
    allocated(gpu=True)
    from vllm import LLM, SamplingParams
    root=Path(campaign["root"])
    units=Units(root/model/"annotations")
    pending=[r for r in rows if units.state(r["uid"]) is None]
    if not pending:
        return {"status":"reused","units":len(rows)}
    a=campaign["annotator"]
    check_files(campaign["code_files"])
    renderer=Renderer(campaign)
    started=time.monotonic()
    llm=LLM(model=a["path"],tokenizer=a["path"],tensor_parallel_size=2,dtype="bfloat16",
        max_model_len=49152,max_num_seqs=64,
        enforce_eager=True,enable_prefix_caching=True,async_scheduling=False,
        enable_chunked_prefill=True,max_num_batched_tokens=8192,gpu_memory_utilization=.90,
        kv_cache_dtype="bfloat16",attention_config={"backend":"FLASH_ATTN"},
        language_model_only=True,generation_config="vllm")
    load_seconds=time.monotonic()-started
    tokenizer=llm.get_tokenizer()
    try:
        from vllm.reasoning import ReasoningParserManager
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        parser=ReasoningParserManager.get_reasoning_parser("qwen3")(tokenizer)
        for offset in range(0,len(pending),a["config"]["batch_size"]):
            require(time.time()<deadline-60,"deadline before original annotator assignment")
            assigned=[r for r in pending[offset:offset+a["config"]["batch_size"]] if units.begin(r,os.environ["SLURM_JOB_ID"])]
            if not assigned:continue
            tick=time.monotonic()
            try:
                batch=[];parameters=[]
                for row in assigned:
                    conversation=renderer.request("render",inputs=row["inputs"])["conversation"]
                    prompt=tokenizer.apply_chat_template(conversation,tokenize=True,add_generation_prompt=True,
                        enable_thinking=a["config"]["enable_thinking"],preserve_thinking=a["config"]["preserve_thinking"],
                        reasoning_effort=a["config"]["reasoning_effort"])
                    if len(prompt)>=49152:
                        units.commit(row,{"status":"upstream_unavailable","label":None,"unknown":True,
                            "reason":"complete original annotation context exceeds original reader context","input_tokens":len(prompt)})
                        continue
                    params=SamplingParams(**a["decoding"],max_tokens=min(a["config"]["max_tokens"],49152-len(prompt)),
                        seed=int(digest([SEED,"reader",row["uid"]])[:8],16),skip_special_tokens=False)
                    batch.append((row,conversation,prompt));parameters.append(params)
                if not batch:continue
                responses=llm.generate([{"prompt_token_ids":p} for r,c,p in batch],parameters,use_tqdm=False)
                require(len(responses)==len(batch),"original annotator batch attribution differs")
                elapsed=time.monotonic()-tick
                batch_id=digest([r["uid"] for r,c,p in batch])
                for response,(row,conversation,prompt) in zip(responses,batch):
                    require(response.prompt_token_ids==prompt and len(response.outputs)==1,"original annotator input mapping differs")
                    output=response.outputs[0]
                    raw=tokenizer.decode(output.token_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    request=ChatCompletionRequest(model=a["config"]["judge_model"],messages=conversation,
                        reasoning_effort=a["config"]["reasoning_effort"])
                    reasoning,content=parser.extract_reasoning(raw,request)
                    label=None;status="truncated" if output.finish_reason!="stop" else "invalid";error=None
                    if output.finish_reason=="stop":
                        parsed=renderer.request("parse",content=content or "")
                        label=parsed.get("label");error=parsed.get("parse_error")
                        if label in CLASSES:status="complete"
                    units.commit(row,{"status":status,"label":label,"unknown":label is None,
                        "raw_completion":raw,"finish_reason":output.finish_reason,
                        "generated_tokens":len(output.token_ids),"input_tokens":len(prompt),
                        "seconds":elapsed/len(batch),"batch_seconds":elapsed,"batch_size":len(batch),"batch_sha256":batch_id,
                        "load_seconds":load_seconds,"parser_error":error,"program_sha256":a["config"]["program_sha256"],
                        "reader_revision":a["revision"],"configuration":a["config"],"native_context":49152,
                        "engine_changes":a["engine_changes"]})
            except Exception as exc:
                for row in assigned:
                    if units.state(row["uid"])["status"]=="started":
                        units.commit(row,{"status":"execution_failed","label":None,"unknown":True,
                            "error_type":type(exc).__name__,"error":str(exc)[:2000]})
                raise
    finally:
        llm.llm_engine.engine_core.shutdown()
        renderer.close()
    return {"status":"complete","units":len(rows)}
