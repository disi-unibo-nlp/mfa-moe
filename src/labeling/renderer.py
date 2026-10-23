"""The reader process for the judge: the original DSPy program, renderer and parser, run in a child process.

`Renderer` is `research_campaign_v3_r6.annotation.Renderer` (with the longer first-answer timeout) and `token_ids` is
`research_campaign_v3_r6.reader_contract.token_ids`, both vendored verbatim. The child runs as the module
`labeling.renderer_child`.
"""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
from receipts import check_files, require


class Renderer:
    def __init__(self,campaign):
        check_files(campaign["renderer"]["files"])
        self.child=subprocess.Popen([sys.executable,"-B","-m","labeling.renderer_child",
            campaign["annotator"]["program"]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,
            env={**os.environ,"PYTHONPATH":os.pathsep.join(filter(None,[str(Path(__file__).resolve().parents[1]),os.environ.get("PYTHONPATH")])),
                 "DSPY_CACHEDIR":os.environ["TMPDIR"]+"/dspy","LITELLM_LOCAL_MODEL_COST_MAP":"True"})
        self.first_response=True

    def request(self,operation,**value):
        self.child.stdin.write(json.dumps({'operation':operation,**value},ensure_ascii=False)+'\n')
        self.child.stdin.flush()
        require(select.select([self.child.stdout],[],[],180 if self.first_response else 45)[0],
                'original renderer child response timed out')
        line=self.child.stdout.readline()
        require(line,'original renderer child exited')
        result=json.loads(line)
        self.first_response=False
        return result

    def close(self):
        if self.child.stdin and not self.child.stdin.closed:self.child.stdin.close()
        try:self.child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.child.terminate()
            try:self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:self.child.kill();self.child.wait(timeout=10)


def token_ids(tokenizer, conversation, config):
    tokens = tokenizer.apply_chat_template(conversation, tokenize=True,
        add_generation_prompt=True, return_dict=False,
        enable_thinking=config['enable_thinking'], preserve_thinking=config['preserve_thinking'],
        reasoning_effort=config['reasoning_effort'])
    require(isinstance(tokens, (list, tuple)) and tokens and
            all(type(t) is int and t >= 0 for t in tokens), 'reader requires integer token IDs')
    return list(tokens)
