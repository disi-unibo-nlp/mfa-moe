"""The reader process for the judge: the frozen DSPy program, its prompt renderer and its parser, run in a
child process.

`Renderer` starts the module `labeling.renderer_child` with the pinned copy of DSPy that the parent record
(CAMPAIGN.json of the seven-model run) names, and asks it over a pipe to render a judge conversation or to parse
a completion into a label (the first answer may take up to 180 s while the program loads, later ones 45 s).
`token_ids` renders a conversation to the token ids of the judge prompt.
"""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
from store import check_files, require


class Renderer:
    def __init__(self,parent):
        check_files(parent["renderer"]["files"])
        # The pinned DSPy copy is named by the parent record (the files just checked), not looked up in the code tree:
        # a worktree or source snapshot has no gitignored .research-runtime.
        self.child=subprocess.Popen([sys.executable,"-B","-m","labeling.renderer_child",
            parent["annotator"]["program"],parent["renderer"]["path"]],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,
            env={**os.environ,"PYTHONPATH":os.pathsep.join(filter(None,[str(Path(__file__).resolve().parents[1]),os.environ.get("PYTHONPATH")])),
                 "DSPY_CACHEDIR":os.environ["TMPDIR"]+"/dspy","LITELLM_LOCAL_MODEL_COST_MAP":"True"})
        self.first_response=True

    def request(self,operation,**value):
        self.child.stdin.write(json.dumps({'operation':operation,**value},ensure_ascii=False)+'\n')
        self.child.stdin.flush()
        require(select.select([self.child.stdout],[],[],180 if self.first_response else 45)[0],
                'renderer child response timed out')
        line=self.child.stdout.readline()
        require(line,'renderer child exited')
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
