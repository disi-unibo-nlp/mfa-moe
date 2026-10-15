"""Original reader process and protocol with bounded cold-start supervision."""
import json
import select
import subprocess
from moe_exp.research_campaign_v3_r6.annotation import Renderer as OriginalRenderer
from .common import require


class Renderer(OriginalRenderer):
    def __init__(self,campaign):
        super().__init__(campaign)
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
