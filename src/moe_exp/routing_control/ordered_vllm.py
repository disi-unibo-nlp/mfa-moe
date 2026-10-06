"""GPU-worker-only entry point: moe_exp.routing_control.ordered_vllm.OrderedWorkerExtension.

Requires a separately frozen combined source tree and separate GPU qualification.
It does not modify the installed agent runtime, vLLM files or legacy frozen tree.
"""
import os
import socket

if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
    raise RuntimeError('ordered routing worker may only be imported on allocated compute nodes')

from .worker_adapter import install

E = install()


class OrderedWorkerExtension(E.SteerWorkerExtension):
    def steer_ordered_status(self):
        return {'version':'ordered-worker-v1','status':self.steer_status()}
