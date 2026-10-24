"""Routing interventions on the running model: the hooks, the generation engine and the analyses."""
import os

# The hooks patch the router modules of the model inside this process, so vLLM must not move its
# engine core to a child process. Set before any module imports vllm.
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
