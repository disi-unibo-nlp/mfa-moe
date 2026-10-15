# Serve a model

Inside an allocated Slurm GPU job:

```bash
scripts/inference/serve.sh /path/to/model --tensor-parallel-size 2
```

The script loads the existing CUDA-compatible vLLM environment and starts
`vllm serve` on localhost. Additional arguments go directly to vLLM.
It runs for the allocation lifetime. Routing interventions use the study's
Python worker because they need access to expert selection during execution.
