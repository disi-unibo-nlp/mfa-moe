ARG VLLM_IMAGE=vllm/vllm-openai:v0.29.0
FROM ${VLLM_IMAGE}

# Keep vLLM's compatible torch/transformers stack; install the project entry
# points so every inference worker discovers the custom architecture plugin.
WORKDIR /opt/moe-experiments
COPY pyproject.toml ./
COPY src/moe_exp ./src/moe_exp
RUN python3 -m pip install --no-cache-dir --no-deps .
WORKDIR /workspace
ENTRYPOINT ["python3"]
