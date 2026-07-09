# Base image and torch wheel index are parametrized so the same file builds
# for both targets:
#   cluster (default):      CUDA 12.2 base + cu121 torch (host driver limit)
#   Blackwell (RTX 5090+):  needs CUDA 12.8+ and torch >= 2.7 cu128 wheels:
#     docker build \
#       --build-arg CUDA_BASE=12.8.1-devel-ubuntu22.04 \
#       --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 \
#       -t moe-mfa-experiments:blackwell .
ARG CUDA_BASE=12.2.0-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_BASE}
LABEL maintainer="Leonardo Tassinari"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /workspace
ENV APP_PATH=/workspace

# Install general-purpose dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
                        git \
                        bash \
                        nano \
                        python3.11 \
                        python3.11-distutils \
                        python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Remap python, python3, and pip to Python 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python3.11 -m pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown

# Install PyTorch from the wheel index matching the CUDA base
# (cu121 is the closest stable wheel to the default CUDA 12.2 base).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}

# Copy project metadata and install dependencies
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir ".[dev]"

# Copy the rest of the project
COPY . .

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
