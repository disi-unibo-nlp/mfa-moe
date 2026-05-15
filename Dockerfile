FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
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

# Install PyTorch with CUDA 12.1 support (closest stable wheel to CUDA 12.2)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Copy project metadata and install dependencies
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir ".[dev]"

# Copy the rest of the project
COPY . .

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
