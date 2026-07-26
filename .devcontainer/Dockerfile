FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Install C/C++ compilation tools, OpenBLAS development headers, and general utilities
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gdb \
        make \
        gcc \
        g++ \
        libopenblas-dev \
        valgrind \
        pkg-config \
        curl \
        wget \
        bzip2 \
        git \
        htop \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set default working directory
WORKDIR /workspace
