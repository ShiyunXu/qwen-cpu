FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      build-essential \
      cmake \
      wget \
      curl \
      libcurl4-openssl-dev \
      python3.8 \
      python3.8-distutils \
      python3-pip && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp
WORKDIR /opt/llama.cpp
RUN cmake -B build -DLLAMA_CURL=ON && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /models/qwen2.5-coder-0.5b && \
    wget -O /models/qwen2.5-coder-0.5b/qwen2.5-coder-0.5b-instruct-fp16.gguf \
      https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-fp16.gguf

EXPOSE 8080
WORKDIR /opt/llama.cpp/build/bin
ENTRYPOINT ["./llama-server", \
             "--model", "/models/qwen2.5-coder-0.5b/qwen2.5-coder-0.5b-instruct-fp16.gguf", \
             "-c", "2048", \
             "--host", "0.0.0.0"]
