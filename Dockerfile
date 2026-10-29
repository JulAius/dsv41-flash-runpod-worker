# llama-server (DeepSeek-V4.1 support) + RunPod queue-based worker.
#
# The model is NOT baked in (~350 GB): it is read from the network volume
# mounted at /runpod-volume (see README.md).
#
#   docker build --platform linux/amd64 -t <user>/dsv41-flash-llamacpp:0.1 .
#
# B200 (sm_100) needs CUDA >= 12.8, so do not lower CUDA_VERSION.
ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# llama.cpp PR #30817 at the commit orcarouter tested its GGUFs with (also what huihui-ai asks for).
# Mainline has DeepSeek-V4.1 since PR #31086: --build-arg LLAMA_CPP_REF=b11981 (or any later tag).
ARG LLAMA_CPP_REF=b72050d03ae14bb3cb5138ecefab3f3f65157d31
# 90 = H100/H200, 100 = B200. Fewer archs = faster build (RunPod GitHub builds cap docker build at 30 min).
ARG CUDA_ARCHS="90;100"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git cmake build-essential libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git init llama.cpp \
    && git -C llama.cpp remote add origin https://github.com/ggml-org/llama.cpp \
    && git -C llama.cpp fetch --depth 1 origin "${LLAMA_CPP_REF}" \
    && git -C llama.cpp checkout FETCH_HEAD

# libcuda.so only exists on the GPU host, hence --allow-shlib-undefined (same as llama.cpp's own cuda.Dockerfile).
RUN cmake -S llama.cpp -B out \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build out --config Release -j"$(nproc)" --target llama-server

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/out/bin/llama-server /usr/local/bin/llama-server

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /requirements.txt

COPY src /src

ENV PYTHONUNBUFFERED=1
CMD ["python3", "/src/start.py"]
