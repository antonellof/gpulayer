# GPUlayer image: the CLI, the control plane (`gpulayer up`) and a worker (`gpulayer join`).
#
#   docker build -t gpulayer .                                   # CPU base (python:3.12-slim)
#   docker build -t gpulayer:cuda \
#     --build-arg BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
#     --build-arg EXTRAS=pytorch-cuda .                          # NVIDIA worker with torch
#   docker build -t gpulayer:torch-cpu --build-arg EXTRAS=torch-cpu .  # CPU worker with torch
#   docker build -t gpulayer:rocm --build-arg BASE=rocm/pytorch:latest \
#     --build-arg EXTRAS=base-torch .                            # AMD worker, the base's torch
#
# State lives in /data (GPULAYER_HOME). The image runs as an unprivileged user.
ARG BASE=python:3.12-slim
FROM ${BASE}

ARG EXTRAS=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GPULAYER_HOME=/data \
    PATH=/opt/gpulayer/bin:$PATH

# CUDA base images ship without Python; python:* images already have it.
RUN if ! command -v python3 >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
      && rm -rf /var/lib/apt/lists/*; \
    fi \
 && if [ "$EXTRAS" = "base-torch" ]; then python3 -m venv --system-site-packages /opt/gpulayer; \
    else python3 -m venv /opt/gpulayer; fi

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN if [ -z "$EXTRAS" ] || [ "$EXTRAS" = "base-torch" ]; then pip install . ; \
    elif [ "$EXTRAS" = "pytorch-cuda" ]; then pip install . torch numpy ; \
    elif [ "$EXTRAS" = "torch-cpu" ]; then pip install . numpy \
      && pip install torch --index-url https://download.pytorch.org/whl/cpu ; \
    else pip install ".[$EXTRAS]" ; fi \
 && rm -rf /src

RUN useradd --create-home --uid 10001 gpulayer && mkdir -p /data && chown gpulayer /data
USER gpulayer
WORKDIR /home/gpulayer
VOLUME ["/data"]
EXPOSE 8470

ENTRYPOINT ["gpulayer"]
CMD ["--help"]
