# One image for both products:
#   Serverless worker: default CMD runs worker/handler.py
#   Pod batch job:     start command overridden to batch/embed_corpus.py (see batch/launch_pod.py)
#
# The model (~130 MB of weights) is baked into the image, so a fresh worker needs no
# network fetch and no network volume. README "Model in the image vs. network volume"
# explains the tradeoff.
#
# GPU build (what Runpod runs):
#   docker buildx build --platform linux/amd64 -t runpod-demo:gpu .
# CPU build (for running the image on a laptop):
#   docker buildx build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t runpod-demo:cpu .

FROM python:3.12-slim

# cu126 wheels run on hosts with NVIDIA driver >= 560 (and >= 525 with CUDA minor-version
# compatibility). The default PyPI torch 2.14 wheel targets CUDA 13, which needs a newer
# driver and so narrows the set of Runpod hosts a worker can land on.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.14.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"

COPY requirements-worker.txt .
RUN pip install -r requirements-worker.txt

COPY worker/download_model.py worker/download_model.py
RUN python worker/download_model.py /models/bge-small-en-v1.5

# Read the baked weights only; never reach out to the Hub at runtime.
ENV MODEL_DIR=/models/bge-small-en-v1.5 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY worker/ worker/
COPY batch/ batch/

CMD ["python", "-u", "/app/worker/handler.py"]
