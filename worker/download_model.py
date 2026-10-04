"""Download the model into a plain directory at image build time (no HF cache layout).

Pinned to a commit so a rebuilt image serves the same weights. Skips the ONNX export and
the duplicate pytorch_model.bin, which keeps about 260 MB out of the image.
"""

import os
import sys

from huggingface_hub import snapshot_download

MODEL_ID = "BAAI/bge-small-en-v1.5"
REVISION = os.environ.get("MODEL_REVISION", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/models/bge-small-en-v1.5"
    path = snapshot_download(
        MODEL_ID,
        revision=REVISION,
        local_dir=target,
        allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*"],
    )
    print(f"downloaded {MODEL_ID}@{REVISION[:8]} to {path}")
