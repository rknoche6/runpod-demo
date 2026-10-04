"""Model loading and input validation, shared by the Serverless handler and the Pod batch job.

The model is BAAI/bge-small-en-v1.5: 33M parameters, 384-dim embeddings, MIT licence.
Inside the Docker image it is read from MODEL_DIR (baked in at build time). Outside the
image it falls back to the Hugging Face model id and the local HF cache.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
# bge v1.5 recommends this prefix for short retrieval queries, not for passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MAX_TEXTS_PER_REQUEST = 256
MAX_CHARS_PER_TEXT = 8_000  # the model truncates at 512 tokens anyway; this caps payload size


class InputError(ValueError):
    """Raised for a request the handler should reject without touching the model."""


def pick_device() -> str:
    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and os.environ.get("USE_MPS") == "1":
        return "mps"
    return "cpu"


@dataclass
class Embedder:
    model_source: str
    device: str
    load_seconds: float
    import_seconds: float
    model: object = field(repr=False)

    @classmethod
    def load(cls) -> "Embedder":
        t_imp = time.perf_counter()
        from sentence_transformers import SentenceTransformer  # pulls in torch and transformers

        import_seconds = time.perf_counter() - t_imp

        source = os.environ.get("MODEL_DIR") or MODEL_ID
        device = pick_device()
        if os.environ.get("REQUIRE_CUDA") == "1" and device != "cuda":
            import torch

            raise RuntimeError(f"REQUIRE_CUDA=1 but torch sees no GPU (torch {torch.__version__}, "
                               f"built for CUDA {torch.version.cuda}); check the endpoint's allowed CUDA versions")
        t0 = time.perf_counter()
        model = SentenceTransformer(source, device=device)
        if device == "cuda":
            model.half()  # fp16 on GPU: less memory and faster; values differ slightly from fp32 (not measured here)
        # One tiny forward pass so CUDA kernels / lazy init happen during load, not on request 1.
        model.encode(["warmup"], normalize_embeddings=True)
        return cls(model_source=source, device=device, load_seconds=time.perf_counter() - t0,
                   import_seconds=import_seconds, model=model)

    def encode(self, texts: list[str], *, kind: str = "passage", normalize: bool = True, batch_size: int = 64):
        if kind == "query":
            texts = [QUERY_PREFIX + t for t in texts]
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def validate_input(job_input: object) -> dict:
    """Return a normalised request dict or raise InputError.

    Accepted shape: {"texts": [str, ...], "kind": "query"|"passage", "normalize": bool}
    A single {"text": str} is accepted as a convenience.
    """
    if not isinstance(job_input, dict):
        raise InputError("input must be a JSON object")
    texts = job_input.get("texts")
    if texts is None and isinstance(job_input.get("text"), str):
        texts = [job_input["text"]]
    if not isinstance(texts, list) or not texts:
        raise InputError("input.texts must be a non-empty list of strings")
    if len(texts) > MAX_TEXTS_PER_REQUEST:
        raise InputError(f"at most {MAX_TEXTS_PER_REQUEST} texts per request (got {len(texts)}); use the batch job for bulk work")
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            raise InputError(f"input.texts[{i}] must be a non-empty string")
        if len(t) > MAX_CHARS_PER_TEXT:
            raise InputError(f"input.texts[{i}] is longer than {MAX_CHARS_PER_TEXT} characters")
    kind = job_input.get("kind", "passage")
    if kind not in ("query", "passage"):
        raise InputError("input.kind must be 'query' or 'passage'")
    normalize = job_input.get("normalize", True)
    if not isinstance(normalize, bool):
        raise InputError("input.normalize must be a boolean")
    return {"texts": texts, "kind": kind, "normalize": normalize}
