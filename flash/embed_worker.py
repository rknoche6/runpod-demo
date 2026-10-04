"""The same embedding endpoint as worker/handler.py, deployed with Runpod Flash
instead of a Docker image: no image to build or push, Flash packages the code and
installs the dependencies on the worker.

Wire contract matches the Docker worker, because Flash calls the function with the
keys of `input` as keyword arguments:
    POST https://api.runpod.ai/v2/<endpoint-id>/runsync
    {"input": {"texts": ["how do I reset my password"], "kind": "query"}}

Trade-off vs the Docker image: the model downloads from Hugging Face on each cold
worker instead of being baked in. For a 135 MB model that is a few seconds.

Deploy: flash deploy   (needs RUNPOD_API_KEY)
Local check of the function body: python embed_worker.py
"""
from runpod_flash import Endpoint, GpuGroup


@Endpoint(
    name="runpod-demo-embed",
    gpu=[GpuGroup.AMPERE_16, GpuGroup.AMPERE_24],  # 33M-param model fits anything; take the cheapest supply
    workers=(0, 2),          # scale to zero, at most two workers
    idle_timeout=30,         # seconds a worker stays warm after its last request
    dependencies=["sentence-transformers==5.1.1", "torch"],
)
async def embed(texts: list, kind: str = "passage", normalize: bool = True) -> dict:
    # Everything the function needs lives in its body: Flash ships only the body.
    import os
    import time

    model_id = "BAAI/bge-small-en-v1.5"
    query_prefix = "Represent this sentence for searching relevant passages: "

    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        return {"error": "texts must be a non-empty list of non-empty strings"}
    if len(texts) > 256:
        return {"error": "at most 256 texts per request; use the batch job for bulk work"}
    if kind not in ("query", "passage"):
        return {"error": "kind must be 'query' or 'passage'"}

    # Load once per worker (module global set inside the body; see Flash gotcha 11).
    global _STATE
    try:
        _STATE
    except NameError:
        t0 = time.perf_counter()
        import torch
        from sentence_transformers import SentenceTransformer

        t_imported = time.perf_counter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_id, device=device)
        if device == "cuda":
            model.half()
        model.encode(["warmup"], normalize_embeddings=True)
        _STATE = {
            "model": model, "device": device, "served": 0, "started": time.time(),
            "import_s": round(t_imported - t0, 3), "load_s": round(time.perf_counter() - t_imported, 3),
        }

    s = _STATE
    if kind == "query":
        texts = [query_prefix + t for t in texts]
    t0 = time.perf_counter()
    vectors = s["model"].encode(texts, batch_size=64, normalize_embeddings=normalize,
                                convert_to_numpy=True, show_progress_bar=False)
    infer_ms = (time.perf_counter() - t0) * 1000
    first = s["served"] == 0
    s["served"] += 1
    return {
        "model": model_id,
        "dim": int(vectors.shape[1]),
        "count": len(texts),
        "embeddings": [[round(float(x), 6) for x in v] for v in vectors],
        "timing": {
            "inference_ms": round(infer_ms, 2),
            "first_request_on_worker": first,
            "worker_import_s": s["import_s"],
            "worker_model_load_s": s["load_s"],
            "worker_age_s": round(time.time() - s["started"], 3),
            "device": s["device"],
        },
        "worker_id": os.environ.get("RUNPOD_POD_ID", "local"),
    }


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(embed(["how do I reset my password"], kind="query"))["timing"])
