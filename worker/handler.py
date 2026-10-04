"""Runpod Serverless handler: text -> bge-small-en-v1.5 embeddings.

The model loads at import time, once per worker. Runpod bills from worker start, so the
load is part of every cold start; requests on a warm worker skip it.

Local testing (no Runpod account needed):
    python worker/handler.py --test_input '{"input": {"texts": ["hello world"]}}'
    python worker/handler.py --rp_serve_api --rp_api_port 8000   # local /run, /runsync, /status
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder import EMBED_DIM, Embedder, InputError, validate_input  # noqa: E402

PROCESS_START = time.time()  # wall clock, comparable across processes on one host
_t_import = time.perf_counter()
EMBEDDER = Embedder.load()
STARTUP_SECONDS = time.perf_counter() - _t_import
_requests_served = 0


def handler(job: dict) -> dict:
    global _requests_served
    try:
        req = validate_input(job.get("input"))
    except InputError as exc:
        # Returning {"error": ...} makes Runpod mark the job FAILED with this message.
        return {"error": str(exc)}

    t0 = time.perf_counter()
    vectors = EMBEDDER.encode(req["texts"], kind=req["kind"], normalize=req["normalize"])
    infer_ms = (time.perf_counter() - t0) * 1000

    first_on_worker = _requests_served == 0
    _requests_served += 1
    return {
        "model": "BAAI/bge-small-en-v1.5",
        "dim": EMBED_DIM,
        "count": len(req["texts"]),
        "embeddings": [[round(float(x), 6) for x in v] for v in vectors],
        "timing": {
            "inference_ms": round(infer_ms, 2),
            # These let a client tell a cold worker from a warm one without platform metrics.
            "first_request_on_worker": first_on_worker,
            "worker_import_s": round(EMBEDDER.import_seconds, 3),
            "worker_model_load_s": round(EMBEDDER.load_seconds, 3),
            "worker_startup_s": round(STARTUP_SECONDS, 3),
            "worker_age_s": round(time.time() - PROCESS_START, 3),
            "device": EMBEDDER.device,
        },
        "worker_id": os.environ.get("RUNPOD_POD_ID", "local"),
    }


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
