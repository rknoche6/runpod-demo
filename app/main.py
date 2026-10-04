"""FastAPI front for the embedding endpoint.

The browser or the customer's backend talks to this service; this service holds the
Runpod API key and talks to the Serverless endpoint. The key never reaches the client.

    RUNPOD_MODE=local uvicorn app.main:app --port 8080          # against the local handler
    RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=... uvicorn app.main:app --port 8080
"""

from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.runpod_client import ConfigError, EndpointConfig, JobFailed, RunpodClient

MAX_TEXTS = 256
MAX_SEARCH_DOCS = 200


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS)
    kind: str = Field(default="passage", pattern="^(query|passage)$")


class JobRequest(EmbedRequest):
    webhook: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    documents: list[str] = Field(min_length=1, max_length=MAX_SEARCH_DOCS)
    top_k: int = Field(default=5, ge=1, le=50)


def _timing(body: dict) -> dict:
    """Collect the timings a customer cares about from one job body.

    delayTime / executionTime are milliseconds reported by Runpod (queue + cold start, and
    handler run time). The local SDK server does not return them, so they may be None.
    """
    out = body.get("output") or {}
    return {
        "client_roundtrip_ms": body.get("client_roundtrip_ms"),
        "runpod_delay_ms": body.get("delayTime"),
        "runpod_execution_ms": body.get("executionTime"),
        "handler": out.get("timing"),
        "polled_status": body.get("client_polled"),
    }


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def create_app(client: RunpodClient | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal client
        own = client is None
        if own:
            try:
                client = RunpodClient(EndpointConfig.from_env())
            except ConfigError as exc:
                # Fail at startup with the reason, not on the first request.
                raise ConfigError(f"[runpod-demo] configuration error: {exc}") from None
        app.state.client = client
        yield
        if own:
            await client.aclose()

    app = FastAPI(title="runpod-demo embedding front", lifespan=lifespan)

    async def call(job_input: dict) -> dict:
        try:
            return await app.state.client.runsync(job_input)
        except JobFailed as exc:
            raise HTTPException(status_code=422 if exc.status == "FAILED" else 504,
                                detail={"status": exc.status, "error": exc.body.get("error")})
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            raise HTTPException(status_code=502, detail=f"Runpod API returned {code}: {exc.response.text[:300]}")
        except (httpx.TransportError, TimeoutError) as exc:
            raise HTTPException(status_code=504, detail=f"endpoint unreachable or timed out: {exc!r}")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (Path(__file__).parent / "static" / "index.html").read_text()

    @app.get("/healthz")
    async def healthz():
        c: RunpodClient = app.state.client
        info = {"mode": c.config.mode, "endpoint_id": c.config.endpoint_id}
        try:
            info["endpoint_health"] = await c.health()
        except httpx.HTTPError as exc:
            info["endpoint_health"] = {"error": repr(exc)}
        return info

    @app.post("/v1/embed")
    async def embed(req: EmbedRequest):
        body = await call({"texts": req.texts, "kind": req.kind})
        out = body["output"]
        return {"embeddings": out["embeddings"], "dim": out["dim"], "model": out["model"],
                "job_id": body.get("id"), "timing": _timing(body)}

    @app.post("/v1/jobs", status_code=202)
    async def submit(req: JobRequest):
        try:
            return await app.state.client.run({"texts": req.texts, "kind": req.kind}, webhook=req.webhook)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=repr(exc))

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str):
        try:
            return await app.state.client.status(job_id)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=repr(exc))

    @app.post("/v1/search")
    async def search(req: SearchRequest):
        """Rank documents against a query. Two jobs (query and passages) sent concurrently."""
        q_body, d_body = await asyncio.gather(
            call({"texts": [req.query], "kind": "query"}),
            call({"texts": req.documents, "kind": "passage"}),
        )
        qv = q_body["output"]["embeddings"][0]
        scored = [
            {"index": i, "score": round(_cosine(qv, dv), 4), "text": req.documents[i]}
            for i, dv in enumerate(d_body["output"]["embeddings"])
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return {"results": scored[: req.top_k],
                "timing": {"query": _timing(q_body), "documents": _timing(d_body)}}

    return app


app = create_app()
