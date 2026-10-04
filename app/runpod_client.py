"""Thin async client for a Runpod queue-based Serverless endpoint.

Two modes, chosen by RUNPOD_MODE:

  runpod (default)  https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID, auth with $RUNPOD_API_KEY
  local             $RUNPOD_LOCAL_URL (default http://localhost:8000), i.e. the same handler
                    served by `python worker/handler.py --rp_serve_api`. No key needed.

The local server from the runpod SDK exposes /run, /runsync and POST /status/{id} at the
root, without the /v2/{endpoint_id} prefix and without /health, so the differences are
handled here and nowhere else.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import httpx

RUNPOD_API_BASE = "https://api.runpod.ai/v2"
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


class ConfigError(RuntimeError):
    pass


class JobFailed(RuntimeError):
    def __init__(self, status: str, body: dict):
        super().__init__(f"job {body.get('id')} ended with status {status}: {body.get('error')}")
        self.status = status
        self.body = body


@dataclass(frozen=True)
class EndpointConfig:
    mode: str
    base_url: str
    api_key: str | None
    endpoint_id: str | None

    @classmethod
    def from_env(cls, env: dict | None = None) -> "EndpointConfig":
        env = os.environ if env is None else env
        mode = env.get("RUNPOD_MODE", "runpod").strip().lower()
        if mode == "local":
            return cls(mode="local", base_url=env.get("RUNPOD_LOCAL_URL", "http://localhost:8000").rstrip("/"),
                       api_key=None, endpoint_id=None)
        if mode != "runpod":
            raise ConfigError(f"RUNPOD_MODE must be 'runpod' or 'local', got {mode!r}")
        missing = [k for k in ("RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID") if not env.get(k)]
        if missing:
            raise ConfigError(
                "missing " + ", ".join(missing) + ". Set them to talk to a deployed endpoint "
                "(see DEPLOY.md), or set RUNPOD_MODE=local and run "
                "`python worker/handler.py --rp_serve_api` to use the local handler."
            )
        endpoint_id = env["RUNPOD_ENDPOINT_ID"].strip()
        return cls(mode="runpod", base_url=f"{RUNPOD_API_BASE}/{endpoint_id}",
                   api_key=env["RUNPOD_API_KEY"].strip(), endpoint_id=endpoint_id)


class RunpodClient:
    def __init__(self, config: EndpointConfig, *, transport: httpx.AsyncBaseTransport | None = None,
                 timeout_s: float = 120.0):
        self.config = config
        headers = {"content-type": "application/json"}
        if config.api_key:
            headers["authorization"] = f"Bearer {config.api_key}"
        self._http = httpx.AsyncClient(base_url=config.base_url, headers=headers,
                                       timeout=timeout_s, transport=transport)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, path: str, payload: dict | None = None) -> dict:
        r = await self._http.post(path, json=payload)
        r.raise_for_status()
        return r.json()

    async def run(self, job_input: dict, *, webhook: str | None = None) -> dict:
        """Async submit. Returns {"id": ..., "status": "IN_QUEUE"|"IN_PROGRESS"}."""
        body: dict = {"input": job_input}
        if webhook:
            body["webhook"] = webhook
        return await self._post("/run", body)

    async def status(self, job_id: str) -> dict:
        if self.config.mode == "local":
            return await self._post(f"/status/{job_id}")  # SDK's local server only has POST
        r = await self._http.get(f"/status/{job_id}")
        r.raise_for_status()
        return r.json()

    async def health(self) -> dict:
        if self.config.mode == "local":
            return {"mode": "local", "note": "the SDK's local server has no /health route"}
        r = await self._http.get("/health")
        r.raise_for_status()
        return r.json()

    async def wait(self, job_id: str, *, poll_s: float = 0.5, timeout_s: float = 300.0) -> dict:
        deadline = time.monotonic() + timeout_s
        while True:
            body = await self.status(job_id)
            if body.get("status") in TERMINAL_STATUSES:
                return body
            if time.monotonic() > deadline:
                raise TimeoutError(f"job {job_id} still {body.get('status')} after {timeout_s}s")
            await asyncio.sleep(poll_s)

    async def runsync(self, job_input: dict, *, timeout_s: float = 300.0) -> dict:
        """Submit with /runsync and return the finished job body, with client timing added.

        /runsync can return while the job is still IN_QUEUE or IN_PROGRESS (for example
        when a cold worker is still loading). In that case we fall back to polling /status.
        Raises JobFailed if the job does not complete.
        """
        t0 = time.perf_counter()
        body = await self._post("/runsync", {"input": job_input})
        polled = False
        if body.get("status") not in TERMINAL_STATUSES and body.get("id"):
            polled = True
            body = await self.wait(body["id"], timeout_s=timeout_s)
        body["client_roundtrip_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        body["client_polled"] = polled
        if body.get("status") != "COMPLETED":
            raise JobFailed(body.get("status", "UNKNOWN"), body)
        return body
