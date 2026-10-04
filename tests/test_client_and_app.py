"""Client and FastAPI tests against a fake Runpod API (httpx.MockTransport). No model, no network."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.runpod_client import ConfigError, EndpointConfig, JobFailed, RunpodClient


def test_config_missing_key_is_explicit():
    with pytest.raises(ConfigError, match="RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID"):
        EndpointConfig.from_env({})
    with pytest.raises(ConfigError, match="RUNPOD_ENDPOINT_ID"):
        EndpointConfig.from_env({"RUNPOD_API_KEY": "k"})


def test_config_modes():
    c = EndpointConfig.from_env({"RUNPOD_API_KEY": "k", "RUNPOD_ENDPOINT_ID": "abc123"})
    assert c.base_url == "https://api.runpod.ai/v2/abc123" and c.api_key == "k"
    local = EndpointConfig.from_env({"RUNPOD_MODE": "local", "RUNPOD_LOCAL_URL": "http://h:9/"})
    assert local.base_url == "http://h:9" and local.api_key is None
    with pytest.raises(ConfigError):
        EndpointConfig.from_env({"RUNPOD_MODE": "cloud"})


def fake_runpod(script):
    """script: list of (method, path_suffix, response_json). Asserts calls happen in order."""
    calls = []

    def handle(request: httpx.Request):
        method, suffix, resp = script[len(calls)]
        calls.append(request)
        assert request.method == method, (request.method, request.url)
        assert request.url.path.endswith(suffix), request.url.path
        return httpx.Response(200, json=resp)

    return httpx.MockTransport(handle), calls


def emb_output(n, dim=3):
    return {"model": "m", "dim": dim, "count": n, "embeddings": [[1.0, 0, 0]] * n,
            "timing": {"inference_ms": 1.0, "first_request_on_worker": False}}


@pytest.mark.anyio
async def test_runsync_sends_auth_and_returns_timings():
    transport, calls = fake_runpod([
        ("POST", "/v2/ep1/runsync", {"id": "j1", "status": "COMPLETED", "delayTime": 1200,
                                     "executionTime": 15, "output": emb_output(1)}),
    ])
    cfg = EndpointConfig.from_env({"RUNPOD_API_KEY": "secret", "RUNPOD_ENDPOINT_ID": "ep1"})
    c = RunpodClient(cfg, transport=transport)
    body = await c.runsync({"texts": ["a"]})
    await c.aclose()
    assert calls[0].headers["authorization"] == "Bearer secret"
    assert json.loads(calls[0].content) == {"input": {"texts": ["a"]}}
    assert body["delayTime"] == 1200 and body["client_polled"] is False


@pytest.mark.anyio
async def test_runsync_falls_back_to_status_polling():
    transport, calls = fake_runpod([
        ("POST", "/runsync", {"id": "j2", "status": "IN_QUEUE"}),
        ("GET", "/status/j2", {"id": "j2", "status": "IN_PROGRESS"}),
        ("GET", "/status/j2", {"id": "j2", "status": "COMPLETED", "output": emb_output(1)}),
    ])
    cfg = EndpointConfig("runpod", "https://api.runpod.ai/v2/ep1", "k", "ep1")
    c = RunpodClient(cfg, transport=transport)
    c_wait = c.wait

    async def fast_wait(job_id, **kw):
        return await c_wait(job_id, poll_s=0.0, timeout_s=5)

    c.wait = fast_wait
    body = await c.runsync({"texts": ["a"]})
    await c.aclose()
    assert body["status"] == "COMPLETED" and body["client_polled"] is True
    assert len(calls) == 3


@pytest.mark.anyio
async def test_local_mode_uses_post_status_and_no_auth():
    transport, calls = fake_runpod([("POST", "/status/test-1", {"id": "test-1", "status": "COMPLETED"})])
    c = RunpodClient(EndpointConfig("local", "http://localhost:8000", None, None), transport=transport)
    await c.status("test-1")
    await c.aclose()
    assert "authorization" not in calls[0].headers
    assert str(calls[0].url) == "http://localhost:8000/status/test-1"


@pytest.mark.anyio
async def test_failed_job_raises():
    transport, _ = fake_runpod([("POST", "/runsync", {"id": "j3", "status": "FAILED", "error": "bad input"})])
    c = RunpodClient(EndpointConfig("runpod", "https://x/v2/e", "k", "e"), transport=transport)
    with pytest.raises(JobFailed, match="bad input"):
        await c.runsync({"texts": []})
    await c.aclose()


def make_app(handler):
    client = RunpodClient(EndpointConfig("runpod", "https://api.runpod.ai/v2/ep", "k", "ep"),
                          transport=httpx.MockTransport(handler))
    return TestClient(create_app(client))


def test_app_search_ranks_by_cosine():
    def handler(req: httpx.Request):
        body = json.loads(req.content)["input"]
        if body["kind"] == "query":
            vecs = [[1.0, 0.0, 0.0]]
        else:
            vecs = [[0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.5, 0.5, 0.0]]
        return httpx.Response(200, json={"id": "j", "status": "COMPLETED", "delayTime": 5, "executionTime": 3,
                                         "output": {**emb_output(len(vecs)), "embeddings": vecs}})

    with make_app(handler) as tc:
        r = tc.post("/v1/search", json={"query": "q", "documents": ["a", "b", "c"], "top_k": 2})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert [x["text"] for x in res] == ["b", "c"]
    assert r.json()["timing"]["query"]["runpod_execution_ms"] == 3


def test_app_maps_failed_job_to_422_and_api_error_to_502():
    def failed(req):
        return httpx.Response(200, json={"id": "j", "status": "FAILED", "error": "input.texts[0] must be..."})

    with make_app(failed) as tc:
        r = tc.post("/v1/embed", json={"texts": ["a"]})
    assert r.status_code == 422 and "input.texts" in r.text

    def unauthorized(req):
        return httpx.Response(401, text="unauthorized")

    with make_app(unauthorized) as tc:
        r = tc.post("/v1/embed", json={"texts": ["a"]})
    assert r.status_code == 502 and "401" in r.text


def test_app_validates_before_calling_endpoint():
    def boom(req):
        raise AssertionError("endpoint should not be called")

    with make_app(boom) as tc:
        assert tc.post("/v1/embed", json={"texts": []}).status_code == 422
        assert tc.post("/v1/embed", json={"texts": ["a"], "kind": "doc"}).status_code == 422


def test_app_refuses_to_start_without_config(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    monkeypatch.delenv("RUNPOD_MODE", raising=False)
    with pytest.raises(ConfigError, match="RUNPOD_API_KEY"):
        with TestClient(create_app()):
            pass


@pytest.fixture
def anyio_backend():
    return "asyncio"
