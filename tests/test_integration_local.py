"""End to end on this machine: real handler served by the runpod SDK's local API,
FastAPI front in local mode in front of it. Exercises /runsync, /run and /status."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def local_worker():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "worker" / "handler.py"), "--rp_serve_api",
                             "--rp_api_port", str(port), "--rp_api_host", "127.0.0.1"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            httpx.get(url + "/", timeout=1)
            break
        except httpx.TransportError:
            if proc.poll() is not None:
                pytest.fail("local worker exited")
            time.sleep(0.3)
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.fixture
def front(local_worker, monkeypatch):
    monkeypatch.setenv("RUNPOD_MODE", "local")
    monkeypatch.setenv("RUNPOD_LOCAL_URL", local_worker)
    from app.main import create_app

    with TestClient(create_app()) as tc:
        yield tc


def test_embed_via_front(front):
    r = front.post("/v1/embed", json={"texts": ["hello", "world"]})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["dim"] == 384 and len(j["embeddings"]) == 2
    assert j["timing"]["client_roundtrip_ms"] > 0


def test_search_via_front(front):
    r = front.post("/v1/search", json={
        "query": "why is the first request slow",
        "documents": ["Cold starts happen when a worker has to boot and load the model.",
                      "Bananas are rich in potassium."],
    })
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["text"].startswith("Cold starts")


def test_async_job_via_front(front):
    r = front.post("/v1/jobs", json={"texts": ["async please"]})
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]
    s = front.get(f"/v1/jobs/{job_id}")
    assert s.json()["status"] == "COMPLETED"
    assert len(s.json()["output"]["embeddings"][0]) == 384


def test_bad_input_surfaces_handler_error(front):
    r = front.post("/v1/embed", json={"texts": ["x" * 9000]})
    assert r.status_code == 422 and "longer than" in r.text
