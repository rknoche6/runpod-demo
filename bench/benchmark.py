"""Measure cold-start and warm latency, then turn them into cost per 1k requests.

Targets:
  local-process   start `python worker/handler.py --rp_serve_api` as a subprocess. Cold start
                  = process spawn until the first /runsync succeeds (model load included).
  local-docker    same, but `docker run` of the image (CPU only on a laptop).
  runpod          a deployed endpoint (needs RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID). Cold start
                  = request sent to an endpoint with zero running workers until it completes.
                  Runpod's delayTime / executionTime are recorded alongside client time.

    python bench/benchmark.py local-process --cold-runs 3 --warm 50
    python bench/benchmark.py local-docker --image runpod-demo:cpu --cold-runs 3 --warm 50
    python bench/benchmark.py runpod --cold-runs 5 --warm 100 --gpu-class 16GB

Results go to bench/results/<target>-<UTC time>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runpod_client import EndpointConfig, RunpodClient  # noqa: E402
from bench import cost  # noqa: E402

QUERY = {"texts": ["how do I keep the first request after a quiet period fast?"], "kind": "query"}
BATCH32 = {"texts": [f"Passage {i}: workers scale to zero after the idle timeout, and the next request "
                     f"waits for a worker to start and load the model." for i in range(32)], "kind": "passage"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(ms: list[float]) -> dict:
    return {"n": len(ms), "p50_ms": round(pct(ms, 0.5), 2), "p90_ms": round(pct(ms, 0.9), 2),
            "p99_ms": round(pct(ms, 0.99), 2), "min_ms": round(min(ms), 2), "max_ms": round(max(ms), 2),
            "mean_ms": round(statistics.fmean(ms), 2)}


async def wait_ready_and_first(client: RunpodClient, t_spawn: float, timeout_s: float, proc=None) -> dict:
    """Poll /runsync until it answers; return first successful body with cold timing."""
    deadline = time.monotonic() + timeout_s
    while True:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"worker exited early with code {proc.returncode}")
        try:
            body = await client.runsync(QUERY, timeout_s=timeout_s)
            body["cold_total_s"] = round(time.perf_counter() - t_spawn, 3)
            return body
        except (httpx.TransportError, httpx.HTTPStatusError):
            if time.monotonic() > deadline:
                raise TimeoutError("worker did not become ready")
            await asyncio.sleep(0.1)


async def warm_series(client: RunpodClient, payload: dict, n: int) -> dict:
    ms, exec_ms, infer_ms = [], [], []
    for _ in range(n):
        body = await client.runsync(payload)
        ms.append(body["client_roundtrip_ms"])
        infer_ms.append(body["output"]["timing"]["inference_ms"])
        if body.get("executionTime") is not None:
            exec_ms.append(float(body["executionTime"]))
    out = {"client": summarize(ms), "handler_inference": summarize(infer_ms)}
    if exec_ms:
        out["runpod_execution"] = summarize(exec_ms)
    return out


def start_local_process(port: int):
    cmd = [sys.executable, "-u", str(ROOT / "worker" / "handler.py"), "--rp_serve_api",
           "--rp_api_port", str(port), "--rp_api_host", "127.0.0.1"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)


def start_local_docker(port: int, image: str, platform_arg: str | None):
    cmd = ["docker", "run", "-d", "--rm", "-p", f"127.0.0.1:{port}:8000"]
    if platform_arg:
        cmd += ["--platform", platform_arg]
    cmd += [image, "python", "-u", "/app/worker/handler.py", "--rp_serve_api", "--rp_api_host", "0.0.0.0",
            "--rp_api_port", "8000"]
    cid = subprocess.check_output(cmd, text=True).strip()
    return cid


async def bench_local(a) -> dict:
    colds, warm = [], None
    for i in range(a.cold_runs):
        port = free_port()
        client = RunpodClient(EndpointConfig("local", f"http://127.0.0.1:{port}", None, None), timeout_s=30)
        t_spawn = time.perf_counter()
        proc = cid = None
        if a.target == "local-process":
            proc = start_local_process(port)
        else:
            cid = start_local_docker(port, a.image, a.platform)
        try:
            first = await wait_ready_and_first(client, t_spawn, a.ready_timeout, proc)
            t = first["output"]["timing"]
            colds.append({"cold_total_s": first["cold_total_s"],
                          "worker_startup_s": t["worker_startup_s"],
                          "import_s": t["worker_import_s"],
                          "model_load_s": t["worker_model_load_s"],
                          "first_inference_ms": t["inference_ms"]})
            print(f"cold run {i + 1}: {first['cold_total_s']:.2f}s total, imports {t['worker_import_s']:.2f}s, "
                  f"model load {t['worker_model_load_s']:.2f}s", flush=True)
            if i == a.cold_runs - 1:
                warm = {"query_1": await warm_series(client, QUERY, a.warm),
                        "passages_32": await warm_series(client, BATCH32, max(10, a.warm // 5))}
        finally:
            await client.aclose()
            if proc:
                proc.terminate()
                proc.wait(10)
            if cid:
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
    return {"cold": colds, "warm": warm}


async def bench_runpod(a) -> dict:
    client = RunpodClient(EndpointConfig.from_env())
    colds = []
    try:
        for i in range(a.cold_runs):
            # Wait until no worker is running, so the next request has to start one.
            t_wait = time.monotonic()
            while True:
                h = await client.health()
                w = h.get("workers", {})
                busy = sum(int(w.get(k, 0) or 0) for k in ("running", "idle", "ready", "initializing"))
                if busy == 0:
                    break
                if time.monotonic() - t_wait > a.scale_down_timeout:
                    print(f"workers still up after {a.scale_down_timeout}s: {w}; is active workers > 0?",
                          file=sys.stderr)
                    break
                await asyncio.sleep(5)
            body = await client.runsync(QUERY, timeout_s=a.ready_timeout)
            t = body["output"]["timing"]
            rec = {"client_roundtrip_s": round(body["client_roundtrip_ms"] / 1000, 3),
                   "runpod_delay_s": (body.get("delayTime") or 0) / 1000,
                   "runpod_execution_ms": body.get("executionTime"),
                   "first_request_on_worker": t["first_request_on_worker"],
                   "worker_age_s": t["worker_age_s"], "import_s": t["worker_import_s"],
                   "model_load_s": t["worker_model_load_s"],
                   "health_before": h}
            colds.append(rec)
            print(f"cold run {i + 1}: client {rec['client_roundtrip_s']}s, delayTime {rec['runpod_delay_s']}s, "
                  f"first_on_worker={rec['first_request_on_worker']}", flush=True)
        warm = {"query_1": await warm_series(client, QUERY, a.warm),
                "passages_32": await warm_series(client, BATCH32, max(10, a.warm // 5))}
    finally:
        await client.aclose()
    return {"cold": colds, "warm": warm}


def add_costs(res: dict, a) -> dict:
    prices = cost.load_prices()
    ps = cost.flex_price_per_s(prices, a.gpu_class)
    key = "cold_total_s" if a.target.startswith("local") else "client_roundtrip_s"
    cold_s = statistics.median(c[key] for c in res["cold"])
    w = res["warm"]["query_1"]
    exec_s = (w.get("runpod_execution") or w["client"])["p50_ms"] / 1000
    exec32_s = (res["warm"]["passages_32"].get("runpod_execution") or res["warm"]["passages_32"]["client"])["p50_ms"] / 1000
    return {
        "gpu_class": a.gpu_class,
        "flex_usd_per_s": round(ps, 8),
        "prices_fetched_on": prices["fetched_on"],
        "inputs": {"cold_start_s_median": cold_s, "warm_exec_s_p50": exec_s, "warm_exec_32_s_p50": exec32_s,
                   "idle_timeout_s": a.idle_timeout},
        "per_1k_queries_busy_worker_usd": round(cost.per_1k_busy_worker(exec_s, ps), 5),
        "per_1k_queries_all_cold_usd": round(cost.per_1k_isolated_requests(cold_s, exec_s, a.idle_timeout, ps), 5),
        "per_1k_batch32_busy_worker_usd": round(cost.per_1k_busy_worker(exec32_s, ps), 5),
        "active_worker_month_usd_no_discount": round(cost.active_worker_monthly(ps), 2),
        "breakeven_req_per_day_active_vs_all_cold": round(
            cost.breakeven_requests_per_day(cold_s, exec_s, a.idle_timeout, ps)),
        "caveat": ("local targets: timings are CPU on this machine, priced at a GPU rate only to show the "
                   "arithmetic; do not quote them as Runpod costs") if a.target.startswith("local") else None,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", choices=["local-process", "local-docker", "runpod"])
    p.add_argument("--image", default="runpod-demo:cpu")
    p.add_argument("--platform", default=None, help="e.g. linux/amd64 for local-docker")
    p.add_argument("--cold-runs", type=int, default=3)
    p.add_argument("--warm", type=int, default=50)
    p.add_argument("--gpu-class", default="16GB")
    p.add_argument("--idle-timeout", type=float, default=5.0, help="endpoint idle timeout, for the cost model")
    p.add_argument("--ready-timeout", type=float, default=600)
    p.add_argument("--scale-down-timeout", type=float, default=180)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if a.target == "runpod":
        EndpointConfig.from_env()  # fail fast with a clear message if the key or endpoint id is missing
    res = asyncio.run(bench_local(a) if a.target.startswith("local") else bench_runpod(a))
    res["cost"] = add_costs(res, a)
    res["meta"] = {"target": a.target, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "host": f"{platform.system()} {platform.machine()}", "python": platform.python_version(),
                   "image": a.image if a.target == "local-docker" else None,
                   "endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID") if a.target == "runpod" else None}
    out = a.out or ROOT / "bench" / "results" / f"{a.target}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({"warm": res["warm"], "cost": res["cost"]}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
