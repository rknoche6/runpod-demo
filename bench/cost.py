"""Cost arithmetic for Serverless and Pod deployments, from per-second prices.

Every function takes seconds and $/s or $/hr explicitly so the README numbers can be
recomputed by hand. Prices come from bench/prices.json (see the sources and date there).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PRICES_PATH = Path(__file__).with_name("prices.json")
SECONDS_PER_MONTH = 30 * 24 * 3600


def load_prices(path: Path = PRICES_PATH) -> dict:
    return json.loads(path.read_text())


def flex_price_per_s(prices: dict, gpu_class: str) -> float:
    return prices["serverless_flex_per_hr"][gpu_class]["usd_per_hr"] / 3600


def per_1k_busy_worker(exec_s: float, price_per_s: float) -> float:
    """Lower bound: workers are never idle, every billed second does work."""
    return 1000 * exec_s * price_per_s


def per_1k_isolated_requests(cold_start_s: float, exec_s: float, idle_timeout_s: float, price_per_s: float) -> float:
    """Upper bound for sparse traffic: each request wakes a scaled-down worker.

    Billed per worker session = start + execution + idle timeout, rounded up to a second.
    """
    session = math.ceil(cold_start_s + exec_s + idle_timeout_s)
    return 1000 * session * price_per_s


def active_worker_monthly(price_per_s: float, workers: int = 1, discount: float = 1.0) -> float:
    return SECONDS_PER_MONTH * price_per_s * discount * workers


def breakeven_requests_per_day(cold_start_s: float, exec_s: float, idle_timeout_s: float,
                               price_per_s: float, active_discount: float = 1.0) -> float:
    """Daily volume above which one always-on worker costs less than flex with every request cold.

    Real traffic sits between the two cases (some requests reuse a warm flex worker), so the
    true break-even is higher than this number.
    """
    active_per_day = 86400 * price_per_s * active_discount
    flex_per_request = per_1k_isolated_requests(cold_start_s, exec_s, idle_timeout_s, price_per_s) / 1000
    return active_per_day / flex_per_request


def pod_batch_cost(n_docs: int, docs_per_s: float, pod_per_hr: float, overhead_s: float = 0.0) -> dict:
    run_s = n_docs / docs_per_s + overhead_s
    cost = run_s / 3600 * pod_per_hr
    return {"run_s": run_s, "cost_usd": cost, "usd_per_1m_docs": cost / n_docs * 1_000_000}


def serverless_batch_cost(n_docs: int, docs_per_s: float, price_per_s: float, overhead_s: float = 0.0) -> dict:
    run_s = n_docs / docs_per_s + overhead_s
    cost = run_s * price_per_s
    return {"run_s": run_s, "cost_usd": cost, "usd_per_1m_docs": cost / n_docs * 1_000_000}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Print cost scenarios for given timings.")
    p.add_argument("--gpu-class", default="16GB")
    p.add_argument("--exec-s", type=float, required=True, help="warm execution time per request")
    p.add_argument("--cold-s", type=float, required=True, help="cold start time (worker start to ready)")
    p.add_argument("--idle-timeout-s", type=float, default=5)
    a = p.parse_args()
    pr = load_prices()
    ps = flex_price_per_s(pr, a.gpu_class)
    print(f"flex {a.gpu_class}: ${ps:.7f}/s (prices fetched {pr['fetched_on']})")
    print(f"per 1k, busy worker:       ${per_1k_busy_worker(a.exec_s, ps):.4f}")
    print(f"per 1k, every request cold: ${per_1k_isolated_requests(a.cold_s, a.exec_s, a.idle_timeout_s, ps):.4f}")
    print(f"1 active worker / month (no discount): ${active_worker_monthly(ps):.2f}")
    print(f"break-even vs all-cold flex: {breakeven_requests_per_day(a.cold_s, a.exec_s, a.idle_timeout_s, ps):.0f} req/day")
