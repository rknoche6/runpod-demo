import math

from bench import cost


def test_prices_file_has_sources_and_date():
    p = cost.load_prices()
    assert p["fetched_on"] and "runpod.io/pricing" in p["sources"]["pricing_page"]
    assert math.isclose(cost.flex_price_per_s(p, "16GB"), 0.58 / 3600)


def test_busy_worker_cost():
    # 20 ms per request at $0.0001/s -> 1000 * 0.02 * 0.0001 = $0.002
    assert math.isclose(cost.per_1k_busy_worker(0.020, 0.0001), 0.002)


def test_isolated_requests_round_up_session():
    # 3.2 s start + 0.02 s exec + 5 s idle = 8.22 s -> billed 9 s
    assert math.isclose(cost.per_1k_isolated_requests(3.2, 0.02, 5, 0.0001), 1000 * 9 * 0.0001)


def test_breakeven():
    ps = 0.0001
    per_req = 9 * ps
    assert math.isclose(cost.breakeven_requests_per_day(3.2, 0.02, 5, ps), 86400 * ps / per_req)


def test_pod_batch_cost():
    r = cost.pod_batch_cost(1_000_000, docs_per_s=1000, pod_per_hr=0.36, overhead_s=0)
    assert math.isclose(r["run_s"], 1000) and math.isclose(r["cost_usd"], 0.1)
