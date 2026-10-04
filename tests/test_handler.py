"""Handler tests. These load the real model (CPU) once per session."""

import math

import pytest

from embedder import MAX_TEXTS_PER_REQUEST, InputError, validate_input


@pytest.fixture(scope="module")
def handler_mod():
    import handler

    return handler


@pytest.mark.parametrize("bad, msg", [
    (None, "JSON object"),
    ({}, "non-empty list"),
    ({"texts": []}, "non-empty list"),
    ({"texts": ["ok", ""]}, "texts[1]"),
    ({"texts": ["ok", 3]}, "texts[1]"),
    ({"texts": ["x" * 9000]}, "longer than"),
    ({"texts": ["a"] * (MAX_TEXTS_PER_REQUEST + 1)}, "at most"),
    ({"texts": ["a"], "kind": "doc"}, "kind"),
    ({"texts": ["a"], "normalize": "yes"}, "normalize"),
])
def test_validate_rejects(bad, msg):
    with pytest.raises(InputError, match=msg.replace("[", r"\[").replace("]", r"\]")):
        validate_input(bad)


def test_validate_single_text_shortcut():
    assert validate_input({"text": "hi"}) == {"texts": ["hi"], "kind": "passage", "normalize": True}


def test_handler_returns_normalised_384d(handler_mod):
    out = handler_mod.handler({"id": "t1", "input": {"texts": ["first", "second"]}})
    assert out["dim"] == 384 and out["count"] == 2
    for v in out["embeddings"]:
        assert len(v) == 384
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, abs_tol=1e-3)
    assert out["timing"]["inference_ms"] > 0
    assert out["timing"]["worker_model_load_s"] > 0


def test_handler_error_becomes_failed_job(handler_mod):
    out = handler_mod.handler({"id": "t2", "input": {"texts": []}})
    assert set(out) == {"error"}


def test_first_request_flag_flips(handler_mod):
    handler_mod.handler({"id": "a", "input": {"text": "warm me"}})
    out = handler_mod.handler({"id": "b", "input": {"text": "again"}})
    assert out["timing"]["first_request_on_worker"] is False


def test_query_prefix_changes_embedding_and_ranks_relevant_doc(handler_mod):
    q = handler_mod.handler({"id": "q", "input": {"texts": ["how to make cold starts faster"], "kind": "query"}})
    d = handler_mod.handler({"id": "d", "input": {"texts": [
        "FlashBoot reduces cold start time by retaining worker state.",
        "The cafeteria serves soup on Tuesdays.",
    ]}})
    qv = q["embeddings"][0]
    scores = [sum(a * b for a, b in zip(qv, dv)) for dv in d["embeddings"]]
    assert scores[0] > scores[1] + 0.1


def test_require_cuda_fails_loudly_on_cpu(monkeypatch):
    from embedder import Embedder

    monkeypatch.setenv("REQUIRE_CUDA", "1")
    monkeypatch.setenv("DEVICE", "cpu")
    with pytest.raises(RuntimeError, match="REQUIRE_CUDA=1"):
        Embedder.load()
