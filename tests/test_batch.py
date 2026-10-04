import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "batch"))

import embed_corpus  # noqa: E402
import launch_pod  # noqa: E402


class FakeEmbedder:
    device = "fake"

    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kw):
        self.calls += 1
        v = np.ones((len(texts), 384), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def write_corpus(path: Path, n: int):
    path.write_text("".join(json.dumps({"id": f"d{i}", "text": f"text {i}"}) + "\n" for i in range(n)))


def test_shards_and_resume(tmp_path):
    corpus = tmp_path / "c.jsonl"
    write_corpus(corpus, 25)
    out = tmp_path / "out"
    fe = FakeEmbedder()
    m = embed_corpus.run(corpus, out, shard_size=10, batch_size=4, embedder=fe)
    assert m["docs_embedded"] == 25 and fe.calls == 3
    assert np.load(out / "shard_00002.npy").shape == (5, 384)
    assert np.load(out / "shard_00000.npy").dtype == np.float16
    assert json.loads((out / "shard_00001.ids.json").read_text())[0] == "d10"

    # Simulate an interrupted run: last shard lost. Only that shard is recomputed.
    (out / "shard_00002.npy").unlink()
    fe2 = FakeEmbedder()
    m2 = embed_corpus.run(corpus, out, shard_size=10, batch_size=4, embedder=fe2)
    assert fe2.calls == 1 and m2["docs_skipped_existing"] == 20 and m2["docs_embedded"] == 5


def test_missing_text_is_an_error(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"id": "x"}\n')
    with pytest.raises(ValueError, match="missing 'text'"):
        embed_corpus.run(corpus, tmp_path / "o", 10, 4, embedder=FakeEmbedder())


def test_terminate_pod_without_env_is_noop(monkeypatch):
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    assert embed_corpus.terminate_pod().startswith("not terminated")


def test_launch_pod_dry_run_needs_no_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    r = subprocess.run([sys.executable, str(ROOT / "batch" / "launch_pod.py"), "--image", "img:1",
                        "--network-volume-id", "vol1", "--price-per-hr", "0.27", "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["networkVolumeId"] == "vol1" and payload["volumeMountPath"] == "/workspace"
    assert "--terminate-pod-when-done" in payload["dockerStartCmd"][-1]


def test_launch_pod_without_key_fails_clearly(monkeypatch):
    env = {k: v for k, v in __import__("os").environ.items() if k != "RUNPOD_API_KEY"}
    r = subprocess.run([sys.executable, str(ROOT / "batch" / "launch_pod.py"), "--image", "img:1",
                        "--network-volume-id", "vol1", "--price-per-hr", "0.27"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "RUNPOD_API_KEY is not set" in r.stderr
