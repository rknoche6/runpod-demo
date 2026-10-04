"""Pod batch job: embed a JSONL corpus into .npy shards.

Same Docker image as the Serverless worker, different start command. On a Runpod Pod the
input and output live on a network volume mounted at /workspace, so results survive the
Pod being stopped or terminated, and a restarted run resumes at the first missing shard.

    python batch/embed_corpus.py --input data/sample_corpus.jsonl --out out/ --shard-size 2000

Input: one JSON object per line with "id" and "text".
Output: shard_00000.npy (float16, L2-normalised, shape [n, 384]), shard_00000.ids.json,
and manifest.json with throughput numbers.

--terminate-pod-when-done deletes the Pod through the REST API after the last shard. It
needs RUNPOD_POD_ID (Runpod sets it inside every Pod) and RUNPOD_API_KEY (you pass it).
Without this a finished Pod keeps billing until someone stops it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker"))

from embedder import EMBED_DIM, Embedder  # noqa: E402


def read_corpus(path: Path):
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "text" not in rec:
                raise ValueError(f"{path}:{line_no}: missing 'text'")
            yield str(rec.get("id", line_no)), rec["text"]


def shards(records, size: int):
    buf: list[tuple[str, str]] = []
    for rec in records:
        buf.append(rec)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def terminate_pod() -> str:
    pod_id, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not pod_id or not key:
        return "not terminated: RUNPOD_POD_ID or RUNPOD_API_KEY missing"
    req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod_id}", method="DELETE",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return f"DELETE pod {pod_id}: HTTP {r.status}"


def run(input_path: Path, out_dir: Path, shard_size: int, batch_size: int, embedder: Embedder | None = None) -> dict:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    embedder = embedder or Embedder.load()
    load_s = time.perf_counter() - t_start

    done_docs = skipped_docs = 0
    encode_s = 0.0
    for idx, shard in enumerate(shards(read_corpus(input_path), shard_size)):
        npy = out_dir / f"shard_{idx:05d}.npy"
        ids_path = out_dir / f"shard_{idx:05d}.ids.json"
        if npy.exists() and ids_path.exists():
            skipped_docs += len(shard)
            continue
        ids, texts = zip(*shard)
        t0 = time.perf_counter()
        vecs = embedder.encode(list(texts), kind="passage", normalize=True, batch_size=batch_size)
        encode_s += time.perf_counter() - t0
        tmp = npy.with_suffix(".tmp.npy")
        np.save(tmp, vecs.astype(np.float16))
        ids_path.write_text(json.dumps(list(ids)))
        tmp.rename(npy)  # write ids first, rename vectors last: a shard counts only when both exist
        done_docs += len(shard)
        rate = done_docs / encode_s if encode_s else 0
        print(f"shard {idx}: {len(shard)} docs, {rate:.0f} docs/s so far", flush=True)

    total_s = time.perf_counter() - t_start
    manifest = {
        "model": "BAAI/bge-small-en-v1.5",
        "dim": EMBED_DIM,
        "dtype": "float16",
        "device": embedder.device,
        "docs_embedded": done_docs,
        "docs_skipped_existing": skipped_docs,
        "model_load_s": round(load_s, 2),
        "encode_s": round(encode_s, 2),
        "total_s": round(total_s, 2),
        "docs_per_s_encode": round(done_docs / encode_s, 1) if encode_s else None,
        "shard_size": shard_size,
        "batch_size": batch_size,
    }
    price = os.environ.get("POD_PRICE_PER_HR")
    if price:
        manifest["pod_price_per_hr"] = float(price)
        manifest["est_cost_usd_this_run"] = round(float(price) * total_s / 3600, 5)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-size", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--terminate-pod-when-done", action="store_true")
    a = p.parse_args(argv)
    if not a.input.exists():
        print(f"input not found: {a.input}", file=sys.stderr)
        return 2
    manifest = run(a.input, a.out, a.shard_size, a.batch_size)
    print(json.dumps(manifest, indent=2))
    if a.terminate_pod_when_done:
        print(terminate_pod(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
