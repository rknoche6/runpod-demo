"""Start the batch job on a Runpod GPU Pod through the REST API (https://rest.runpod.io/v1).

Prints the request body and exits with --dry-run, which needs no key. Without --dry-run
it needs RUNPOD_API_KEY, creates the Pod, and polls it until it is gone (the job deletes
its own Pod when it finishes) or until --max-minutes passes, then prints the wall time
and the cost implied by --price-per-hr.

    python batch/launch_pod.py --image docker.io/USER/runpod-demo:0.1.0 \
        --network-volume-id VOLUME_ID --gpu "NVIDIA RTX A5000" --price-per-hr 0.27 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

REST = "https://rest.runpod.io/v1"


def build_payload(a) -> dict:
    cmd = ""
    if a.sample_docs:
        # No upload needed: generate a synthetic corpus on the volume first (skipped if present).
        cmd += (f"[ -f {a.input} ] || python /app/batch/make_sample_corpus.py --n {a.sample_docs} "
                f"--out {a.input}; ")
    cmd += (f"python -u /app/batch/embed_corpus.py --input {a.input} --out {a.out} "
           f"--shard-size {a.shard_size} --batch-size {a.batch_size} --terminate-pod-when-done")
    payload = {
        "name": a.name,
        "imageName": a.image,
        "gpuTypeIds": [a.gpu],
        "gpuCount": 1,
        "cloudType": a.cloud_type,
        "containerDiskInGb": 10,
        "networkVolumeId": a.network_volume_id,
        "volumeMountPath": "/workspace",
        "dockerStartCmd": ["bash", "-c", cmd],
        "env": {"POD_PRICE_PER_HR": str(a.price_per_hr), "REQUIRE_CUDA": "1"},
        "ports": [],
    }
    if a.data_center_id:
        payload["dataCenterIds"] = [a.data_center_id]
    return payload


def api(method: str, path: str, key: str, body: dict | None = None) -> dict | None:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(REST + path, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--network-volume-id", required=True)
    p.add_argument("--gpu", default="NVIDIA RTX A5000")
    p.add_argument("--cloud-type", default="SECURE", choices=["SECURE", "COMMUNITY"])
    p.add_argument("--data-center-id", help="must match the network volume's data center")
    p.add_argument("--input", default="/workspace/corpus.jsonl")
    p.add_argument("--out", default="/workspace/embeddings")
    p.add_argument("--shard-size", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sample-docs", type=int, default=0,
                   help="generate this many synthetic docs at --input if it does not exist")
    p.add_argument("--name", default="runpod-demo-batch")
    p.add_argument("--price-per-hr", type=float, required=True, help="Pod $/hr from runpod.io/pricing, for the cost line")
    p.add_argument("--max-minutes", type=float, default=60)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    payload = build_payload(a)
    key = os.environ.get("RUNPOD_API_KEY")
    if a.dry_run:
        payload["env"]["RUNPOD_API_KEY"] = "<set from $RUNPOD_API_KEY at launch>"
        print(json.dumps(payload, indent=2))
        return 0
    if not key:
        print("RUNPOD_API_KEY is not set. Create a key in the Runpod console (Settings > API Keys) "
              "and export it, or use --dry-run to print the request.", file=sys.stderr)
        return 2
    # The job deletes its own Pod at the end, so it needs a key inside the Pod.
    # Use a key restricted to Pods if your account supports scoped keys.
    payload["env"]["RUNPOD_API_KEY"] = key

    t0 = time.time()
    pod = api("POST", "/pods", key, payload)
    pod_id = pod["id"]
    print(f"created pod {pod_id} ({a.gpu}); polling", flush=True)
    deadline = t0 + a.max_minutes * 60
    while time.time() < deadline:
        time.sleep(15)
        try:
            info = api("GET", f"/pods/{pod_id}", key)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                break  # the job deleted the Pod: done
            raise
        print(f"  {time.time() - t0:6.0f}s  desiredStatus={info.get('desiredStatus')}", flush=True)
    else:
        print(f"pod {pod_id} still exists after {a.max_minutes} min; stopping it so it stops billing GPU time",
              file=sys.stderr)
        api("POST", f"/pods/{pod_id}/stop", key)
        return 1
    wall = time.time() - t0
    print(json.dumps({"pod_id": pod_id, "wall_s": round(wall, 1),
                      "est_cost_usd": round(wall / 3600 * a.price_per_hr, 4),
                      "note": "wall time includes image pull and scheduling; see manifest.json on the volume"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
