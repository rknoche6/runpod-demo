# runpod-demo: one embedding model, served two ways on Runpod

A small text-embedding model (`BAAI/bge-small-en-v1.5`, 33M parameters, 384 dimensions,
MIT licence) packaged as one Docker image that runs as:

- a **Runpod Serverless endpoint** for real-time requests (embed a search query, rank a
  handful of documents), behind a small **FastAPI front** that holds the API key, and
- a **GPU Pod batch job** that embeds a whole corpus from a network volume and deletes its
  own Pod when it finishes.

Plus a benchmark that measures cold start and warm latency and turns them into cost per
1,000 requests using Runpod's published per-second prices, and a write-up of how a
customer would decide between flex workers, active workers and Pods.

**Status:** everything that can run without a Runpod account has been run on a laptop
(numbers below). Nothing has been deployed to Runpod yet; the numbers that need a real
endpoint are in a table of TODOs, each with the command that fills it. Deploy steps are
in [DEPLOY.md](DEPLOY.md).

## The customer story

A team runs semantic search over their support articles and tickets. They have two
embedding workloads with opposite shapes:

| | Query embeddings | Corpus embeddings |
|---|---|---|
| Shape | one short text per user search, a few per second at peak, near zero at night | hundreds of thousands of passages, when the corpus or the model changes |
| Cares about | latency of each request, including the first one after a quiet period | throughput and total cost; latency of a single item is irrelevant |
| Runpod product | Serverless endpoint (flex workers, maybe one active worker) | a GPU Pod for the duration of the job |

The same image serves both, so queries and corpus get the same weights and the same
code. Embeddings need that. Vectors from two different model builds can't be compared.

```
 browser / customer backend
        │  POST /v1/search, /v1/embed, /v1/jobs
        ▼
 FastAPI front (app/)  ── holds RUNPOD_API_KEY ──►  https://api.runpod.ai/v2/{endpoint}/runsync | run | status
                                                            │ queue
                                                            ▼
                                              Serverless workers: runpod-demo image
                                              CMD python worker/handler.py

 batch/launch_pod.py ── REST https://rest.runpod.io/v1/pods ──►  GPU Pod: same image
                                                                 start cmd: batch/embed_corpus.py
                                                                 /workspace = network volume
                                                                   corpus.jsonl ─► embeddings/shard_*.npy
                                                                 then DELETE /v1/pods/{own id}
```

## Layout

| Path | What |
|---|---|
| `worker/embedder.py` | model load, device choice, input validation; shared by handler and batch job |
| `worker/handler.py` | Serverless handler. Loads the model at import (once per worker), returns embeddings plus timing fields that show whether the worker was cold |
| `worker/download_model.py` | bakes the model into the image at a pinned Hugging Face revision |
| `Dockerfile` | `python:3.12-slim` + torch 2.14.1 (cu126 wheels) + model. One image, two start commands |
| `app/runpod_client.py` | async client for `/run`, `/runsync`, `/status`, `/health`; `RUNPOD_MODE=local` points it at the SDK's local server |
| `app/main.py`, `app/static/index.html` | FastAPI front: `/v1/embed`, `/v1/search`, `/v1/jobs`, `/healthz`, and a one-page demo UI |
| `batch/embed_corpus.py` | Pod batch job: JSONL in, float16 `.npy` shards out, resumable, optional self-delete |
| `batch/launch_pod.py` | creates the Pod through the REST API (or prints the request with `--dry-run`) |
| `batch/make_sample_corpus.py` | deterministic synthetic corpus for throughput tests |
| `bench/benchmark.py` | cold/warm latency against a local process, a local container, or a Runpod endpoint |
| `bench/cost.py`, `bench/prices.json` | cost formulas and the prices they use, with source URLs and fetch date |
| `tests/` | 39 pytest tests: handler, client and front (fake Runpod API), cost maths, batch job, and an end-to-end test through the local handler server |

## Running it locally

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

# Handler, once, with a test input (the runpod SDK's local test mode)
.venv/bin/python worker/handler.py --test_input '{"input": {"texts": ["hello world"], "kind": "query"}}'

# Handler as a local API (/run, /runsync, /status), then the front in local mode
.venv/bin/python worker/handler.py --rp_serve_api --rp_api_port 8010
RUNPOD_MODE=local RUNPOD_LOCAL_URL=http://127.0.0.1:8010 .venv/bin/uvicorn app.main:app --port 8090
open http://localhost:8090

# Tests
.venv/bin/python -m pytest -q

# Batch job on a synthetic corpus
.venv/bin/python batch/make_sample_corpus.py --n 5000 --out data/sample_corpus.jsonl
.venv/bin/python batch/embed_corpus.py --input data/sample_corpus.jsonl --out out/sample --shard-size 1000

# Benchmarks
.venv/bin/python bench/benchmark.py local-process --cold-runs 5 --warm 100
docker buildx build --load --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t runpod-demo:cpu .
.venv/bin/python bench/benchmark.py local-docker --image runpod-demo:cpu --cold-runs 5 --warm 100
```

Without `RUNPOD_MODE=local`, the front refuses to start unless `RUNPOD_API_KEY` and
`RUNPOD_ENDPOINT_ID` are set, and says which one is missing. `bench/benchmark.py runpod`
and `batch/launch_pod.py` (without `--dry-run`) fail the same way.

## Measured locally

Measured on 2026-10-04 on an Apple M2 laptop (8 cores, 16 GB RAM, macOS) that was also
running other applications. CPU only: no GPU was involved in any of these numbers. Docker
Desktop's VM had 1 vCPU and 2 GB RAM. Raw results are in `bench/results/`.

**Tests.** `pytest -q`: 39 passed in 17 s.

**Image sizes** (`docker image inspect`, and `docker save | gzip -6 | wc -c` as a stand-in
for the pull size, since registries store gzip layers):

| Image | Platform | Size | gzip | Build time |
|---|---|---|---|---|
| `runpod-demo:gpu` (torch cu126) | linux/amd64 | 7.51 GB | 4.00 GB | 296 s |
| `runpod-demo:cpu` (torch cpu) | linux/arm64 | 1.63 GB | 0.51 GB | 131 s |

Layers of the GPU image: torch plus the CUDA libraries it pulls in 6.68 GB, the other
Python dependencies 569 MB, the model 135 MB, the code under 25 kB. The model is 2% of
the image; torch and its CUDA libraries are 89%. Inside that layer: torch 1.7 GB, cuDNN
1.0 GB, cuBLAS 573 MB, cuSPARSELt 432 MB, NCCL 377 MB, cuSPARSE 281 MB, cuFFT 269 MB,
cuSOLVER 233 MB, NVSHMEM 195 MB (`du -sh` inside the container).

**Cold start and warm latency.** `bench/benchmark.py`. "Cold" is the time from starting
the process (or `docker run`) to the first successful `/runsync` response. `imports` is
`from sentence_transformers import SentenceTransformer` (which loads torch and
transformers), and `model load` is reading the weights plus one warm-up forward pass,
both reported by the handler itself. Warm numbers are sequential requests to the same
worker through the SDK's local HTTP server.

| | Local process (8 cores) | Docker, CPU image (1 vCPU) |
|---|---|---|
| Cold, 5 runs | 3.78 to 5.33 s (median 3.89) | 3.77 to 5.41 s (median 4.08) |
| of which imports | 3.02 to 4.20 s | 2.86 to 4.12 s |
| of which model load | 0.14 to 0.37 s | 0.11 to 0.23 s |
| Warm, 1 query, client p50 / p90 (n=100) | 13.7 / 14.8 ms | 27.6 / 28.9 ms |
| Warm, 1 query, handler inference p50 | 11.9 ms | 22.2 ms |
| Warm, 32 passages, client p50 / p90 (n=20) | 97.9 / 101.6 ms | 587.6 / 635.5 ms |

In both cases the first cold run is the slowest by about 1.5 s; the later ones benefit
from the OS file cache already holding torch's shared libraries.

What this says for Runpod, and what it does not: for a model this small, loading the
weights is not the cold-start cost. Importing torch is, and on Runpod the image pull
(about 4 GB) comes before either. Those are the parts that FlashBoot and host-level image
caching address. The local cold start excludes image pull, container scheduling and GPU
initialisation, so it is a lower bound for the in-container part only.

The GPU image also runs on the laptop under amd64 emulation, on CPU: one cold start took
11.0 s (imports 10.2 s, emulated). With `REQUIRE_CUDA=1` it exits at startup with
`RuntimeError: REQUIRE_CUDA=1 but torch sees no GPU (torch 2.14.1+cu126, built for CUDA 12.6)`,
which is what a worker on a host with the wrong driver should do.

**Batch job.** 5,000 synthetic documents (61 tokens on average, 34 to 92) in shards of
1,000. Three runs on the laptop CPU: 107 to 141 documents/s while encoding. Rerunning on
a finished output directory skips all 5 shards in 4.9 s (model load included). Output is
float16, 768 bytes per document.

**Front.** With the front and the local handler both running under uvicorn, a `/v1/search`
call (one query job and one 3-document job, sent concurrently) took 37 ms and 50 ms per
job as seen by the front.

## Numbers that only exist after deploying

Every cell below is empty because it needs a Runpod endpoint or Pod. Each row lists the
command (from [DEPLOY.md](DEPLOY.md)) that produces it; results land in `bench/results/`.

| Measurement | Value | Command |
|---|---|---|
| First request on a fresh endpoint (image pull + start + load), client time | TODO | `curl .../runsync` from DEPLOY.md step 4, with `time` |
| Cold start, FlashBoot on: client p50 over 5 runs; Runpod `delayTime` p50 | TODO | `python bench/benchmark.py runpod --cold-runs 5 --warm 100 --gpu-class 16GB` |
| Cold start, FlashBoot off: same | TODO | PATCH `flashboot:false`, then `python bench/benchmark.py runpod --cold-runs 5 --warm 20 --out bench/results/runpod-noflashboot.json` |
| Share of "cold" runs that were FlashBoot revivals (`first_request_on_worker=false` with a large `worker_age_s`) | TODO | same output file, `cold[*]` |
| Warm, 1 query: client p50 / p90, Runpod `executionTime` p50 | TODO | same as row 2, `warm.query_1` |
| Warm, 32 passages: client p50 / p90, `executionTime` p50 | TODO | same, `warm.passages_32` |
| GPU the workers landed on | TODO | `runpodctl serverless get $RUNPOD_ENDPOINT_ID --include-workers` |
| Cost per 1k queries, busy worker / every request cold | TODO | same as row 2, `cost.*` |
| Batch: docs/s on an RTX A5000 Pod, 200k synthetic docs | TODO | `python batch/launch_pod.py ... --sample-docs 200000`, then read `manifest.json` (DEPLOY.md step 7) |
| Batch: Pod wall time incl. pull and scheduling, and its cost | TODO | output of the same `launch_pod.py` run |

I expect FlashBoot on/off to be the largest difference in the table and the GPU type to
barely matter for a 33M-parameter model, but that is a guess until the rows are filled.

## Cost model

Prices fetched on 2026-10-04 from [runpod.io/pricing](https://www.runpod.io/pricing)
(page says "Updated September 27, 2026") and the per-second table in the
[endpoint settings docs](https://docs.runpod.io/serverless/endpoints/endpoint-configurations).
All copied into `bench/prices.json`.

| | Price |
|---|---|
| Serverless flex, 16 GB class (A4000, A4500, RTX 4000, RTX 2000) | $0.58/hr = $0.000161/s |
| Serverless flex, 24 GB class (L4, A5000, 3090) | $0.69/hr = $0.000192/s |
| Serverless active workers | not published; "discounts available through sales inquiry" |
| Pod, RTX A5000 (as listed on the pricing page) | $0.27/hr |
| Network volume | $0.07/GB/month under 1 TB |

Billing rules from the [Serverless pricing docs](https://docs.runpod.io/serverless/pricing):
a worker is billed from start until it fully stops, rounded up to the second, and that
includes start time, execution and the idle timeout (default 5 s).

`bench/cost.py` has the formulas. With `t` = warm execution time, `c` = cold start,
`i` = idle timeout and `p` = $/s:

- **busy worker** (traffic keeps workers occupied): `1000 × t × p` per 1k requests
- **every request cold** (sparse traffic, each request wakes a worker): `1000 × ceil(c + t + i) × p`
- **one active worker**: `2,592,000 s × p × discount` per month, which is $417.60 at the
  16 GB flex rate with no discount
- **break-even** for one active worker vs. all-cold flex: `86,400 × p × discount / (ceil(c + t + i) × p)` requests per day

The two bounds are far apart. To show how far, take the local CPU timings, which are
not Runpod numbers. With `t` = 13.7 ms, a busy 16 GB worker costs $0.0022 per 1k queries. If every query instead
arrives alone and wakes a worker with a 4 s cold start and the 5 s idle timeout, each one
bills 10 s, which is $1.61 per 1k, about 700 times more. At sparse traffic the bill is
mostly cold start plus idle timeout, and execution time barely shows. The real `c` will
be measured in the table above. The break-even for a 10 s session is 8,640 requests/day
at an undiscounted active price, i.e. one request every 10 s. Below that, flex with
scale-to-zero is cheaper. Above it, an always-on worker costs less, and it also removes
the cold start.

For the batch side, the listed prices alone make the case: an A5000 Pod at $0.27/hr
against Serverless 24 GB flex at $0.69/hr is 2.6 times cheaper per GPU-second, assuming
the two prices are for comparable tiers (the pricing page doesn't say which Pod tier the
$0.27 is). A long batch keeps its GPU busy, so it doesn't need scale-to-zero.

## Model in the image vs. network volume vs. cached models

| | Baked into image (this repo) | Network volume | Runpod cached model |
|---|---|---|---|
| Where the weights come from at start | image layers, already on the host after the pull | `/runpod-volume`, over the data center network | host-local cache at `/runpod-volume/huggingface-cache/hub/...` |
| Cold start cost | larger pull on a host that hasn't seen the image; then local disk | smaller image, but every start reads weights over the network | Runpod says faster than network volume, and download time isn't billed |
| Regions | any data center with the GPU | **only the volume's data center** | any, for models on Hugging Face |
| Model update | new image tag, rolling release | copy new weights to the volume, no rebuild | change the model reference |
| Fits | small models; models that aren't on Hugging Face | large or frequently swapped models, shared datasets | public, gated or private Hugging Face models |

For this model the choice is easy: 135 MB is 2% of a 7.5 GB image, so baking it in adds
nothing measurable to the pull and avoids tying the endpoint to one data center. For a
20 GB LLM the comparison changes: the weights would dominate the image, and cached models
or a network volume would be the better fit. The network-volume row is from the
[endpoint settings docs](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)
("adds network latency and restricts your endpoint to the volume's data center"). The
cached-models row is from the [cached models docs](https://docs.runpod.io/serverless/endpoints/model-caching).
I have not measured either on Runpod.

The cheaper way to shrink cold starts here is the image itself. Most of its 6.7 GB
torch layer is libraries a single-GPU inference job does not call: NCCL and NVSHMEM are
for multi-GPU communication, cuFFT and cuSOLVER for FFTs and linear solvers. Options, none tried yet: build on a Runpod base
image that hosts may already have cached, or export the model to ONNX and serve it with
`onnxruntime-gpu`, which needs a much smaller set of CUDA libraries.

## How a customer would adopt this

### 1. Pick the product per workload

| If the workload is... | Use | Because |
|---|---|---|
| Spiky or low-volume requests, latency of a few seconds on the first request is acceptable | Serverless, flex workers only (`workersMin=0`) | nothing billed while idle |
| User-facing requests where the first request after a quiet period must also be fast | Serverless with 1 active worker, flex workers above it for peaks | the active worker absorbs the cold start; peaks still scale |
| Steady traffic that keeps a GPU busy most of the day | active workers sized to the base load, or a Pod if they also want to manage the server | a busy GPU doesn't benefit from scale-to-zero |
| Large one-off or nightly jobs (re-embedding a corpus, offline scoring) | a Pod for the duration of the job, deleted at the end | cheapest $/GPU-second in the price list above; no request size or timeout limits |
| Jobs submitted by users that take minutes | Serverless `/run` + webhook or `/status` polling | queue, retries and per-job billing without running a scheduler |

The break-even formula above turns the second and third rows into a number for the
customer's own traffic. Bring their request logs to the conversation.

### 2. Endpoint settings that matter for this model

- **FlashBoot**: on (default). It keeps worker state after scale-down so a revived worker
  starts faster. It helps most when workers cycle often, which is the pattern of a
  search box. The benchmark's `first_request_on_worker` and `worker_age_s` fields show
  whether a "cold" request got a revived worker or a fresh one.
- **Idle timeout**: 5 s default. Every request that wakes a worker pays for this time,
  so raising it to 30 s or 60 s costs more per isolated request but turns bursts of
  queries into warm requests. Pick it from the gap between requests in the customer's logs.
- **Scaler**: `QUEUE_DELAY` (default 4 s) waits before adding workers, which is too
  slow for an interactive search box. `REQUEST_COUNT` with value 1 adds a worker per
  queued request.
- **Max workers**: a cost cap as much as a capacity setting. Runpod suggests about 20%
  above expected peak concurrency.
- **GPU types**: list two or three classes. This model needs under 1 GB of VRAM, so the
  cheapest class with availability wins. Availability matters more than speed here.
- **Execution timeout**: 30 s rather than the 10 min default. A stuck embedding call
  should fail fast and free the worker.
- **Data centers**: restrict only for data residency (e.g. an EU customer), because it
  shrinks the GPU pool. A network volume restricts it too.

### 3. Rollout

1. Build and push the image with a version tag. Deploy with `workersMin=0`.
2. Run `bench/benchmark.py runpod` and fill in the table above with their real cold start.
3. Point a copy of production traffic at the front (or replay logs), look at
   `/health` (`inQueue`, `throttled`) and at `delayTime` vs `executionTime`.
4. Decide on active workers using the break-even number and their latency target.
5. Embed the corpus with the Pod job using the same image tag, so query and corpus
   vectors come from the same build.

### 4. Failure modes to plan for

| Symptom | Likely cause | What to do |
|---|---|---|
| First request after a quiet period takes tens of seconds | image pull on a host that hasn't cached the 4 GB image, or a FlashBoot miss | active worker, smaller image, longer idle timeout |
| Requests sit `IN_QUEUE`, `/health` shows `throttled` workers | no capacity for the selected GPU type in the allowed data centers | add GPU classes, allow more data centers |
| Worker runs but is slow; `timing.device` is `cpu` | host driver older than the image's CUDA build | set allowed CUDA versions on the endpoint; `REQUIRE_CUDA=1` makes the worker fail visibly instead |
| `/runsync` returns `IN_PROGRESS` instead of a result | the job took longer than the sync wait (common on cold starts) | poll `/status` (the client here does) or use `/run` + webhook |
| Results gone when polled later | results are kept 30 min after `/run`, 1 min after `/runsync` | fetch promptly, or use a webhook |
| Job deleted mid-run, status returns 404 | job TTL (default 24 h) counts queue time too | set `policy.ttl` for long queues |
| Endpoint stops scaling after a quiet week | Runpod lowers max workers after 3 and 7 days without requests | raise max workers again; monitor it |
| Batch Pod keeps billing after the job | the container exited but the Pod was not stopped or deleted | the job deletes its own Pod; `launch_pod.py` stops it after `--max-minutes` as a backstop |
| Bills higher than expected at low traffic | idle timeout and cold starts billed per wake-up | use the "every request cold" formula, not the busy-worker one, for sparse traffic |

## Limits

- No Runpod measurements yet. Every local latency is CPU on a laptop; GPU latency, image
  pull time and FlashBoot behaviour are unknown until the TODO table is filled.
- The GPU image was built and run under emulation on CPU only. It has not touched a GPU.
- GPU type IDs and some `runpodctl` flags in DEPLOY.md come from the docs, not from a
  working account. `runpodctl gpu list` will tell.
- The batch corpus is synthetic text, used only to time the job. The retrieval
  quality of a 33M-parameter model is limited: for the demo page's default query ("How
  do I avoid slow first requests?") the idle-timeout sentence ranks first (0.67) and the
  FlashBoot sentence second (0.59), with little separation from the third (0.55). This
  repo makes no claim about retrieval quality.
- The front has no authentication or rate limiting. It must not be exposed publicly as is.
- The active-worker price is not published, so active-worker costs use the flex rate.
- The Pod self-delete needs an API key inside the Pod. That key is visible to anything
  running in the container.
