# Deploy guide

Steps to put this on Runpod yourself. Nothing here has been run against Runpod yet: there
is no account or API key behind this repo. API shapes and flags are taken from
docs.runpod.io as of 2026-10-04 (links inline). Where I was not able to confirm something,
it says so.

Costs: a Serverless endpoint with 0 active workers costs nothing while no worker is
running. A network volume bills for storage until it is deleted. Every step below that
starts a worker or a Pod bills per second. The account
default spend limit is $80/hour ([pricing docs](https://docs.runpod.io/serverless/pricing)).

## 0. Prerequisites

- A Runpod account with credit, and an API key (console → Settings → API Keys).
- A container registry you can push to (Docker Hub used below).
- `docker` with `buildx`, `curl`, `jq`. Optional: `runpodctl`
  (`brew install runpod/runpodctl/runpodctl`, [docs](https://docs.runpod.io/runpodctl/overview)),
  and the AWS CLI for reading files off a network volume.

```sh
export RUNPOD_API_KEY=...            # never commit this
export DOCKER_USER=...               # your Docker Hub user
export IMAGE=docker.io/$DOCKER_USER/runpod-demo:0.1.0
```

## 1. Build and push the image

Runpod hosts are x86_64, so build for `linux/amd64` even on an Apple Silicon laptop.

```sh
docker buildx build --platform linux/amd64 -t $IMAGE --push .
```

Use a version tag, not `latest`. Workers cache images by tag, and a new tag is the
reliable way to roll out a change (Runpod also does [rolling releases](https://docs.runpod.io/serverless/endpoints/rolling-releases)
when the template's image changes).

Optional check before pushing: run the GPU image's CPU fallback locally.
```sh
docker run --rm --platform linux/amd64 runpod-demo:gpu \
  python /app/worker/handler.py --test_input '{"input":{"texts":["hello"]}}'
```

## 2. Create a Serverless template

REST ([reference](https://docs.runpod.io/api-reference/templates/POST/templates)):

```sh
TEMPLATE_ID=$(curl -s https://rest.runpod.io/v1/templates \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d "{\"name\":\"runpod-demo-embed-0.1.0\",\"imageName\":\"$IMAGE\",\"isServerless\":true,
       \"containerDiskInGb\":10,\"env\":{\"REQUIRE_CUDA\":\"1\"}}" | jq -r .id)
echo $TEMPLATE_ID
```

or `runpodctl template create --name runpod-demo-embed-0.1.0 --image $IMAGE --serverless --container-disk-in-gb 10`
(then add `REQUIRE_CUDA=1` as an environment variable in the console).

`REQUIRE_CUDA=1` makes the worker exit at startup if PyTorch cannot see a GPU, instead of
quietly serving from CPU at a fraction of the speed while billing GPU seconds.

A Serverless template can be bound to only one endpoint at a time
([runpodctl docs](https://docs.runpod.io/runpodctl/reference/runpodctl-serverless)).

## 3. Create the endpoint

Settings and why ([endpoint settings reference](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)):

| Setting | Value | Reason |
|---|---|---|
| GPU | 16 GB class first, 24 GB class as fallback | the model needs < 1 GB VRAM; a fallback improves availability |
| Active workers (`workersMin`) | 0 | scale to zero; change only after measuring cold starts |
| Max workers | 3 | cost cap; raise with traffic |
| Idle timeout | 5 s (default) | billed while idle; longer keeps workers warm for bursty traffic |
| Scaler | `QUEUE_DELAY`, 4 s | default; `REQUEST_COUNT` with value 1 reacts faster for short requests |
| FlashBoot | on | default for new endpoints |
| Execution timeout | 30 s | an embedding call that runs 30 s is broken; fail it |
| Data centers | EU only if the customer needs EU data residency; otherwise all | restricting shrinks the GPU pool |
| Allowed CUDA versions | 12.6 and newer | image uses cu126 torch wheels |

REST ([reference](https://docs.runpod.io/api-reference/endpoints/POST/endpoints)). GPU type
ids must match `runpodctl gpu list` output; the ones below are what I expect the 16 GB and
24 GB classes to be called, not verified:

```sh
ENDPOINT_ID=$(curl -s https://rest.runpod.io/v1/endpoints \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d "{\"name\":\"runpod-demo-embed\",\"templateId\":\"$TEMPLATE_ID\",
       \"gpuTypeIds\":[\"NVIDIA RTX A4000\",\"NVIDIA RTX A4500\",\"NVIDIA RTX A5000\",\"NVIDIA L4\"],
       \"gpuCount\":1,\"workersMin\":0,\"workersMax\":3,\"idleTimeout\":5,
       \"scalerType\":\"QUEUE_DELAY\",\"scalerValue\":4,\"flashboot\":true,
       \"executionTimeoutMs\":30000,\"allowedCudaVersions\":[\"12.6\",\"12.7\",\"12.8\",\"12.9\",\"13.0\"]}" \
  | jq -r .id)
export RUNPOD_ENDPOINT_ID=$ENDPOINT_ID
```

or `runpodctl serverless create --name runpod-demo-embed --template-id $TEMPLATE_ID --gpu-id "NVIDIA RTX A4000" --workers-min 0 --workers-max 3`
(check `runpodctl serverless create --help` for the idle-timeout / FlashBoot flags).

Console route: Serverless → New Endpoint → Docker image → paste `$IMAGE` → choose 16 GB
and 24 GB GPUs → set the values from the table → Deploy.

## 4. Smoke test

```sh
curl -s https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/health -H "Authorization: Bearer $RUNPOD_API_KEY" | jq
time curl -s https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"texts":["hello world"],"kind":"query"}}' | jq '{status, delayTime, executionTime, timing: .output.timing}'
```

The first call starts a worker: expect the image pull (about 4 GB gzip-compressed, see README)
on top of model load. `output.timing.device` should say `cuda`. With `REQUIRE_CUDA=1` a
worker that cannot see a GPU (for example a driver older than the CUDA build) fails at
startup and shows in the worker logs, rather than answering slowly from CPU.

## 5. Run the front against the endpoint

```sh
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --port 8080      # reads RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID
open http://localhost:8080
```

## 6. Fill the "after deploy" table in the README

Each row of that table names its command. All of them write a JSON file to `bench/results/`.

```sh
# cold start (FlashBoot on), warm latency, cost per 1k
.venv/bin/python bench/benchmark.py runpod --cold-runs 5 --warm 100 --gpu-class 16GB

# same with FlashBoot off: edit the endpoint, then rerun with a different output file
curl -s -X PATCH https://rest.runpod.io/v1/endpoints/$RUNPOD_ENDPOINT_ID \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' -d '{"flashboot":false}'
.venv/bin/python bench/benchmark.py runpod --cold-runs 5 --warm 20 --out bench/results/runpod-noflashboot.json
```

Each cold run waits until `/health` shows no running or idle workers, so with the default
5 s idle timeout a 5-run benchmark takes a few minutes. Turn FlashBoot back on afterwards.

## 7. Pod batch job

Network volumes are tied to one data center. Pick one that also has the S3-compatible API
so results can be read without starting a Pod; EU-RO-1 does
([S3 API docs](https://docs.runpod.io/storage/s3-api)).

```sh
runpodctl network-volume create --name runpod-demo-data --size 10 --data-center-id EU-RO-1
export VOLUME_ID=...   # from the output, or runpodctl network-volume list

# Dry run prints the Pod request without sending it
.venv/bin/python batch/launch_pod.py --image $IMAGE --network-volume-id $VOLUME_ID \
  --data-center-id EU-RO-1 --gpu "NVIDIA RTX A5000" --price-per-hr 0.27 \
  --sample-docs 200000 --dry-run

# Real run: generates a 200k-doc synthetic corpus on the volume, embeds it, deletes its own Pod
.venv/bin/python batch/launch_pod.py --image $IMAGE --network-volume-id $VOLUME_ID \
  --data-center-id EU-RO-1 --gpu "NVIDIA RTX A5000" --price-per-hr 0.27 --sample-docs 200000
```

For real data, upload a JSONL file (`{"id": ..., "text": ...}` per line) to the volume
first and drop `--sample-docs`:

```sh
aws s3 cp corpus.jsonl s3://$VOLUME_ID/corpus.jsonl --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io
```

Read the result:

```sh
aws s3 cp s3://$VOLUME_ID/embeddings/manifest.json - --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io
```

The job passes your API key into the Pod's environment so it can delete its own Pod. If
your account offers restricted keys, create one limited to Pods for this. If the delete
fails, `launch_pod.py` stops the Pod after `--max-minutes`.

## 8. Clean up

```sh
curl -s -X DELETE https://rest.runpod.io/v1/endpoints/$RUNPOD_ENDPOINT_ID -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -s -X DELETE https://rest.runpod.io/v1/templates/$TEMPLATE_ID -H "Authorization: Bearer $RUNPOD_API_KEY"
runpodctl network-volume delete $VOLUME_ID     # the volume bills $0.07/GB/month until deleted
```

An endpoint left without requests is scaled down by Runpod after 3 days (max workers 2)
and 7 days (max workers 0).

## 9. Hosting the front on a p001.ai subdomain

The front is a stateless FastAPI app; any host that runs a Python container works. It
needs `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` as secrets. Before making it public, add
a per-IP rate limit and lower `MAX_TEXTS` / `MAX_SEARCH_DOCS` in `app/main.py`: every
request spends GPU seconds on the account.

## Option B: deploy with Runpod Flash (no Docker)

`flash/embed_worker.py` is the same endpoint written for [Runpod Flash](https://docs.runpod.io/flash/overview).
Flash packages the Python code and installs the dependencies on the worker, so there is no image
to build or push. The request and response shapes match the Docker worker. The trade-off: the
model downloads from Hugging Face on every cold worker (about 135 MB) instead of being baked in.

```sh
python3.12 -m venv .flash-venv && .flash-venv/bin/pip install runpod-flash==1.20.0
export RUNPOD_API_KEY=...            # or keep it in .env (git-ignored)
cd flash
../.flash-venv/bin/flash build --python-version 3.12   # local check: ~136 MB artifact, torch excluded
../.flash-venv/bin/flash deploy --python-version 3.12  # prints the endpoint id
```

Then point everything else at it: `export RUNPOD_ENDPOINT_ID=<id>` and run the smoke test and
`bench/benchmark.py runpod` as in the steps above. Tear down with `flash app delete <app>`.
