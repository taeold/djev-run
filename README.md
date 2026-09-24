# djev-run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taeold/djev-run/blob/main/colab.ipynb)


Serve DiffusionGemma-Jev (`djev`) on Cloud Run
with an NVIDIA RTX PRO 6000 Blackwell GPU, built on
[mmastrac/djev](https://github.com/mmastrac/djev).

Built-in demo apps, inspired by:

- `/snake`: [mizorewww/laya-coreml](https://github.com/mizorewww/laya-coreml)
- `/dino`: [virajbhartiya/laya-vs-jev](https://github.com/virajbhartiya/laya-vs-jev)
- `/tetris`: [trungdq88/jev-tetris](https://github.com/trungdq88/jev-tetris)

<img width="640" height="360" alt="djev snake" src="https://github.com/user-attachments/assets/2e9a5321-f8a9-4734-b6f2-4d6f47193390" />

--------------------------------------------------------------------------------

## Deploy on Google Cloud Run

Follows
[Cloud Run GPU best practices](https://docs.cloud.google.com/run/docs/configuring/services/gpu-best-practices).

### Step 1: Upload Model to GCS

```bash
export BUCKET="your-gcs-bucket"
export REGION="us-central1" # Supported RTX PRO 6000 regions: us-central1, europe-west4, asia-southeast1, asia-south2

gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}"
hf download nvidia/diffusiongemma-26B-A4B-it-NVFP4 --local-dir /tmp/dgemma
gcloud storage cp -r /tmp/dgemma/* "gs://${BUCKET}/dgemma/"
```

### Step 2: Deploy to Cloud Run

You have two options for deployment:

#### Option A: Pre-built `djev-run` Container (Recommended)
Deploy the pre-built image with `/dev/shm` staging and all `vllm serve` flags baked in.
```bash
gcloud beta run deploy djev-dgemma \
  --region="us-central1" \
  --image=ghcr.io/taeold/djev-run:latest \
  --gpu=1 --gpu-type=nvidia-rtx-pro-6000 --no-gpu-zonal-redundancy \
  --cpu=20 --memory=80Gi --no-cpu-throttling \
  --concurrency=32 --min-instances=0 --max-instances=1 \
  --port=8080 \
  --network=default --subnet=default --vpc-egress=all-traffic \
  --add-volume=name=weights,type=cloud-storage,bucket="${BUCKET}",readonly=false,mount-options=enable-buffered-read=true \
  --add-volume-mount=volume=weights,mount-path=/mnt/gcs \
  --startup-probe=httpGet.path=/health,httpGet.port=8080,initialDelaySeconds=5,periodSeconds=2,timeoutSeconds=2,failureThreshold=120
```

#### Option B: Raw `vLLM` Nightly Container (Zero custom Dockerfile)
Directly deploy the upstream Docker container running pure OpenAI completions.
```bash
gcloud beta run deploy djev-dgemma \
  --region="us-central1" \
  --image=docker.io/vllm/vllm-openai:nightly \
  --gpu=1 --gpu-type=nvidia-rtx-pro-6000 --no-gpu-zonal-redundancy \
  --cpu=20 --memory=80Gi --no-cpu-throttling \
  --concurrency=32 --min-instances=0 --max-instances=1 \
  --port=8000 \
  --network=default --subnet=default --vpc-egress=all-traffic \
  --add-volume=name=weights,type=cloud-storage,bucket="${BUCKET}",readonly=false,mount-options=enable-buffered-read=true \
  --add-volume-mount=volume=weights,mount-path=/mnt/gcs \
  --startup-probe=httpGet.path=/health,httpGet.port=8000,initialDelaySeconds=5,periodSeconds=2,timeoutSeconds=2,failureThreshold=120 \
  --set-env-vars="VLLM_FLASHINFER_MOE_BACKEND=masked_gemm,VLLM_ENABLE_V1_MULTIPROCESSING=0" \
  --command="/bin/bash" \
  --args="-c","cp -r /mnt/gcs/dgemma /dev/shm/dgemma && exec vllm serve /dev/shm/dgemma --served-model-name djev-dgemma --allowed-origins '[\"*\"]' --trust-remote-code --enforce-eager --language-model-only --attention-backend TRITON_ATTN --kv-cache-memory 2G --max-num-seqs 32 --max-model-len 4096 --diffusion-config '{\"canvas_length\":128}' --override-generation-config '{\"max_new_tokens\":null}'"

# Cloud Run & Storage flags:
# --image=docker.io/vllm/vllm-openai:nightly: serves standard vLLM diffusion natively without a custom Dockerfile
# --no-gpu-zonal-redundancy: required for standard regional RTX PRO 6000 quota
# --no-cpu-throttling: keeps all 20 vCPUs active during weight loading and vLLM scheduling
# --network=default --subnet=default --vpc-egress=all-traffic: streams weights from GCS over Google internal networking (~1.05 GiB/s)
# mount-options=enable-buffered-read=true & cp -r to /dev/shm: prefetches 18 GB safetensors shards sequentially from GCS into RAM before vLLM starts
#
# vLLM cold-start & runtime flags (--set-env-vars / --args):
# VLLM_ENABLE_V1_MULTIPROCESSING=0: runs EngineCore in-process so Python/CUDA modules are not imported twice
# VLLM_FLASHINFER_MOE_BACKEND=masked_gemm: selects the low-latency FlashInfer MoE kernel on Blackwell SM120
# --enforce-eager: skips torch.compile and CUDA graph capture on startup
# --language-model-only: skips loading and profiling the unused SigLIP vision encoder
# --kv-cache-memory=2G: pre-allocates a fixed 2 GiB KV cache, skipping the startup memory-profiling forward pass
# --attention-backend=TRITON_ATTN: uses Triton bidirectional attention required by DiffusionGemma
# --diffusion-config='{"canvas_length":128}': configures the 128-token parallel diffusion canvas
```

### Step 3: Querying the Model

You have two options depending on which deployment you chose in Step 2:

#### Option A: Query with `/v1/systemone` (`ghcr.io/taeold/djev-run:latest`)
The pre-built container includes an ASGI middleware that natively parses the high-level `state` + `questions` JSON schema, calculates token slots dynamically, builds the 128-token canvas, invokes the internal vLLM generation endpoint, and normalizes the logprobs into concrete category and score outputs.

```bash
curl -s https://<your-cloud-run-url>/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Classify: Payment issues\nlabel:",
    "steps": 1,
    "questions": [
      {
        "id": "category",
        "type": "choice",
        "choices": [["Billing", "Billing"], ["Technical", "Technical"], ["Other", "Other"]],
        "labels": ["Billing", "Technical", "Other"]
      }
    ]
  }'
```

```json
{
  "answers": {
    "category": {
      "choice": "Billing",
      "probabilities": {
        "Billing": 0.85,
        "Technical": 0.10,
        "Other": 0.05
      },
      "confidence": 0.85
    }
  },
  "diagnostics": {
    "timing": {
      "total_ms": 116.3
    }
  }
}
```

#### Option B: Query with Standard vLLM Diffusion (`POST /tokenize` + `POST /v1/chat/completions`)
When using the raw `vllm-openai:nightly` container directly, you must manually pre-tokenize the prompt before invoking the completions schema explicitly passing the parallel `diffusion_seed_canvas` natively routing through `vllm_xargs`.

```bash
# 1. Tokenize query
curl -s https://<your-cloud-run-url>/tokenize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Classify: Payment issues\nlabel:"}'
```

```json
{"count":8,"max_model_len":4096,"tokens":[4335,1891,236787,35032,4342,107,2491,236787],"token_strs":null}
```

```bash
# 2. Diffusion Read
curl -s https://<your-cloud-run-url>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "djev-dgemma",
    "messages": [
      {"role": "user", "content": "Classify: Payment issues\nlabel:"}
    ],
    "max_tokens": 8,
    "vllm_xargs": {
      "diffusion_seed_canvas": [4335, 1891, 236787, 35032, 4342, 107, 2491, 236787, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
      "diffusion_pinned": [0, 1, 2],
      "diffusion_max_steps": 1,
      "diffusion_read_only": true
    }
  }'
```

```json
{
  "id": "chatcmpl",
  "object": "chat.completion",
  "created": 1727834,
  "model": "djev-dgemma",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": " Billing"}, "finish_reason": "length"}]
}
```
