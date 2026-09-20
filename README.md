# djev-run

Serve DiffusionGemma-Jev (`djev`) on a TypeSafe AI compatible API on Cloud Run
with an NVIDIA RTX PRO 6000 Blackwell GPU. Built on
[Matt Mastracci (@mmastrac)'s `djev-spark`](https://github.com/mmastrac/djev-spark)
([announcement](https://x.com/mmastrac/status/2100373761195401724)).

```typescript
import { createTypeSafeAi } from '@ai-sdk/typesafe-ai';
import { experimental_evaluate, type Experimental_EvaluationModel } from 'ai';

const typeSafeAi = createTypeSafeAi({
  baseURL: 'https://<your-cloud-run-url>/v1',
});

async function triage(model: Experimental_EvaluationModel, message: string) {
  return experimental_evaluate({
    model,
    state: { message },
    questions: {
      department: {
        type: 'choice',
        instructions: 'Which team should handle this?',
        criteria: {
          billing: 'Payments and refunds',
          support: 'Other requests',
        },
      },
      severity: {
        type: 'score',
        instructions: 'How severe is the issue?',
        criteria: ['Cosmetic', 'Workaround exists', 'Blocking; no workaround'],
      },
      requestsRefund: {
        type: 'boolean',
        instructions: 'Is the customer requesting money back?',
      },
    },
  });
}

const result = await triage(
  typeSafeAi.evaluationModel('jev-latest'),
  'I was charged twice and my account is locked',
);
console.log(result);
// {
//   department: {
//     type: 'choice',
//     choice: 'billing',
//     probabilities: { billing: 0.9988, support: 0.0012 }
//   },
//   severity: {
//     type: 'score',
//     score: 1.9995,
//     probabilities: { '0': 0.0002, '1': 0.0001, '2': 0.9997 }
//   },
//   requestsRefund: {
//     type: 'boolean',
//     probability: 0.9928
//   }
// }
```

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

```bash
gcloud beta run deploy djev-dgemma \
  --region="${REGION}" \
  --image=ghcr.io/taeold/djev-run:latest \
  --gpu=1 \
  --gpu-type=nvidia-rtx-pro-6000 \
  --no-gpu-zonal-redundancy \
  --cpu=20 \
  --memory=80Gi \
  --no-cpu-throttling \
  --concurrency=32 \
  --min-instances=0 \
  --max-instances=1 \
  --port=8080 \
  --network=default \
  --subnet=default \
  --vpc-egress=all-traffic \
  --add-volume=name=weights,type=cloud-storage,bucket="${BUCKET}",readonly=false,mount-options=enable-buffered-read=true \
  --add-volume-mount=volume=weights,mount-path=/mnt/gcs \
  --startup-probe=httpGet.path=/health,httpGet.port=8080,initialDelaySeconds=5,periodSeconds=2,timeoutSeconds=2,failureThreshold=120 \
  --set-env-vars="MODEL=/mnt/gcs/dgemma,CANVAS=128,MAX_SEQS=32,MAX_MODEL_LEN=4096,GPU_UTIL=0.40,KV_CACHE_GB=2,ATTN=TRITON_ATTN,COPY_TO_SHM=1,ENFORCE_EAGER=1,DISABLE_MM=1,TORCH_COMPILE_DISABLE=1,VLLM_WORKER_MULTIPROC_METHOD=fork,VLLM_UF_EAGER_ALL=1,VLLM_FLASHINFER_MOE_BACKEND=masked_gemm,CUDA_MODULE_LOADING=LAZY"

# --image=ghcr.io/taeold/djev-run:latest: prebuilt from github.com/mmastrac/djev-spark (upstream does not publish a registry image)
# --no-gpu-zonal-redundancy: required for standard regional RTX PRO 6000 quota
# --no-cpu-throttling: keeps all 20 vCPUs active during weight loading and vLLM scheduling
# --network=default --subnet=default --vpc-egress=all-traffic: streams weights from GCS over Google internal networking (~1.05 GiB/s)
# mount-options=enable-buffered-read=true: prefetches 18 GB safetensors shards sequentially from GCS
# COPY_TO_SHM=1: copies the 17.53 GiB model into /dev/shm RAM in the background while Python imports torch/vllm so safetensors mmap loads in 5.94s
# VLLM_WORKER_MULTIPROC_METHOD=fork: forks EngineCore from APIServer with torch/vllm already imported in RAM (saves 20s vs spawn)
# ENFORCE_EAGER=1 & TORCH_COMPILE_DISABLE=1: skips torch.compile, CUDA graph capture, and redundant startup profiling/autotuning (saves 116s)
# DISABLE_MM=1: passes --language-model-only --skip-mm-profiling to skip 51s of SigLIP vision/video encoder profiling
```

### Run the Sample Code

```bash
npm install
DJEV_BASE_URL="https://<your-cloud-run-url>/v1" TYPESAFE_AI_API_KEY="$(gcloud auth print-identity-token)" npm start
```

Open `snake.html` in a browser (or visit `https://<your-cloud-run-url>/`) and
paste your Cloud Run URL to run the live 1-step diffusion Snake demo.

--------------------------------------------------------------------------------

## Performance

-   **Warm single-step evaluation (`c=1, steps=1, samples=1`)**: **62-64 ms**
    server inference (`63.0 ms` mean, `62.5 ms` median; **163 ms** for
    `samples="auto"` with 4 parallel samples).
-   **Concurrent batch (`c=32, steps=1, samples=1`)**: **79-123 RPS** (`0.68s` -
    `0.81s` for 64 requests across 32 workers; **157-181 ms** per 32-request
    batch).
-   **Cold start (`53.5s` (`0m 53s`) vs. `4m 05s` (`245s`) baseline — 4.6x
    faster)**:
    -   **Parallel GCS to `/dev/shm` staging (`COPY_TO_SHM=1`, saves `16.6s` on
        critical path)**: Copies `config.json` and tokenizer files first
        (`0.05s`) and streams the 17.53 GiB `safetensors` shards into `/dev/shm`
        (`1.05 GiB/s`, finishes at `t=20.0s`) in the background while Python
        imports `torch` and `vllm`, followed by a `5.94s` `safetensors` mmap
        load from RAM into VRAM (`t=47.9s`).
    -   **`VLLM_WORKER_MULTIPROC_METHOD=fork` (saves `20.0s`)**: Forks
        `EngineCore` from `APIServer` after `torch`, `vllm`, `transformers`, and
        `flashinfer` are already imported in memory instead of spawning a second
        Python interpreter from scratch.
    -   **`DISABLE_MM=1` (saves `51.0s`)**: Skips SigLIP vision/video encoder
        profiling (`--language-model-only --skip-mm-profiling`).
    -   **`ENFORCE_EAGER=1` + `TORCH_COMPILE_DISABLE=1` + skipping redundant
        `profile_run()` / `kernel_warmup` (saves `104s`, cutting `init engine`
        from `73.1s` to `1.98s`)**: Because `--kv-cache-memory` (`2 GiB`) and
        `--enforce-eager` are explicitly configured for 1-step block diffusion
        (`steps=1`), skipping `torch.compile`, CUDA graph capture, dummy
        4096-token memory-profiling runs, and 21-bucket FlashInfer startup
        autotuning brings `init engine` down to `1.98s` and total container
        startup to **`53.5s`**.

--------------------------------------------------------------------------------

## Pricing

1 NVIDIA RTX PRO 6000 GPU (20 vCPU, 80 GiB RAM) costs $3.19 per hour while
active.
