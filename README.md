# djev-run

Serve DiffusionGemma-Jev (`djev`) on a TypeSafe AI compatible API on Cloud Run
with an NVIDIA RTX PRO 6000 Blackwell GPU. Built on
[Matt Mastracci (@mmastrac)'s DiffusionGemma structured evaluation mode](https://x.com/mmastrac/status/2100373761195401724).

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
  --startup-probe=httpGet.path=/health,httpGet.port=8080,initialDelaySeconds=15,periodSeconds=10,timeoutSeconds=5,failureThreshold=90 \
  --set-env-vars="MODEL=/mnt/gcs/dgemma,CANVAS=128,MAX_SEQS=32,MAX_MODEL_LEN=4096,GPU_UTIL=0.40,KV_CACHE_GB=2,ATTN=TRITON_ATTN,COPY_TO_SHM=1,ENFORCE_EAGER=1,DISABLE_MM=1,VLLM_UF_EAGER_ALL=1,VLLM_FLASHINFER_MOE_BACKEND=masked_gemm,VLLM_WORKER_MULTIPROC_METHOD=spawn,CUDA_MODULE_LOADING=LAZY"

# --no-gpu-zonal-redundancy: required for standard regional RTX PRO 6000 quota
# --no-cpu-throttling: keeps all 20 vCPUs active during weight loading and vLLM scheduling
# --network=default --subnet=default --vpc-egress=all-traffic: streams weights from GCS over Google internal networking (~1.05 GiB/s)
# mount-options=enable-buffered-read=true: prefetches 18 GB safetensors shards sequentially from GCS
# COPY_TO_SHM=1: copies the 17.53 GiB model into /dev/shm RAM in 16.6s so safetensors mmap loads in 5.97s
# ENFORCE_EAGER=1: passes --enforce-eager to vllm serve to skip torch.compile and 35-batch CUDA graph capture (saves 52s on cold start)
# DISABLE_MM=1: passes --language-model-only --skip-mm-profiling to skip SigLIP vision/video encoder profiling (saves another 30s on cold start)
```

`ghcr.io/taeold/djev-run:latest` is built from the `Dockerfile` in this repo
(`vllm/vllm-openai` patched with
[vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250)).

--------------------------------------------------------------------------------

## Performance

-   **Warm single-step evaluation (`steps=1`)**: **62-71 ms** (`65.9 ms` mean
    for `samples=1`; **160-168 ms** for `samples="auto"` with 4 parallel
    samples).
-   **Cold start (`2m 39s` (`159s`) vs. `4m 05s` (`245s`) baseline)**:
    -   **GCS to `/dev/shm` stage (`COPY_TO_SHM=1`)**: `16.6s` for 17.53 GiB
        (`1.05 GiB/s`), saturating Cloud Run's ~10 Gbps container network link,
        followed by `5.97s` `safetensors` weight loading from RAM.
    -   **`ENFORCE_EAGER=1` (saves `52s`, `220s` -> `168s`)**: Standard LLM
        deployments run `torch.compile` and record 35 CUDA graphs across batch
        sizes to save ~2 ms per token across 500 generated tokens. Because
        `djev` evaluates all questions in a single forward pass (`steps=1`),
        `ENFORCE_EAGER=1` skips `torch.compile` and CUDA graph capture while
        adding only ~3-5 ms per request.
    -   **`DISABLE_MM=1` (saves `30s`, `168s` -> `138s`)**: DiffusionGemma
        includes a SigLIP vision/video tower that vLLM profiles with 3 dummy
        video items at startup. Passing `--language-model-only
        --skip-mm-profiling` runs the model in text-only mode and cuts `init
        engine` from `73.1s` to `49.1s`.

--------------------------------------------------------------------------------

## Pricing

1 NVIDIA RTX PRO 6000 GPU (20 vCPU, 80 GiB RAM) costs $3.19 per hour while
active.
