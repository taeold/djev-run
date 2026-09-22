# djev-run

Serve DiffusionGemma-Jev (`djev`) on a TypeSafe AI compatible API on Cloud Run
with an NVIDIA RTX PRO 6000 Blackwell GPU. Built on
[`mmastrac/djev-spark`](https://github.com/mmastrac/djev-spark)
([`@mmastrac`](https://x.com/mmastrac/status/2100373761195401724)) with the
Snake demo inspired by
[`mizorewww/laya-coreml`](https://github.com/mizorewww/laya-coreml).

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
  --set-env-vars="MODEL=/mnt/gcs/dgemma,CANVAS=128,MAX_SEQS=32,MAX_MODEL_LEN=4096,GPU_UTIL=0.40,KV_CACHE_GB=2,ATTN=TRITON_ATTN,TEST_PAGE=1,COPY_TO_SHM=1,ENFORCE_EAGER=1,DISABLE_MM=1,TORCH_COMPILE_DISABLE=1,VLLM_WORKER_MULTIPROC_METHOD=fork,VLLM_UF_EAGER_ALL=1,VLLM_FLASHINFER_MOE_BACKEND=masked_gemm,CUDA_MODULE_LOADING=LAZY"

# --image=ghcr.io/taeold/djev-run:latest: prebuilt from github.com/mmastrac/djev-spark (upstream does not publish a registry image)
# --no-gpu-zonal-redundancy: required for standard regional RTX PRO 6000 quota
# --no-cpu-throttling: keeps all 20 vCPUs active during weight loading and vLLM scheduling
# --network=default --subnet=default --vpc-egress=all-traffic: streams weights from GCS over Google internal networking (~1.05 GiB/s)
# mount-options=enable-buffered-read=true: prefetches 18 GB safetensors shards sequentially from GCS
# TEST_PAGE=1: enables the built-in /snake and /playground web UIs
# COPY_TO_SHM=1: stages the 17.5 GB model into /dev/shm RAM in the background while Python imports torch/vllm
# VLLM_WORKER_MULTIPROC_METHOD=fork: forks EngineCore from APIServer without re-importing Python
# ENFORCE_EAGER=1 & TORCH_COMPILE_DISABLE=1: skips torch.compile, CUDA graph capture, and redundant startup profiling
# DISABLE_MM=1: skips SigLIP vision/video encoder profiling for text-only evaluation
```

### Step 3: Play the Built-in Snake Demo (Zero Dependencies)

`snake.html` is a standalone HTML file with zero dependencies (no Node.js,
`npm`, or AI SDK required). It calls `POST /v1/systemone` directly from the
browser via `fetch()` at ~15 moves/sec:

-   **Hosted on Cloud Run**: Open `https://<your-cloud-run-url>/snake` in your
    browser.
-   **Local file**: Open `snake.html` directly in a browser and set
    `CLOUD_RUN_URL` at the top of `<script>`.

--------------------------------------------------------------------------------

## Use with Vercel AI SDK (Optional)

Because `djev-spark` implements the `/v1/systemone` endpoint contract, you can
also point `@ai-sdk/typesafe-ai` at your Cloud Run URL from Node.js or
TypeScript (`index.ts`):

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

```bash
npm install
CLOUD_RUN_URL="https://<your-cloud-run-url>" TYPESAFE_AI_API_KEY="$(gcloud auth print-identity-token)" npm start
```

--------------------------------------------------------------------------------

## Performance

-   **Cold start from zero instances**: **~47.5s** (down from `4m 05s` baseline)
-   **Single-request latency**: **~35-60 ms** (`steps=1, samples=1`; **~160 ms**
    with `samples="auto"`)
-   **Batch throughput**: **~100-123 requests/sec** at `concurrency=32`

To reach a 47.5-second cold start on Cloud Run, the container streams the 17.5 GB
`safetensors` weights from GCS into `/dev/shm` RAM in the background (`1.05
GiB/s`) while Python imports `torch` and `vllm`, forks `EngineCore` from
`APIServer` (`VLLM_WORKER_MULTIPROC_METHOD=fork`) so modules are not imported
twice, disables unused SigLIP vision profiling (`DISABLE_MM=1`), and skips
`torch.compile`, CUDA graph capture, and redundant memory-profiling passes
(`ENFORCE_EAGER=1`, `TORCH_COMPILE_DISABLE=1`, `--kv-cache-memory`).

Empirical benchmarks on Cloud Run with the RTX PRO 6000 confirmed:
- Streaming over VPC egress from GCS FUSE into `/dev/shm` (~47.5s) outperforms baking weights into the container image (80-115s), avoiding heavy overlay filesystem read overhead and long image import operations.
- `--cpu-boost` is omitted because host CPU frequency scaling under boosted allocation slows down 20 vCPU container initialization on GPU nodes (73-75s vs 47.5s).

--------------------------------------------------------------------------------

## Pricing

1 NVIDIA RTX PRO 6000 GPU (20 vCPU, 80 GiB RAM) costs $3.19 per hour while
active and scales to $0 when idle with `--min-instances=0`.
