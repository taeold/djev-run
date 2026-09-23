# djev-run

Serve DiffusionGemma-Jev (`djev`) on a TypeSafe AI compatible API on Cloud Run
with an NVIDIA RTX PRO 6000 Blackwell GPU. Built on
[`mmastrac/djev`](https://github.com/mmastrac/djev) and
[`mmastrac/djev-spark`](https://github.com/mmastrac/djev-spark)
([`@mmastrac`](https://x.com/mmastrac/status/2100373761195401724),
[`vllm#57250`](https://github.com/vllm-project/vllm/pull/57250),
[`vllm#58216`](https://github.com/vllm-project/vllm/pull/58216),
[`vllm#58226`](https://github.com/vllm-project/vllm/pull/58226)), with built-in
browser demos inspired by
[`mizorewww/laya-coreml`](https://github.com/mizorewww/laya-coreml) (`/snake`),
[`virajbhartiya/laya-vs-jev`](https://github.com/virajbhartiya/laya-vs-jev)
(`/dino`), and [`trungdq88/jev-tetris`](https://github.com/trungdq88/jev-tetris)
(`/tetris`).

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
# TEST_PAGE=1: enables the built-in /snake, /dino, /tetris, and /playground web UIs
# COPY_TO_SHM=1: stages the 17.5 GB model into /dev/shm RAM in the background while Python imports torch/vllm
# VLLM_WORKER_MULTIPROC_METHOD=fork: forks EngineCore from APIServer without re-importing Python
# ENFORCE_EAGER=1 & TORCH_COMPILE_DISABLE=1: skips torch.compile, CUDA graph capture, and redundant startup profiling
# DISABLE_MM=1: skips SigLIP vision/video encoder profiling for text-only evaluation
```

### Step 3: Play the Built-in Snake, Chrome Dino, and Tetris Demos (Zero Dependencies)

`snake.html` (`/snake`), `dino.html` (`/dino`), and `tetris.html` (`/tetris`) are
standalone HTML files with zero external dependencies. They call
`POST /v1/systemone` directly from the browser via `fetch()`:

-   **Snake Arena (`/snake`)**: Open `https://<your-cloud-run-url>/snake` in
    your browser (12x12 grid with flood-fill safety analysis and live move
    probability bars).
-   **Chrome T-Rex Dino Arena (`/dino`)**: Open
    `https://<your-cloud-run-url>/dino` in your browser (`unassisted` raw
    `/v1/systemone` mode with 3-way pipelined in-flight requests and
    `model + live shield` mode, 2x HiDPI Chromium sprites, exact two-stage
    pixel-box collision checks, and unthrottled 60 FPS Web Worker physics loop).
-   **Tetris Arena (`/tetris`)**: Open `https://<your-cloud-run-url>/tetris` in
    your browser (10x20 Tetris board adapted from
    [`trungdq88/jev-tetris`](https://github.com/trungdq88/jev-tetris) with
    single-pass 4-question speculative fan-out over `placement`, `strategy`,
    `board_health`, and `next_piece_fits`, plus next-piece `/v1/systemone`
    prefetching for 0 ms inter-piece wait at 60 FPS).
-   **Extractive Spans (`span` / `spans`), Constrained Readout & Fused Sampler**:
    Built on [`mmastrac/djev`](https://github.com/mmastrac/djev)
    (`span-answer-type`),
    [`vllm-project/vllm#58216`](https://github.com/vllm-project/vllm/pull/58216)
    (`diffusion_constrained` + `diffusion_pinned`), and
    [`vllm-project/vllm#58226`](https://github.com/vllm-project/vllm/pull/58226)
    (one-pass Triton `_row_stats_kernel` sampler, `49/49` `span_battery.py` in
    `10.7s`).

--------------------------------------------------------------------------------

## Use with Vercel AI SDK (Optional)

Because `djev-run` implements the `/v1/systemone` endpoint contract (`noul` /
`boolean`, `choice`, `score`, `span`, and `spans`), you can also point
`@ai-sdk/typesafe-ai` at your Cloud Run URL from Node.js or TypeScript
(`index.ts`):

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
```

```bash
npm install
CLOUD_RUN_URL="https://<your-cloud-run-url>" TYPESAFE_AI_API_KEY="$(gcloud auth print-identity-token)" npm start
```

--------------------------------------------------------------------------------

## Performance & JevBench v1.3.0 (`N = 231`)

-   **Cold start from zero instances**: **47.5s** (`29.7s` engine init with
    `VLLM_ENABLE_V1_MULTIPROCESSING=0` in-process vLLM engine + 32-worker
    parallel rootfs prefetch + 2-shard background `/dev/shm` streaming, down
    from `4m 05s` baseline)
-   **Single-request latency**: **~61 ms server / ~116 ms WAN RTT** (`steps=1, samples=1`;
    **~121 ms p50 WAN RTT** with `samples="auto"`)
-   **JevBench v1.3.0 (`N = 231` public suite, `diffusion_constrained=True`)**:
    -   **Fast Mode (`steps=1, samples=1`)**: **81.39% Overall Accuracy**
        (**100.00% Easy**, **95.83% Standard**, **63.96% Hard**), **0.2739 Mean
        Brier**, **0.0963 ECE**, **80.74 Calibration Score**, **79.29
        Intelligence Score**, **77.34 Composite Score**.
    -   **Auto Mode (`steps=1, samples="auto"`)**: **81.82% Overall Accuracy**
        (**100.00% Easy**, **95.83% Standard**, **64.86% Hard**), **0.2687 Mean
        Brier**, **0.0989 ECE**, **80.22 Calibration Score**, **79.70
        Intelligence Score**, **77.01 Composite Score**.

--------------------------------------------------------------------------------

## Pricing

1 NVIDIA RTX PRO 6000 GPU (20 vCPU, 80 GiB RAM) costs $3.19 per hour while
active and scales to $0 when idle with `--min-instances=0`.
