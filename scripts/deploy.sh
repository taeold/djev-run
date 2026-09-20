#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-danielylee-joonix}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-djev-dgemma}"
BUCKET="${BUCKET:-danielylee-run-mount}"
IMAGE="${IMAGE:-ghcr.io/taeold/djev-run:latest}"
# Supported GPU values on nvidia-rtx-pro-6000:
#   1    (96 GB VRAM, min 20 vCPU / 80Gi RAM) - GA default
#   0.5  (48 GB VRAM, min 10 vCPU / 40Gi RAM) - requires project in AllowVGPU_FeatureSettings.gcl
#   0.25 (24 GB VRAM, min 4 vCPU / 16Gi RAM; 8 vCPU / 32Gi max) - requires AllowVGPU + AllowQuarterVGPU
GPU="${GPU:-1}"

case "${GPU}" in
  0.5|500m)
    CPU="${CPU:-10}"
    MEMORY="${MEMORY:-40Gi}"
    COPY_TO_SHM="${COPY_TO_SHM:-0}"
    GPU_UTIL="${GPU_UTIL:-0.75}"
    GCLOUD_RUN=(gcloud alpha run)
    ;;
  0.25|250m)
    CPU="${CPU:-8}"
    MEMORY="${MEMORY:-32Gi}"
    COPY_TO_SHM="${COPY_TO_SHM:-0}"
    GPU_UTIL="${GPU_UTIL:-0.90}"
    GCLOUD_RUN=(gcloud alpha run)
    ;;
  *)
    CPU="${CPU:-20}"
    MEMORY="${MEMORY:-80Gi}"
    COPY_TO_SHM="${COPY_TO_SHM:-1}"
    GPU_UTIL="${GPU_UTIL:-0.40}"
    GCLOUD_RUN=(gcloud run)
    ;;
esac

"${GCLOUD_RUN[@]}" deploy "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --gpu="${GPU}" \
  --gpu-type=nvidia-rtx-pro-6000 \
  --no-gpu-zonal-redundancy \
  --cpu="${CPU}" \
  --memory="${MEMORY}" \
  --no-cpu-throttling \
  --concurrency=32 \
  --timeout=600 \
  --min-instances=0 \
  --max-instances=1 \
  --port=8080 \
  --network=default \
  --subnet=default \
  --vpc-egress=all-traffic \
  --add-volume=name=weights,type=cloud-storage,bucket="${BUCKET}",readonly=false,mount-options=enable-buffered-read=true \
  --add-volume-mount=volume=weights,mount-path=/mnt/gcs \
  --startup-probe=httpGet.path=/health,httpGet.port=8080,initialDelaySeconds=15,periodSeconds=10,timeoutSeconds=5,failureThreshold=90 \
  --set-env-vars="MODEL=/mnt/gcs/dgemma,CANVAS=128,MAX_SEQS=32,MAX_MODEL_LEN=4096,GPU_UTIL=${GPU_UTIL},KV_CACHE_GB=2,ATTN=TRITON_ATTN,COPY_TO_SHM=${COPY_TO_SHM},TEST_PAGE=1,VLLM_UF_EAGER_ALL=1,VLLM_FLASHINFER_MOE_BACKEND=masked_gemm,VLLM_WORKER_MULTIPROC_METHOD=spawn,CUDA_MODULE_LOADING=LAZY" \
  --no-allow-unauthenticated
