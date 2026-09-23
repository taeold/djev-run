#!/usr/bin/env bash
set -euo pipefail

BUCKET="${BUCKET:?Set BUCKET to your GCS bucket containing gs://\$BUCKET/dgemma/}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-djev-dgemma}"
IMAGE="${IMAGE:-ghcr.io/taeold/djev-run:latest}"

gcloud beta run deploy "${SERVICE}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
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
  --add-volume="name=weights,type=cloud-storage,bucket=${BUCKET},readonly=false,mount-options=enable-buffered-read=true" \
  --add-volume-mount=volume=weights,mount-path=/mnt/gcs \
  --startup-probe=httpGet.path=/health,httpGet.port=8080,initialDelaySeconds=5,periodSeconds=2,timeoutSeconds=2,failureThreshold=120 \
  --set-env-vars="MODEL=/mnt/gcs/dgemma,CANVAS=128,MAX_SEQS=32,MAX_MODEL_LEN=4096,GPU_UTIL=0.40,KV_CACHE_GB=2,ATTN=TRITON_ATTN,TEST_PAGE=1,COPY_TO_SHM=1,ENFORCE_EAGER=1,DISABLE_MM=1,TORCH_COMPILE_DISABLE=1,VLLM_WORKER_MULTIPROC_METHOD=fork,VLLM_UF_EAGER_ALL=1,VLLM_FLASHINFER_MOE_BACKEND=masked_gemm,CUDA_MODULE_LOADING=LAZY"
