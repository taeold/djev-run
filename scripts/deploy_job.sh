#!/usr/bin/env bash
set -euo pipefail

# Cloud Run GPU Job deployment script for batch evaluation and weight/JIT staging.
# Follows Cloud Run GPU best practices for jobs:
#   https://docs.cloud.google.com/run/docs/configuring/jobs/gpu-best-practices
#   - --gpu=1 --gpu-type=nvidia-rtx-pro-6000 --no-gpu-zonal-redundancy
#   - --cpu=20 --memory=80Gi (required minimum for 1x RTX PRO 6000 + /dev/shm staging)
#   - --max-retries=0 to avoid duplicate GPU billing on deterministic errors
#   - --tasks=1 --parallelism=1 (scale tasks/parallelism for sharded JSONL datasets)
#   - Cloud Storage FUSE mount with enable-buffered-read=true

PROJECT="${PROJECT:-danielylee-joonix}"
REGION="${REGION:-us-central1}"
JOB_NAME="${JOB_NAME:-djev-dgemma-eval-job}"
BUCKET="${BUCKET:-danielylee-run-mount}"
IMAGE="${IMAGE:-ghcr.io/taeold/djev-run:latest}"
TASKS="${TASKS:-1}"
PARALLELISM="${PARALLELISM:-1}"
TASK_TIMEOUT="${TASK_TIMEOUT:-3600s}"

gcloud run jobs deploy "${JOB_NAME}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --gpu=1 \
  --gpu-type=nvidia-rtx-pro-6000 \
  --no-gpu-zonal-redundancy \
  --cpu=20 \
  --memory=80Gi \
  --tasks="${TASKS}" \
  --parallelism="${PARALLELISM}" \
  --max-retries=0 \
  --task-timeout="${TASK_TIMEOUT}" \
  --network=default \
  --subnet=default \
  --vpc-egress=all-traffic \
  --add-volume=name=weights,type=cloud-storage,bucket="${BUCKET}",readonly=false,mount-options=enable-buffered-read=true \
  --add-volume-mount=volume=weights,mount-path=/mnt/gcs \
  --set-env-vars="MODEL=/mnt/gcs/dgemma,CANVAS=128,MAX_SEQS=32,MAX_MODEL_LEN=4096,GPU_UTIL=0.40,KV_CACHE_GB=2,ATTN=TRITON_ATTN,COPY_TO_SHM=1,VLLM_UF_EAGER_ALL=1,VLLM_FLASHINFER_MOE_BACKEND=masked_gemm,VLLM_WORKER_MULTIPROC_METHOD=spawn,CUDA_MODULE_LOADING=LAZY"
