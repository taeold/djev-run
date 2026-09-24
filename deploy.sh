#!/usr/bin/env bash
set -euo pipefail
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-your-bucket-name}"
MODE="${MODE:-djev}"

common_flags="--region=${REGION} --gpu=1 --gpu-type=nvidia-rtx-pro-6000 --no-gpu-zonal-redundancy --cpu=20 --memory=80Gi --no-cpu-throttling --concurrency=32 --min-instances=0 --max-instances=1 --network=default --subnet=default --vpc-egress=all-traffic --add-volume=name=weights,type=cloud-storage,bucket=${BUCKET},readonly=false,mount-options=enable-buffered-read=true --add-volume-mount=volume=weights,mount-path=/mnt/gcs"
probe="--startup-probe=httpGet.path=/health,httpGet.port=8000,initialDelaySeconds=5,periodSeconds=2,timeoutSeconds=2,failureThreshold=120"
probe_8080="--startup-probe=httpGet.path=/health,httpGet.port=8080,initialDelaySeconds=5,periodSeconds=2,timeoutSeconds=2,failureThreshold=120"

if [ "$MODE" = "raw" ]; then
  gcloud beta run deploy djev-dgemma \
    $common_flags $probe \
    --image=docker.io/vllm/vllm-openai:nightly \
    --port=8000 \
    --set-env-vars="VLLM_FLASHINFER_MOE_BACKEND=masked_gemm,VLLM_ENABLE_V1_MULTIPROCESSING=0,LD_LIBRARY_PATH=/usr/local/cuda/compat" \
    --command="/bin/bash" \
    --args="-c","cp -r /mnt/gcs/dgemma /dev/shm/dgemma && exec vllm serve /dev/shm/dgemma --served-model-name djev-dgemma --allowed-origins * --trust-remote-code --enforce-eager --language-model-only --attention-backend TRITON_ATTN --kv-cache-memory 2G --max-num-seqs 32 --max-model-len 4096 --diffusion-config '{\"canvas_length\":128}' --override-generation-config '{\"max_new_tokens\":null}'"
else
  gcloud beta run deploy djev-dgemma \
    $common_flags $probe_8080 \
    --image=ghcr.io/taeold/djev-run:latest \
    --port=8080
fi
