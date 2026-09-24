#!/usr/bin/env bash
set -euo pipefail

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_FLASHINFER_MOE_BACKEND=masked_gemm
export PYTHONPATH="/opt/dgemma:${PYTHONPATH:-}"

cp -r "${MODEL:-/mnt/gcs/dgemma}" /dev/shm/dgemma

exec vllm serve /dev/shm/dgemma \
  --middleware server.SystemOneMiddleware \
  --port "${PORT:-8080}" \
  --served-model-name djev-dgemma \
  --allowed-origins * \
  --trust-remote-code \
  --enforce-eager \
  --language-model-only \
  --attention-backend TRITON_ATTN \
  --kv-cache-memory 2G \
  --max-num-seqs 32 \
  --max-model-len 4096 \
  --diffusion-config '{"canvas_length":128}' \
  --override-generation-config '{"max_new_tokens":null}' \
  "$@"
