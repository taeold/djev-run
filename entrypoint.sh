#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/mnt/gcs/dgemma}"
CANVAS="${CANVAS:-128}"
PORT="${PORT:-8080}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
JIT_CACHE_ARCHIVE="/mnt/gcs/jit-cache/rtx-pro-6000-cache.tar.gz"

mkdir -p /root/.cache/flashinfer /root/.triton /root/.cache/vllm

# 1. Restore pre-warmed JIT cache from GCS if available
if [ -f "$JIT_CACHE_ARCHIVE" ]; then
  echo "[init] Restoring FlashInfer/Triton/vLLM JIT cache from GCS..."
  tar -xzf "$JIT_CACHE_ARCHIVE" -C /root/ || echo "[warn] Failed to extract JIT cache, continuing..."
fi

# 2. Optional parallel copy of weights from GCS FUSE into /dev/shm RAM disk
if [ "${COPY_TO_SHM:-1}" = "1" ] && [ -d "/mnt/gcs/dgemma" ] && [ -f "/mnt/gcs/dgemma/config.json" ]; then
  echo "[init] Copying weights from /mnt/gcs/dgemma to /dev/shm/dgemma..."
  mkdir -p /dev/shm/dgemma
  find /mnt/gcs/dgemma -maxdepth 1 -type f | xargs -P 16 -I {} cp -f {} /dev/shm/dgemma/
  MODEL="/dev/shm/dgemma"
  echo "[init] Weights staged in /dev/shm/dgemma"
fi

# 3. Start internal vLLM server on 127.0.0.1:8000
VLLM_EXTRA_ARGS=()
if [ "$ENFORCE_EAGER" = "1" ]; then
  echo "[init] ENFORCE_EAGER=1 enabled: passing --enforce-eager to vLLM..."
  VLLM_EXTRA_ARGS+=(--enforce-eager)
fi

echo "[init] Starting vLLM serve for $MODEL..."
vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name dgemma \
  --trust-remote-code \
  --max-num-seqs "${MAX_SEQS:-32}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --attention-backend "${ATTN:-TRITON_ATTN}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.40}" \
  --kv-cache-memory "$(( ${KV_CACHE_GB:-2} * 1073741824 ))" \
  --max-logprobs 32 \
  --enable-prefix-caching \
  --diffusion-config "{\"canvas_length\": ${CANVAS}}" \
  --override-generation-config '{"max_new_tokens": null}' \
  --async-scheduling \
  "${VLLM_EXTRA_ARGS[@]}" &
VLLM_PID=$!

# 4. Poll vLLM health endpoint before launching structured_server.py
echo "[init] Waiting for vLLM at http://127.0.0.1:8000/health ..."
for i in $(seq 1 450); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[error] vLLM process exited prematurely" >&2
    exit 1
  fi
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "[init] vLLM is healthy after $((i * 2))s"
    break
  fi
  sleep 2
done

if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "[error] vLLM did not become healthy within 900s" >&2
  exit 1
fi

# 5. If JIT cache archive does not exist yet, warm up /v1/systemone on 127.0.0.1:8011
#    and persist JIT caches to GCS before opening 0.0.0.0:$PORT.
if [ ! -f "$JIT_CACHE_ARCHIVE" ] && [ -d "/mnt/gcs" ]; then
  echo "[init] Pre-warming sampler and persisting JIT cache to $JIT_CACHE_ARCHIVE..."
  TEST_PAGE=1 python3 /opt/dgemma/structured_server.py \
    --upstream http://127.0.0.1:8000 \
    --model dgemma \
    --tokenizer "$MODEL" \
    --canvas "$CANVAS" \
    --host 127.0.0.1 \
    --port 8011 &
  WARM_PID=$!
  for j in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8011/health >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  curl -sf http://127.0.0.1:8011/v1/systemone \
    -H 'Content-Type: application/json' \
    -d '{"model":"dgemma","state":"warmup","samples":1,"steps":1,"questions":{"ok":{"type":"noul","instructions":"ok?"}}}' >/dev/null 2>&1 || true
  curl -sf http://127.0.0.1:8011/v1/systemone \
    -H 'Content-Type: application/json' \
    -d '{"model":"dgemma","state":"warmup","samples":2,"steps":1,"questions":{"ok":{"type":"noul","instructions":"ok?"}}}' >/dev/null 2>&1 || true
  kill "$WARM_PID" 2>/dev/null || true
  wait "$WARM_PID" 2>/dev/null || true
  mkdir -p /mnt/gcs/jit-cache
  tar -czf /tmp/rtx-pro-6000-cache.tar.gz -C /root .cache/flashinfer .triton .cache/vllm 2>/dev/null || true
  cp -f /tmp/rtx-pro-6000-cache.tar.gz "$JIT_CACHE_ARCHIVE" 2>/dev/null || true
  rm -f /tmp/rtx-pro-6000-cache.tar.gz
  echo "[init] Saved JIT cache to $JIT_CACHE_ARCHIVE"
fi

# 6. Launch structured_server.py on 0.0.0.0:$PORT (Cloud Run Service ingress port)
echo "[init] Starting structured_server.py on 0.0.0.0:${PORT}..."
export TEST_PAGE="${TEST_PAGE:-1}"
exec python3 /opt/dgemma/structured_server.py \
  --upstream http://127.0.0.1:8000 \
  --model dgemma \
  --tokenizer "$MODEL" \
  --canvas "$CANVAS" \
  --host 0.0.0.0 \
  --port "$PORT"

