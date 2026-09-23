#!/usr/bin/env bash
set -euo pipefail

echo "[ultra-fast-init] Starting at $(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)"

PY_BIN="python3"
if command -v python3.12 >/dev/null 2>&1; then
  PY_BIN="python3.12"
fi
SITE_PKG="/usr/local/lib/python3.12/dist-packages"
NV_LIBS=$(find "$SITE_PKG/nvidia" -maxdepth 2 -type d -name "lib" 2>/dev/null | tr '\n' ':' || true)
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${SITE_PKG}/torch/lib:${NV_LIBS}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SITE_PKG}:/opt/dgemma:${PYTHONPATH:-}"

# 1. Stage small config/tokenizer/processor files immediately
if [ ! -L /dev/shm/dgemma ]; then
  mkdir -p /dev/shm/dgemma
fi
for f in chat_template.jinja config.json generation_config.json hf_quant_config.json model.safetensors.index.json processor_config.json preprocessor_config.json tokenizer.json tokenizer_config.json; do
  if [ -f "/mnt/gcs/dgemma/$f" ]; then
    cp -f "/mnt/gcs/dgemma/$f" "/dev/shm/dgemma/$f" &
  fi
done

# 2. Restore pre-compiled .pyc cache + pre-patched gpu_worker.py from GCS and prefetch all vllm/transformers/torch .py/.pyc/.so into RAM via 32 parallel workers (~2.5s)
if [ -f "/mnt/gcs/fast-init/pyc-cache.tar" ]; then
  tar -xf /mnt/gcs/fast-init/pyc-cache.tar -C / 2>/dev/null || true
fi
if [ -d "/opt/dgemma/v9_baked/vllm" ]; then
  cp -rf /opt/dgemma/v9_baked/vllm/* /usr/local/lib/python3.12/dist-packages/vllm/
fi
if [ -d "/opt/dgemma/v9_baked/dgemma" ]; then
  cp -rf /opt/dgemma/v9_baked/dgemma/* /opt/dgemma/
fi
if [ -d "/opt/dgemma/vllm_overlay" ]; then
  cp -rf /opt/dgemma/vllm_overlay/* /usr/local/lib/python3.12/dist-packages/vllm/
fi
"$PY_BIN" -m compileall -f -q \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/diffusion_gemma.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/ops/diffusion_sampler.py \
  /usr/local/lib/python3.12/dist-packages/vllm/utils/diffusion.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/diffusion_scheduler.py \
  /opt/dgemma/structured_server.py 2>/dev/null || true
find /usr/local/lib/python3.12/dist-packages/vllm \
     /usr/local/lib/python3.12/dist-packages/transformers \
     /usr/local/lib/python3.12/dist-packages/tokenizers \
     /usr/local/lib/python3.12/dist-packages/safetensors \
     /opt/dgemma \
     \( -name "*.py" -o -name "*.pyc" -o -name "*.so" \) 2>/dev/null | xargs -P 32 -n 64 cat > /dev/null 2>&1 || true
find /usr/local/lib/python3.12/dist-packages/torch -maxdepth 2 \
     \( -name "*.py" -o -name "*.pyc" -o -name "*.so" \) 2>/dev/null | xargs -P 32 -n 64 cat > /dev/null 2>&1 || true
wait
echo "[ultra-fast-init] Parallel 32-way rootfs prefetch & small configs staged at $(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)"

# 3. Background 2-process streaming cp of the two .safetensors shards into /dev/shm/dgemma (~14.3s, runs concurrently with Python startup)
if [ ! -f "/dev/shm/dgemma/.ready" ]; then
  (
    for f in /mnt/gcs/dgemma/*.safetensors; do
      [ -f "$f" ] && cp -f "$f" "/dev/shm/dgemma/$(basename "$f")" &
    done
    wait
    touch /dev/shm/dgemma/.ready
    echo "[ultra-fast-init] Parallel 2-shard cp to /dev/shm/dgemma complete at $(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)"
  ) &
fi

# 3. Single-process structured_server.py + In-Process vLLM Engine (VLLM_ENABLE_V1_MULTIPROCESSING=0)
cat << 'PYEOF' > /tmp/run_inproc_server.py
import os
import pathlib
import sys
import threading
import time

T_BOOT = time.time()
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["TEST_PAGE"] = "1"

# Patch gpu_worker.py in-place before importing vllm (only touches 1 file in overlayfs)
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")
s = p.read_text()
if "while os.path.exists('/dev/shm/dgemma')" not in s:
    s = s.replace(
        "        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n"
        "            # still need a profile run which compiles the model for\n"
        "            # max_num_batched_tokens\n"
        "            self.model_runner.profile_run()",
        "        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n"
        "            if not self.model_config.enforce_eager:\n"
        "                self.model_runner.profile_run()"
    )
    s = s.replace(
        "        kernel_warmup(self)\n\n"
        "        if self.use_v2_model_runner:\n"
        "            # A workspace resize after capture frees what the graphs point at.\n"
        "            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)",
        "        if not self.model_config.enforce_eager:\n"
        "            kernel_warmup(self)\n"
        "            if self.use_v2_model_runner:\n"
        "                warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)"
    )
    s = s.replace(
        "self.model_runner.load_model(load_dummy_weights=load_dummy_weights)",
        "import os, time\n"
        "            while os.path.exists('/dev/shm/dgemma') and not os.path.exists('/dev/shm/dgemma/.ready'): time.sleep(0.02)\n"
        "            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)"
    )
    p.write_text(s)

sys.path.insert(0, "/opt/dgemma")
import structured_server as S

ENGINE_READY = False
ENGINE_LOCK = threading.Lock()
LLM_ENGINE = None

# Ensure all HTML demo pages are registered
S.TEST_PAGE = True
S.PAGES["/playground.html"] = "playground.html"
if os.path.exists("/opt/dgemma/snake.html"):
    S.PAGES["/snake"] = "snake.html"
    S.PAGES["/snake.html"] = "snake.html"
if os.path.exists("/opt/dgemma/dino.html"):
    S.PAGES["/dino"] = "dino.html"
    S.PAGES["/dino.html"] = "dino.html"
if os.path.exists("/opt/dgemma/tetris.html"):
    S.PAGES["/tetris"] = "tetris.html"
    S.PAGES["/tetris.html"] = "tetris.html"

S.ARGS = type("Args", (), {
    "upstream": "http://127.0.0.1:8000",
    "model": "dgemma",
    "tokenizer": "/dev/shm/dgemma",
    "canvas": int(os.environ.get("CANVAS", "256")),
    "canvas_step": 16,
    "constrained": os.environ.get("NO_CONSTRAINED", "0") != "1",
    "host": "0.0.0.0",
    "port": int(os.environ.get("PORT", "8080")),
    "tls_port": 0,
})()
S.CANVAS_LEN = S.ARGS.canvas
S.CANVAS_STEP = S.ARGS.canvas_step

# Patch /health on Handler to return 200 OK as soon as ENGINE_READY is True (suppress BrokenPipeError)
orig_do_GET = S.Handler.do_GET
def patched_do_GET(self):
    try:
        if self.path == "/health":
            if ENGINE_READY:
                return self._json(200, {"status": "ok", "mode": "inproc-vllm"})
            return self._json(503, {"status": "starting"})
        return orig_do_GET(self)
    except BrokenPipeError:
        pass
S.Handler.do_GET = patched_do_GET

# Start HTTP server immediately on 0.0.0.0:8080 and load AutoTokenizer in background thread
srv = S.Server((S.ARGS.host, S.ARGS.port), S.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"[ultra-fast-init] HTTP listener active on {S.ARGS.host}:{S.ARGS.port} at t={time.time() - T_BOOT:.2f}s", flush=True)

def _bg_load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/dev/shm/dgemma")
    if tok.chat_template is None and os.path.exists("/dev/shm/dgemma/chat_template.jinja"):
        tok.chat_template = open("/dev/shm/dgemma/chat_template.jinja").read()
    S.init_tokenizer(tok)
    print(f"[ultra-fast-init] Tokenizer initialized at t={time.time() - T_BOOT:.2f}s", flush=True)

tok_thread = threading.Thread(target=_bg_load_tokenizer, daemon=True)
tok_thread.start()

from vllm import LLM, SamplingParams
from vllm.renderers.base import BaseRenderer
BaseRenderer.warmup = lambda self, *a, **kw: None

def _extract_top_dict(lp_dict):
    return {int(k): float(v.logprob if hasattr(v, "logprob") else v) for k, v in lp_dict.items()}

def inproc_read_many(schema, template, slots, sys_text, state_content, seed, n, prefix=None, thinking=False):
    if prefix is not None:
        prompt_ids = prefix
    elif isinstance(state_content, str):
        prompt_ids = S.chat_prompt_ids(sys_text, state_content, thinking=thinking)
    else:
        prompt_ids = S.chat_messages_ids(
            [{"role": "system", "content": sys_text}, {"role": "user", "content": state_content}],
            thinking=thinking,
        )
    prompts = [{"prompt_token_ids": prompt_ids} for _ in range(n)]
    lids = S.label_id_union(slots)
    num_lp = len(lids) if lids else S.TOPK
    cwidth = S.canvas_width(template)
    sp_list = [
        SamplingParams(
            max_tokens=len(template) + 1,
            detokenize=False,
            logprobs=num_lp,
            logprob_token_ids=lids,
            extra_args={
                "diffusion_seed_canvas": S.build_canvas(template, slots, seed + k * 7919),
                "diffusion_canvas_length": cwidth,
                "diffusion_max_steps": schema["steps"],
                "diffusion_read_only": True,
                **S.pin_xargs(template, slots, schema["steps"]),
                **S.constrained_xargs(),
            },
        )
        for k in range(n)
    ]
    max_len = int(os.environ.get("MAX_MODEL_LEN", "4096"))
    if len(prompt_ids) + S.canvas_width(template) > max_len:
        raise S.SchemaError(
            f"prompt ({len(prompt_ids)} tokens) + canvas ({S.canvas_width(template)}) exceeds max_model_len ({max_len})"
        )
    with ENGINE_LOCK:
        outputs = LLM_ENGINE.generate(prompts, sp_list, use_tqdm=False)
    results = []
    usages = []
    for ro in outputs:
        lp_rows = ro.outputs[0].logprobs
        out = []
        for s in slots:
            top = _extract_top_dict(lp_rows[s["pos"]])
            out.append(S.slot_distribution(top, s["label_ids"]))
        results.append(out)
        usages.append({"prompt_tokens": len(prompt_ids)})
    return results, usages

orig_read_many = S.read_many
S.read_many = inproc_read_many

def inproc_upstream_completions(body, timeout=600):
    prompt_ids = body["prompt"]
    xargs = body.get("vllm_xargs")
    lids = body.get("logprob_token_ids")
    num_lp = len(lids) if lids else max(1, int(body.get("logprobs") or 1))
    sp = SamplingParams(
        max_tokens=int(body.get("max_tokens", 64)),
        detokenize=False,
        logprobs=num_lp,
        logprob_token_ids=lids,
        stop_token_ids=body.get("stop_token_ids"),
        extra_args=xargs,
    )
    with ENGINE_LOCK:
        ro = LLM_ENGINE.generate([{"prompt_token_ids": prompt_ids}], [sp], use_tqdm=False)[0]
    out = ro.outputs[0]
    tokens = [f"token_id:{tid}" for tid in out.token_ids]
    rows = []
    if out.logprobs:
        for lp_dict in out.logprobs:
            rows.append({f"token_id:{k}": float(v.logprob if hasattr(v, "logprob") else v) for k, v in lp_dict.items()})
    return {
        "choices": [{"logprobs": {"tokens": tokens, "top_logprobs": rows}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt_ids)},
    }

def inproc_upstream_chat(body, timeout=600):
    msgs = body.get("messages", [])
    thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", False)
    prompt_ids = S.chat_messages_ids(msgs, thinking=thinking)
    xargs = body.get("vllm_xargs")
    lids = body.get("logprob_token_ids")
    top_lp = len(lids) if lids else int(body.get("top_logprobs") or (S.TOPK if body.get("logprobs") else 1))
    sp = SamplingParams(
        max_tokens=int(body.get("max_tokens", 64)),
        detokenize=False,
        logprobs=max(1, top_lp),
        logprob_token_ids=lids,
        stop_token_ids=body.get("stop_token_ids"),
        extra_args=xargs,
    )
    with ENGINE_LOCK:
        ro = LLM_ENGINE.generate([{"prompt_token_ids": prompt_ids}], [sp], use_tqdm=False)[0]
    out = ro.outputs[0]
    content = []
    for pos_idx, tid in enumerate(out.token_ids):
        lp_dict = out.logprobs[pos_idx] if (out.logprobs and pos_idx < len(out.logprobs)) else {tid: 0.0}
        top_list = [
            {"token": f"token_id:{k}", "logprob": float(v.logprob if hasattr(v, "logprob") else v)}
            for k, v in lp_dict.items()
        ]
        content.append({
            "token": f"token_id:{tid}",
            "logprob": top_list[0]["logprob"] if top_list else 0.0,
            "top_logprobs": top_list,
        })
    return {
        "choices": [{
            "message": {"role": "assistant", "content": S.TOK.decode(out.token_ids)},
            "logprobs": {"content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": len(prompt_ids)},
    }

S.upstream_completions = inproc_upstream_completions
S.upstream_chat = inproc_upstream_chat

import torch
gpu_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 24.0
gpu_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (8, 9)
low_vram = gpu_vram_gib < 20.0
dtype = os.environ.get("DTYPE", "float16" if gpu_cap[0] < 8 else "auto")
cpu_offload_gb = float(os.environ.get("CPU_OFFLOAD_GB", "5.0" if low_vram else "0.0"))
disable_mm = os.environ.get("DISABLE_MM", "1") == "1"
default_kv_gb = "0.5" if low_vram else ("1.5" if gpu_vram_gib < 30.0 else "2.0")
kv_cache_gb = float(os.environ.get("KV_CACHE_GB", default_kv_gb))
default_gpu_util = "0.85" if low_vram else ("0.90" if gpu_vram_gib < 30.0 else ("0.85" if gpu_vram_gib < 60.0 else "0.40"))
gpu_util = float(os.environ.get("GPU_UTIL", default_gpu_util))
if torch.cuda.is_available():
    free_b, total_b = torch.cuda.mem_get_info(0)
    max_safe_util = round((free_b / total_b) * 0.94, 3)
    if gpu_util > max_safe_util:
        gpu_util = max_safe_util
    # Ensure gpu_util budget comfortably covers model weights + kv_cache_gb
    min_needed_gib = (17.55 - cpu_offload_gb) + kv_cache_gb + 0.8
    if gpu_util * gpu_vram_gib < min_needed_gib and max_safe_util * gpu_vram_gib >= min_needed_gib:
        gpu_util = min(max_safe_util, round((min_needed_gib + 0.5) / gpu_vram_gib, 3))
    if (17.55 - cpu_offload_gb) + kv_cache_gb > gpu_util * gpu_vram_gib - 0.6:
        kv_cache_gb = max(0.5, round(gpu_util * gpu_vram_gib - (17.55 - cpu_offload_gb) - 0.8, 2))
print(
    f"[ultra-fast-init] Initializing in-process vLLM LLM engine "
    f"(vram={gpu_vram_gib:.1f}GiB, sm={gpu_cap[0]}.{gpu_cap[1]}, dtype={dtype}, "
    f"gpu_util={gpu_util}, kv_cache_gb={kv_cache_gb}, cpu_offload_gb={cpu_offload_gb}, disable_mm={disable_mm}) at t={time.time() - T_BOOT:.2f}s...",
    flush=True,
)
LLM_ENGINE = LLM(
    model="/dev/shm/dgemma",
    dtype=dtype,
    cpu_offload_gb=cpu_offload_gb,
    skip_tokenizer_init=disable_mm,
    trust_remote_code=True,
    max_num_seqs=int(os.environ.get("MAX_SEQS", "32")),
    max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
    max_num_batched_tokens=4096,
    attention_backend=os.environ.get("ATTN", "TRITON_ATTN"),
    gpu_memory_utilization=gpu_util,
    kv_cache_memory_bytes=int(kv_cache_gb * 1073741824),
    max_logprobs=128,
    enable_prefix_caching=True,
    enforce_eager=True,
    language_model_only=disable_mm,
    skip_mm_profiling=True,
    limit_mm_per_prompt={"image": 0 if disable_mm else 1, "video": 0},
    diffusion_config={"canvas_length": S.CANVAS_LEN},
    override_generation_config={"max_new_tokens": None},
    kernel_config={
        "enable_flashinfer_autotune": False,
        "enable_cutedsl_warmup": False,
        "enable_jit_warmup": False,
    },
    async_scheduling=True,
    disable_log_stats=False,
)
tok_thread.join()
ENGINE_READY = True
print(f"[ultra-fast-init] ENGINE_READY=True (in-process vLLM ready) at t={time.time() - T_BOOT:.2f}s", flush=True)

while True:
    time.sleep(3600)
PYEOF

exec "$PY_BIN" /tmp/run_inproc_server.py
