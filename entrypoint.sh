#!/usr/bin/env bash
set -euo pipefail

echo "[djev-init] Starting at $(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)"

PY_BIN="python3"
if [ -x "/opt/djev_venv/bin/python" ]; then
  PY_BIN="/opt/djev_venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
  PY_BIN="python3.12"
fi
SITE_PKG=$("$PY_BIN" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
NV_LIBS=$(find "$SITE_PKG/nvidia" -maxdepth 2 -type d -name "lib" 2>/dev/null | tr '\n' ':' || true)
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${SITE_PKG}/torch/lib:${NV_LIBS}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SITE_PKG}:/opt/dgemma:${PYTHONPATH:-}"

WEIGHTS_SRC="${MODEL:-/mnt/gcs/dgemma}"
if [ ! -d "$WEIGHTS_SRC" ] && [ -d "/gcs/dgemma" ]; then
  WEIGHTS_SRC="/gcs/dgemma"
fi

# 1. Stage small config/tokenizer/processor files immediately into /dev/shm/dgemma
if [ ! -L /dev/shm/dgemma ]; then
  mkdir -p /dev/shm/dgemma
fi
for f in chat_template.jinja config.json generation_config.json hf_quant_config.json model.safetensors.index.json processor_config.json preprocessor_config.json tokenizer.json tokenizer_config.json; do
  if [ -f "$WEIGHTS_SRC/$f" ]; then
    cp -f "$WEIGHTS_SRC/$f" "/dev/shm/dgemma/$f" &
  fi
done

# 2. Prefetch vllm/transformers/torch .py/.pyc/.so into page cache via 32 parallel workers
find "$SITE_PKG/vllm" \
     "$SITE_PKG/transformers" \
     "$SITE_PKG/tokenizers" \
     "$SITE_PKG/safetensors" \
     \( -name "*.py" -o -name "*.pyc" -o -name "*.so" \) 2>/dev/null | xargs -P 32 -n 64 cat > /dev/null 2>&1 || true
find "$SITE_PKG/torch" -maxdepth 2 \
     \( -name "*.py" -o -name "*.pyc" -o -name "*.so" \) 2>/dev/null | xargs -P 32 -n 64 cat > /dev/null 2>&1 || true
wait
echo "[djev-init] Parallel rootfs prefetch & tokenizer configs staged at $(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)"

# 3. Background 2-process streaming copy of .safetensors shards into /dev/shm/dgemma
if [ ! -f "/dev/shm/dgemma/.ready" ]; then
  (
    for f in "$WEIGHTS_SRC"/*.safetensors; do
      [ -f "$f" ] && cp -f "$f" "/dev/shm/dgemma/$(basename "$f")" &
    done
    wait
    touch /dev/shm/dgemma/.ready
    echo "[djev-init] Parallel 2-shard copy to /dev/shm/dgemma complete at $(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)"
  ) &
fi

# 4. Launch in-process vLLM server (/v1/chat/completions, /v1/completions, /tokenize, /snake, /dino, /tetris, /health)
cat << 'PYEOF' > /tmp/run_inproc_server.py
import importlib.util
import json
import math
import os
import pathlib
import random
import string
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

T_BOOT = time.time()
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

# Ensure vllm#57250 block-diffusion files and eager startup hook are present in vllm
vllm_spec = importlib.util.find_spec("vllm")
vllm_dir = pathlib.Path(vllm_spec.origin).resolve().parent
dg_path = vllm_dir / "model_executor/models/diffusion_gemma.py"
if not dg_path.exists() or "diffusion_read_only" not in dg_path.read_text():
    VLLM_57250 = "f831226339b87554b1de9341276e7bee7270fa8a"
    for rel in (
        "model_executor/models/diffusion_gemma.py",
        "utils/diffusion.py",
        "v1/core/sched/diffusion_scheduler.py",
        "config/diffusion.py",
        "v1/worker/gpu/model_runner.py",
        "v1/worker/gpu/states.py",
    ):
        dest = vllm_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/mmastrac/vllm/{VLLM_57250}/vllm/{rel}",
            dest,
        )

p = vllm_dir / "v1/worker/gpu_worker.py"
if p.exists():
    s = p.read_text()
    if "while os.path.exists('/dev/shm/dgemma')" not in s:
        s = s.replace(
            "        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n"
            "            # still need a profile run which compiles the model for\n"
            "            # max_num_batched_tokens\n"
            "            self.model_runner.profile_run()",
            "        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n"
            "            if not self.model_config.enforce_eager:\n"
            "                self.model_runner.profile_run()",
        )
        s = s.replace(
            "        kernel_warmup(self)\n\n"
            "        if self.use_v2_model_runner:\n"
            "            # A workspace resize after capture frees what the graphs point at.\n"
            "            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)",
            "        if not self.model_config.enforce_eager:\n"
            "            kernel_warmup(self)\n"
            "            if self.use_v2_model_runner:\n"
            "                warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)",
        )
        s = s.replace(
            "self.model_runner.load_model(load_dummy_weights=load_dummy_weights)",
            "import os, time\n"
            "            while os.path.exists('/dev/shm/dgemma') and not os.path.exists('/dev/shm/dgemma/.ready'): time.sleep(0.02)\n"
            "            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)",
        )
        p.write_text(s)

PORT = int(os.environ.get("PORT", "8080"))
CANVAS_LEN = int(os.environ.get("CANVAS", "256"))
CANVAS_STEP = 16
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
TURN_CLOSE = 106
PAD = 0
VOCAB = 262144

ENGINE_READY = False
ENGINE_LOCK = threading.Lock()
LLM_ENGINE = None
TOK = None
SCAFFOLD = []
ALL_LABEL_TOKEN_IDS = []

PAGES = {
    "/": "snake.html",
    "/snake": "snake.html",
    "/snake.html": "snake.html",
    "/dino": "dino.html",
    "/dino.html": "dino.html",
    "/tetris": "tetris.html",
    "/tetris.html": "tetris.html",
}


def enc(text):
    return [int(t) for t in TOK.encode(text, add_special_tokens=False)]


def canvas_width(need_tokens):
    return min(CANVAS_LEN, max(CANVAS_STEP, -(-(need_tokens) // CANVAS_STEP) * CANVAS_STEP))


def chat_messages_ids(messages, thinking=False):
    out = TOK.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
    )
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return [int(t) for t in ids]


def normalize_diffusion_xargs(xargs, prompt_ids):
    if not xargs or "diffusion_seed_canvas" not in xargs:
        return dict(xargs or {}), prompt_ids
    x = dict(xargs)
    user_canvas = [int(t) for t in x["diffusion_seed_canvas"]]
    if SCAFFOLD and user_canvas[: len(SCAFFOLD)] != SCAFFOLD and prompt_ids[-len(SCAFFOLD) :] != SCAFFOLD:
        prompt_ids = list(prompt_ids) + list(SCAFFOLD)
    user_len = len(user_canvas)
    pinned_raw = x.get("diffusion_pinned")
    if pinned_raw is not None:
        pinned_set = {int(i) for i in pinned_raw}
        for i in range(user_len):
            if i not in pinned_set and user_canvas[i] < 256000:
                user_canvas[i] = random.randint(256000, VOCAB - 1)
    cwidth = int(x.get("diffusion_canvas_length") or canvas_width(user_len + 1))
    if len(user_canvas) < cwidth:
        full_canvas = user_canvas + [TURN_CLOSE] + [PAD] * max(0, cwidth - user_len - 1)
        full_canvas = full_canvas[:cwidth]
    else:
        full_canvas = user_canvas[:cwidth]
    x["diffusion_seed_canvas"] = full_canvas
    x["diffusion_canvas_length"] = cwidth
    x.setdefault("diffusion_max_steps", 1)
    x.setdefault("diffusion_read_only", True)
    if int(x["diffusion_max_steps"]) <= 1:
        x.pop("diffusion_pinned", None)
    elif pinned_raw is not None:
        x["diffusion_pinned"] = sorted({int(i) for i in pinned_raw} | set(range(user_len, cwidth)))
    return x, prompt_ids


class Server(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        try:
            path_clean = self.path.split("?")[0]
            if path_clean == "/health":
                if ENGINE_READY:
                    return self._json(200, {"status": "ok", "mode": "inproc-vllm"})
                return self._json(503, {"status": "starting"})
            if path_clean in PAGES:
                for base_dir in ("/opt/dgemma", os.getcwd()):
                    fpath = os.path.join(base_dir, PAGES[path_clean])
                    if os.path.exists(fpath):
                        data = open(fpath, "rb").read()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
            self._json(404, {"error": "not found"})
        except BrokenPipeError:
            pass

    def do_POST(self):
        try:
            if not ENGINE_READY:
                return self._json(503, {"error": "engine starting"})
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            path_clean = self.path.split("?")[0]

            if path_clean == "/tokenize":
                add_special = bool(body.get("add_special_tokens", False))
                if "prompts" in body:
                    tok_lists = [
                        [int(t) for t in TOK.encode(p, add_special_tokens=add_special)]
                        for p in body["prompts"]
                    ]
                    return self._json(200, {
                        "tokens": tok_lists,
                        "counts": [len(t) for t in tok_lists],
                        "max_model_len": MAX_MODEL_LEN,
                    })
                prompt_text = body.get("prompt", "")
                toks = [int(t) for t in TOK.encode(prompt_text, add_special_tokens=add_special)]
                return self._json(200, {
                    "tokens": toks,
                    "count": len(toks),
                    "max_model_len": MAX_MODEL_LEN,
                    "token_strs": [TOK.decode([t]) for t in toks],
                })

            if path_clean == "/v1/chat/completions":
                t0 = time.time()
                msgs = body.get("messages", [])
                thinking = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking", False))
                prompt_ids = chat_messages_ids(msgs, thinking=thinking)
                raw_xargs = (body.get("extra_body") or {}).get("vllm_xargs") or body.get("vllm_xargs") or {}
                xargs, prompt_ids = normalize_diffusion_xargs(raw_xargs, prompt_ids)
                lids = body.get("logprob_token_ids") or ALL_LABEL_TOKEN_IDS
                top_lp = max(len(lids), int(body.get("top_logprobs") or 20))
                max_toks = int(body.get("max_tokens") or 64)
                if "diffusion_canvas_length" in xargs:
                    max_toks = max(max_toks, int(xargs["diffusion_canvas_length"]))
                sp = SamplingParams(
                    max_tokens=max_toks,
                    detokenize=False,
                    logprobs=top_lp,
                    logprob_token_ids=lids,
                    stop_token_ids=body.get("stop_token_ids"),
                    extra_args=xargs if xargs else None,
                )
                with ENGINE_LOCK:
                    ro = LLM_ENGINE.generate([{"prompt_token_ids": prompt_ids}], [sp], use_tqdm=False)[0]
                out = ro.outputs[0]
                content = []
                for pos_idx, tid in enumerate(out.token_ids):
                    lp_dict = (out.logprobs[pos_idx] if (out.logprobs and pos_idx < len(out.logprobs)) else None) or {tid: 0.0}
                    top_list = []
                    for k, v in lp_dict.items():
                        raw_t = TOK.decode([int(k)])
                        top_list.append({
                            "token": raw_t.strip(),
                            "raw_token": raw_t,
                            "token_id": int(k),
                            "logprob": float(v.logprob if hasattr(v, "logprob") else v),
                        })
                    raw_tid = TOK.decode([int(tid)])
                    content.append({
                        "token": raw_tid.strip(),
                        "raw_token": raw_tid,
                        "token_id": int(tid),
                        "logprob": top_list[0]["logprob"] if top_list else 0.0,
                        "top_logprobs": top_list,
                    })
                return self._json(200, {
                    "id": f"chatcmpl-{int(t0 * 1000)}",
                    "object": "chat.completion",
                    "created": int(t0),
                    "model": body.get("model", "djev-dgemma"),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": TOK.decode(out.token_ids)},
                        "logprobs": {"content": content},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(out.token_ids),
                        "total_tokens": len(prompt_ids) + len(out.token_ids),
                    },
                    "timing_ms": round((time.time() - t0) * 1000, 2),
                })

            if path_clean == "/v1/systemone":
                # Translates /v1/systemone into the same in-process vLLM diffusion read
                t0 = time.time()
                qs_raw = body.get("questions", {})
                qs = []
                if isinstance(qs_raw, dict):
                    for qid, qspec in qs_raw.items():
                        qtype = qspec.get("type", "choice")
                        crit = qspec.get("criteria")
                        if qtype == "choice":
                            opts = list(crit.items()) if isinstance(crit, dict) else [(str(x), None) for x in (crit or [])]
                            lbls = list(string.ascii_lowercase[: len(opts)])
                        elif qtype == "score":
                            opts = [(str(x), str(x)) for x in (crit or [])]
                            lbls = [str(i + 1) for i in range(len(opts))]
                        else:
                            opts = [("yes", None), ("no", None)]
                            lbls = ["yes", "no"]
                        qs.append({"id": qid, "type": qtype, "instructions": qspec.get("instructions", ""), "choices": opts, "labels": lbls})
                sys_lines = [
                    "Answer a fixed set of questions about the state the user provides. "
                    "Each question lists its allowed answers; reply with exactly one label per question.\n"
                ]
                for q in qs:
                    sys_lines.append(f"\nQuestion {q['id']}: {q['instructions'].strip()}")
                    for (name, desc), lbl in zip(q["choices"], q["labels"]):
                        if q["type"] == "noul":
                            sys_lines.append(f"  {lbl}")
                        elif desc:
                            sys_lines.append(f"  {lbl}: {name} ({desc})")
                        else:
                            sys_lines.append(f"  {lbl}: {name}")
                sys_lines.append('\nReply with one line per question, in this order, formatted as "id: label".')
                sys_text = "\n".join(sys_lines)
                base_str = "\n".join(f"{q['id']}: {q['labels'][0]}" for q in qs)
                alt_str = "\n".join(f"{q['id']}: {q['labels'][min(1, len(q['labels']) - 1)]}" for q in qs)
                base_toks = enc(base_str)
                alt_toks = enc(alt_str)
                slot_positions = [i for i in range(len(base_toks)) if base_toks[i] != alt_toks[i]]
                seed_canvas = list(base_toks)
                rng = random.Random(42)
                for pos in slot_positions:
                    seed_canvas[pos] = rng.randrange(VOCAB)
                pinned = [i for i in range(len(base_toks)) if i not in set(slot_positions)]
                state_val = body.get("state", "")
                user_text = state_val if isinstance(state_val, str) else json.dumps(state_val)
                prompt_ids = chat_messages_ids([{"role": "system", "content": sys_text}, {"role": "user", "content": user_text}])
                xargs, prompt_ids = normalize_diffusion_xargs({
                    "diffusion_seed_canvas": seed_canvas,
                    "diffusion_pinned": pinned,
                    "diffusion_max_steps": int(body.get("steps", 1)),
                    "diffusion_read_only": True,
                }, prompt_ids)
                sp = SamplingParams(
                    max_tokens=len(base_toks) + 1,
                    detokenize=False,
                    logprobs=max(20, len(ALL_LABEL_TOKEN_IDS)),
                    logprob_token_ids=ALL_LABEL_TOKEN_IDS,
                    extra_args=xargs,
                )
                with ENGINE_LOCK:
                    ro = LLM_ENGINE.generate([{"prompt_token_ids": prompt_ids}], [sp], use_tqdm=False)[0]
                out = ro.outputs[0]
                answers = {}
                for q, pos in zip(qs, slot_positions):
                    lp_dict = out.logprobs[pos]
                    lbl_map = {}
                    for k, v in lp_dict.items():
                        t_str = TOK.decode([int(k)]).strip()
                        val = float(v.logprob if hasattr(v, "logprob") else v)
                        if t_str not in lbl_map or val > lbl_map[t_str]:
                            lbl_map[t_str] = val
                    floor = (min(lbl_map.values()) if lbl_map else -20.0) - 5.0
                    lp_t = [lbl_map.get(lbl, floor) for lbl in q["labels"]]
                    mx = max(lp_t)
                    ex = [math.exp(x - mx) for x in lp_t]
                    probs = [e / sum(ex) for e in ex]
                    if q["type"] == "choice":
                        p_dict = {name: round(p, 6) for (name, _), p in zip(q["choices"], probs)}
                        best_idx = max(range(len(probs)), key=lambda i: probs[i])
                        answers[q["id"]] = {"choice": q["choices"][best_idx][0], "probabilities": p_dict, "confidence": round(probs[best_idx], 6)}
                    elif q["type"] == "score":
                        p_dict = {str(i + 1): round(p, 6) for i, p in enumerate(probs)}
                        sc = sum((i + 1) * p for i, p in enumerate(probs))
                        answers[q["id"]] = {"score": round(sc, 6), "probabilities": p_dict, "confidence": round(max(probs), 6)}
                    else:
                        answers[q["id"]] = {"noul": round(probs[0], 6), "probability": round(probs[0], 6)}
                total_ms = round((time.time() - t0) * 1000, 2)
                return self._json(200, {"answers": answers, "diagnostics": {"timing": {"total_ms": total_ms}}})

            self._json(404, {"error": f"unknown route {path_clean}"})
        except Exception as e:
            self._json(500, {"error": str(e)})


srv = Server(("0.0.0.0", PORT), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"[djev-init] HTTP listener active on 0.0.0.0:{PORT} at t={time.time() - T_BOOT:.2f}s", flush=True)


def _bg_load_tokenizer():
    global TOK, SCAFFOLD, ALL_LABEL_TOKEN_IDS
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/dev/shm/dgemma")
    if tok.chat_template is None and os.path.exists("/dev/shm/dgemma/chat_template.jinja"):
        tok.chat_template = open("/dev/shm/dgemma/chat_template.jinja").read()
    TOK = tok
    SCAFFOLD = [int(t) for t in tok.encode("<|channel>thought\n<channel|>", add_special_tokens=False)]
    cands = set()
    for s in list(string.ascii_lowercase) + [str(d) for d in range(10)] + ["yes", "no", "true", "false"]:
        for prefix in (" ", ""):
            ids = tok.encode(prefix + s, add_special_tokens=False)
            if ids:
                cands.add(int(ids[-1]))
    ALL_LABEL_TOKEN_IDS = sorted(cands)[:128]
    print(f"[djev-init] Tokenizer initialized ({len(ALL_LABEL_TOKEN_IDS)} label token IDs) at t={time.time() - T_BOOT:.2f}s", flush=True)


tok_thread = threading.Thread(target=_bg_load_tokenizer, daemon=True)
tok_thread.start()

from vllm import LLM, SamplingParams
from vllm.renderers.base import BaseRenderer
BaseRenderer.warmup = lambda self, *a, **kw: None

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
    min_needed_gib = (17.55 - cpu_offload_gb) + kv_cache_gb + 0.8
    if gpu_util * gpu_vram_gib < min_needed_gib and max_safe_util * gpu_vram_gib >= min_needed_gib:
        gpu_util = min(max_safe_util, round((min_needed_gib + 0.5) / gpu_vram_gib, 3))
    if (17.55 - cpu_offload_gb) + kv_cache_gb > gpu_util * gpu_vram_gib - 0.6:
        kv_cache_gb = max(0.5, round(gpu_util * gpu_vram_gib - (17.55 - cpu_offload_gb) - 0.8, 2))
print(
    f"[djev-init] Initializing in-process vLLM LLM engine "
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
    max_model_len=MAX_MODEL_LEN,
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
    diffusion_config={"canvas_length": CANVAS_LEN},
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
print(f"[djev-init] ENGINE_READY=True (in-process vLLM ready) at t={time.time() - T_BOOT:.2f}s", flush=True)

while True:
    time.sleep(3600)
PYEOF

exec "$PY_BIN" /tmp/run_inproc_server.py
