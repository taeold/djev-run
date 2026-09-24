
import json, random, time, os, string, asyncio, urllib.request
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

ALL_LABEL_TOKEN_IDS = None
TOK = None

def init_globals():
    global ALL_LABEL_TOKEN_IDS, TOK
    if TOK is not None:
        return True
    try:
        from transformers import AutoTokenizer
        if not os.path.exists("/dev/shm/dgemma"):
            return False
        TOK = AutoTokenizer.from_pretrained("/dev/shm/dgemma")
        if TOK.chat_template is None and os.path.exists("/dev/shm/dgemma/chat_template.jinja"):
            with open("/dev/shm/dgemma/chat_template.jinja") as f:
                TOK.chat_template = f.read()
        cands = set()
        for s in list(string.ascii_lowercase) + [str(d) for d in range(10)] + ["yes", "no", "true", "false"]:
            for prefix in (" ", ""):
                ids = TOK.encode(prefix + s, add_special_tokens=False)
                if ids: cands.add(int(ids[-1]))
        ALL_LABEL_TOKEN_IDS = sorted(cands)[:128]
        return True
    except Exception as e:
        print("Tokenizer load err:", e)
        return False

class SystemOneMiddleware:
    def __init__(self, app):
        self.app = app
        self.pages = {
            "/": "snake.html", "/snake": "snake.html", "/snake.html": "snake.html",
            "/dino": "dino.html", "/dino.html": "dino.html",
            "/tetris": "tetris.html", "/tetris.html": "tetris.html",
        }

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
            
        req = Request(scope, receive)
        path = req.url.path

        if req.method == "GET" and path in self.pages:
            p = os.path.join("/opt/dgemma", self.pages[path])
            if not os.path.exists(p): p = os.path.join(os.getcwd(), self.pages[path])
            try:
                with open(p, "rb") as f:
                    return await HTMLResponse(content=f.read())(scope, receive, send)
            except Exception as e:
                return await HTMLResponse(content=str(e), status_code=500)(scope, receive, send)

        if req.method == "POST" and path == "/v1/systemone":
            try:
                body = await req.json()
            except:
                return await JSONResponse({"error": "invalid json"}, status_code=400)(scope, receive, send)
            
            t0 = time.time()
            qs = body.get("questions", [])
            if not init_globals():
                return await JSONResponse({"error": "tokenizer loading"}, status_code=503)(scope, receive, send)
            
            sys_lines = ["\ntype Question = " + q["type"] for q in qs]
            sys_lines.append("\n" + "\n".join(f"{q['id']}: Question" for q in qs))
            sys_text = "\n".join(sys_lines)
            
            base_str = "\n".join(f"{q['id']}: {q['labels'][0]}" for q in qs)
            alt_str = "\n".join(f"{q['id']}: {q['labels'][min(1, len(q['labels']) - 1)]}" for q in qs)
            base_toks = TOK.encode(base_str, add_special_tokens=False)
            alt_toks = TOK.encode(alt_str, add_special_tokens=False)
            
            slot_positions = [i for i in range(len(base_toks)) if base_toks[i] != alt_toks[i]]
            seed_canvas = list(base_toks)
            
            rng = random.Random(42)
            for pos in slot_positions:
                seed_canvas[pos] = rng.randrange(262144)
            pinned = [i for i in range(len(base_toks)) if i not in set(slot_positions)]
            
            while len(seed_canvas) < 128:
                seed_canvas.append(1)

            state_val = body.get("state", "")
            user_text = state_val if isinstance(state_val, str) else json.dumps(state_val)

            def do_req():
                port = os.environ.get("PORT", "8080")
                payload = {
                    "model": "djev-dgemma",
                    "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": user_text}],
                    "max_tokens": len(base_toks),
                    "logprobs": True,
                    "top_logprobs": 20,
                    "logprob_token_ids": ALL_LABEL_TOKEN_IDS,
                    "vllm_xargs": {
                        "diffusion_seed_canvas": seed_canvas,
                        "diffusion_pinned": pinned,
                        "diffusion_max_steps": int(body.get("steps", 1)),
                        "diffusion_constrained": True
                    }
                }
                rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                            data=json.dumps(payload).encode(),
                                            headers={"Content-Type": "application/json"})
                return json.loads(urllib.request.urlopen(rq, timeout=10).read())
            
            try:
                out = await asyncio.to_thread(do_req)
            except Exception as e:
                return await JSONResponse({"error": str(e)}, status_code=500)(scope, receive, send)

            import math
            answers = {}
            for q, pos in zip(qs, slot_positions):
                choices_array = out["choices"][0]["logprobs"]["content"]
                if pos >= len(choices_array):
                    lbl_map = {lbl: -25.0 for lbl in q["labels"]}
                else:    
                    lp_dict = choices_array[pos]["top_logprobs"]
                    lbl_map = {}
                    for top in lp_dict:
                        t_str = str(top["token"]).strip()
                        val = float(top["logprob"])
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
            answers_json = {"answers": answers, "diagnostics": {"timing": {"total_ms": total_ms}}}
            return await JSONResponse(answers_json)(scope, receive, send)

        return await self.app(scope, receive, send)
