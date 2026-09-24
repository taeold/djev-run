
import json, random, time, os, asyncio, urllib.request
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

_TOK_CACHE = {}

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
            qs = []
            questions_data = body.get("questions", {})
            if isinstance(questions_data, dict):
                for qid, qspec in questions_data.items():
                    qtype = qspec.get("type", "choice")
                    crit = qspec.get("criteria", [])
                    opts, labels = [], []
                    if qtype == "choice":
                        if isinstance(crit, list):
                            opts = [[str(x), None] for x in crit]
                        else:
                            opts = [[str(k), str(v) if v is not None else None] for k, v in crit.items()]
                        labels = [chr(97 + i) for i in range(len(opts))]
                    elif qtype == "score":
                        if isinstance(crit, list):
                            opts = [[str(x), str(x)] for x in crit]
                        else:
                            opts = []
                        labels = [str(i + 1) for i in range(len(opts))]
                    else: # noul
                        opts = [["yes", None], ["no", None]]
                        labels = ["yes", "no"]
                    qs.append({
                        "id": str(qid),
                        "type": qtype,
                        "instructions": qspec.get("instructions", ""),
                        "choices": opts,
                        "labels": labels
                    })
            elif isinstance(questions_data, list):
                qs = questions_data

            sys_text = "Answer a fixed set of questions about the state the user provides. Each question lists its allowed answers; reply with exactly one label per question.\n"
            for q in qs:
                sys_text += f"\nQuestion {q['id']}: {q.get('instructions', '').strip()}\n"
                for idx, (name, desc) in enumerate(q.get('choices', [])):
                    lbl = q.get('labels', [])[idx]
                    if q['type'] == "noul":
                        sys_text += f"  {lbl}\n"
                    elif desc:
                        sys_text += f"  {lbl}: {name} ({str(desc).strip()})\n"
                    else:
                        sys_text += f"  {lbl}: {name}\n"
            sys_text += '\nReply with one line per question, in this order, formatted as "id: label".'
            
            base_str = "\n".join(f"{q['id']}: {q['labels'][0]}" for q in qs)
            alt_str = "\n".join(f"{q['id']}: {q['labels'][min(1, len(q['labels']) - 1)]}" for q in qs)

            port = os.environ.get("PORT", "8080")
            cache_key = f"{base_str}|{alt_str}"
            if cache_key in _TOK_CACHE:
                base_toks, alt_toks = _TOK_CACHE[cache_key]
            else:
                def do_tok():
                    b_toks = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps({"prompt": base_str, "add_special_tokens": False}).encode(), headers={"Content-Type": "application/json"})).read())["tokens"]
                    a_toks = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps({"prompt": alt_str, "add_special_tokens": False}).encode(), headers={"Content-Type": "application/json"})).read())["tokens"]
                    return b_toks, a_toks
                try:
                    base_toks, alt_toks = await asyncio.to_thread(do_tok)
                    _TOK_CACHE[cache_key] = (base_toks, alt_toks)
                except Exception as e:
                    return await JSONResponse({"error": f"tokenize error {str(e)}"}, status_code=500)(scope, receive, send)
            
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
                payload = {
                    "model": "djev-dgemma",
                    "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": user_text}],
                    "max_tokens": len(base_toks),
                    "logprobs": True,
                    "top_logprobs": 20,
                    "vllm_xargs": {
                        "diffusion_seed_canvas": seed_canvas,
                        "diffusion_pinned": pinned,
                        "diffusion_max_steps": int(body.get("steps", 1)),
                        "diffusion_read_only": True
                    }
                }
                rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                            data=json.dumps(payload).encode(),
                                            headers={"Content-Type": "application/json"})
                return json.loads(urllib.request.urlopen(rq, timeout=10).read())
            
            try:
                out = await asyncio.to_thread(do_req)
            except Exception as e:
                return await JSONResponse({"error": f"vllm timeout {str(e)}"}, status_code=500)(scope, receive, send)

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
                    p_dict = {name: round(p, 6) for (name, _), p in zip(q.get("choices", []), probs)}
                    best_idx = max(range(len(probs)), key=lambda i: probs[i])
                    answers[q["id"]] = {"choice": q.get("choices", [])[best_idx][0], "probabilities": p_dict, "confidence": round(probs[best_idx], 6)}
                elif q["type"] == "score":
                    p_dict = {str(i + 1): round(p, 6) for i, p in enumerate(probs)}
                    sc = sum((i + 1) * p for i, p in enumerate(probs))
                    answers[q["id"]] = {"score": round(sc, 6), "probabilities": p_dict, "confidence": round(max(probs), 6)}
                else:
                    answers[q["id"]] = {"noul": round(probs[0], 6), "probability": round(probs[0], 6)}
            
            total_ms = round((time.time() - t0) * 1000, 2)
            answers_json = {"model": "djev-dgemma", "answers": answers, "diagnostics": {"timing": {"total_ms": total_ms}}}
            return await JSONResponse(answers_json)(scope, receive, send)

        return await self.app(scope, receive, send)
