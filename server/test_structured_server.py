"""End-to-end check of structured_server.py against a fake vLLM upstream.

Needs only the tokenizer: run inside the dgemma image with /models/dgemma
mounted, or set TOKENIZER to a local copy and CHAT_TEMPLATE to a jinja file when
that copy ships without one.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json, math, os, sys, threading, time, typing, urllib.request

sys.path.insert(
    0, os.environ.get("SERVER_DIR", os.path.dirname(os.path.abspath(__file__)))
)
S: typing.Any = __import__("structured_server")

SEEN: list[typing.Any] = []
CONF = [0.7]  # first-label probability the fake returns at every slot
THOUGHT = None  # token ids the fake writes when asked to think


class Fake(BaseHTTPRequestHandler):

  def log_message(self, format: str, *args: typing.Any) -> None:
    pass

  def do_POST(self):
    req = json.loads(self.rfile.read(int(self.headers["content-length"])))
    req["_path"] = self.path
    SEEN.append(req)
    if (
        "vllm_xargs" not in req
        and self.path.endswith("/v1/chat/completions")
        and "stop_token_ids" in req
    ):
      # a thought written through the chat endpoint (image states)
      toks = [
          {"token": f"token_id:{i}", "logprob": -0.1}
          for i in S.THOUGHT_OPEN + THOUGHT + S.THOUGHT_CLOSE
      ][: req["max_tokens"]]
      body = json.dumps({
          "choices": [{
              "message": {"content": ""},
              "logprobs": {"content": toks},
              "finish_reason": "stop",
          }],
          "usage": {"prompt_tokens": 400},
      }).encode()
      self.send_response(200)
      self.send_header("content-type", "application/json")
      self.send_header("content-length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
      return
    if "vllm_xargs" not in req and self.path.endswith("/v1/chat/completions"):
      # a raw chat completion passed through
      body = json.dumps({
          "choices": [
              {"message": {"role": "assistant", "content": "raw reply"}}
          ],
          "usage": {"prompt_tokens": 5},
      }).encode()
      self.send_response(200)
      self.send_header("content-type", "application/json")
      self.send_header("content-length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
      return
    if "vllm_xargs" not in req:
      # a thought: the fake writes a fixed one and closes the channel
      toks = [f"token_id:{i}" for i in THOUGHT + S.THOUGHT_CLOSE][
          : req["max_tokens"]
      ]
      body = json.dumps({
          "choices": [{"logprobs": {"tokens": toks}, "finish_reason": "stop"}],
          "usage": {},
      }).encode()
      self.send_response(200)
      self.send_header("content-type", "application/json")
      self.send_header("content-length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
      return
    x = req["vllm_xargs"]
    canvas = x["diffusion_seed_canvas"]
    assert len(canvas) == x["diffusion_canvas_length"] <= S.CANVAS_LEN
    assert x["diffusion_read_only"] is True and x["diffusion_max_steps"] == 1
    # Every position carries the seed id near-certain plus, for each label
    # family, the first label at CONF and the rest sharing the remainder,
    # so the fake needs no knowledge of where the slots are.
    content = []
    for pos, tid in enumerate(canvas[: req["max_tokens"]]):
      top = [{"token": f"token_id:{tid}", "logprob": -0.01}]
      for ids in FAMILIES:
        p = [CONF[0]] + [(1 - CONF[0]) / (len(ids) - 1)] * (len(ids) - 1)
        top += [
            {"token": f"token_id:{i}", "logprob": math.log(pi)}
            for i, pi in zip(ids, p)
        ]
      content.append(
          {"token": f"token_id:{tid}", "logprob": -0.01, "top_logprobs": top}
      )
    if self.path.endswith("/v1/completions"):
      rows = [
          {tp["token"]: tp["logprob"] for tp in c["top_logprobs"]}
          for c in content
      ]
      body = json.dumps({
          "choices": [{"logprobs": {"top_logprobs": rows}}],
          "usage": {"prompt_tokens": 321},
      }).encode()
    else:
      body = json.dumps({
          "choices": [{"logprobs": {"content": content}}],
          "usage": {"prompt_tokens": 321},
      }).encode()
    self.send_response(200)
    self.send_header("content-type", "application/json")
    self.send_header("content-length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)


SCHEMA = {
    "questions": [
        {
            "id": "urgent",
            "type": "noul",
            "instructions": "Does the customer need a reply within the hour?",
        },
        {
            "id": "bucket",
            "type": "choice",
            "instructions": "Which team owns this?",
            "options": [
                {"name": "billing"},
                {"name": "outage", "description": "service down"},
                {"name": "feature"},
            ],
        },
        {
            "id": "tone",
            "type": "score",
            "instructions": "How angry is the customer?",
            "levels": ["calm", "annoyed", "furious"],
        },
    ],
    "samples": 3,
}

S.ARGS = type(
    "A", (), {"upstream": "http://127.0.0.1:8998", "model": "dgemma"}
)()
try:
  tok = __import__("transformers").AutoTokenizer.from_pretrained(
      os.environ.get("TOKENIZER", os.environ.get("MODEL", "/models/dgemma"))
  )
  if tok.chat_template is None and os.environ.get("CHAT_TEMPLATE"):
    tok.chat_template = open(os.environ["CHAT_TEMPLATE"]).read()
except ModuleNotFoundError:
  import re

  class _FallbackTokenizer:
    _PAT = re.compile(
        r"<\|channel>thought\n|<channel\|>|<\|think\|>|w\d+|yes|no|[A-Za-z_]{1,4}|:\s|\n|\s|."
    )

    def __init__(self):
      self._s2i: dict[str, int] = {}
      self._i2s: dict[int, str] = {}

    def _id(self, s: str) -> int:
      if s not in self._s2i:
        idx = len(self._s2i) + 200
        self._s2i[s] = idx
        self._i2s[idx] = s
      return self._s2i[s]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
      return [self._id(m.group(0)) for m in self._PAT.finditer(text)]

    def decode(self, ids: list[int]) -> str:
      return "".join(self._i2s.get(i, "") for i in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, typing.Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> list[int]:
      prefix = "<|think|>\n" if enable_thinking else ""
      body = "\n".join(
          m.get("content", "") if isinstance(m.get("content"), str) else ""
          for m in messages
      )
      return self.encode(prefix + body)

  tok = _FallbackTokenizer()
S.init_tokenizer(tok)
THOUGHT = S.enc("The customer says everything is down, so this is urgent.")
schema = S.parse_schema(SCHEMA)
TEMPLATE, SLOTS = S.resolve_template(
    schema["questions"], S.SCAFFOLD, "", "lines"
)
FAMILIES = [s["label_ids"] for s in SLOTS]
print(
    "template tokens",
    len(TEMPLATE),
    "slots",
    [
        (q["id"], s["pos"], [S.TOK.decode([i]) for i in s["label_ids"]])
        for q, s in zip(schema["questions"], SLOTS)
    ],
)
print("system prompt:\n" + S.system_text(schema))

threading.Thread(
    target=ThreadingHTTPServer(("127.0.0.1", 8998), Fake).serve_forever,
    daemon=True,
).start()
threading.Thread(
    target=ThreadingHTTPServer(("127.0.0.1", 8999), S.Handler).serve_forever,
    daemon=True,
).start()
time.sleep(0.3)


def post(body):
  req = urllib.request.Request(
      "http://127.0.0.1:8999/v1/chat/completions",
      data=json.dumps(body).encode(),
      headers={"content-type": "application/json"},
  )
  try:
    r = urllib.request.urlopen(req)
    return r.status, json.load(r)
  except urllib.error.HTTPError as e:
    return e.code, json.load(e)


code, d = post({
    "model": "x",
    "messages": [
        {"role": "system", "content": json.dumps(SCHEMA)},
        {
            "role": "user",
            "content": json.dumps(
                {"ticket": "Everything is down and I am furious, fix it now"}
            ),
        },
    ],
})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
print(json.dumps(out["answers"], indent=1))
assert (
    out["answers"]["urgent"]["label"] == "yes"
    and abs(out["answers"]["urgent"]["noul"] - 0.7) < 1e-6
)
assert (
    out["answers"]["bucket"]["choice"] == "billing"
    and out["answers"]["tone"]["level"] == "calm"
)
assert (
    out["diagnostics"]["samples"]["n"] == 3
    and out["answers"]["urgent"]["stderr"] < 1e-9
    and out["answers"]["urgent"]["agreement"] == 1.0
)
assert (
    len(SEEN) == 3
    and len({tuple(r["vllm_xargs"]["diffusion_seed_canvas"]) for r in SEEN})
    == 3
), "reads must differ in their noise"
assert all(
    r["chat_template_kwargs"] == {"enable_thinking": False}
    and "ignore_eos" not in r
    for r in SEEN
)
assert (
    out["diagnostics"]["thought"] is None
    and d["usage"]["completion_tokens"] == len(TEMPLATE) + 1
)
print(
    "fixed samples ok; upstream saw",
    len(SEEN),
    "reads, max_tokens",
    SEEN[0]["max_tokens"],
)

# auto policy: an uncertain first read extends to max, a confident one stops at one
auto = dict(SCHEMA)
auto.pop("samples")
for conf, want in [(0.7, 4), (0.999, 1)]:
  SEEN.clear()
  CONF[0] = conf
  code, d = post({
      "messages": [
          {"role": "system", "content": json.dumps(auto)},
          {"role": "user", "content": "{}"},
      ]
  })
  out = json.loads(d["choices"][0]["message"]["content"])
  print(
      "auto conf",
      conf,
      out["diagnostics"]["samples"]["policy"],
      "reads",
      len(SEEN),
  )
  assert out["diagnostics"]["samples"]["n"] == want and len(SEEN) == want
  dq = out["diagnostics"]["questions"]
  assert [dq[q]["pos"] for q in ("urgent", "bucket", "tone")] == [
      s["pos"] for s in SLOTS
  ]
  assert all(len(dq[q]["entropy"]) == want for q in dq)
  assert out["diagnostics"]["samples"]["policy"]["first_read_entropy"] == {
      q: dq[q]["entropy"][0] for q in dq
  }
CONF[0] = 0.7

# think: one generation in the thought channel, then reads that carry it in their prompt
SEEN.clear()
code, d = post({
    "messages": [
        {
            "role": "system",
            "content": json.dumps(dict(SCHEMA, samples=2, think=64)),
        },
        {"role": "user", "content": "{}"},
    ]
})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
gen, reads = SEEN[0], SEEN[1:]
assert (
    gen["_path"].endswith("/v1/completions")
    and gen["max_tokens"] == 64
    and gen["stop_token_ids"] == S.THOUGHT_CLOSE
)
assert gen["prompt"][
    -len(S.THOUGHT_OPEN) :
] == S.THOUGHT_OPEN and "<|think|>" in S.TOK.decode(
    gen["prompt"]
), "thinking on, open tag last"
assert len(reads) == 2 and all(
    r["prompt"] == gen["prompt"] + THOUGHT + S.THOUGHT_CLOSE for r in reads
), "reads continue the closed thought"
assert all(
    r["vllm_xargs"]["diffusion_seed_canvas"][: len(S.SCAFFOLD)] != S.SCAFFOLD
    for r in reads
), "no second thought block on the canvas"
th = out["diagnostics"]["thought"]
assert (
    th["tokens"] == len(THOUGHT) and th["closed"] and "urgent" in th["text"]
), th
assert (
    out["answers"]["urgent"]["label"] == "yes"
    and out["diagnostics"]["samples"]["n"] == 2
)
assert d["usage"]["completion_tokens"] == len(THOUGHT) + reads[0]["max_tokens"]
print("think ok: thought", repr(th["text"]), "then", len(reads), "reads")

# chunking: twelve yes/no questions do not fit 32 rows, so the server splits them
SEEN.clear()
CONF[0] = 0.7
many = {
    "questions": [
        {"id": f"w{i}", "type": "noul", "instructions": f"word {i}?"}
        for i in range(12)
    ],
    "samples": 1,
    "chunk_rows": 32,
}
assert (
    schema["format"] == "lines" and S.parse_schema(many)["format"] == "indexed"
)
code, d = post({
    "messages": [
        {
            "role": "system",
            "content": json.dumps(dict(many, chunk_prompt="shared")),
        },
        {"role": "user", "content": "{}"},
    ]
})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
chunks = out["diagnostics"]["chunks"]
assert len(chunks) > 1 and sum(len(c) for c in chunks) == 12, chunks
assert all(out["answers"][f"w{i}"]["label"] == "yes" for i in range(12))
assert (
    len(SEEN) == len(chunks)
    and len({r["messages"][0]["content"] for r in SEEN}) == 1
), "chunks share one system prompt"
assert all(r["vllm_xargs"]["diffusion_canvas_length"] <= 32 for r in SEEN)
print("chunked:", chunks, "reads", len(SEEN))
SEEN.clear()
code, d = post({
    "messages": [
        {"role": "system", "content": json.dumps(many)},
        {"role": "user", "content": "{}"},
    ]
})
assert code == 200 and len({r["messages"][0]["content"] for r in SEEN}) == len(
    chunks
), "own prompts (the default) differ per chunk"
print("chunked with own prompts ok")

# parallel chunks each think for themselves
SEEN.clear()
code, d = post({
    "messages": [
        {"role": "system", "content": json.dumps(dict(many, think=64))},
        {"role": "user", "content": "{}"},
    ]
})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
assert len(SEEN) == 2 * len(chunks) and sum(
    "vllm_xargs" not in r for r in SEEN
) == len(chunks)
assert [t["tokens"] for t in out["diagnostics"]["thought"]] == [
    len(THOUGHT)
] * len(chunks)
print("chunked with a thought per chunk ok")

# sequential chunks: chunk two continues chunk one's answer in the prompt
SEEN.clear()
code, d = post({
    "messages": [
        {"role": "system", "content": json.dumps(dict(many, sequential=True))},
        {"role": "user", "content": "{}"},
    ]
})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
assert out["diagnostics"]["sequential"] and len(SEEN) == len(chunks)
first, second = SEEN[0], SEEN[1]
assert first["_path"].endswith("/v1/chat/completions") and "messages" in first
assert second["_path"].endswith("/v1/completions") and isinstance(
    second["prompt"], list
)
tail = S.TOK.decode(second["prompt"][-40:])
assert (
    "<channel|>" in tail
    and tail.endswith(chunks[0][-1] + "yes")
    and "w0yes w1yes" in tail
), tail
print("sequential ok: chunk 2 prompt ends", repr(tail[-40:]))

# sequential chunks share one thought, written under the full question list
SEEN.clear()
code, d = post({
    "messages": [
        {
            "role": "system",
            "content": json.dumps(dict(many, sequential=True, think=64)),
        },
        {"role": "user", "content": "{}"},
    ]
})
assert code == 200, d
out = json.loads(d["choices"][0]["message"]["content"])
gen, first, second = SEEN[0], SEEN[1], SEEN[2]
assert len(SEEN) == 1 + len(chunks) and "vllm_xargs" not in gen
assert (
    first["prompt"] == gen["prompt"] + THOUGHT + S.THOUGHT_CLOSE
), "chunk 1 reads right after the thought"
assert second["prompt"][: len(first["prompt"])] == first[
    "prompt"
] and S.TOK.decode(second["prompt"][-40:]).endswith(chunks[0][-1] + "yes")
assert isinstance(out["diagnostics"]["thought"], dict) and out["diagnostics"][
    "thought"
]["tokens"] == len(THOUGHT)
assert d["usage"]["completion_tokens"] == len(THOUGHT) + sum(
    r["max_tokens"] for r in SEEN[1:]
)
print("sequential with one thought ok")


# Jev's contract on /v1/systemone
def post_s1(body):
  req = urllib.request.Request(
      "http://127.0.0.1:8999/v1/systemone",
      data=json.dumps(body).encode(),
      headers={"content-type": "application/json"},
  )
  try:
    r = urllib.request.urlopen(req)
    return r.status, json.load(r)
  except urllib.error.HTTPError as e:
    return e.code, json.load(e)


SEEN.clear()
jev = {
    "model": "jev-latest",
    "state": {"ticket": "Everything is down and I am furious"},
    "questions": {
        "urgent": {
            "type": "noul",
            "instructions": "Does the customer need a reply within the hour?",
            "criteria": {"true": "needs a reply now", "false": "can wait"},
        },
        "bucket": {
            "type": "choice",
            "instructions": "Which team owns this?",
            "criteria": {
                "billing": None,
                "outage": "service down",
                "feature": None,
            },
        },
        "tone": {
            "type": "score",
            "instructions": "How angry is the customer?",
            "criteria": ["calm", "annoyed", "furious"],
        },
    },
    "samples": 2,
}
code, d = post_s1(jev)
assert code == 200, d
assert d["model"] == "dgemma" and list(d["answers"]) == [
    "urgent",
    "bucket",
    "tone",
]
u, b, t = d["answers"]["urgent"], d["answers"]["bucket"], d["answers"]["tone"]
assert set(u) == {"type", "noul"} and abs(u["noul"] - 0.7) < 1e-6
assert (
    b["type"] == "choice"
    and b["choice"] == "billing"
    and list(b["probabilities"]) == ["billing", "outage", "feature"]
    and abs(b["confidence"] - 0.7) < 1e-6
)
assert (
    t["type"] == "score"
    and t["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}
    and list(t["probabilities"]) == ["0", "1", "2"]
    and abs(t["score"] - 0.45) < 1e-6
)
assert d["usage"] == {
    "input_tokens": 321,
    "output_tokens": len(TEMPLATE) + 1,
}, d["usage"]
assert d["diagnostics"]["samples"]["n"] == 2 and len(SEEN) == 2
sysprompt = SEEN[0]["messages"][0]["content"]
assert (
    "yes: needs a reply now" in sysprompt
    and "no: can wait" in sysprompt
    and "outage (service down)" in sysprompt
), sysprompt
assert SEEN[0]["messages"][1]["content"] == json.dumps(jev["state"])
code, d = post_s1(dict(jev, state="Everything is down, fix it now", samples=1))
assert (
    code == 200
    and SEEN[-1]["messages"][1]["content"] == "Everything is down, fix it now"
), "a text state goes through as text"
for body, want in [
    (dict(jev, questions={}), "non-empty map"),
    (
        dict(
            jev,
            questions={
                "a": {
                    "type": "choice",
                    "instructions": "?",
                    "criteria": ["x", "y"],
                }
            },
        ),
        "map option names",
    ),
    (
        dict(
            jev,
            questions={
                "a": {
                    "type": "score",
                    "instructions": "?",
                    "criteria": ["only"],
                }
            },
        ),
        "at least two",
    ),
    (
        dict(jev, questions={"a": {"type": "rank", "instructions": "?"}}),
        "unknown type",
    ),
    ({"model": "jev-latest", "questions": jev["questions"]}, "state: required"),
]:
  code, d = post_s1(body)
  assert code == 422 and want in d["error"]["message"], (code, d)
print("systemone ok")

# images: multipart file parts beside the request JSON, or data URLs in "images"
import base64

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
PNG_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()
BOUNDARY = "xxDJEVxx"


def multipart(parts):
  out = b""
  for name, filename, ctype, data in parts:
    head = (
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"'
        + (f'; filename="{filename}"' if filename else "")
        + f"\r\nContent-Type: {ctype}\r\n\r\n"
    )
    out += head.encode() + data + b"\r\n"
  return out + f"--{BOUNDARY}--\r\n".encode()


def post_mp(parts):
  req = urllib.request.Request(
      "http://127.0.0.1:8999/v1/systemone",
      data=multipart(parts),
      headers={"content-type": f"multipart/form-data; boundary={BOUNDARY}"},
  )
  try:
    r = urllib.request.urlopen(req)
    return r.status, json.load(r)
  except urllib.error.HTTPError as e:
    return e.code, json.load(e)


one = dict(jev, samples=1)
SEEN.clear()
code, d = post_mp([
    ("request", None, "application/json", json.dumps(one).encode()),
    ("photo", "a.png", "image/png", PNG),
    ("photo2", "b.png", "image/png", PNG),
])
assert code == 200, d
content = SEEN[-1]["messages"][1]["content"]
assert content == [{"type": "image_url", "image_url": {"url": PNG_URL}}] * 2 + [
    {"type": "text", "text": json.dumps(one["state"])}
], content
assert (
    d["usage"]["input_tokens"] == 321 and d["answers"]["urgent"]["noul"] > 0.5
)
SEEN.clear()
code, d = post_s1(
    dict(
        one,
        images=[
            PNG_URL,
            {
                "content_type": "image/png",
                "base64": base64.b64encode(PNG).decode(),
            },
        ],
    )
)
assert (
    code == 200
    and SEEN[-1]["messages"][1]["content"][:2]
    == [{"type": "image_url", "image_url": {"url": PNG_URL}}] * 2
), d
code, d = post_s1(dict(one, images=["not an image"]))
assert code == 422 and "images[0]" in d["error"]["message"], d
# a thought with an image: written through the chat endpoint, seeded into the canvas
SEEN.clear()
code, d = post_s1(dict(one, images=[PNG_URL], think=64))
assert code == 200, d
gen, read = SEEN[0], SEEN[1]
assert (
    gen["_path"].endswith("/v1/chat/completions")
    and gen["stop_token_ids"] == S.THOUGHT_CLOSE
    and gen["chat_template_kwargs"] == {"enable_thinking": True}
)
assert (
    gen["messages"][1]["content"][0]["type"] == "image_url"
), "the thought sees the image"
canvas = read["vllm_xargs"]["diffusion_seed_canvas"]
assert (
    canvas[: len(S.THOUGHT_OPEN) + len(THOUGHT) + 1]
    == S.THOUGHT_OPEN + THOUGHT + S.THOUGHT_CLOSE
), "the read's canvas starts with the thought"
assert (
    read["chat_template_kwargs"] == {"enable_thinking": True}
    and read["messages"][1]["content"][0]["type"] == "image_url"
)
th = d["diagnostics"]["thought"]
assert (
    th["tokens"] == len(THOUGHT)
    and th["closed"]
    and th["budget"] <= 64
    and d["answers"]["urgent"]["noul"] > 0.5
), th
code, d = post_s1(dict(one, images=[PNG_URL], think=4096))
assert (
    code == 200 and d["diagnostics"]["thought"]["budget"] < 4096
), "the canvas bounds the thought"
code, d = post_s1(dict(one, images=[PNG_URL], sequential=True, chunk_rows=8))
assert (
    code == 200
    and d["diagnostics"]["conditioning"] == "restated"
    and len(SEEN) >= 4
), d["diagnostics"]["chunks"]
assert (
    "Answers so far:"
    in typing.cast(typing.Any, SEEN[-1])["messages"][1]["content"][-1]["text"]
), "sequential chunks with images restate the earlier answers"
code, d = post_mp([
    ("request", None, "application/json", b"{}"),
    ("notes", "n.txt", "text/plain", b"hi"),
])
assert code == 400 and "neither" in d["error"]["message"], d
print("images ok")


# the playground page, only with TEST_PAGE=1
def get(path):
  try:
    r = urllib.request.urlopen("http://127.0.0.1:8999" + path)
    return r.status, r.read()
  except urllib.error.HTTPError as e:
    return e.code, e.read()


S.TEST_PAGE = False
assert get("/")[0] == 404 and get("/health")[0] == 200
S.TEST_PAGE = True
code, page = get("/")
assert code == 200 and b"djev playground" in page and b"/v1/systemone" in page
assert (
    get("/playground")[0] == 200
    and get("/playground.html")[0] == 200
    and get("/other")[0] == 404
)
code, page = get("/walk")
assert code == 200 and b"all clear ahead" in page and b"facingMode" in page
assert get("/cube")[0] == 200 and b"Cube Rule Live" in get("/cube")[1]
S.TEST_PAGE = False
assert get("/walk")[0] == 404 and get("/cube")[0] == 404
print("playground ok")


# raw passthrough and the bearer token
def post_raw(path, body, headers=None):
  req = urllib.request.Request(
      "http://127.0.0.1:8999" + path,
      data=json.dumps(body).encode(),
      headers={"content-type": "application/json", **(headers or {})},
  )
  try:
    r = urllib.request.urlopen(req)
    return r.status, json.load(r)
  except urllib.error.HTTPError as e:
    return e.code, json.load(e)


SEEN.clear()
code, d = post_raw(
    "/v1/raw/chat/completions",
    {
        "model": "dgemma",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    },
)
assert (
    code == 200
    and d["choices"][0]["message"]["content"] == "raw reply"
    and SEEN[-1]["max_tokens"] == 8
), d
S.API_KEY = "s3cret"
assert post_raw("/v1/systemone", jev)[0] == 401
assert (
    post_raw("/v1/systemone", jev, {"authorization": "Bearer wrong"})[0] == 401
)
assert (
    post_raw(
        "/v1/systemone",
        dict(jev, samples=1),
        {"authorization": "Bearer s3cret"},
    )[0]
    == 200
)
assert post_raw("/v1/raw/chat/completions", {"messages": []})[0] == 401
assert get("/health")[0] == 200, "health stays open"
S.API_KEY = ""
print("raw passthrough and api key ok")

# https listener with a self-signed certificate
import ssl, tempfile

S.serve_tls("127.0.0.1", 8997, tempfile.mkdtemp())
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
r = urllib.request.urlopen("https://127.0.0.1:8997/health", context=ctx)
assert r.status == 200 and json.load(r) == {"status": "ok"}
print("tls ok")

# dependencies: stages, gating, alone reads, and the conditioning of later stages
DEPS = {
    "model": "jev-latest",
    "state": {"ticket": "Everything is down and I am furious"},
    "samples": 1,
    "questions": {
        "urgent": {
            "type": "noul",
            "instructions": "Does the customer need a reply within the hour?",
        },
        "bucket": {
            "type": "choice",
            "instructions": "Which team owns this?",
            "criteria": {"billing": None, "outage": None, "feature": None},
        },
        "escalate": {
            "type": "noul",
            "instructions": "Escalate to the on-call lead?",
            "depends_on": ["urgent", "bucket"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Offer a refund?",
            "ask_if": {"bucket": ["billing"]},
        },
        "outage_note": {
            "type": "noul",
            "instructions": "Post a status page note?",
            "ask_if": {"bucket": ["outage"]},
            "depends_on": ["urgent"],
        },
    },
}
SEEN.clear()
code, d = post_s1(DEPS)
assert code == 200, d
dg = d["diagnostics"]
assert dg["stages"] == [["urgent", "bucket"], ["escalate", "refund"]], dg[
    "stages"
]
assert (
    d["answers"]["outage_note"] is None
    and dg["skipped"]["outage_note"]["because"] == "bucket"
    and dg["skipped"]["outage_note"]["was"] == "billing"
)
assert (
    d["answers"]["refund"]["type"] == "noul"
    and d["answers"]["escalate"]["type"] == "noul"
)
assert list(d["answers"]) == [
    "urgent",
    "bucket",
    "escalate",
    "refund",
    "outage_note",
], "every id, in schema order"
assert len(SEEN) == 2 and dg["conditioning"] == "prefill"
first, second = SEEN
assert (
    first["_path"].endswith("/v1/chat/completions")
    and "Question refund" in first["messages"][0]["content"]
), "stage prompts list every question"
assert second["_path"].endswith("/v1/completions")
tail = S.TOK.decode(second["prompt"][-30:])
assert "urgent: yes" in tail and "bucket: A" in tail, tail
print("dependencies ok:", dg["stages"], "skipped", list(dg["skipped"]))

# alone: an isolated question reads on its own beside the joint read
SEEN.clear()
alone = json.loads(json.dumps(DEPS))
alone["questions"] = {
    "urgent": alone["questions"]["urgent"],
    "bucket": dict(alone["questions"]["bucket"], alone=True),
    "tone": {
        "type": "score",
        "instructions": "How angry?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}
code, d = post_s1(alone)
assert (
    code == 200
    and d["diagnostics"]["stages"] == [["urgent", "bucket", "tone"]]
    and d["diagnostics"]["chunks"] == [["urgent"], ["bucket"], ["tone"]]
), d["diagnostics"]["chunks"]
assert len(SEEN) == 3 and all(
    "Question bucket" not in r["messages"][0]["content"]
    for r in SEEN
    if "Question urgent" in r["messages"][0]["content"]
), "own prompts per read"
print("alone ok")

# images: later stages restate the earlier answers in the state text
SEEN.clear()
code, d = post_s1(dict(DEPS, images=[PNG_URL]))
assert code == 200, d
assert d["diagnostics"]["conditioning"] == "restated" and len(SEEN) == 2
second = SEEN[1]
assert second["_path"].endswith("/v1/chat/completions")
content = second["messages"][1]["content"]
assert (
    content[0]["type"] == "image_url"
    and "Answers so far:" in content[-1]["text"]
    and "bucket: billing" in content[-1]["text"]
    and "urgent: yes" in content[-1]["text"]
), content[-1]["text"]
print("dependencies with images ok")

# refusals
for body, want in [
    (
        dict(
            DEPS,
            questions=dict(
                DEPS["questions"],
                escalate=dict(
                    DEPS["questions"]["escalate"], depends_on=["nope"]
                ),
            ),
        ),
        "unknown question",
    ),
    (
        dict(
            DEPS,
            questions={
                "a": {"type": "noul", "instructions": "?", "depends_on": ["b"]},
                "b": {"type": "noul", "instructions": "?", "depends_on": ["a"]},
            },
        ),
        "cycle",
    ),
    (
        dict(
            DEPS,
            questions=dict(
                DEPS["questions"],
                refund=dict(
                    DEPS["questions"]["refund"], ask_if={"bucket": ["legal"]}
                ),
            ),
        ),
        "must be among",
    ),
    (dict(DEPS, ask=["escalate"]), "not everything it depends on"),
]:
  code, d = post_s1(body)
  assert code == 422 and want in d["error"]["message"], (code, d)
print("dependency refusals ok")

# bad requests
for body, want in [
    ({"messages": [{"role": "user", "content": "{}"}]}, "exactly two"),
    (
        {
            "messages": [
                {"role": "system", "content": "hello"},
                {"role": "user", "content": "{}"},
            ]
        },
        "JSON question schema",
    ),
    (
        {
            "messages": [
                {"role": "system", "content": json.dumps(SCHEMA)},
                {"role": "user", "content": "not json"},
            ]
        },
        "JSON question schema",
    ),
    (
        {
            "messages": [
                {
                    "role": "system",
                    "content": json.dumps({
                        "questions": [
                            {"id": "a", "type": "choice", "options": ["x"]}
                        ]
                    }),
                },
                {"role": "user", "content": "{}"},
            ]
        },
        "at least two",
    ),
    (
        {
            "messages": [
                {
                    "role": "system",
                    "content": json.dumps(
                        {"questions": [{"id": "a", "type": "noul"}] * 2}
                    ),
                },
                {"role": "user", "content": "{}"},
            ]
        },
        "duplicate",
    ),
    (
        {
            "messages": [
                {
                    "role": "system",
                    "content": json.dumps(
                        {"questions": [{"id": "q" * 300, "type": "noul"}]}
                    ),
                },
                {"role": "user", "content": "{}"},
            ]
        },
        "canvas holds",
    ),
    (
        {
            "messages": [
                {
                    "role": "system",
                    "content": json.dumps(dict(SCHEMA, think="lots")),
                },
                {"role": "user", "content": "{}"},
            ]
        },
        "think must be",
    ),
]:
  code, d = post(body)
  assert code == 400 and want in d["error"]["message"], (code, d)
  print("400 ok:", d["error"]["message"][:90])

# GET /v1/models
code, models_raw = get("/v1/models")
models_data = json.loads(models_raw)
assert code == 200 and models_data["object"] == "list", models_data
assert [m["id"] for m in models_data["data"]] == ["dgemma", "jev-latest"]
print("/v1/models ok")

# POST /v1/evaluate with "boolean", "choice", and "score" + providerMetadata
SEEN.clear()
eval_req = {
    "model": "jev-latest",
    "state": {
        "prompt": "Reset password",
        "response": "Click Settings > Security",
    },
    "samples": 1,
    "questions": {
        "faithful": {
            "type": "boolean",
            "instructions": "Is the response faithful to the prompt?",
        },
        "route": {
            "type": "choice",
            "instructions": "Which category fits best?",
            "criteria": {
                "billing": None,
                "security": "password or auth",
                "feature": None,
            },
        },
        "quality": {
            "type": "score",
            "instructions": "Rate overall response quality.",
            "criteria": ["poor", "acceptable", "excellent"],
        },
    },
}
code, eval_res = post_raw("/v1/evaluate", eval_req)
assert code == 200, eval_res
assert eval_res["answers"]["faithful"]["type"] == "boolean"
assert abs(eval_res["answers"]["faithful"]["noul"] - 0.7) < 1e-6
assert abs(eval_res["answers"]["faithful"]["probability"] - 0.7) < 1e-6
assert eval_res["answers"]["route"]["choice"] == "billing"
assert (
    abs(
        eval_res["providerMetadata"]["typesafe"]["confidence"]["faithful"] - 0.7
    )
    < 1e-6
)
print("/v1/evaluate ok")

# POST /v1/chat/completions with OpenAI / Vercel AI SDK response_format: json_schema
SEEN.clear()
code, chat_js = post({
    "model": "dgemma",
    "messages": [
        {
            "role": "system",
            "content": (
                "Evaluate the assistant response for faithfulness and tone."
            ),
        },
        {
            "role": "user",
            "content": (
                "User: How do I reset my password? Assistant: Go to Settings >"
                " Security."
            ),
        },
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "evaluation_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "faithful": {
                        "type": "boolean",
                        "description": "Is the answer grounded and accurate?",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["billing", "security", "feature"],
                        "description": "Primary support domain",
                    },
                    "score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                        "description": "Quality score from 1 to 3",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Calibrated summary",
                    },
                },
                "required": ["faithful", "category", "score", "reason"],
            },
        },
    },
})
assert code == 200, chat_js
parsed_obj = json.loads(chat_js["choices"][0]["message"]["content"])
assert parsed_obj["faithful"] is True, parsed_obj
assert parsed_obj["category"] == "billing", parsed_obj
assert parsed_obj["score"] in (1, 2, 3), parsed_obj
assert "Calibrated 1-step diffusion evaluation" in parsed_obj["reason"]
assert (
    abs(chat_js["providerMetadata"]["typesafe"]["confidence"]["faithful"] - 0.7)
    < 1e-6
)
assert "answers" in chat_js["djev"] and "diagnostics" in chat_js["djev"]
print("/v1/chat/completions json_schema ok")

print("ALL OK")
