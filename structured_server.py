"""Structured decisions in front of a vLLM DiffusionGemma server.

POST /v1/systemone takes Jev's request body: {"model", "state", "questions"}.
"questions" maps an id to {"type", "instructions", "criteria"}, where "type"
is "noul", "choice", "score", "span" or "spans" and the criteria shape
follows the type:
  noul:   optional {"true": ..., "false": ...} descriptions
  choice: option name -> description or null
  score:  ordered list of levels
  span:   optional {"max_tokens": n}; the answer is a piece of the text
  spans:  optional {"max_tokens": n, "max_items": n}; every such piece
Answers take Jev's shapes, with this server's diagnostics alongside:
  noul:   {"noul": p}
  choice: {"choice", "probabilities", "confidence"}
  score:  {"score", "legend", "probabilities", "confidence"}
  span:   {"found", "text", "start", "end", "confidence", "coverage"}
  spans:  {"found", "items": [{"text", "start", "end", "confidence"}]}
A span is a substring of the state's text, by construction: the model fills
a pinned blank by copying, only the text's own token ids are read, and the
decode walks the text with those tokens. "start" and "end" are character
offsets into the state when it is a string, else into its "text" field.
The request body may also carry the schema keys "instructions", "samples",
"auto_max", "auto_threshold", "steps", "think", "ask", "chunk_rows",
"chunk_prompt" and "sequential" as extensions. Images go ahead of the
state, either as multipart/form-data with the JSON body in a part named
"request" and each image as a file part, or as an "images" array of data
URLs in the JSON body.

POST /v1/chat/completions makes the same decision from an OpenAI-shaped
call. The system message is the schema JSON below and the user message is
the state. The reply's `content` is the JSON answer set.

POST /v1/raw/chat/completions passes the body to vLLM's chat completions
unchanged, for plain generation through this port.

With API_KEY set in the environment, every POST needs "Authorization:
Bearer <key>". --tls-port adds an HTTPS listener with a self-signed
certificate kept in --cert-dir, for clients that need a secure origin.

Each answer is one distribution per question, from one denoise step over a
seeded canvas, averaged over a few noise draws. This server handles the
canvas, tokenizer, slot resolution, noise draws and averaging.

Schema (system message):
  {"questions": [
     {"id": "urgent", "type": "noul", "instructions": "..."},
     {"id": "bucket", "type": "choice", "instructions": "...",
      "options": [{"name": "billing", "description": "..."}, ...]},
     {"id": "tone", "type": "score", "instructions": "...",
      "levels": ["calm", "annoyed", "furious"]}],
   "instructions": "optional context",
   "samples": "auto" | N, "auto_threshold": 0.1, "auto_max": 4,
   "steps": 1, "think": 0}

A question may also declare:
  "depends_on": [ids]        answered after those, with their answers in
                             its prompt
  "ask_if": {id: [answers]}  asked only when that question's answer is
                             among them (a skipped answer is null)
  "alone": true              a read of its own
Questions run in stages by these dependencies. Each stage is one joint
read. Later stages continue the earlier answers, prefilled for a text
state and restated for an image.

Up to ten questions answer as "id: label" lines. Past that the id runs
straight into the label, space separated, one row fewer per question. The
server splits a schema whose answer template does not fit the canvas into
chunks that run together, each with its own question list. "chunk_rows"
sets the rows per chunk, "ask" picks a subset of question ids for one
read, "sequential": true runs the chunks in order with the earlier answers
prefilled, and "chunk_prompt": "shared" lists every question in each
chunk's prompt. "think": N lets the model write up to N tokens in its
thought channel, as an ordinary generation, and the read then runs with
that thought in its prompt. With images the model writes the thought with
the image in view and the server seeds it into the canvas ahead of the
answer, so the canvas bounds it. The noise draws of a decision share one
thought.

Serve the model with a canvas that holds the answer template, for example:
  vllm serve google/diffusiongemma-26B-A4B-it \
      --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 \
      --enable-prefix-caching
then run this in front of it:
  python structured_server.py --upstream http://127.0.0.1:8000 \
      --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64 --port 8011
"""

import argparse
import json
import math
import os
import random
import re
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import pybase64 as base64
except ImportError:
    import base64
from transformers import AutoTokenizer

ARGS = None
TOK = None
TEST_PAGE = os.environ.get("TEST_PAGE", "") == "1"
PAGES = {
    "/": "snake.html",
    "/snake": "snake.html",
    "/snake.html": "snake.html",
    "/dino": "dino.html",
    "/dino.html": "dino.html",
    "/tetris": "tetris.html",
    "/tetris.html": "tetris.html",
    "/playground": "playground.html",
    "/playground.html": "playground.html",
    "/walk": "walk.html",
    "/cube": "cube.html",
}
API_KEY = os.environ.get(
    "API_KEY", ""
)  # when set, POST routes need "Authorization: Bearer <key>"
CANVAS_LEN = 64  # the served canvas length. A request may be narrower.
CANVAS_STEP = 16  # request widths are multiples of this
VOCAB = 262144
TURN_CLOSE = 106
PAD = 0
TOPK = 20
MAX_QUESTIONS = 64  # per request
MAX_SAMPLES = 32  # reads per question, fixed or auto
MAX_PARALLEL = 16  # question groups read at once
SPAN_STEPS = (
    4  # denoise steps for a span blank; one step spells it position by position
)
SPAN_REGION = 24  # default blank for one span, in tokens
SPAN_LIST_REGION = 64  # default blank for a list of spans
SPAN_ID_CAP = 128  # vLLM's logprob_token_ids cap: the text's ids must fit
SPAN_MIN_CONF = 0.3  # below this a span read is repeated with the next seed
SPAN_MAX_READS = 3
SPAN_PASSES = 2  # list reads per window, merged by offset
SPAN_FLOOR = -30.0
SPAN_BOUNDARY = set(" \t\n,;:.!?()[]{}\"'#$")
EOS = 1
NL = None
SPAN_SYSTEM = (
    "Answer with the exact value copied from the text, character for character, "
    "or none if the text has no such value.{context}\n\nQuestion answer: {q}\n"
    "  (the value only)\n\n"
    'Reply with one line per question, in this order, formatted as "id: label".'
)
SPAN_LIST_SYSTEM = (
    "List every value of the requested kind found in the text, one per line, "
    "each copied exactly as it appears, character for character, in the order "
    "they appear. Write none if there is none.{context}\n\nQuestion answer: {q}\n"
    "  (one value per line)\n\n"
    'Reply with the answer lines, formatted as "id: value".'
)
SPAN_CHOOSE_SYSTEM = (
    "Several candidate answers were copied from the text. Choose the one that "
    "is exactly the right answer, no more and no less.{context}\n\n"
    "Question answer: {q}\nCandidates:\n{options}\n\n"
    'Reply with one line, formatted as "id: label".'
)
# the empty thought block the chat template leaves to the model
SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"
SCAFFOLD = None
THOUGHT_OPEN = None
THOUGHT_CLOSE = None


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------


class SchemaError(ValueError):
    pass


def parse_schema(value):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("questions"), list)
        or not value["questions"]
    ):
        raise SchemaError("schema: needs a non-empty questions array")
    if len(value["questions"]) > MAX_QUESTIONS:
        raise SchemaError(f"schema: at most {MAX_QUESTIONS} questions")
    qs = []
    seen = set()
    for q in value["questions"]:
        qid = str(q.get("id", "")).strip()
        if not qid or ":" in qid or "\n" in qid:
            raise SchemaError(
                f"question id {qid!r} must be non-empty, no ':' or newline"
            )
        if qid in seen:
            raise SchemaError(f"duplicate question id {qid!r}")
        seen.add(qid)
        kind = q.get("type")
        if kind in ("noul", "bool", "boolean"):
            kind = "noul"
            crit = q.get("criteria") or {}
            choices = [("yes", crit.get("true")), ("no", crit.get("false"))]
            labels = ["yes", "no"]
        elif kind == "choice":
            opts = q.get("options") or []
            choices = [
                (o["name"], o.get("description"))
                if isinstance(o, dict)
                else (str(o), None)
                for o in opts
            ]
            labels = [chr(ord("A") + i) for i in range(len(choices))]
        elif kind == "score":
            choices = [(str(level), None) for level in (q.get("levels") or [])]
            labels = (
                [str(i + 1) for i in range(len(choices))]
                if len(choices) <= 9
                else [chr(ord("A") + i) for i in range(len(choices))]
            )
        elif kind in ("span", "spans"):
            default = SPAN_REGION if kind == "span" else SPAN_LIST_REGION
            span_cfg = {
                "max_tokens": q.get("max_tokens", default),
                "max_items": q.get("max_items", 16),
            }
            for key, hi in (("max_tokens", 96), ("max_items", 64)):
                v = span_cfg[key]
                if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= hi:
                    raise SchemaError(f"question {qid!r}: {key} must be 1 to {hi}")
            choices, labels = [], []
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        if kind not in ("span", "spans") and len(choices) < 2:
            raise SchemaError(f"question {qid!r}: needs at least two alternatives")
        if len(choices) > 26:
            raise SchemaError(f"question {qid!r}: at most 26 alternatives")
        deps = q.get("depends_on") or []
        ask_if = q.get("ask_if") or {}
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise SchemaError(
                f"question {qid!r}: depends_on must be a list of question ids"
            )
        if not isinstance(ask_if, dict) or not all(
            isinstance(v, list) and v for v in ask_if.values()
        ):
            raise SchemaError(
                f"question {qid!r}: ask_if must map a question id to a "
                "non-empty list of its answers"
            )
        if kind in ("span", "spans") and (deps or ask_if):
            raise SchemaError(
                f"question {qid!r}: a span question cannot depend on another question"
            )
        qs.append(
            {
                "id": qid,
                "type": kind,
                "instructions": str(q.get("instructions", "")),
                "choices": choices,
                "labels": labels,
                "depends_on": list(dict.fromkeys(list(deps) + list(ask_if))),
                "ask_if": ask_if,
                "alone": bool(q.get("alone", False)),
                "span": span_cfg if kind in ("span", "spans") else None,
            }
        )
    by_id = {q["id"]: q for q in qs}
    for q in qs:
        for dep in q["depends_on"]:
            if dep not in by_id or dep == q["id"]:
                raise SchemaError(
                    f"question {q['id']!r}: depends on unknown question {dep!r}"
                )
            if by_id[dep]["span"]:
                raise SchemaError(
                    f"question {q['id']!r}: cannot depend on the span question {dep!r}"
                )
        for dep, vals in q["ask_if"].items():
            names = [c[0] for c in by_id[dep]["choices"]]
            if any(v not in names for v in vals):
                raise SchemaError(
                    f"question {q['id']!r}: ask_if values for {dep!r} "
                    f"must be among {names}"
                )
    schedule(qs)  # refuses a cycle
    samples = value.get("samples", "auto")
    if samples == "auto":
        policy = {
            "mode": "auto",
            "max": max(1, min(int(value.get("auto_max", 4)), MAX_SAMPLES)),
            "threshold": float(value.get("auto_threshold", 0.1)),
        }
    elif isinstance(samples, int) and samples >= 1:
        policy = {"mode": "fixed", "n": min(samples, MAX_SAMPLES)}
    else:
        raise SchemaError('schema: samples must be a positive count or "auto"')
    ask = value.get("ask")
    if ask is not None:
        if not isinstance(ask, list) or not ask or any(a not in seen for a in ask):
            raise SchemaError("schema: ask must list question ids from this schema")
        for q in qs:
            if q["id"] in ask and any(d not in ask for d in q["depends_on"]):
                raise SchemaError(
                    f"schema: ask names {q['id']!r} but not everything it depends on"
                )
    chunk_rows = value.get("chunk_rows")
    if chunk_rows is not None and (not isinstance(chunk_rows, int) or chunk_rows < 8):
        raise SchemaError("schema: chunk_rows must be an integer of at least 8")
    chunk_prompt = value.get("chunk_prompt", "own")
    if chunk_prompt not in ("shared", "own"):
        raise SchemaError('schema: chunk_prompt must be "shared" or "own"')
    sequential = bool(value.get("sequential", False))
    think = value.get("think", 0)
    if isinstance(think, bool) or not isinstance(think, int) or not 0 <= think <= 4096:
        raise SchemaError("schema: think must be a thought budget in tokens, 0 to 4096")
    return {
        "questions": qs,
        "instructions": value.get("instructions"),
        "policy": policy,
        "steps": max(1, min(int(value.get("steps", 1)), 8)),
        "think": think,
        "ask": ask,
        "chunk_rows": chunk_rows,
        "chunk_prompt": chunk_prompt,
        "sequential": sequential,
        "format": "lines" if len([q for q in qs if not q["span"]]) <= 10 else "indexed",
    }


# Answer template shape: (join between questions, what precedes the label,
# reply instruction). A small schema gets "lines", which is readable.
# "indexed" ("0yes 1no") costs three tokens a question against four or five
# and agreed with "lines" on every set tried: 42 booleans, ten 26-way
# choices, twenty 5-level scores. Past ten questions the saved rows keep a
# schema in one read. Two tokens a question, or no id at all, loses
# alignment beyond about twenty questions, because the id ties a label to
# its question.
FORMATS = {
    "lines": (
        "\n",
        "{id}: ",
        'Reply with one line per question, in this order, formatted as "id: label".',
    ),
    "indexed": (
        " ",
        "{id}",
        "Reply on one line with each question's id immediately followed by its "
        "label, separated by single spaces.",
    ),
}


def system_text(schema, chunked=False):
    s = (
        "Answer a fixed set of questions about the state the user provides. "
        "Each question lists its allowed answers; reply with exactly one label "
        "per question.\n"
    )
    if schema.get("instructions"):
        s += "\n" + str(schema["instructions"]).strip() + "\n"
    for q in schema["questions"]:
        s += f"\nQuestion {q['id']}: {q['instructions'].strip()}\n"
        for (name, desc), label in zip(q["choices"], q["labels"]):
            if q["type"] == "noul":
                s += f"  {label}: {str(desc).strip()}\n" if desc else f"  {label}\n"
            elif desc:
                s += f"  {label}: {name} ({str(desc).strip()})\n"
            else:
                s += f"  {label}: {name}\n"
    s += "\n" + FORMATS[schema.get("format", "lines")][2]
    if chunked:
        s += (
            " A reply may cover only some of the questions; answer every line "
            "that is present."
        )
    return s


def answer_text(qs, labels, fmt="lines"):
    join, lead, _ = FORMATS[fmt]
    return join.join(
        lead.format(id=q["id"]) + q["labels"][i] for q, i in zip(qs, labels)
    )


def enc(text):
    return TOK.encode(text, add_special_tokens=False)


def init_tokenizer(tok):
    global TOK, SCAFFOLD, THOUGHT_OPEN, THOUGHT_CLOSE, EOS, NL
    TOK = tok
    if tok.eos_token_id is not None:
        EOS = int(tok.eos_token_id)
    NL = enc("\n")[0]
    THOUGHT_OPEN = enc("<|channel>thought\n")
    THOUGHT_CLOSE = enc("<channel|>")
    SCAFFOLD = enc(SCAFFOLD_TEXT)
    assert THOUGHT_OPEN + THOUGHT_CLOSE == SCAFFOLD, (
        "the thought tags must tokenize apart"
    )


def resolve_template(qs, head, lead, fmt):
    """Tokenize the answer template and find each question's slot. Every label
    must change exactly one token, at the same position for all of a question's
    labels, or this raises SchemaError. ``head`` is the token run the canvas
    starts with: the empty thought block for a plain read, and empty when the
    prompt already ends the thought channel. ``lead`` is the text before the
    first answer: the join when earlier answers are in the prompt, so the
    tokens match one joint template."""
    base_labels = [0] * len(qs)
    base = head + enc(lead + answer_text(qs, base_labels, fmt))
    if len(base) + 1 > CANVAS_LEN:
        raise SchemaError(
            f"answer template is {len(base)} tokens; the canvas holds {CANVAS_LEN - 1}"
        )
    if len(qs) == 1 and len(base) + 1 > CANVAS_LEN:
        raise SchemaError(
            f"question {qs[0]['id']!r} alone needs {len(base) + 1} canvas rows"
        )
    slots = []
    for qi, q in enumerate(qs):
        pos = None
        ids = [0] * len(q["labels"])
        for li in range(1, len(q["labels"])):
            labels = list(base_labels)
            labels[qi] = li
            e = head + enc(lead + answer_text(qs, labels, fmt))
            if len(e) != len(base):
                raise SchemaError(
                    f"question {q['id']!r}: label {q['labels'][li]!r} is not a "
                    "single token"
                )
            diffs = [i for i in range(len(e)) if e[i] != base[i]]
            if len(diffs) != 1 or (pos is not None and diffs[0] != pos):
                raise SchemaError(
                    f"question {q['id']!r}: labels do not share one template slot"
                )
            pos = diffs[0]
            ids[li] = e[pos]
        ids[0] = base[pos]
        if len(set(ids)) != len(ids):
            raise SchemaError(
                f"question {q['id']!r}: two labels tokenize to the same id"
            )
        slots.append({"pos": pos, "label_ids": ids})
    return base, slots


_template_cache = {}


def template_for(schema, head, lead):
    fmt = schema.get("format", "lines")
    key = json.dumps(
        [head, lead, fmt] + [(q["id"], q["labels"]) for q in schema["questions"]]
    )
    if key not in _template_cache:
        _template_cache[key] = resolve_template(schema["questions"], head, lead, fmt)
    return _template_cache[key]


# ----------------------------------------------------------------------------
# Reads
# ----------------------------------------------------------------------------


def canvas_width(template):
    """Smallest multiple of CANVAS_STEP that holds the template and the turn close."""
    need = len(template) + 1
    return min(CANVAS_LEN, -(-need // CANVAS_STEP) * CANVAS_STEP)


def pin_xargs(template, slots, steps):
    """Past one denoise step the template must be held, or accept/renoise
    rewrites it: pin every canvas position that is not an answer slot."""
    if steps <= 1:
        return {}
    free = {s["pos"] for s in slots}
    return {
        "diffusion_pinned": [p for p in range(canvas_width(template)) if p not in free]
    }


def build_canvas(template, slots, seed):
    rng = random.Random(seed)
    canvas = list(template) + [TURN_CLOSE]
    canvas += [PAD] * (canvas_width(template) - len(canvas))
    for s in slots:
        canvas[s["pos"]] = rng.randrange(VOCAB)
    return canvas


def label_id_union(slots):
    ids = sorted({i for s in slots for i in s["label_ids"]})
    return ids[:128]  # vLLM's cap per request. A schema needs far fewer.


def upstream_chat(body, timeout=600):
    req = urllib.request.Request(
        ARGS.upstream.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def upstream_completions(body, timeout=600):
    req = urllib.request.Request(
        ARGS.upstream.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def chat_prompt_ids(sys_text, state_text, thinking=False):
    """The prompt the chat endpoint would build, as token ids, ending after
    the model turn marker. Text states only. ``thinking`` turns the chat
    template's thinking marker on."""
    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": state_text},
    ]
    out = TOK.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
    )
    ids = (
        out["input_ids"] if hasattr(out, "keys") else out
    )  # newer transformers return a dict
    return [int(t) for t in ids]


def think(sys_text, state_text, budget):
    """A read prefix that ends with a thought the model wrote: the chat prompt
    with thinking on, the open tag, up to ``budget`` generated tokens, the
    close tag. Returns the prefix and a diagnostics dict for the thought."""
    prompt = chat_prompt_ids(sys_text, state_text, thinking=True) + THOUGHT_OPEN
    started = time.time()
    d = upstream_completions(
        {
            "model": ARGS.model,
            "prompt": prompt,
            "max_tokens": budget,
            "logprobs": 0,
            "return_tokens_as_token_ids": True,
            "stop_token_ids": THOUGHT_CLOSE,
        }
    )
    ids = [int(t.split(":")[1]) for t in d["choices"][0]["logprobs"]["tokens"]]
    closed = THOUGHT_CLOSE[0] in ids
    if closed:
        ids = ids[: ids.index(THOUGHT_CLOSE[0])]
    info = {
        "tokens": len(ids),
        "closed": closed,
        "ms": (time.time() - started) * 1e3,
        "text": TOK.decode(ids),
    }
    return prompt + ids + THOUGHT_CLOSE, info


def think_chat(sys_text, state_content, budget):
    """A thought written with the image in view: the chat endpoint with
    thinking on, capped at ``budget`` tokens and cut at the close tag.
    Returns the thought's token ids and a diagnostics dict."""
    body = {
        "model": ARGS.model,
        "messages": [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": state_content},
        ],
        "max_tokens": budget,
        "logprobs": True,
        "top_logprobs": 0,
        "return_tokens_as_token_ids": True,
        "stop_token_ids": THOUGHT_CLOSE,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    started = time.time()
    d = upstream_chat(body)
    choice = d["choices"][0]
    ids = [
        int(t["token"].split(":")[1])
        for t in (choice.get("logprobs") or {}).get("content") or []
    ]
    if ids[: len(THOUGHT_OPEN)] == THOUGHT_OPEN:  # the model opens the channel itself
        ids = ids[len(THOUGHT_OPEN) :]
    closed = THOUGHT_CLOSE[0] in ids
    if closed:
        ids = ids[: ids.index(THOUGHT_CLOSE[0])]
    return ids, {
        "tokens": len(ids),
        "closed": closed,
        "ms": (time.time() - started) * 1e3,
        "text": TOK.decode(ids),
    }


def one_read(
    schema, template, slots, sys_text, state_content, seed, prefix=None, thinking=False
):
    if prefix is not None:
        return one_read_continuation(schema, template, slots, prefix, seed)
    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": state_content},
    ]
    body = {
        "model": ARGS.model,
        "messages": messages,
        "max_tokens": len(template) + 1,
        "logprobs": True,
        "top_logprobs": TOPK,
        # Exact logprobs for every label at every position. With a long
        # option list most labels never rank in the top-k, and the model's
        # mass sits on tokens that spell the option name instead.
        "logprob_token_ids": label_id_union(slots),
        "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "vllm_xargs": {
            "diffusion_seed_canvas": build_canvas(template, slots, seed),
            "diffusion_canvas_length": canvas_width(template),
            "diffusion_max_steps": schema["steps"],
            "diffusion_read_only": True,
            **pin_xargs(template, slots, schema["steps"]),
        },
    }
    d = upstream_chat(body)
    content = d["choices"][0]["logprobs"]["content"]
    out = []
    for q, s in zip(schema["questions"], slots):
        top = {
            int(t["token"].split(":")[1]): t["logprob"]
            for t in content[s["pos"]]["top_logprobs"]
        }
        out.append(slot_distribution(top, s["label_ids"]))
    return out, d.get("usage", {})


def slot_distribution(top, label_ids):
    """Label probabilities at one slot from the returned logprobs: every
    label's own value plus the argmax token. Read-only logprobs are at
    temperature 1, so the label softmax uses them directly. The entropy is
    over that returned set."""
    floor = min(top.values()) - 5.0
    lp_t = [top.get(i, floor) for i in label_ids]
    mx = max(lp_t)
    ex = [math.exp(x - mx) for x in lp_t]
    probs = [e / sum(ex) for e in ex]
    top_p = [math.exp(v) for v in top.values()]
    return {
        "probs": probs,
        "label_mass": sum(math.exp(x) for x in lp_t),
        "entropy": -sum(p * math.log(p) for p in top_p if p > 0),
        "argmax_is_label": max(top, key=top.get) in label_ids,
    }


def one_read_continuation(schema, template, slots, prompt_ids, seed):
    """A read whose prompt already holds the thought scaffold and earlier
    answer lines, sent as token ids so the chat template cannot alter it."""
    body = {
        "model": ARGS.model,
        "prompt": prompt_ids,
        "max_tokens": len(template) + 1,
        "logprobs": TOPK,
        "logprob_token_ids": label_id_union(slots),
        "return_tokens_as_token_ids": True,
        "vllm_xargs": {
            "diffusion_seed_canvas": build_canvas(template, slots, seed),
            "diffusion_canvas_length": canvas_width(template),
            "diffusion_max_steps": schema["steps"],
            "diffusion_read_only": True,
            **pin_xargs(template, slots, schema["steps"]),
        },
    }
    d = upstream_completions(body)
    rows = d["choices"][0]["logprobs"]["top_logprobs"]
    out = []
    for q, sl in zip(schema["questions"], slots):
        top = {int(k.split(":")[1]): v for k, v in rows[sl["pos"]].items()}
        out.append(slot_distribution(top, sl["label_ids"]))
    return out, d.get("usage", {})


def read_many(
    schema,
    template,
    slots,
    sys_text,
    state_content,
    seed,
    n,
    prefix=None,
    thinking=False,
):
    results = [None] * n
    errors = [None] * n
    usages = [None] * n

    def run(k):
        try:
            results[k], usages[k] = one_read(
                schema,
                template,
                slots,
                sys_text,
                state_content,
                seed + k * 7919,
                prefix,
                thinking,
            )
        except Exception as e:  # raised again below
            errors[k] = e

    threads = [threading.Thread(target=run, args=(k,)) for k in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in errors:
        if e is not None:
            raise e
    return results, usages


def schedule(qs):
    """Questions in stages: a question's stage comes after the stages of
    everything it depends on. Declaration order is kept within a stage."""
    ids = {q["id"] for q in qs}
    pending = list(qs)
    done: set = set()
    levels = []
    while pending:
        level = [
            q
            for q in pending
            if all(d in done or d not in ids for d in q["depends_on"])
        ]
        if not level:
            raise SchemaError(
                "schema: dependency cycle among " + ", ".join(q["id"] for q in pending)
            )
        levels.append(level)
        done |= {q["id"] for q in level}
        pending = [q for q in pending if q["id"] not in done]
    return levels


def chunk_groups(schema, qs):
    """``qs`` split, in order, into the fewest groups whose answer templates
    fit ``chunk_rows`` (the canvas by default). A question marked alone gets
    its own group."""
    limit = schema.get("chunk_rows") or CANVAS_LEN
    groups, group = [], []
    for q in qs:
        if q["alone"]:
            if group:
                groups.append(group)
                group = []
            groups.append([q])
            continue
        trial = group + [q]
        rows = (
            len(SCAFFOLD)
            + len(
                enc(answer_text(trial, [0] * len(trial), schema.get("format", "lines")))
            )
            + 1
        )
        if rows > limit and group:
            groups.append(group)
            group = [q]
        else:
            group = trial
    if group:
        groups.append(group)
    return groups


def answer_name(q, a):
    """The answer as the name ask_if compares against: yes or no, an option
    name, or a level name."""
    if a is None:
        return None
    return a["label"] if q["type"] == "noul" else a.get("choice", a.get("level"))


def decide(schema, state_content, seed):
    """One decision. Questions run in stages by their dependencies. A stage
    is one joint read, chunked by the canvas, with a question marked alone
    in its own read. Later stages condition on every earlier answer: a
    prefilled continuation for a text state, or the answers restated in the
    state for an image. A question whose ask_if condition failed is skipped
    and its answer is null."""
    started = time.time()
    qs = [
        q
        for q in schema["questions"]
        if not schema.get("ask") or q["id"] in schema["ask"]
    ]
    span_qs = [q for q in qs if q["span"]]
    qs = [q for q in qs if not q["span"]]
    # Span questions are reads of their own, run beside the label reads.
    span_pool = ThreadPoolExecutor(max_workers=1) if span_qs else None
    span_future = (
        span_pool.submit(decide_spans, schema, span_qs, state_content, seed)
        if span_qs
        else None
    )
    schema = dict(schema, questions=[q for q in schema["questions"] if not q["span"]])
    if not qs:
        span_answers, span_diag = span_future.result()
        span_pool.shutdown(wait=False)
        diagnostics = {
            "steps": schema["steps"],
            "stages": [],
            "skipped": {},
            "chunks": [],
            "spans": span_diag,
            "timing": {
                "total_ms": (time.time() - started) * 1e3,
                "reads": span_diag["reads"],
            },
            "questions": {},
            "engine": "vllm",
        }
        return {"answers": span_answers, "diagnostics": diagnostics}, span_diag["rows"]
    levels = schedule(qs)
    text_state = isinstance(state_content, str)
    fmt = schema["format"]
    join = FORMATS[fmt][0]
    # More than one read in sequence needs the full question list in every
    # prompt, so that later reads continue one answer.
    chained = len(levels) > 1 or schema["sequential"]
    sys_full = system_text(schema, chunked=not text_state and chained)
    base_ids, thought = None, None
    if chained and text_state:
        if schema["think"]:
            base_ids, thought = think(sys_full, state_content, schema["think"])
        else:
            base_ids = chat_prompt_ids(sys_full, state_content) + SCAFFOLD
    shared = schema["chunk_prompt"] == "shared"
    answered, lines, earlier = {}, [], []
    parts, stages, chunks, skipped = [], [], [], {}
    by_id = {q["id"]: q for q in schema["questions"]}

    def run(group, k, conditioned):
        sub = dict(schema, questions=group)
        # The thought is written once: in the prefix of a chained text decision,
        # or in the first read of anything else.
        sub["think"] = 0 if (chained and text_state) or conditioned else schema["think"]
        if text_state:
            if conditioned:
                prefix, lead, sys_text = (
                    base_ids + enc(join.join(lines)),
                    join,
                    sys_full,
                )
            elif chained:
                prefix, lead, sys_text = (base_ids if thought else None), "", sys_full
            else:
                prefix, lead, sys_text = (
                    None,
                    "",
                    (system_text(schema, chunked=True) if shared else system_text(sub)),
                )
            return decide_group(
                sub, sys_text, state_content, seed + 104729 * k, prefix, lead
            )
        state = state_content
        if conditioned:
            text = next(
                (p["text"] for p in state_content if p.get("type") == "text"), ""
            )
            text += "\n\nAnswers so far:\n" + "\n".join(earlier)
            state = [p for p in state_content if p.get("type") != "text"] + [
                {"type": "text", "text": text}
            ]
            sys_text = sys_full
        else:
            sys_text = sys_full if (chained or shared) else system_text(sub)
        return decide_group(sub, sys_text, state, seed + 104729 * k)

    def absorb(group, body, rows):
        parts.append((body, rows))
        chunks.append([q["id"] for q in group])
        answered.update(body["answers"])
        lines.append(
            answer_text(
                group,
                [q["labels"].index(body["answers"][q["id"]]["label"]) for q in group],
                fmt,
            )
        )
        earlier.extend(
            f"{q['id']}: {answer_name(q, body['answers'][q['id']])}" for q in group
        )

    k = 0
    for level in levels:
        asked = []
        for q in level:
            failed = next(
                (
                    (dep, vals)
                    for dep, vals in q["ask_if"].items()
                    if answer_name(by_id[dep], answered.get(dep)) not in vals
                ),
                None,
            )
            if failed:
                answered[q["id"]] = None
                skipped[q["id"]] = {
                    "because": failed[0],
                    "was": answer_name(by_id[failed[0]], answered.get(failed[0])),
                    "wanted": failed[1],
                }
                continue
            asked.append(q)
        if not asked:
            continue
        stages.append([q["id"] for q in asked])
        groups = chunk_groups(schema, asked)
        conditioned = bool(lines)
        if schema["sequential"] or len(groups) == 1:
            for group in groups:
                body, rows = run(
                    group, k, conditioned or (schema["sequential"] and bool(lines))
                )
                absorb(group, body, rows)
                k += 1
        else:
            with ThreadPoolExecutor(max_workers=min(len(groups), MAX_PARALLEL)) as ex:
                results = list(
                    ex.map(
                        lambda gk, conditioned=conditioned: run(
                            gk[1], gk[0], conditioned
                        ),
                        [(k + i, g) for i, g in enumerate(groups)],
                    )
                )
            for group, (body, rows) in zip(groups, results):
                absorb(group, body, rows)
            k += len(groups)

    answers = {q["id"]: answered.get(q["id"]) for q in qs}
    diag_q = {}
    for body, _ in parts:
        diag_q.update(body["diagnostics"]["questions"])
    # A thought written here (a chained text decision) is outside every
    # group's row count. One written inside a group is already counted there.
    extra_rows = thought["tokens"] if thought else 0
    if thought is None:
        thoughts = [b["diagnostics"].get("thought") for b, _ in parts]
        thought = (
            thoughts[0] if len(parts) == 1 else ([t for t in thoughts if t] or None)
        )
    one = len(parts) == 1 and not skipped
    diagnostics = {
        "steps": schema["steps"],
        "stages": stages,
        "skipped": skipped,
        "chunks": chunks,
        "chunk_prompt": "full" if chained else schema["chunk_prompt"],
        "sequential": schema["sequential"],
        "conditioning": (
            None
            if len(stages) <= 1 and not schema["sequential"]
            else ("prefill" if text_state else "restated")
        ),
        "thought": thought,
        "samples": (
            parts[0][0]["diagnostics"]["samples"]
            if one
            else {
                "n": [b["diagnostics"]["samples"]["n"] for b, _ in parts],
                "tops": [b["diagnostics"]["samples"]["tops"] for b, _ in parts],
                "policy": [b["diagnostics"]["samples"]["policy"] for b, _ in parts],
            }
        ),
        "timing": {
            "total_ms": (time.time() - started) * 1e3,
            "reads": sum(b["diagnostics"]["timing"]["reads"] for b, _ in parts),
        },
        "prompt_tokens": max(
            (b["diagnostics"].get("prompt_tokens") or 0) for b, _ in parts
        )
        or None,
        "questions": diag_q,
        "engine": "vllm",
    }
    total_rows = sum(rows for _, rows in parts) + extra_rows
    if span_future is not None:
        try:
            span_answers, span_diag = span_future.result()
        finally:
            span_pool.shutdown(wait=False)
        answers.update(span_answers)
        diagnostics["spans"] = span_diag
        diagnostics["timing"]["reads"] += span_diag["reads"]
        diagnostics["timing"]["total_ms"] = (time.time() - started) * 1e3
        total_rows += span_diag["rows"]
    order = [q["id"] for q in qs] + [q["id"] for q in span_qs]
    answers = {qid: answers.get(qid) for qid in order}
    return {"answers": answers, "diagnostics": diagnostics}, total_rows


# ----------------------------------------------------------------------------
# Spans
# ----------------------------------------------------------------------------


def span_source(state_content):
    """The text a span's offsets refer to: the state when it is a string,
    else the text part of an image state. A JSON state is grounded as the
    JSON the model reads."""
    if isinstance(state_content, str):
        return state_content
    return next((p["text"] for p in state_content if p.get("type") == "text"), "")


_piece_cache = {}
_piece_lock = threading.Lock()


def span_pieces(text):
    """id -> the string that token contributes: every token of the text in
    both spacings, plus each word's own first token with and without a
    leading space, since the model opens a value like "A-1042" with " A"
    while the text only holds "#A"."""
    with _piece_lock:
        if text in _piece_cache:
            return _piece_cache[text]
    variants = [" " + text, text]
    for w in set(text.split()):
        variants += [" " + w, w]
        bare = w.lstrip("#$([\"'")
        if bare and bare != w:
            variants += [" " + bare, bare]
    pieces = {}
    for v in variants:
        ids = enc(v)
        for tid, piece in zip(ids, TOK.convert_ids_to_tokens(ids)):
            if piece and not piece.startswith("<"):
                pieces[tid] = piece.replace("\u2581", " ")
    with _piece_lock:
        if len(_piece_cache) > 256:
            _piece_cache.clear()
        _piece_cache[text] = pieces
    return pieces


def span_read_canvas(sys_text, state_content, body_ids, free, allowed, seed, steps):
    """One pinned read of a canvas whose body follows the scaffold, with the
    positions in ``free`` left as noise. Returns per body position the
    logprobs of the ``allowed`` ids, and the emitted ids."""
    rng = random.Random(seed)
    shown = [rng.randrange(VOCAB) if i in free else t for i, t in enumerate(body_ids)]
    template = SCAFFOLD + shown
    width = canvas_width(template)
    if len(template) + 1 > width:
        raise SchemaError(
            f"a span blank of {len(body_ids)} rows does not fit the canvas"
        )
    canvas = template + [TURN_CLOSE]
    canvas += [PAD] * (width - len(canvas))
    want = sorted(allowed)
    if len(want) > SPAN_ID_CAP:
        raise SchemaError(
            f"{len(want)} distinct token ids; a read allows {SPAN_ID_CAP}"
        )
    pinned = [p for p in range(width) if (p - len(SCAFFOLD)) not in free]
    d = upstream_chat(
        {
            "model": ARGS.model,
            "messages": [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": state_content},
            ],
            "max_tokens": width,
            "logprobs": True,
            "top_logprobs": 1,
            "logprob_token_ids": want,
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": {
                "diffusion_seed_canvas": canvas,
                "diffusion_canvas_length": width,
                "diffusion_max_steps": steps,
                "diffusion_read_only": True,
                "diffusion_pinned": pinned,
            },
        }
    )
    content = d["choices"][0]["logprobs"]["content"]
    rows, emitted = [], []
    for c in content[len(SCAFFOLD) : len(SCAFFOLD) + len(body_ids)]:
        rows.append(
            {int(t["token"].split(":")[1]): t["logprob"] for t in c["top_logprobs"]}
        )
        emitted.append(int(c["token"].split(":")[1]))
    return rows, emitted, width


def span_decode(text, rows, pieces, none_id, max_slop=2, slop_cost=-1.5, min_start=0):
    """The best contiguous substring under the blank's logprobs. Each token
    scores as regret against the model's own top pick at its position, so a
    long span pays only where it departs from what the model wanted to
    write. Up to ``max_slop`` blank positions may be skipped at ``slop_cost``
    each without advancing in the text, so an inserted token (a fix, a
    closing quote) does not break the walk. Returns (score, start, end,
    confidence, coverage); start is None for none. Confidence is the weakest
    normalised probability along the span and its end; coverage is the mass
    the allowed ids held there."""
    n = len(rows)
    L = len(text)
    end_ids = {TURN_CLOSE, EOS, NL}
    matches = [[] for _ in range(L + 1)]
    starts = [[] for _ in range(L + 1)]
    for tid, pc in pieces.items():
        c = text.find(pc)
        while c != -1:
            matches[c].append((tid, len(pc)))
            c = text.find(pc, c + 1)
        if pc.startswith(" ") and len(pc) > 1:
            q = pc[1:]
            c = text.find(q)
            while c != -1:
                starts[c].append((tid, len(q)))
                # one space from the model where the text has a run of whitespace
                w = c
                while w > 0 and text[w - 1].isspace():
                    w -= 1
                if c - w > 1:
                    matches[w].append((tid, c - w + len(q)))
                c = text.find(q, c + 1)
    space_ids = [tid for tid, pc in pieces.items() if pc == " "]
    z = [sum(math.exp(v) for v in r.values()) or 1e-9 for r in rows]
    top = [max(r.values()) if r else SPAN_FLOOR for r in rows]
    end_lp = [
        max(r.get(e, SPAN_FLOOR) for e in end_ids) - top[k] for k, r in enumerate(rows)
    ]
    punct = {
        tid
        for tid, pc in pieces.items()
        if pc.strip() and all(ch in SPAN_BOUNDARY for ch in pc.strip())
    }
    end_conf = [
        sum(math.exp(r[t]) for t in r if t in end_ids or t in punct) / z[k]
        for k, r in enumerate(rows)
    ]
    cov = [min(1.0, zk) for zk in z]
    best = None
    for a in range(min_start, L):
        if text[a].isspace() or not (a == 0 or text[a - 1] in SPAN_BOUNDARY):
            continue
        dp = {(a, 0): (0.0, 1.0)}
        for k in range(n - 1):
            nxt = {}
            for (c, used), (sc, mn) in dp.items():
                opts = matches[c]
                if k == 0:  # a value may open with a lone space token
                    opts = opts + starts[c] + [(t, 0) for t in space_ids]
                for tid, ln in opts:
                    lp = rows[k].get(tid, SPAN_FLOOR)
                    if lp <= SPAN_FLOOR:
                        continue
                    cand = (sc + lp - top[k], min(mn, math.exp(lp) / z[k]))
                    key = (c + ln, used)
                    if key not in nxt or cand[0] > nxt[key][0]:
                        nxt[key] = cand
                if used < max_slop and c > a:
                    key = (c, used + 1)
                    cand = (sc + slop_cost, mn)
                    if key not in nxt or cand[0] > nxt[key][0]:
                        nxt[key] = cand
            dp = nxt
            if not dp:
                break
            for (c, used), (sc, mn) in dp.items():
                if c == a or not (c == L or text[c] in SPAN_BOUNDARY):
                    continue
                total = sc + end_lp[k + 1]
                if best is None or total > best[0]:
                    best = (
                        total,
                        a,
                        c,
                        min(mn, end_conf[k + 1]),
                        sum(cov[: k + 2]) / (k + 2),
                    )
    none = rows[0].get(none_id, SPAN_FLOOR) - top[0] + (end_lp[1] if n > 1 else 0.0)
    if best is None or none > best[0]:
        return (
            none,
            None,
            None,
            math.exp(rows[0].get(none_id, SPAN_FLOOR)) / z[0],
            cov[0],
        )
    return best


def span_trim(text, a, b):
    """Trailing sentence punctuation the model copied along ("No!" -> "No")."""
    while b - a > 1 and text[b - 1] in ".,;:!?" and not text[b - 2].isdigit():
        b -= 1
    return b


def span_windows(text, reserve=0):
    """(start, end) windows whose token ids fit one read, split at sentence
    ends and newlines with one sentence of overlap. The whole text when it
    fits."""
    budget = SPAN_ID_CAP - 4 - reserve
    if len(span_pieces(text)) <= budget:
        return [(0, len(text))]
    units = [
        (m.start(), m.end())
        for m in re.finditer(r"[^.!?\n]*[.!?\n]+\s*|[^.!?\n]+$", text)
        if text[m.start() : m.end()].strip()
    ]
    split = []
    for s_, e in units:
        if len(span_pieces(text[s_:e])) <= budget:
            split.append((s_, e))
            continue
        cur = s_
        for m in re.finditer(r"\S+\s*", text[s_:e]):
            if (
                len(span_pieces(text[cur : s_ + m.end()])) > budget
                and s_ + m.start() > cur
            ):
                split.append((cur, s_ + m.start()))
                cur = s_ + m.start()
        split.append((cur, e))
    windows, cur = [], []
    for s_, e in split:
        if cur and len(span_pieces(text[cur[0][0] : e])) > budget:
            windows.append((cur[0][0], cur[-1][1]))
            cur = [cur[-1]] if len(span_pieces(text[cur[-1][0] : e])) <= budget else []
        cur.append((s_, e))
    if cur:
        windows.append((cur[0][0], cur[-1][1]))
    return windows


def span_context(schema):
    ins = schema.get("instructions")
    return ("\n\n" + str(ins).strip()) if ins else ""


def span_choose(schema, q, state_content, candidates, seed):
    """A one-step choice over candidate strings with the whole state in
    view: the verify step when windows disagree. Returns (index, probs)."""
    letters = [chr(ord("A") + i) for i in range(len(candidates))]
    options = "\n".join(f"  {ltr}: {c}" for ltr, c in zip(letters, candidates))
    sys_text = SPAN_CHOOSE_SYSTEM.format(
        q=q["instructions"], context=span_context(schema), options=options
    )
    ids = [enc(" " + ltr)[0] for ltr in letters]
    body = enc("answer:") + [PAD]
    rows, _, _ = span_read_canvas(
        sys_text,
        state_content,
        body,
        {len(body) - 1},
        set(ids) | {TURN_CLOSE, EOS, NL},
        seed,
        1,
    )
    dist = slot_distribution(rows[-1], ids)
    probs = dict(zip(candidates, dist["probs"]))
    return max(range(len(candidates)), key=lambda i: dist["probs"][i]), probs


def read_span_window(schema, q, sys_text, state_content, text, window, seed):
    """One span question in one window: up to SPAN_MAX_READS reads with
    successive seeds until the answer clears SPAN_MIN_CONF. A rare bad read
    places the end early and shows up as low confidence."""
    wtext = text[window[0] : window[1]]
    pieces = span_pieces(wtext)
    none_id = enc(" none")[0]
    allowed = set(pieces) | {TURN_CLOSE, EOS, NL, none_id}
    head = enc("answer:")
    region = max(
        1, min(q["span"]["max_tokens"], CANVAS_LEN - len(SCAFFOLD) - len(head) - 1)
    )
    body = head + [PAD] * region
    free = set(range(len(head), len(body)))
    steps = max(SPAN_STEPS, schema["steps"])
    best, reads, rows_used = None, 0, 0
    for i in range(SPAN_MAX_READS):
        rows, _, width = span_read_canvas(
            sys_text, state_content, body, free, allowed, seed + i, steps
        )
        reads += 1
        rows_used += width
        _, a, b, conf, cov = span_decode(wtext, rows[len(head) :], pieces, none_id)
        if a is None:
            ans = {
                "type": "span",
                "found": False,
                "text": None,
                "start": None,
                "end": None,
            }
        else:
            b = span_trim(wtext, a, b)
            ans = {
                "type": "span",
                "found": True,
                "text": wtext[a:b],
                "start": a + window[0],
                "end": b + window[0],
            }
        ans["confidence"] = round(conf, 4)
        ans["coverage"] = round(cov, 4)
        if best is None or ans["confidence"] > best["confidence"]:
            best = ans
        if ans["confidence"] >= SPAN_MIN_CONF:
            break
    best["reads"] = reads
    return best, rows_used


def read_span(schema, q, state_content, seed):
    """One span question: the whole text in the prompt, the read restricted
    to each window's ids, windows in parallel, and one verify choice when
    they disagree."""
    text = span_source(state_content)
    sys_text = SPAN_SYSTEM.format(q=q["instructions"], context=span_context(schema))
    windows = span_windows(text)
    with ThreadPoolExecutor(max_workers=min(8, len(windows))) as ex:
        parts = list(
            ex.map(
                lambda w: read_span_window(
                    schema, q, sys_text, state_content, text, w, seed
                ),
                windows,
            )
        )
    rows = sum(r for _, r in parts)
    found = [a for a, _ in parts if a["found"]]
    if not found:
        a = max((a for a, _ in parts), key=lambda a: a["confidence"] * a["coverage"])
    else:
        found.sort(key=lambda a: -a["confidence"] * a["coverage"])
        distinct = []
        for a in found:
            if a["text"] not in [d["text"] for d in distinct]:
                distinct.append(a)
        a = distinct[0]
        if len(distinct) > 1:
            i, probs = span_choose(
                schema, q, state_content, [d["text"] for d in distinct[:8]], seed
            )
            a = distinct[i]
            a["resolved"] = probs
            a["reads"] += 1
    a["reads"] = sum(p["reads"] for p, _ in parts) + (1 if "resolved" in a else 0)
    if len(windows) > 1:
        a["windows"] = len(windows)
    return a, rows


def read_spans_window(schema, q, sys_text, state_content, text, window, seed):
    """A list of spans in one window: one read writes the values one per
    line, each line is decoded as a span whose start follows the previous
    item, so repeated mentions map to successive occurrences."""
    wtext = text[window[0] : window[1]]
    pieces = span_pieces(wtext)
    none_id = enc(" none")[0]
    allowed = set(pieces) | {TURN_CLOSE, EOS, NL, none_id}
    head = enc("answer:")
    region = max(
        1, min(q["span"]["max_tokens"], CANVAS_LEN - len(SCAFFOLD) - len(head) - 1)
    )
    body = head + [PAD] * region
    free = set(range(len(head), len(body)))
    steps = max(SPAN_STEPS, schema["steps"])
    rows, emitted, width = span_read_canvas(
        sys_text, state_content, body, free, allowed, seed, steps
    )
    region_rows, region_em = rows[len(head) :], emitted[len(head) :]
    items, seg_start, min_start = [], 0, 0
    for k, t in enumerate(region_em + [TURN_CLOSE]):
        if t not in (TURN_CLOSE, EOS, NL) and t != PAD:
            continue
        # the model repeats an "answer:" label on later lines: its word is
        # outside the allowed set, its colon may not be (the text can hold
        # "02:14"), so skip unreadable tokens and a colon after them
        if seg_start < k and region_em[seg_start] not in allowed:
            while seg_start < k and region_em[seg_start] not in allowed:
                seg_start += 1
            if seg_start < k and pieces.get(region_em[seg_start], "").strip() == ":":
                seg_start += 1
        seg = region_rows[seg_start : k + 1]
        if k > seg_start and seg:
            _, a, b, conf, cov = span_decode(wtext, seg, pieces, none_id)
            dup = False
            if a is not None and a < min_start:
                if wtext.find(wtext[a:b], min_start) != -1:
                    _, a, b, conf, cov = span_decode(
                        wtext, seg, pieces, none_id, min_start=min_start
                    )
                else:
                    a, dup = None, True  # the model repeated a line it wrote
            if a is not None:
                b = span_trim(wtext, a, b)
                items.append(
                    {
                        "text": wtext[a:b],
                        "start": a + window[0],
                        "end": b + window[0],
                        "confidence": round(conf, 4),
                        "coverage": round(cov, 4),
                    }
                )
                min_start = b
            elif not items and not dup:
                break  # the first line is none
        seg_start = k + 1
        if t != NL or len(items) >= q["span"]["max_items"]:
            break
    return items, width


def read_spans(schema, q, state_content, seed):
    """Every span of a kind: windows and SPAN_PASSES seeds in parallel,
    merged by offset, low-confidence lines dropped."""
    text = span_source(state_content)
    sys_text = SPAN_LIST_SYSTEM.format(
        q=q["instructions"], context=span_context(schema)
    )
    windows = span_windows(text)
    jobs = [(w, seed + p) for p in range(SPAN_PASSES) for w in windows]
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        parts = list(
            ex.map(
                lambda j: read_spans_window(
                    schema, q, sys_text, state_content, text, j[0], j[1]
                ),
                jobs,
            )
        )
    items = []
    for part, _ in parts:
        for it in part:
            if it["confidence"] < SPAN_MIN_CONF:
                continue
            same = [
                o for o in items if it["start"] < o["end"] and it["end"] > o["start"]
            ]
            if same:
                if it["confidence"] > same[0]["confidence"]:
                    same[0].update(it)
                continue
            items.append(it)
    items.sort(key=lambda i: i["start"])
    a = {"type": "spans", "found": bool(items), "items": items, "reads": len(jobs)}
    if len(windows) > 1:
        a["windows"] = len(windows)
    return a, sum(r for _, r in parts)


def decide_spans(schema, span_qs, state_content, seed):
    """Every span question of a decision, in parallel."""
    started = time.time()
    if not span_source(state_content).strip():
        raise SchemaError("a span question needs a text state to point into")

    def run(iq):
        i, q = iq
        fn = read_spans if q["type"] == "spans" else read_span
        return fn(schema, q, state_content, seed + 104729 * (i + 1))

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(span_qs))) as ex:
        results = list(ex.map(run, enumerate(span_qs)))
    answers = {q["id"]: a for q, (a, _) in zip(span_qs, results)}
    diag = {
        "steps": max(SPAN_STEPS, schema["steps"]),
        "reads": sum(a["reads"] for a, _ in results),
        "rows": sum(r for _, r in results),
        "source": "state" if isinstance(state_content, str) else "text part",
        "ms": (time.time() - started) * 1e3,
    }
    return answers, diag


def decide_group(schema, sys_text, state_content, seed, prefix=None, lead=""):
    started = time.time()
    thought = None
    head = SCAFFOLD if prefix is None else []
    thinking = False
    if prefix is None and schema["think"]:
        if isinstance(state_content, str):
            prefix, thought = think(sys_text, state_content, schema["think"])
            head = []
        else:
            # With images the model writes the thought with the image in view, and
            # the server seeds it into the canvas ahead of the answer, so the read
            # keeps the image. The canvas bounds the thought.
            answer = enc(
                answer_text(
                    schema["questions"],
                    [0] * len(schema["questions"]),
                    schema.get("format", "lines"),
                )
            )
            fits = CANVAS_LEN - 1 - len(THOUGHT_OPEN) - len(THOUGHT_CLOSE) - len(answer)
            if fits < 8:
                raise SchemaError(
                    f"think: the canvas leaves {fits} rows for a thought "
                    "beside this template"
                )
            ids, thought = think_chat(
                sys_text, state_content, min(schema["think"], fits)
            )
            thought["budget"] = min(schema["think"], fits)
            head = THOUGHT_OPEN + ids + THOUGHT_CLOSE
            thinking = True
    template, slots = template_for(schema, head, lead)
    policy = schema["policy"]
    if policy["mode"] == "fixed":
        reads, usages = read_many(
            schema,
            template,
            slots,
            sys_text,
            state_content,
            seed,
            policy["n"],
            prefix,
            thinking,
        )
        extended = None
        first_entropy = None
    else:
        reads, usages = read_many(
            schema, template, slots, sys_text, state_content, seed, 1, prefix, thinking
        )
        first_entropy = {
            q["id"]: r["entropy"] for q, r in zip(schema["questions"], reads[0])
        }
        extended = (
            max(first_entropy.values()) > policy["threshold"] and policy["max"] > 1
        )
        if extended:
            more, more_usages = read_many(
                schema,
                template,
                slots,
                sys_text,
                state_content,
                seed + 1,
                policy["max"] - 1,
                prefix,
                thinking,
            )
            reads += more
            usages += more_usages
    prompt_tokens = next(
        (u["prompt_tokens"] for u in usages if u and u.get("prompt_tokens")), None
    )
    elapsed_ms = (time.time() - started) * 1e3

    answers = {}
    diag_q = {}
    n = len(reads)
    for qi, q in enumerate(schema["questions"]):
        per = [r[qi]["probs"] for r in reads]
        mean = [sum(p[i] for p in per) / n for i in range(len(q["labels"]))]
        top = max(range(len(mean)), key=lambda i: mean[i])
        a = {
            "type": q["type"],
            "label": q["labels"][top],
            "confidence": mean[top],
            "probabilities": {c[0]: m for c, m in zip(q["choices"], mean)},
        }
        if q["type"] == "noul":
            a["noul"] = mean[0]
        elif q["type"] == "choice":
            a["choice"] = q["choices"][top][0]
        else:
            a["score"] = sum((i + 1) * m for i, m in enumerate(mean))
            a["level"] = q["choices"][top][0]
        if n > 1:
            var = sum((p[top] - mean[top]) ** 2 for p in per) / (n - 1)
            a["stderr"] = (var / n) ** 0.5
            a["agreement"] = (
                sum(1 for p in per if max(range(len(p)), key=lambda i: p[i]) == top) / n
            )
        answers[q["id"]] = a
        diag_q[q["id"]] = {
            "pos": slots[qi]["pos"],
            "entropy": [r[qi]["entropy"] for r in reads],
            "label_mass": reads[0][qi]["label_mass"],
            "argmax_is_label": reads[0][qi]["argmax_is_label"],
        }
    tops = [
        {
            q["id"]: [
                q["labels"][
                    max(range(len(r[qi]["probs"])), key=lambda i: r[qi]["probs"][i])
                ],
                max(r[qi]["probs"]),
                r[qi]["entropy"],
            ]
            for qi, q in enumerate(schema["questions"])
        }
        for r in reads
    ]
    return {
        "answers": answers,
        "diagnostics": {
            "steps": schema["steps"],
            "samples": {
                "n": n,
                "tops": tops,
                "policy": dict(
                    policy, extended=extended, first_read_entropy=first_entropy
                ),
            },
            "timing": {"total_ms": elapsed_ms, "reads": n},
            "thought": thought,
            "prompt_tokens": prompt_tokens,
            "questions": diag_q,
            "engine": "vllm",
        },
    }, len(template) + 1 + (thought["tokens"] if thought else 0)


# ----------------------------------------------------------------------------
# Jev's contract
# ----------------------------------------------------------------------------

JEV_EXTENSIONS = (
    "instructions",
    "samples",
    "auto_max",
    "auto_threshold",
    "steps",
    "think",
    "ask",
    "chunk_rows",
    "chunk_prompt",
    "sequential",
)


def jev_schema(body):
    """This server's schema from a Jev request body."""
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise SchemaError("questions: needs a non-empty map of id -> question")
    out = []
    for qid, q in qs.items():
        if not isinstance(q, dict):
            raise SchemaError(f"question {qid!r}: must be an object")
        kind, crit, ins = q.get("type"), q.get("criteria"), q.get("instructions", "")
        item = {
            "id": qid,
            "type": kind,
            "instructions": ins if isinstance(ins, str) else json.dumps(ins),
        }
        if kind == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise SchemaError(
                    f"question {qid!r}: noul criteria must be an object with "
                    "true and false"
                )
            item["criteria"] = crit
        elif kind == "choice":
            if not isinstance(crit, dict) or not crit:
                raise SchemaError(
                    f"question {qid!r}: choice criteria must map option names "
                    "to descriptions"
                )
            item["options"] = [
                {"name": str(n), "description": d} for n, d in crit.items()
            ]
        elif kind == "score":
            if not isinstance(crit, list):
                raise SchemaError(
                    f"question {qid!r}: score criteria must be an ordered list "
                    "of levels"
                )
            item["levels"] = crit
        elif kind in ("span", "spans"):
            if crit is not None and not isinstance(crit, dict):
                raise SchemaError(
                    f"question {qid!r}: {kind} criteria must be an object, "
                    'e.g. {"max_tokens": 24}'
                )
            for key in ("max_tokens", "max_items"):
                if crit and key in crit:
                    item[key] = crit[key]
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        for key in ("depends_on", "ask_if", "alone"):
            if key in q:
                item[key] = q[key]
        out.append(item)
    schema = {k: body[k] for k in JEV_EXTENSIONS if k in body}
    schema["questions"] = out
    return parse_schema(schema)


def image_part(content_type, data):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{content_type};base64," + base64.b64encode(data).decode()
        },
    }


def jev_images(value):
    """Image parts from the body's "images": data URLs, or objects with
    content_type and base64."""
    parts = []
    for i, im in enumerate(value or []):
        if isinstance(im, str) and im.startswith("data:image/"):
            parts.append({"type": "image_url", "image_url": {"url": im}})
        elif (
            isinstance(im, dict)
            and str(im.get("content_type", "")).startswith("image/")
            and isinstance(im.get("base64"), str)
        ):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{im['content_type']};base64,{im['base64']}"
                    },
                }
            )
        else:
            raise SchemaError(
                f"images[{i}]: a data:image/... URL or an object with "
                "content_type and base64"
            )
    return parts


def jev_state(body, image_parts=()):
    """The user message: the state as text (as given, or as JSON), with any
    images ahead of it."""
    state = body.get("state")
    if state is None:
        raise SchemaError("state: required")
    text = state if isinstance(state, str) else json.dumps(state)
    if not image_parts:
        return text
    return list(image_parts) + [{"type": "text", "text": text}]


def jev_answer(q, a):
    if a is None:
        return None
    if q["type"] == "noul":
        return {"type": "noul", "noul": a["noul"]}
    if q["type"] == "choice":
        return {
            "type": "choice",
            "choice": a["choice"],
            "probabilities": a["probabilities"],
            "confidence": a["confidence"],
        }
    if q["type"] in ("span", "spans"):
        return a
    names = [c[0] for c in q["choices"]]
    probs = {str(i): a["probabilities"][n] for i, n in enumerate(names)}
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs.values())),
        "legend": {str(i): n for i, n in enumerate(names)},
        "probabilities": probs,
        "confidence": a["confidence"],
    }


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------


def answer_brief(v):
    """One word of an answer for the log line."""
    if v is None:
        return "skipped"
    if "label" in v:
        return v["label"]
    if "items" in v:
        return f"{len(v['items'])}spans"
    return repr(v.get("text")) if v.get("found") else "none"


def message_text(m):
    c = m.get("content", "")
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c if isinstance(c, str) else ""


class Server(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        path_clean = self.path.split("?")[0]
        if path_clean in PAGES and TEST_PAGE:
            target_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), PAGES[path_clean]
            )
            if os.path.exists(target_file):
                body = open(target_file, "rb").read()
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        return self._json(404, {"error": {"message": "unknown route"}})

    def _read_request(self):
        """-> (body, image parts): a JSON body, or multipart/form-data with the
        JSON in a part named request and each image as a file part, in order."""
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        ctype = self.headers.get("content-type", "")
        if not ctype.lower().startswith("multipart/form-data"):
            return json.loads(raw), []
        from email.parser import BytesParser
        from email.policy import HTTP

        msg = BytesParser(policy=HTTP).parsebytes(
            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw
        )
        body, images = None, []
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            data = part.get_payload(decode=True)
            if name == "request":
                body = json.loads(data)
            elif part.get_content_type().startswith("image/"):
                images.append(image_part(part.get_content_type(), data))
            else:
                raise ValueError(
                    f"part {name!r}: neither the request JSON nor an image"
                )
        if body is None:
            raise ValueError(
                "multipart needs a part named request holding the JSON body"
            )
        return body, images

    def do_POST(self):
        if API_KEY and self.headers.get("authorization", "") != f"Bearer {API_KEY}":
            return self._json(
                401,
                {
                    "error": {
                        "message": "missing or wrong bearer token",
                        "type": "authentication_error",
                    }
                },
            )
        if self.path == "/v1/raw/chat/completions":
            return self._raw_chat()
        try:
            req, images = self._read_request()
        except Exception as e:
            return self._json(
                400,
                {
                    "error": {
                        "message": f"invalid body: {e}",
                        "type": "invalid_request_error",
                    }
                },
            )
        if self.path == "/v1/systemone":
            return self._systemone(req, images)
        if self.path == "/v1/chat/completions":
            return self._chat(req)
        return self._json(404, {"error": {"message": "unknown route"}})

    def _raw_chat(self):
        """Pass the body and status through to vLLM's chat completions."""
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        req = urllib.request.Request(
            ARGS.upstream.rstrip("/") + "/v1/chat/completions",
            data=raw,
            headers={
                "content-type": self.headers.get("content-type", "application/json")
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                code, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, body = e.code, e.read()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _decide(self, schema, state, seed):
        """-> (status, body) with the error body already shaped."""
        try:
            return 200, decide(schema, state, seed)
        except SchemaError as e:
            return 422, {"error": {"message": str(e), "type": "validation_error"}}
        except urllib.error.HTTPError as e:
            return 502, {
                "error": {
                    "message": (
                        f"upstream {e.code}: {e.read()[:300].decode(errors='replace')}"
                    ),
                    "type": "server_error",
                }
            }
        except Exception as e:
            return 500, {"error": {"message": repr(e), "type": "server_error"}}

    def _systemone(self, req, images):
        try:
            schema = jev_schema(req)
            state = jev_state(req, images + jev_images(req.get("images")))
        except SchemaError as e:
            return self._json(
                422, {"error": {"message": str(e), "type": "validation_error"}}
            )
        code, result = self._decide(schema, state, int(req.get("seed", 42)))
        if code != 200:
            return self._json(code, result)
        body, completion_tokens = result
        answers = {
            q["id"]: jev_answer(q, body["answers"][q["id"]])
            for q in schema["questions"]
        }
        labels = " ".join(f"{k}={answer_brief(v)}" for k, v in body["answers"].items())
        print(
            f"systemone: {labels} "
            f"reads={body['diagnostics']['timing']['reads']} "
            f"{body['diagnostics']['timing']['total_ms']:.0f}ms",
            flush=True,
        )
        self._json(
            200,
            {
                "model": ARGS.model,
                "answers": answers,
                # vLLM's prompt count when the reads reported one, since it covers
                # images. Otherwise the tokenizer's count of the text prompt.
                "usage": {
                    "input_tokens": body["diagnostics"].get("prompt_tokens")
                    or (
                        len(chat_prompt_ids(system_text(schema), state))
                        if isinstance(state, str)
                        else 0
                    ),
                    "output_tokens": completion_tokens,
                },
                "diagnostics": body["diagnostics"],
            },
        )

    def _chat(self, req):
        msgs = req.get("messages") or []
        if (
            len(msgs) != 2
            or msgs[0].get("role") not in ("system", "developer")
            or msgs[1].get("role") != "user"
        ):
            return self._json(
                400,
                {
                    "error": {
                        "message": (
                            "a structured request is exactly two messages: the "
                            "schema (system) and the state JSON (user)"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )
        try:
            schema_value = json.loads(message_text(msgs[0]))
            schema = parse_schema(schema_value)
            content = msgs[1].get("content", "")
            has_image = isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") in ("image_url", "image")
                for p in content
            )
            if has_image:
                # image parts pass through to vLLM unchanged, with text parts as context
                state = content
            else:
                state = message_text(msgs[1]).strip()
                json.loads(state)
        except SchemaError as e:
            return self._json(
                400, {"error": {"message": str(e), "type": "invalid_request_error"}}
            )
        except Exception as e:
            return self._json(
                400,
                {
                    "error": {
                        "message": (
                            "system must be a JSON question schema and user "
                            f"must be JSON state or image parts: {e}"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )
        code, result = self._decide(schema, state, int(req.get("seed", 42)))
        if code != 200:
            if code == 422:
                result["error"]["type"] = "invalid_request_error"
                code = 400
            return self._json(code, result)
        body, completion_tokens = result
        content = json.dumps(body, indent=2)
        labels = " ".join(f"{k}={answer_brief(v)}" for k, v in body["answers"].items())
        print(
            f"structured: {labels} "
            f"reads={body['diagnostics']['timing']['reads']} "
            f"{body['diagnostics']['timing']['total_ms']:.0f}ms",
            flush=True,
        )
        self._json(
            200,
            {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", "dgemma-structured"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": completion_tokens,
                    "total_tokens": completion_tokens,
                },
            },
        )


def self_signed(cert_dir):
    """Paths of a self-signed certificate and key in cert_dir, made with
    openssl on first use."""
    os.makedirs(cert_dir, exist_ok=True)
    cert, key = os.path.join(cert_dir, "djev.crt"), os.path.join(cert_dir, "djev.key")
    if not (os.path.exists(cert) and os.path.exists(key)):
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "3650",
                "-subj",
                "/CN=djev",
                "-keyout",
                key,
                "-out",
                cert,
            ],
            check=True,
            capture_output=True,
        )
    return cert, key


def serve_tls(host, port, cert_dir):
    cert, key = self_signed(cert_dir)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    srv = Server((host, port), Handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    global ARGS, CANVAS_LEN, CANVAS_STEP
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default="http://127.0.0.1:8010")
    p.add_argument("--model", default="dgemma")
    p.add_argument("--tokenizer", default="/models/dgemma", help="HF id or local path")
    p.add_argument("--canvas", type=int, default=64, help="the served canvas length")
    p.add_argument(
        "--canvas-step",
        type=int,
        default=16,
        help="request widths round up to a multiple of this",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8011)
    p.add_argument(
        "--tls-port", type=int, default=0, help="also listen with HTTPS here (0 = off)"
    )
    p.add_argument(
        "--cert-dir",
        default=os.path.expanduser("~/.cache/djev"),
        help="directory for the self-signed certificate",
    )
    ARGS = p.parse_args()
    CANVAS_LEN = ARGS.canvas
    CANVAS_STEP = ARGS.canvas_step
    init_tokenizer(AutoTokenizer.from_pretrained(ARGS.tokenizer))
    if ARGS.tls_port:
        serve_tls(ARGS.host, ARGS.tls_port, ARGS.cert_dir)
        print(
            f"structured server https on {ARGS.host}:{ARGS.tls_port} (self-signed)",
            flush=True,
        )
    print(
        f"structured server on {ARGS.host}:{ARGS.port} -> {ARGS.upstream} "
        f"(canvas {CANVAS_LEN})",
        flush=True,
    )
    Server((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
