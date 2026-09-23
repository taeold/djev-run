# DiffusionGemma-Jev (`djev-run`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taeold/djev-run/blob/main/colab.ipynb)

Single-container **`vLLM` block-diffusion server** (`vllm#57250`) for **`nvidia/diffusiongemma-26B-A4B-it-NVFP4`** (`26B` total parameters, `4B` active per token across `128` routed experts, `17.53 GiB` NVFP4 weights).

Instead of generating tokens left-to-right one at a time, **`djev-run`** uses `vLLM`'s native block-diffusion `SamplingParams.extra_args` (`diffusion_seed_canvas`, `diffusion_pinned`, `diffusion_max_steps`, `diffusion_read_only`) on **`POST /v1/chat/completions`** and **`POST /tokenize`**. A client tokenizes a fixed output template, pins the scaffold tokens, fills the answer slots with noise tokens (`256000..262143`), and reads the exact marginal probability distribution at every unpinned slot simultaneously in **a single forward pass (`~45-60 ms`)**.

---

## Repository Layout

```text
djev-run/
├── entrypoint.sh        # Self-contained in-process vLLM server (/tokenize, /v1/chat/completions, /snake, /dino, /tetris)
├── deploy.sh            # One-command Google Cloud Run deployment (NVIDIA RTX PRO 6000 / L4)
├── colab.ipynb          # 4-cell Google Colab Pro notebook (A100 / L4) with embedded games
├── snake.html           # 1-step diffusion Snake demo (/tokenize + /v1/chat/completions extra_body.vllm_xargs)
├── dino.html            # 1-step diffusion Dino Runner demo (/tokenize + /v1/chat/completions extra_body.vllm_xargs)
├── tetris.html          # 1-step diffusion Tetris demo (/tokenize + /v1/chat/completions extra_body.vllm_xargs)
└── ai-sdk-example/      # TypeScript Vercel AI SDK integration example
```

---

## 1-Step Diffusion Canvas Read via Standard `vLLM` (`POST /tokenize` + `POST /v1/chat/completions`)

1. **Tokenize the answer template (`POST /tokenize`)**:
   Prepend the closed thought scaffold (`SCAFFOLD = [100, 45518, 107, 101]` for `<thought>\n</thought>`) to the tokenized template `department: a\nurgency: 1\nrefund_requested: yes`.
2. **Pin scaffold positions (`diffusion_pinned`)**:
   Pin every token index except the answer slots (`department`, `urgency`, `refund_requested`), and place random mask tokens (`256000..262143`) at the unpinned slot indices.
3. **Read all slot distributions in 1 forward step (`POST /v1/chat/completions`)**:
   Pass `extra_body.vllm_xargs` with `diffusion_max_steps: 1` and `diffusion_read_only: true`. Read the exact per-slot probability distributions from `choices[0].logprobs.content[slot_pos].top_logprobs`.

```bash
# 1. Tokenize the template
curl -s http://localhost:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "department: a\nurgency: 1\nrefund_requested: yes", "add_special_tokens": false}'

# 2. Single-pass diffusion read over the pinned canvas
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "djev-dgemma",
    "messages": [
      {
        "role": "system",
        "content": "Answer each question with its single label.\ndepartment: a = billing, b = technical, c = sales\nurgency: 1 = low, 2 = minor, 3 = locked production access, 4 = complete outage\nrefund_requested: yes or no"
      },
      {
        "role": "user",
        "content": "{\"ticket_id\": \"TCK-9042\", \"text\": \"Double-charged $149.00 on invoice INV-2026-8841 and API key is locked.\"}"
      }
    ],
    "max_tokens": 20,
    "logprobs": true,
    "top_logprobs": 16,
    "extra_body": {
      "vllm_xargs": {
        "diffusion_seed_canvas": [100, 45518, 107, 101, 41957, 236787, 256101, 107, 62916, 236787, 256202, 107, 51319, 236779, 43554, 236787, 256303],
        "diffusion_pinned": [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15],
        "diffusion_max_steps": 1,
        "diffusion_read_only": true
      }
    }
  }'
```

---

## Quick Start: Run with Docker or Google Cloud Run

```bash
docker run --gpus all --shm-size 24g -p 8080:8080 ghcr.io/taeold/djev-run:latest
```

Once running, open the built-in browser games powered by 1-step `vLLM` diffusion reads:
- `http://localhost:8080/tetris`
- `http://localhost:8080/dino`
- `http://localhost:8080/snake`

To deploy on Google Cloud Run with an NVIDIA RTX PRO 6000 GPU:

```bash
BUCKET=your-gcs-weights-bucket ./deploy.sh
```
