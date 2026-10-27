# Screen Agent

**A screen-reading agent that "looks" with one Nemotron model and "thinks" with another — served entirely on Nebius Token Factory.**

Most vision demos use a single model for everything: it looks at the screen, and it also decides what to do next. That works until the task needs real reasoning — then an 8B vision model either hallucinates a button or freezes on a long chain of thought.

This project splits the job into two NVIDIA models with different strengths, both called at runtime from Nebius Token Factory:

| Role | Model | Why this one |
|---|---|---|
| **Look** (vision) | `nvidia/nemotron-3.5-super-vl` | 1024K context, strong OCR and layout understanding, returns structured JSON |
| **Think** (reasoning) | `nvidia/nemotron-3-ultra-550b-a55b` | 550B hybrid MoE built for multi-step reasoning and agent orchestration |

The visual model is told to *only describe*, never to infer intent. The reasoning model receives that description as ground truth and is told *not to invent anything not in it*. This separation is what makes the agent honest: when it is wrong, you can see which half was wrong.

## What it does

1. **Upload** a screenshot or a photo.
2. **Look** — the VL model returns structured JSON: scene type, app name, layout, readable text, UI elements with normalized boxes, dominant colors, and an explicit `uncertain` list.
3. **Ask** — the reasoning model answers: what this is, what you probably want, what to click next, and what looks risky.
4. **Correct** — say "no, that's an apple, look again". The VL model re-examines the image with your correction as a hint, and the reasoning model must open its answer by admitting it was wrong, then give the corrected judgement.

Step 4 is the part most vision agents skip. A screen assistant that cannot say "I was wrong" is not usable for anything that matters.

## Architecture

```
browser ──upload──▶ FastAPI ──▶ Nemotron 3.5 Super VL   (structured JSON, "look")
                       │                  │
                       │                  ▼
                       └────────▶ Nemotron 3 Ultra       (reasoning, "think")
                                          │
                                          ▼
                                   answer + trace
```

Both calls go to `https://api.tokenfactory.nebius.com/v1/` (OpenAI-compatible). No local GPU is required to run the demo.

## Quick start

```bash
git clone https://github.com/bo117/screen-agent-nebius.git
cd screen-agent-nebius

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then put your Nebius Token Factory key in .env
export NEBIUS_API_KEY=...   # or `set -a && source .env && set +a`

uvicorn app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Get a key at [Nebius Token Factory](https://nebius.com/services/token-factory) — the API is OpenAI-compatible, so any OpenAI SDK works by changing `base_url`.

Verify the wiring without a browser:

```bash
python scripts/smoke_test.py --image path/to/screenshot.png
```

It prints the VL JSON and the reasoning answer, and fails loudly if the key or model id is wrong.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | key present? which models? how many images cached |
| `/api/upload` | POST | multipart image → `image_id` |
| `/api/look` | POST | `{image_id}` → structured visual JSON |
| `/api/ask` | POST | `{image_id, question, history}` → look + think |
| `/api/correct` | POST | `{image_id, correction}` → re-look + forced acknowledgement |

## Notes on model choice

The Token Factory catalogue makes the tiering explicit:

- **Ultra 550B** — frontier reasoning, deep research, long-running agents.
- **Super 120B** — complex reasoning, multi-agent work.
- **3.5 Super VL** — vision-language, 1024K context.
- **Nano 30B** — fast everyday calls.

A common mistake is to fine-tune a small vision model and expect reasoning to appear. It does not: reasoning needs both the parameter budget and reasoning-shaped training data. Splitting "look" and "think" lets a mid-size VL model do what it is good at, and puts the expensive model only where reasoning is actually needed — which also keeps token spend down.

## Known limits

- The reasoning model cannot see the image directly; it only sees the VL model's JSON. If the VL description is wrong, the answer is wrong — which is why the `uncertain` field exists and why the correction loop matters.
- Coordinates are normalized 0–1 and are approximate; this is a reading assistant, not a click-precision grounding system.
- Images are cached in process memory only (12 MB cap per image); restarting the server clears them.
- The demo is single-user and has no auth. Do not expose it publicly with a live API key without adding rate limiting.

## License

Apache 2.0 — see [LICENSE](LICENSE).
