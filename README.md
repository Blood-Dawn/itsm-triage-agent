---
title: ITSM Triage Agent
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# ITSM Triage Agent

An end-to-end LLM-powered IT ticket triage pipeline. Given a ticket subject and body,
the system classifies it into one of 8 categories and assigns a priority level using
either a fine-tuned DistilBERT model (fast, free, offline) or an LLM baseline
(Claude Haiku or GPT-4o-mini).

Built as a portfolio project by **Kheiven D'Haiti**, B.S. Computer Science, AI Minor —
Florida Atlantic University (expected Dec 2026).

[![CI](https://github.com/Blood-Dawn/itsm-triage-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Blood-Dawn/itsm-triage-agent/actions/workflows/ci.yml)

---

## 🚀 Try the Live Demo

**[huggingface.co/spaces/Bloodawn/itsm-triage-agent](https://huggingface.co/spaces/Bloodawn/itsm-triage-agent)**

### No setup required for the Finetuned backend

The **Finetuned (DistilBERT + LoRA)** backend runs entirely on HF's free CPU hardware.
Just open the Space and start triaging tickets — no account, no API key, no cost.

### To use the Baseline (LLM comparison) backend

The baseline calls an external LLM API for a side-by-side comparison. You need your own
API key — **your key, your cost** (typically $0.001 per call):

**Option A — Anthropic (Claude Haiku):**
1. Create a free account at [console.anthropic.com](https://console.anthropic.com)
2. Go to **API Keys** → **Create Key**
3. Copy your key (starts with `sk-ant-...`)
4. In the Space sidebar, select **Anthropic (Claude Haiku)** and paste your key

**Option B — OpenAI (GPT-4o-mini):**
1. Create an account at [platform.openai.com](https://platform.openai.com)
2. Go to **API keys** → **Create new secret key**
3. Copy your key (starts with `sk-...`)
4. In the Space sidebar, select **OpenAI (GPT-4o-mini)** and paste your key

> Your API key is sent directly to the model server for that request only — it is never
> stored, logged, or shared.

---

## What It Does

An IT helpdesk receives hundreds of tickets per day. A human agent reads each one,
decides what kind of problem it is (network outage? software bug? printer jam?), and
assigns a priority. This project automates that first-pass triage step.

```
Ticket subject + body
        |
        v
  POST /triage
        |
   +---------+
   | backend |
   +---------+
   finetuned |  baseline
   (21ms/$0) |  (2000ms/$0.001)
        |
        v
  TriageResponse
    category: "Network"
    priority: "High"
    confidence: 0.97
    next_action: "Escalate to network team"
```

---

## Architecture

```
itsm-triage-agent/
|
+-- data/
|   +-- schema/ticket.py        # Single source of truth: categories, priorities, schema
|   +-- generator/gen.py        # Synthetic ticket generation (Faker)
|   +-- raw/                    # train.jsonl / val.jsonl / test.jsonl
|
+-- models/
|   +-- baseline/               # M1: Claude Haiku zero-shot classifier
|   |   +-- predict.py
|   |   +-- prompt.py
|   |
|   +-- finetune/               # M2-M3: DistilBERT + LoRA fine-tuning
|   |   +-- dataset.py          # HuggingFace DatasetDict preparation
|   |   +-- model.py            # DualHeadDistilBERT architecture
|   |   +-- train.py            # HuggingFace Trainer loop
|   |   +-- predict.py          # Batch inference wrapper
|   |
|   +-- finetuned/
|       +-- distilbert-lora/
|           +-- adapter/        # Saved LoRA weights (not in git — too large)
|
+-- api/
|   +-- app.py                  # M4: FastAPI server, /health + /triage endpoints
|   +-- schemas.py              # Pydantic request/response models
|
+-- eval/
|   +-- metrics.py              # M5: accuracy, F1, latency percentiles
|   +-- run.py                  # CLI evaluation harness
|   +-- results/                # Timestamped JSON result files
|
+-- tests/
|   +-- test_metrics.py         # 21 pytest unit tests for eval/metrics.py
|
+-- scripts/
|   +-- test_baseline.py        # Quick smoke test for the LLM baseline
|   +-- test_finetune.py        # Quick smoke test for the fine-tuned model
|
+-- Dockerfile                  # M6: python:3.11-slim, CPU torch, HEALTHCHECK
+-- docker-compose.yml          # Single-command local deployment
+-- .github/workflows/ci.yml    # 3-job CI: lint + pytest + docker build
```

---

## Eval Results

Evaluated on 200 randomly sampled test tickets (seed=42) from a held-out test set of
1,016 tickets that were never seen during training.

| Metric | Finetuned | Baseline (Claude Haiku) |
|---|---|---|
| Category Accuracy | **100.0%** | ~95% (zero-shot) |
| Category Macro F1 | **100.0%** | — |
| Priority Accuracy | 53.5% | — |
| Priority Macro F1 | 17.4% | — |
| Latency (mean) | **21 ms** | ~2,000 ms |
| Latency (p99) | **24 ms** | — |
| Cost per ticket | **$0.000** | ~$0.001 |

Category classification is perfect on the test set. Priority prediction is weaker —
priority is inherently subjective and the model skews toward majority classes. This is
a known limitation targeted in [M10 of the roadmap](ROADMAP.md).

---

## Quick Start

### Option 1: Docker (recommended)

```bash
# Copy and fill in your Anthropic API key (only needed for baseline backend)
cp .env.example .env

# Start the server
docker compose up

# Server is live at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### Option 2: Local (Python 3.11+)

```bash
# Install CPU-only PyTorch first (avoids large CUDA download)
pip install torch==2.6.0+cpu torchvision==0.21.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Copy and fill in your Anthropic API key
cp .env.example .env

# Start the server
python -m uvicorn api.app:app --reload

# Swagger UI at http://localhost:8000/docs
```

---

## API

### `GET /health`

Returns server status and which backends are available.

```json
{
  "status": "ok",
  "finetuned_available": true,
  "baseline_available": true,
  "model": "distilbert-base-uncased + LoRA"
}
```

### `POST /triage`

Classifies a ticket and returns triage results.

**Request:**
```json
{
  "subject": "Cannot connect to VPN",
  "body": "Getting timeout errors since this morning. Tried restarting.",
  "backend": "finetuned"
}
```

**Response:**
```json
{
  "category": "Network",
  "priority": "High",
  "confidence": 0.97,
  "next_action": "Escalate to network team — check VPN gateway status",
  "backend_used": "finetuned",
  "latency_ms": 21.4
}
```

---

## Running Evaluations

```bash
# Evaluate finetuned model on 200 tickets
python -m eval.run --backend finetuned --n 200

# Compare both backends on 100 tickets (requires ANTHROPIC_API_KEY)
python -m eval.run --backend both --n 100

# Full test set evaluation
python -m eval.run --backend finetuned --n 1016
```

Results are saved to `eval/results/` as timestamped JSON files.

---

## Training Your Own Model

```bash
# Generate synthetic data
python -m data.generator.gen

# Fine-tune (requires ~4 GB RAM, runs on CPU in ~10 minutes)
python -m models.finetune.train

# Smoke-test the adapter
python scripts/test_finetune.py
```

---

## CI Pipeline

Every push triggers three GitHub Actions jobs:

| Job | Tool | What it checks | Time |
|---|---|---|---|
| Lint | ruff | Unused imports, syntax errors, code style | ~10s |
| Test | pytest | 21 unit tests for eval/metrics.py | ~30s |
| Docker | docker build | Dockerfile validity + /health smoke test | ~3 min |

The Docker job only runs if lint and tests pass.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data generation | Python, Faker |
| LLM baseline | Anthropic API (Claude Haiku) |
| Model | DistilBERT + LoRA (HuggingFace PEFT) |
| Training | HuggingFace Transformers, Trainer API |
| Inference API | FastAPI, Pydantic v2, uvicorn |
| Evaluation | scikit-learn, numpy, rich |
| Containerization | Docker, docker-compose |
| CI | GitHub Actions, ruff, pytest |

---

## Deployment (Hugging Face Spaces)

The live demo runs on Hugging Face Spaces (free CPU tier). Both the FastAPI server
and the Streamlit UI run inside a single Docker container managed by supervisord.

**One-time setup:**

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Space name: `itsm-triage-agent`
   - SDK: **Docker**
   - Hardware: **CPU Basic** (free)

2. Clone the Space repo and copy this project into it:
```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/itsm-triage-agent hf-space
cp -r . hf-space/
cd hf-space
```

3. Copy the HF Spaces Dockerfile into place:
```bash
cp deploy/Dockerfile.spaces Dockerfile
```

4. Add the Space metadata header to the top of `README.md` in the Space repo:
```yaml
---
title: ITSM Triage Agent
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
```

5. Set your Anthropic API key as a Space secret (for the baseline backend):
   - Space → Settings → Variables and secrets → New secret
   - Name: `ANTHROPIC_API_KEY`  Value: `sk-ant-...`

6. Push and HF builds automatically:
```bash
git add -A
git commit -m "deploy: itsm-triage-agent"
git push
```

The Space URL will be: `https://huggingface.co/spaces/YOUR_HF_USERNAME/itsm-triage-agent`

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full milestone breakdown including upcoming work:
- **M8** — Streamlit demo UI
- **M9** — Cloud deployment (Railway/Render)
- **M10** — Priority classification improvement
- **M11** — Monitoring + observability

---

## Author

**Kheiven D'Haiti**
B.S. Computer Science, AI Minor — Florida Atlantic University
[github.com/Blood-Dawn](https://github.com/Blood-Dawn)
