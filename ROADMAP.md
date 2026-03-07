# ITSM Triage Agent — Project Roadmap

This document tracks every milestone from initial idea to production-ready deployment.
Each milestone is a self-contained vertical slice: it has clear inputs, outputs, and a
"why it matters" explanation so the decision-making behind the project is transparent.

---

## Completed Milestones

### M0 — Synthetic Data Generation
**Status:** Complete

**What:** Built a data generator using the `Faker` library to produce realistic ITSM
(IT Service Management) ticket datasets. Generates tickets across 8 categories
(Network, Hardware, Software, Security, Access, Email, Printer, Performance) and 4
priority levels (Low, Medium, High, Critical).

**Why:** Real ITSM ticket data is proprietary and rarely public. Synthetic generation
lets the project demonstrate a full ML pipeline without depending on data that can't
be shared. The generator is schema-driven — every downstream component imports from
`data/schema/ticket.py` so the category/priority definitions are a single source of truth.

**Output:** `data/raw/train.jsonl`, `data/raw/val.jsonl`, `data/raw/test.jsonl`
(~8,000 / 1,000 / 1,016 tickets respectively)

---

### M1 — LLM Baseline
**Status:** Complete

**What:** Implemented a zero-shot classification baseline using the Anthropic API
(Claude Haiku). Given a ticket's subject + body, the model returns a predicted
category, priority, confidence score, reasoning, and suggested next action.

**Why:** Before training a custom model it is important to establish a baseline. The
LLM baseline requires no training data and sets the performance ceiling for "what a
capable general-purpose AI can do out of the box." It becomes the comparison target
for the fine-tuned model in M5.

**Output:** `models/baseline/predict.py`, `models/baseline/prompt.py`
**Cost:** ~$0.001 per ticket | **Latency:** ~2,000 ms per ticket

---

### M2 — Fine-Tuning Dataset Preparation
**Status:** Complete

**What:** Transformed the raw JSONL ticket data into a HuggingFace `DatasetDict`
format suitable for supervised fine-tuning. Applied DistilBERT tokenization, mapped
string labels to integer IDs, and validated split integrity.

**Why:** HuggingFace Trainer expects a specific data contract. Getting data preparation
right before training prevents silent bugs where the model trains on misaligned labels.
The label maps (`ID_TO_CATEGORY`, `ID_TO_PRIORITY`) defined here are shared by both
training and inference to guarantee consistency.

**Output:** `models/finetune/dataset.py`

---

### M3 — Fine-Tuned Model (DualHeadDistilBERT + LoRA)
**Status:** Complete

**What:** Fine-tuned a `distilbert-base-uncased` model with two classification heads
(one for category, one for priority) using LoRA (Low-Rank Adaptation) from the PEFT
library. LoRA freezes most of the base model weights and only trains small rank-4
adapter matrices, making fine-tuning feasible on consumer hardware.

**Why:** DistilBERT is 40% smaller than BERT while retaining 97% of its performance.
LoRA reduces trainable parameters by ~99% compared to full fine-tuning. The result is
a model that trains in minutes on a laptop and runs inference at ~21 ms on CPU —
orders of magnitude faster and cheaper than the LLM baseline.

**Output:** `models/finetune/model.py`, `models/finetune/train.py`,
`models/finetuned/distilbert-lora/adapter/`

---

### M4 — FastAPI Inference Server
**Status:** Complete

**What:** Wrapped both backends (finetuned and baseline) behind a REST API using
FastAPI. Exposes two endpoints: `GET /health` and `POST /triage`. The `/triage`
endpoint accepts a `TriageRequest` and returns a `TriageResponse` with a unified
schema across both backends. Swagger UI auto-generated from Pydantic type annotations.

**Why:** A model that only runs from a script is not usable. The FastAPI server makes
the model consumable by any HTTP client — a frontend, a ticketing system webhook, or
an integration test. The unified response schema means callers don't need to know which
backend is active.

**Output:** `api/app.py`, `api/schemas.py`
**Runs:** `uvicorn api.app:app --reload` → Swagger UI at `http://localhost:8000/docs`

---

### M5 — Evaluation Harness
**Status:** Complete

**What:** Built a CLI evaluation harness that loads test tickets, runs them through
either or both backends, and reports accuracy, macro F1, weighted F1, per-class F1,
and latency percentiles (p50/p95/p99). Results are saved as timestamped JSON files.

**Why:** "The model is good" is not a claim — it is a hypothesis. The eval harness
turns that hypothesis into a number. Per-class F1 exposes which categories the model
struggles with. Latency percentiles (not just mean) reveal tail latency issues that
would appear under real load. The harness also powers the baseline vs. finetuned
comparison that is the project's central result.

**Eval Results (finetuned, n=200, seed=42):**

| Metric | Score |
|---|---|
| Category Accuracy | 100.0% |
| Category Macro F1 | 100.0% |
| Priority Accuracy | 53.5% |
| Priority Macro F1 | 17.4% |
| Latency (mean) | 21.2 ms |
| Latency (p99) | 24.4 ms |

Category classification is perfect. Priority prediction is weaker — priority is
inherently more subjective and the model leans toward majority classes. This is a
known limitation and a target for M8+ improvements.

**Output:** `eval/metrics.py`, `eval/run.py`, `eval/results/`
**Runs:** `python -m eval.run --backend finetuned --n 200`

---

### M6 — Docker + GitHub Actions CI
**Status:** Complete

**What:** Containerized the FastAPI server with a production-ready `Dockerfile` using
`python:3.11-slim` and CPU-only PyTorch (~250 MB vs 2.5 GB CUDA). Added
`docker-compose.yml` for single-command local deployment. Implemented a three-job
GitHub Actions CI pipeline: lint (ruff), unit tests (pytest), and Docker build with
smoke test.

**Why:** Containerization removes "works on my machine" from the equation. A recruiter
or hiring manager can run the project with one command regardless of their environment.
CI ensures that every push is automatically validated — broken Dockerfiles and import
errors are caught before they reach main.

**Output:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`,
`.github/workflows/ci.yml`, `tests/test_metrics.py`
**Runs:** `docker compose up` → server live at `http://localhost:8000`

---

### M7 — Documentation
**Status:** Complete

**What:** Wrote a comprehensive `README.md` covering project overview, architecture,
quick-start instructions, eval results, and tech stack. Wrote this `ROADMAP.md`
explaining the rationale behind every milestone decision.

**Why:** Documentation is the difference between a project that looks complete and one
that is. A recruiter who clones the repo needs to understand what it does, why it
exists, and how to run it within 60 seconds.

---

## Upcoming Milestones

### M8 — Streamlit Demo UI
**Status:** Planned

**What:** A web UI built with Streamlit that lets users type a ticket subject and body,
select a backend (finetuned or baseline), and see the triage result in real time with
confidence scores and a latency readout.

**Why:** A live demo is more compelling than a terminal screenshot in a portfolio. The
UI also surfaces the latency difference between the two backends in a way that is
immediately intuitive — clicking "baseline" vs "finetuned" and watching the response
time change is a better explanation than any chart.

**Planned output:** `app/streamlit_app.py`
**Runs:** `streamlit run app/streamlit_app.py`

---

### M9 — Cloud Deployment
**Status:** Planned

**What:** Deploy the FastAPI server (and optionally the Streamlit UI) to a cloud
platform. Candidates: Railway, Render, or Hugging Face Spaces (free tier for the
Streamlit demo).

**Why:** A live URL on a resume is worth more than a GitHub repo. It proves the project
runs in a real environment, not just on a developer's laptop.

**Planned output:** `railway.toml` or `render.yaml`, live URL

---

### M10 — Priority Classification Improvement
**Status:** Planned

**What:** Address the 53.5% priority accuracy identified in M5. Candidates: class
weighting in the loss function, priority-aware data augmentation, or a second-stage
re-ranking model trained specifically on priority edge cases.

**Why:** The eval harness exists precisely to drive this kind of targeted improvement.
Category classification is already at 100% — priority is the remaining gap between
the current model and a production-ready triage system.

---

### M11 — Monitoring + Observability
**Status:** Planned

**What:** Add structured request logging (ticket text → predicted label → latency →
confidence) to the FastAPI server. Build a lightweight dashboard (Streamlit or
Grafana) showing prediction distributions and latency trends over time.

**Why:** A model in production degrades silently without monitoring. Logging prediction
confidence over time can detect distribution shift — if average confidence drops, the
incoming tickets may look different from training data and the model needs retraining.

---

## Tech Stack Summary

| Layer | Technology | Reason |
|---|---|---|
| Data generation | Python, Faker | Reproducible synthetic data without privacy concerns |
| LLM baseline | Anthropic API (Claude Haiku) | Fast, cheap, no training required |
| Model architecture | DistilBERT + LoRA (PEFT) | 40% smaller than BERT, fine-tunable on CPU |
| Training | HuggingFace Transformers, Trainer | Industry-standard fine-tuning API |
| Inference API | FastAPI, Pydantic, uvicorn | Type-safe, auto-documented, production-grade |
| Evaluation | scikit-learn, numpy, rich | Standard metrics, clean terminal output |
| Containerization | Docker, docker-compose | Reproducible deployment, one-command startup |
| CI | GitHub Actions, ruff, pytest | Automated quality gates on every push |
| Planned UI | Streamlit | Rapid prototyping, no frontend build toolchain |
| Planned deploy | Railway / Render | Free tier sufficient for portfolio demos |
