# Dockerfile
# ──────────
# Containerises the ITSM Triage Agent FastAPI server (M4).
#
# HOW TO BUILD AND RUN:
#
#   # Build the image
#   docker build -t itsm-triage-agent .
#
#   # Run it (pass your API key as an env var, not baked into the image)
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... itsm-triage-agent
#
#   # Or use docker-compose (recommended — handles env and volumes):
#   docker-compose up
#
# WHY python:3.11-slim AND NOT python:3.11-alpine?
#
#   Alpine Linux uses musl libc instead of glibc. PyTorch, numpy, and
#   many other C-extension packages are distributed as pre-compiled
#   wheels that link against glibc. On Alpine, pip either fails to find
#   a compatible wheel and tries to compile from source (which requires
#   gcc, cmake, fortran, etc. and takes 30+ minutes) or fails outright.
#
#   python:3.11-slim is Debian-based (glibc), so all wheels install
#   instantly. "slim" strips docs, tests, and package manager caches,
#   giving us a much smaller image than the full Debian build.
#
# WHY CPU-ONLY TORCH?
#
#   The full torch wheel with CUDA support is ~2.5 GB. A Docker container
#   running on a standard cloud instance (or a CI runner) has no GPU.
#   We install the CPU-only wheel (~250 MB) from PyTorch's CPU index.
#   DistilBERT inference on CPU is still fast enough for an API (~50ms).
#
#   If you ever want GPU support, change the torch install line to:
#   RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
#   and use a CUDA base image instead of python:3.11-slim.

# ── Stage 1: base ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Set working directory inside the container
WORKDIR /app

# Environment variables that affect Python behaviour inside the container:
#
#   PYTHONDONTWRITEBYTECODE=1 — stops Python from writing .pyc files.
#     .pyc files cache compiled bytecode to disk. Inside a container the
#     filesystem is ephemeral, so .pyc files waste space and slow builds.
#
#   PYTHONUNBUFFERED=1 — forces stdout/stderr to be flushed immediately
#     instead of being buffered. Without this, print() and loguru output
#     might not appear in docker logs until the buffer fills up, making
#     debugging very confusing.
#
#   PIP_NO_CACHE_DIR=1 — tells pip not to cache downloaded wheels.
#     Docker layer caching already handles re-use between builds.
#     pip's own cache would just bloat the image for no benefit.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── Stage 2: dependencies ─────────────────────────────────────────────────────
#
# WHY COPY requirements.txt BEFORE COPYING SOURCE CODE?
#
# Docker builds images in layers. Each instruction (RUN, COPY, etc.)
# creates a new layer. Layers are cached — Docker only rebuilds a layer
# if that layer or any layer above it changed.
#
# If we copied all source code first, then installed dependencies, Docker
# would re-run "pip install" every time we changed a single line of Python.
# That would waste minutes on every build.
#
# By copying requirements.txt first and running pip install, the dependency
# layer is cached independently from the source code. As long as
# requirements.txt doesn't change, "pip install" is skipped on subsequent
# builds even if we changed app.py. This is the single most impactful
# Docker caching best practice.

COPY requirements.txt .

# Install torch CPU-only first (separate step for better layer caching)
# This is the largest download (~250 MB) and rarely changes.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies from requirements.txt.
# We exclude torch here since we just installed the CPU build above.
RUN grep -v "^torch" requirements.txt | pip install -r /dev/stdin

# ── Stage 3: application code ─────────────────────────────────────────────────
#
# WHY COPY SOURCE CODE AFTER DEPENDENCIES?
# See the layer caching explanation above.

COPY api/         api/
COPY data/schema/ data/schema/
COPY models/      models/

# ── Stage 4: model weights ───────────────────────────────────────────────────
#
# WHY COPY THE ADAPTER SEPARATELY FROM THE REST OF models/?
#
# The LoRA adapter weights (~4 MB) change when you retrain the model.
# Training checkpoints (models/finetuned/distilbert-lora/checkpoint-*)
# are large (~400 MB each) and NOT needed at inference time.
#
# By being selective about what we COPY, we keep the image small and avoid
# baking in checkpoint files that serve no purpose in production.
#
# In a real deployment, you'd pull the adapter from an artifact store
# (S3, GCS, MLflow) at container startup instead of baking it in.
# For this portfolio project, baking it in is fine and simpler.

COPY models/finetuned/distilbert-lora/adapter/ \
     models/finetuned/distilbert-lora/adapter/

# ── Runtime configuration ─────────────────────────────────────────────────────
#
# ADAPTER_DIR tells api/app.py where to find the fine-tuned model weights.
# We set it as a build-time default that can be overridden at runtime with:
#   docker run -e ADAPTER_DIR=/custom/path ...

ENV ADAPTER_DIR=/app/models/finetuned/distilbert-lora/adapter

# Tell Docker that the container listens on port 8000.
# This is documentation only — it doesn't publish the port.
# Use -p 8000:8000 in docker run or "ports:" in docker-compose to expose it.
EXPOSE 8000

# ── Healthcheck ──────────────────────────────────────────────────────────────
#
# Docker's built-in HEALTHCHECK polls the /health endpoint every 30 seconds.
# If it returns a non-zero exit code (i.e., the server isn't responding),
# Docker marks the container as "unhealthy". Orchestration systems like
# Kubernetes, ECS, and docker-compose can automatically restart unhealthy
# containers.
#
# This is the containerised equivalent of the readiness probe pattern we
# implemented in the /health endpoint — the same design principle, different layer.

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
  || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
#
# WHY --host 0.0.0.0?
#
# By default, uvicorn binds to 127.0.0.1 (localhost), which is only
# reachable from within the same container. To receive traffic from
# outside the container (from the host machine or other containers),
# we must bind to 0.0.0.0 ("all interfaces").
#
# WHY --workers 1?
#
# Multiple workers would spawn multiple Python processes, each loading
# the DistilBERT model into memory. With workers=1 we load it once.
# For a portfolio project, 1 worker is the right tradeoff.
# In production you'd use Gunicorn + uvicorn workers or a proper
# orchestration layer to scale horizontally.

CMD ["python", "-m", "uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
