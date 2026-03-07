"""
api/app.py
──────────
M4 FastAPI inference server — wraps both M1 (LLM baseline) and M3
(fine-tuned) backends behind a single REST endpoint.

HOW TO RUN (from the itsm-triage-agent root):

    # Development mode (auto-reloads on file save)
    uvicorn api.app:app --reload --port 8000

    # Then open in browser:
    #   http://localhost:8000/docs   ← Swagger UI (interactive API explorer)
    #   http://localhost:8000/health ← Health check
    #   http://localhost:8000/redoc  ← ReDoc (alternative API docs)

    # Test with curl (PowerShell):
    Invoke-RestMethod -Method Post -Uri http://localhost:8000/triage `
      -ContentType "application/json" `
      -Body '{"text": "My laptop won'\''t boot", "backend": "finetuned"}'

WHY FASTAPI (not Flask)?

    Flask is the classic Python web framework — simple, minimal, no
    opinions. FastAPI is the modern choice for ML APIs because:

    1. Async-native: built on Starlette + anyio. Each request runs in
       an async event loop, so the server can handle many concurrent
       requests while waiting on I/O (like the Anthropic API call).
       Flask is synchronous by default — one request blocks the thread.

    2. Automatic validation: Pydantic integration means incoming JSON
       is validated and typed before your handler runs. Flask requires
       manual request.get_json() + validation.

    3. Auto-generated docs: FastAPI reads your type annotations and
       generates OpenAPI (Swagger) docs at /docs automatically.
       This is huge for demos and for ML APIs that teammates consume.

    4. Performance: FastAPI is one of the fastest Python frameworks
       (benchmarks show it neck-and-neck with Node.js Express).

    In AI engineering roles, FastAPI is the de-facto standard for
    serving ML models. You'll see it everywhere.

KEY PATTERN — lifespan context manager:

    Modern FastAPI uses a lifespan function (instead of the old
    @app.on_event("startup")) to control what happens when the server
    starts and shuts down. We use startup to preload the fine-tuned
    model into GPU memory so the first request isn't slow.

    @asynccontextmanager
    async def lifespan(app):
        # Code before yield runs at startup
        load_model()
        yield
        # Code after yield runs at shutdown
        cleanup()
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from loguru import logger

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env with override=True so real keys win over empty env vars
# (same pattern as our test scripts)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

from api.schemas import Backend, HealthResponse, TriageRequest, TriageResponse

# ─── CONFIG ───────────────────────────────────────────────────────────────────
#
# WHY os.getenv WITH A DEFAULT (not hardcoded path)?
#
# The adapter directory might be different in different environments:
#   - Local dev: models/finetuned/distilbert-lora/adapter  (our default)
#   - Docker: /app/models/finetuned/adapter
#   - Cloud VM: /mnt/models/adapter
#
# Using an environment variable means you can override the path without
# changing code. This is the 12-factor app principle: config from env.

ADAPTER_DIR = Path(
    os.getenv("ADAPTER_DIR", str(PROJECT_ROOT / "models" / "finetuned" / "distilbert-lora" / "adapter"))
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ─── LIFESPAN ─────────────────────────────────────────────────────────────────
#
# The lifespan function runs startup/shutdown logic around the app's life.
# It replaces the older @app.on_event("startup") pattern (deprecated in
# FastAPI 0.93+).
#
# WHY PRELOAD THE FINE-TUNED MODEL ON STARTUP?
#
# Loading DualHeadDistilBERT + LoRA adapter takes ~1-2 seconds the first
# time (disk read + CUDA memory allocation). If we loaded lazily (on first
# request), the first caller would get a 2-second response and see a
# misleading latency spike. Preloading ensures every request gets the
# fast ~10ms latency.
#
# The LLM baseline (M1) doesn't need preloading because it just creates
# an HTTP client — no heavy weights to load.

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: preload fine-tuned model. Shutdown: nothing to clean up."""

    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info("Server starting up...")

    if ADAPTER_DIR.exists():
        logger.info(f"Preloading fine-tuned model from {ADAPTER_DIR}")
        try:
            # Import here (not at top) so the server can start even if
            # PyTorch isn't installed — it will just serve the baseline.
            from models.finetune.predict import _load_model
            _load_model(ADAPTER_DIR)
            logger.info("Fine-tuned model loaded and ready")
        except Exception as e:
            # Non-fatal: server still starts, finetuned backend just won't work
            logger.warning(f"Fine-tuned model failed to load: {e}")
            logger.warning("Server will start without fine-tuned backend")
    else:
        logger.warning(
            f"Adapter directory not found: {ADAPTER_DIR}\n"
            f"Fine-tuned backend unavailable. Run training first:\n"
            f"  python -m models.finetune.train --epochs 3 --batch-size 32"
        )

    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set. Baseline backend unavailable.")

    yield  # ← server runs while we're here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    # PyTorch + CUDA memory is freed automatically when the process exits.
    # Nothing explicit needed here for this project.
    logger.info("Server shutting down")


# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ITSM Triage Agent API",
    description=(
        "Classifies IT helpdesk tickets into category + priority using "
        "either a fine-tuned DistilBERT model (fast, free, local) or "
        "the Claude LLM baseline (slower, paid, with reasoning)."
    ),
    version="0.1.0",
    lifespan=lifespan,
    # docs_url="/docs" is the default — Swagger UI at http://localhost:8000/docs
    # redoc_url="/redoc" is the default — ReDoc at http://localhost:8000/redoc
)


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """
    Health check — returns which backends are available.

    Use this to verify the server is running and the model is loaded
    before sending traffic. In production you'd poll this from a
    load balancer to know when a new instance is ready to serve.
    """
    from models.finetune.predict import _model as ft_model

    backends_available = []
    if ft_model is not None:
        backends_available.append("finetuned")
    if ANTHROPIC_API_KEY:
        backends_available.append("baseline")

    status = "ok" if backends_available else "degraded"

    return HealthResponse(
        status=status,
        backends_available=backends_available,
        model_loaded=(ft_model is not None),
    )


@app.post("/triage", response_model=TriageResponse, tags=["inference"])
async def triage(req: TriageRequest) -> TriageResponse:
    """
    Classify a ticket into category + priority.

    Send a POST with `text` (the ticket body) and optionally `backend`
    ("finetuned" or "baseline"). Returns category, priority, next_action,
    and backend-specific metadata.

    **finetuned** backend:
    - ~10ms latency, $0 cost, offline-capable
    - Returns confidence scores (how sure the model is)
    - No reasoning text

    **baseline** backend:
    - ~2000ms latency, ~$0.001/ticket
    - Returns chain-of-thought reasoning
    - Requires ANTHROPIC_API_KEY in environment
    """

    # ── ROUTE TO BACKEND ──────────────────────────────────────────────────────

    if req.backend == Backend.FINETUNED:
        return await _run_finetuned(req.text)
    else:
        return await _run_baseline(req.text)


# ─── BACKEND HANDLERS ─────────────────────────────────────────────────────────
#
# WHY SEPARATE PRIVATE FUNCTIONS INSTEAD OF ALL LOGIC IN THE ROUTE?
#
# The route handler is responsible for HTTP concerns: parsing the request,
# returning the response. The backend handlers are responsible for ML
# concerns: calling the right model and mapping its output to TriageResponse.
#
# Keeping these separate makes the code easier to test in isolation and
# easier to read — you can understand each function without the others.

async def _run_finetuned(text: str) -> TriageResponse:
    """
    Run inference using the local fine-tuned model (M3).

    WHY async even though predict() is synchronous?

        FastAPI route handlers must be async (or regular def — FastAPI
        runs regular def routes in a thread pool automatically). We mark
        helper functions as async to be explicit about the async context.

        In a production deployment you'd use asyncio.get_event_loop()
        .run_in_executor() to run the synchronous PyTorch forward pass
        in a thread pool, preventing it from blocking the event loop.
        For a portfolio project, calling it directly is fine.
    """
    if not ADAPTER_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Fine-tuned model not available. "
                "Run training first: python -m models.finetune.train"
            ),
        )

    try:
        from models.finetune.predict import predict as ft_predict

        result = ft_predict(text, adapter_dir=ADAPTER_DIR)

        if not result.success:
            # Return 200 with success=False rather than 500 — the server
            # worked fine, the model just couldn't classify this ticket.
            # The caller can check result.success and handle accordingly.
            return TriageResponse(
                category="other",
                priority="P3",
                next_action="",
                backend=Backend.FINETUNED.value,
                latency_ms=result.latency_ms,
                success=False,
                error=result.error,
            )

        return TriageResponse(
            category=result.category,
            priority=result.priority,
            next_action=result.next_action,
            backend=Backend.FINETUNED.value,
            latency_ms=result.latency_ms,
            success=True,
            cat_confidence=result.cat_confidence,
            pri_confidence=result.pri_confidence,
        )

    except Exception as e:
        logger.exception("Unexpected error in finetuned backend")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_baseline(text: str) -> TriageResponse:
    """
    Run inference using the Claude LLM baseline (M1).

    WHY raise 503 instead of 500 when the key is missing?

        HTTP 503 = Service Unavailable (temporary, may fix itself)
        HTTP 500 = Internal Server Error (something is broken)
        HTTP 401 = Unauthorized (wrong key)

        Missing key is a configuration problem, not a code bug.
        503 is the right status because the baseline will work fine
        once the key is added. A load balancer or retry logic should
        treat 503 differently than 500.
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "Baseline backend unavailable: ANTHROPIC_API_KEY not set. "
                "Add it to your .env file."
            ),
        )

    try:
        from models.baseline.predict import predict as bl_predict

        result = bl_predict(text)

        if not result.success:
            return TriageResponse(
                category="other",
                priority="P3",
                next_action="",
                backend=Backend.BASELINE.value,
                latency_ms=result.latency_ms,
                success=False,
                error=result.error,
            )

        return TriageResponse(
            category=result.category,
            priority=result.priority,
            next_action=result.next_action,
            backend=Backend.BASELINE.value,
            latency_ms=result.latency_ms,
            success=True,
            reasoning=result.reasoning,
            cost_usd=result.cost_usd,
            model=result.model,
        )

    except Exception as e:
        logger.exception("Unexpected error in baseline backend")
        raise HTTPException(status_code=500, detail=str(e))
