"""
api/schemas.py
──────────────
Pydantic request and response models for the triage API.

WHY DEFINE SCHEMAS SEPARATELY FROM app.py?

    Separation of concerns. app.py handles HTTP routing and request
    lifecycle. schemas.py defines the data contracts — what goes in
    and what comes out. This split makes both files easier to read
    and lets you import schemas in tests without importing the whole
    FastAPI application.

    In a larger codebase you might also import schemas into a CLI,
    a batch job, or a frontend type generator — another reason to
    keep them decoupled from HTTP handling.

WHY PYDANTIC?

    Pydantic is FastAPI's native validation library. When FastAPI
    sees a function parameter typed with a Pydantic model, it:
      1. Parses the incoming JSON into a Python dict
      2. Validates every field against its type annotation
      3. Raises a 422 Unprocessable Entity with a clear error message
         if any field is wrong (missing, wrong type, out of range)
      4. Returns the validated data as a typed Python object

    This means you never manually call json.loads(), check for missing
    keys, or validate types yourself. FastAPI + Pydantic handles all
    of it, and the OpenAPI spec is auto-generated from these classes.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ─── ENUMS ────────────────────────────────────────────────────────────────────

class Backend(str, Enum):
    """
    Which model backend to use for inference.

    WHY str, Enum (not plain Enum)?
        Dual inheritance with str makes the enum values JSON-serializable
        by default. FastAPI can include them in the OpenAPI spec and
        serialize them without extra config. Same pattern we used in
        data/schema/ticket.py for Category and Priority.

    baseline:
        M1 — sends ticket to Claude Haiku via Anthropic API.
        Pros: rich reasoning text, no local model required.
        Cons: ~$0.001/ticket, ~2s latency, requires internet + API key.

    finetuned:
        M3 — runs ticket through local DualHeadDistilBERT + LoRA.
        Pros: $0/ticket, ~10ms latency, works offline.
        Cons: no reasoning text, priority accuracy limited by training data.
    """
    BASELINE  = "baseline"
    FINETUNED = "finetuned"


# ─── REQUEST ──────────────────────────────────────────────────────────────────

class TriageRequest(BaseModel):
    """
    Incoming POST /triage request body.

    Fields
    ------
    text : str
        Raw ticket text — the same format as the training data.
        Subject line + body, natural language, any length (truncated
        to 128 tokens internally if longer).
    backend : Backend
        Which model to use. Defaults to "finetuned" (fast, free, local).
        Use "baseline" when you want LLM reasoning or higher accuracy.

    Example JSON body:
        {
          "text": "My laptop won't boot after a Windows update",
          "backend": "finetuned"
        }
    """

    text: str = Field(
        ...,                           # ... means required (no default)
        min_length=10,
        max_length=5000,
        description="Raw ticket text to classify",
        examples=["My laptop won't boot after a Windows update."],
    )

    backend: Backend = Field(
        default=Backend.FINETUNED,
        description="Model backend: 'finetuned' (fast/free) or 'baseline' (LLM)",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        """
        Reject tickets that are all whitespace.

        WHY A VALIDATOR WHEN min_length=10 EXISTS?
            min_length counts characters including whitespace.
            "          " (10 spaces) passes min_length but is meaningless.
            This validator catches that edge case.
        """
        if not v.strip():
            raise ValueError("ticket text cannot be blank or whitespace only")
        return v.strip()    # Return stripped version as the actual value


# ─── RESPONSE ─────────────────────────────────────────────────────────────────

class TriageResponse(BaseModel):
    """
    Outgoing POST /triage response body.

    Combines fields from both TriageResult (M1) and FinetuneResult (M3).
    Backend-specific fields are Optional and null when not applicable.

    WHY ONE UNIFIED RESPONSE INSTEAD OF TWO SEPARATE ONES?

        The caller shouldn't need to change how they parse the response
        based on which backend they chose. A unified schema means:
          - Frontend code stays the same regardless of backend
          - M5 eval harness reads the same fields for both models
          - Adding a third backend later is a drop-in addition

        Fields not applicable to a backend are null (not missing).
        That's cleaner than a discriminated union for this use case.

    Fields (common to both backends)
    ---------------------------------
    category : str
        Predicted ticket category (hardware, software, network, etc.)
    priority : str
        Predicted ticket priority (P1, P2, P3, P4)
    next_action : str
        Recommended next action for the IT agent.
    backend : str
        Which backend was used ("baseline" or "finetuned").
    latency_ms : float
        Inference time in milliseconds (excludes model load time).
    success : bool
        True if prediction succeeded.
    error : Optional[str]
        Error message if success=False.

    Fields (baseline only)
    ----------------------
    reasoning : Optional[str]
        LLM chain-of-thought reasoning before classification.
    cost_usd : Optional[float]
        API cost for this call in USD (~$0.001 for Haiku).
    model : Optional[str]
        Anthropic model ID used (e.g. "claude-haiku-4-5-20251001").

    Fields (finetuned only)
    -----------------------
    cat_confidence : Optional[float]
        Softmax probability for predicted category (0.0–1.0).
    pri_confidence : Optional[float]
        Softmax probability for predicted priority (0.0–1.0).
    """

    # ── Common fields ──────────────────────────────────────────────────────────
    category:    str   = Field(description="Predicted ticket category")
    priority:    str   = Field(description="Predicted ticket priority (P1-P4)")
    next_action: str   = Field(description="Recommended next action for IT agent")
    backend:     str   = Field(description="Which backend was used")
    latency_ms:  float = Field(description="Inference time in milliseconds")
    success:     bool  = Field(description="True if prediction succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    # ── Baseline-only fields ───────────────────────────────────────────────────
    reasoning: Optional[str]   = Field(default=None, description="LLM reasoning (baseline only)")
    cost_usd:  Optional[float] = Field(default=None, description="API cost in USD (baseline only)")
    model:     Optional[str]   = Field(default=None, description="Model ID used (baseline only)")

    # ── Fine-tuned-only fields ─────────────────────────────────────────────────
    cat_confidence: Optional[float] = Field(
        default=None,
        description="Category prediction confidence 0-1 (finetuned only)"
    )
    pri_confidence: Optional[float] = Field(
        default=None,
        description="Priority prediction confidence 0-1 (finetuned only)"
    )


# ─── HEALTH ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """
    GET /health response.

    status : "ok" | "degraded"
        "ok" = all backends available.
        "degraded" = finetuned model loaded but LLM key missing or vice versa.
    backends_available : list[str]
        Which backends are ready to serve requests right now.
    model_loaded : bool
        Whether the fine-tuned model is loaded in memory.
    """
    status:             str        = "ok"
    backends_available: list[str]  = []
    model_loaded:       bool       = False
    version:            str        = "0.1.0"
