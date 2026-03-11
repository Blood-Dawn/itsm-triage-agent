"""
models/baseline/openai_predict.py
──────────────────────────────────
OpenAI GPT baseline for ITSM ticket classification.

This mirrors models/baseline/predict.py (the Anthropic version) but uses
the OpenAI SDK instead. The prompt and parsing logic are shared — only
the API client and response extraction differ.

WHY DUPLICATE FILES INSTEAD OF ONE UNIFIED PREDICTOR?

    Option A: One file with a big if/else for provider
        Pro: DRY. Con: Both SDKs are imported even when only one is used.
        Becomes messy as each provider has different retry patterns,
        error types, token field names, and pricing.

    Option B: Separate files per provider (this approach)
        Pro: Each file is self-contained and easy to understand.
             Adding a new provider is a new file, not a modified one.
             The API server imports whichever it needs.
        Con: Slight duplication of the retry/backoff logic.

    For a portfolio project, Option B is the cleaner choice.

COST REFERENCE (GPT-4o-mini, March 2025)
    Input:  $0.15 / 1M tokens  →  $0.00015 / 1K tokens
    Output: $0.60 / 1M tokens  →  $0.00060 / 1K tokens
    Typical ticket call:  ~400 input + ~100 output ≈ $0.000120
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from models.baseline.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from data.schema.ticket import Category, Priority

load_dotenv()

# ─── PRICING ──────────────────────────────────────────────────────────────────

# GPT-4o-mini pricing (per token, USD)
_PRICE_INPUT_PER_TOKEN  = 0.15  / 1_000_000   # $0.15  per 1M input tokens
_PRICE_OUTPUT_PER_TOKEN = 0.60  / 1_000_000   # $0.60  per 1M output tokens

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _calculate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * _PRICE_INPUT_PER_TOKEN
            + output_tokens * _PRICE_OUTPUT_PER_TOKEN)


# ─── RESULT TYPE ──────────────────────────────────────────────────────────────

@dataclass
class TriageResult:
    """
    Structured output from a single OpenAI classification call.
    Field names match the Anthropic TriageResult so api/app.py can
    treat both providers identically.
    """
    category:      str
    priority:      str
    next_action:   str
    reasoning:     str
    model:         str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    latency_ms:    int
    success:       bool = True
    error:         Optional[str] = None


# ─── RESPONSE PARSER ──────────────────────────────────────────────────────────

def _parse_response(
    response_text: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
) -> TriageResult:
    """Parse the OpenAI response text into a TriageResult."""
    cost = _calculate_cost(input_tokens, output_tokens)

    # Strip markdown code fences if the model wraps the JSON
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OpenAI response as JSON: {e}")
        return TriageResult(
            category="other", priority="P3",
            next_action="", reasoning="",
            model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_usd=cost,
            latency_ms=latency_ms, success=False,
            error=f"JSON parse error: {e}"
        )

    category    = parsed.get("category", "")
    priority    = parsed.get("priority", "")
    next_action = parsed.get("next_action", "")
    reasoning   = parsed.get("reasoning", "")

    valid_categories = {c.value for c in Category}
    valid_priorities = {p.value for p in Priority}

    if category not in valid_categories:
        logger.warning(f"OpenAI returned invalid category: '{category}'")
        return TriageResult(
            category="other", priority="P3",
            next_action=next_action, reasoning=reasoning,
            model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_usd=cost,
            latency_ms=latency_ms, success=False,
            error=f"Invalid category: '{category}'"
        )

    if priority not in valid_priorities:
        logger.warning(f"OpenAI returned invalid priority: '{priority}'")
        return TriageResult(
            category=category, priority="P3",
            next_action=next_action, reasoning=reasoning,
            model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_usd=cost,
            latency_ms=latency_ms, success=False,
            error=f"Invalid priority: '{priority}'"
        )

    return TriageResult(
        category=category,
        priority=priority,
        next_action=next_action,
        reasoning=reasoning,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
        success=True,
    )


# ─── MAIN PREDICTOR ───────────────────────────────────────────────────────────

def predict(
    ticket_text: str,
    model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = 3,
    api_key: Optional[str] = None,
) -> TriageResult:
    """
    Classify a single ITSM ticket using zero-shot OpenAI inference.

    Parameters
    ----------
    ticket_text : str
        Raw ticket text to classify.
    model : str
        OpenAI model to use (default: gpt-4o-mini, cheapest + fastest).
    max_retries : int
        Number of retry attempts on transient failures.
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env var if not provided.

    Returns
    -------
    TriageResult
        Classification result with metadata. Check result.success.
    """
    # Key resolution: caller-supplied > environment variable
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        logger.error("No OpenAI API key found. Pass api_key= or set OPENAI_API_KEY.")
        return TriageResult(
            category="other", priority="P3",
            next_action="", reasoning="",
            model=model, input_tokens=0, output_tokens=0,
            cost_usd=0.0, latency_ms=0,
            success=False, error="OPENAI_API_KEY not set"
        )

    # Import openai here (lazy) so the server starts even if openai isn't
    # installed — the baseline route just won't work for OpenAI tickets.
    try:
        from openai import OpenAI, RateLimitError, AuthenticationError, APIError
    except ImportError:
        return TriageResult(
            category="other", priority="P3",
            next_action="", reasoning="",
            model=model, input_tokens=0, output_tokens=0,
            cost_usd=0.0, latency_ms=0,
            success=False,
            error="openai package not installed. Run: pip install openai"
        )

    client = OpenAI(api_key=resolved_key, timeout=60.0)

    last_error: Optional[str] = None

    for attempt in range(max_retries + 1):

        if attempt > 0:
            wait_seconds = 2 ** attempt
            logger.warning(f"OpenAI retry {attempt}/{max_retries} after {wait_seconds}s...")
            time.sleep(wait_seconds)

        try:
            start = time.time()

            response = client.chat.completions.create(
                model=model,
                max_tokens=512,
                temperature=0.0,          # Deterministic classification
                # response_format forces the model to return valid JSON only
                # (available on gpt-4o-mini and later models)
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(ticket_text)},
                ],
            )

            latency_ms = int((time.time() - start) * 1000)

            # Extract text and token counts from OpenAI response object
            # (field names differ slightly from Anthropic's SDK)
            raw_text      = response.choices[0].message.content or ""
            input_tokens  = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            logger.debug(f"OpenAI raw response (attempt {attempt+1}): {raw_text[:200]}...")

            result = _parse_response(
                response_text=raw_text,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )

            if not result.success and attempt < max_retries:
                last_error = result.error
                logger.warning(f"Parse failed on attempt {attempt+1}: {result.error}. Retrying...")
                continue

            return result

        except RateLimitError as e:
            last_error = f"Rate limit: {e}"
            logger.warning(f"OpenAI rate limit on attempt {attempt+1}. Will retry.")

        except AuthenticationError as e:
            logger.error(f"OpenAI authentication failed — check your API key: {e}")
            return TriageResult(
                category="other", priority="P3",
                next_action="", reasoning="",
                model=model, input_tokens=0, output_tokens=0,
                cost_usd=0.0, latency_ms=0,
                success=False, error=f"OpenAI authentication error: {e}"
            )

        except APIError as e:
            last_error = f"OpenAI API error: {e}"
            logger.warning(f"OpenAI API error on attempt {attempt+1}: {e}")

        except Exception as e:
            last_error = f"Unexpected error: {e}"
            logger.error(f"Unexpected error on attempt {attempt+1}: {e}")

    logger.error(f"All {max_retries} retries failed. Last error: {last_error}")
    return TriageResult(
        category="other", priority="P3",
        next_action="", reasoning="",
        model=model, input_tokens=0, output_tokens=0,
        cost_usd=0.0, latency_ms=0,
        success=False, error=last_error
    )
