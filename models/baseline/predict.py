"""
models/baseline/predict.py
───────────────────────────
Zero-shot ITSM ticket classification using the Anthropic Claude API.

This module provides the baseline predictor — it calls Claude with a
structured prompt and parses the response into a validated TriageResult.

WHY THIS IS THE BASELINE:
    "Zero-shot" means no training. We give Claude the prompt and the
    ticket text, and it classifies using only its pre-trained knowledge.
    This is our performance floor — if the fine-tuned model can't beat
    this, something is wrong with the fine-tuning.

HOW IT FITS IN THE SYSTEM:
    data/raw/ → [this file] → TriageResult
    The eval harness (M5) will call predict() on every test ticket
    and compare the output to the ground-truth labels in the JSONL file.
"""

# ─── STANDARD LIBRARY ────────────────────────────────────────────────────────
#
# dataclasses.dataclass:
#   A decorator that auto-generates __init__, __repr__, and __eq__ for
#   a class based on its annotated fields. We use it for TriageResult
#   because it's lighter weight than Pydantic (no external validation
#   needed for internal results) but still gives us structured, typed data.
#
# json:
#   For parsing Claude's JSON response string into a Python dict.
#   json.loads(string) → dict. This is the inverse of json.dumps().
#
# time.time():
#   For measuring latency. We record time before and after the API call,
#   subtract to get elapsed milliseconds.
#
# os:
#   For reading environment variables. os.getenv("KEY") reads from the
#   system environment, which is populated from our .env file.
#
# typing.Optional:
#   For fields that might be None (like when a partial result is returned
#   on error).

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

# ─── THIRD-PARTY ─────────────────────────────────────────────────────────────
#
# dotenv.load_dotenv():
#   Reads our .env file and injects its contents into the process
#   environment (os.environ). This is why we can then call
#   os.getenv("ANTHROPIC_API_KEY") — load_dotenv() puts it there first.
#
#   WHY CALL load_dotenv() IN EVERY FILE THAT NEEDS ENV VARS?
#
#   Because load_dotenv() is idempotent (safe to call multiple times —
#   it only loads the file once) and because any file might be the
#   ENTRY POINT. If you only load it in main.py but then someone
#   imports predict.py directly in a test or notebook, it won't work.
#   Defensive practice: load it wherever you need it.
#
# anthropic.Anthropic:
#   The official Python SDK client for the Anthropic Claude API.
#   We use the SDK instead of raw HTTP requests (via httpx or requests)
#   because:
#   - It handles authentication automatically (reads ANTHROPIC_API_KEY)
#   - It types the response objects so your IDE knows what fields exist
#   - It handles streaming, retries, and connection pooling for us
#   - It stays up to date with API changes
#
# anthropic.APIStatusError, anthropic.RateLimitError, etc.:
#   Specific exception types from the SDK. Catching these instead of
#   bare Exception means we can respond differently to different errors:
#   - RateLimitError → wait and retry
#   - AuthenticationError → fail immediately (bad API key)
#   - APIError → log and retry

from dotenv import load_dotenv
import anthropic

# ─── PROJECT IMPORTS ─────────────────────────────────────────────────────────
#
# Everything we need from our own modules:
#   - The prompt builders from prompt.py
#   - The schema types for type annotations and validation
#   - Loguru for structured logging

from models.baseline.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    calculate_cost,
    DEFAULT_BASELINE_MODEL,
)
from data.schema.ticket import Category, Priority

# ─── LOGGING SETUP ───────────────────────────────────────────────────────────
#
# WHY LOGURU INSTEAD OF PYTHON'S BUILT-IN logging MODULE?
#
# Python's built-in logging module requires ~10 lines of boilerplate
# to set up a basic logger. Loguru does it in one import. It also
# gives you:
#   - Colored output by default
#   - Structured logging (key=value pairs)
#   - Better exception tracebacks
#   - Zero configuration needed
#
# In production you'd configure log levels and output formats centrally.
# For a student project, loguru's defaults are production-quality out of the box.
#
# logger.debug() → only visible when LOG_LEVEL=DEBUG (not shown in normal runs)
# logger.info()  → normal operational messages
# logger.warning() → something unexpected but non-fatal
# logger.error() → something failed but the program continues

from loguru import logger

# Load .env file immediately when this module is imported
load_dotenv()


# ─── RESULT TYPE ─────────────────────────────────────────────────────────────
#
# WHY A DATACLASS FOR THE RESULT?
#
# We need to return multiple values from predict(): category, priority,
# next_action, AND metadata like latency, cost, tokens. We have options:
#
#   Option A: Return a tuple → (category, priority, next_action, latency, ...)
#     Problem: Tuples are positional. predict()[3] is unreadable.
#
#   Option B: Return a plain dict → {"category": "hardware", ...}
#     Problem: No type hints, no IDE autocomplete, typos silently fail.
#
#   Option C: Return a Pydantic model
#     Problem: Overkill — we don't need external validation for internal data.
#
#   Option D: Return a @dataclass → result.category, result.latency_ms
#     This is what we choose. Named fields, typed, zero dependencies,
#     and Python generates __init__ automatically from the field annotations.
#
# @dataclass tells Python: "generate an __init__ that accepts all these
# fields as arguments, in order." So TriageResult(category="hardware", ...)
# works without us writing any __init__ code.

@dataclass
class TriageResult:
    """
    The structured output from a single ticket classification call.

    Contains both the classification result AND the metadata needed
    for the eval harness to compare models.

    Fields
    ------
    category : str
        Predicted category ("hardware", "software", etc.)
    priority : str
        Predicted priority ("P1", "P2", "P3", "P4")
    next_action : str
        Recommended next step for the support agent
    reasoning : str
        Claude's chain-of-thought reasoning (discarded in eval,
        but useful for debugging bad predictions)
    model : str
        Which Claude model was used (for cost calculation)
    input_tokens : int
        Number of input tokens consumed (for cost tracking)
    output_tokens : int
        Number of output tokens produced (for cost tracking)
    cost_usd : float
        Estimated USD cost of this single API call
    latency_ms : int
        Wall-clock time for the API call in milliseconds
    success : bool
        True if classification succeeded, False if an error occurred
    error : Optional[str]
        Error message if success=False, None otherwise
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


# ─── RESPONSE PARSER ─────────────────────────────────────────────────────────

def _parse_response(response_text: str, model: str, usage, latency_ms: int) -> TriageResult:
    """
    Parse Claude's text response into a validated TriageResult.

    WHY A SEPARATE FUNCTION?

    Parsing is the most error-prone part of this module — Claude might
    return invalid JSON, use wrong field names, or give values not in
    our schema. Isolating this logic makes it easy to test independently:
    just pass in a string and see what comes out.

    WHY NOT USE THE ANTHROPIC SDK's STRUCTURED OUTPUT?

    The Anthropic API supports "tool use" (function calling) which
    guarantees structured output. That's the production approach.
    We're using plain JSON parsing here because:
    1. It's simpler to understand for learning
    2. It demonstrates prompt engineering skill
    3. We can upgrade to tool use in a later milestone

    Parameters
    ----------
    response_text : str
        The raw text from Claude's response message.
    model : str
        Model name for cost calculation.
    usage : anthropic.types.Usage
        Token usage object from the API response.
    latency_ms : int
        Pre-computed latency in milliseconds.

    Returns
    -------
    TriageResult
        Populated result, or an error result if parsing fails.
    """
    input_tokens  = usage.input_tokens
    output_tokens = usage.output_tokens
    cost          = calculate_cost(model, input_tokens, output_tokens)

    # ── STEP 1: Parse JSON ────────────────────────────────────────────────
    #
    # json.loads() converts a JSON string → Python dict.
    # If the string is not valid JSON, it raises json.JSONDecodeError.
    # We catch that and return a failed result instead of crashing.
    #
    # WHY .strip()?
    # Claude occasionally adds a leading/trailing newline to its response.
    # .strip() removes all leading/trailing whitespace (spaces, newlines,
    # tabs). Without it, json.loads() might fail on "\n{...}".

    # ── STEP 1a: Strip markdown code fences ──────────────────────────────
    #
    # Our system prompt says "no markdown code fences", but newer Claude
    # models (Haiku 4.5+) sometimes wrap the JSON in ```json ... ``` anyway.
    # This is a known behavior change introduced as models became more
    # "helpful" — they started adding formatting that developers then have
    # to strip out.
    #
    # The fix: before we attempt json.loads(), strip any code fence wrapper.
    # We do this by checking if the text starts with a backtick fence and,
    # if so, removing the first and last lines.
    #
    # WHY NOT UPDATE THE PROMPT INSTEAD?
    # We could try stronger wording in the system prompt (e.g., repeat the
    # "no fences" instruction three times). But defensive parsing is more
    # reliable — a regex strip is guaranteed to work regardless of how the
    # model phrases its fence. Belt + suspenders.
    #
    # This handles both:
    #   ```json         (with language tag)
    #   {               ...
    #   }
    #   ```
    # and:
    #   ```             (without language tag)
    #   {               ...
    #   }
    #   ```

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        # Remove the opening fence line (e.g. "```json" or "```")
        lines = cleaned.splitlines()
        # Drop first line (the opening fence)
        lines = lines[1:]
        # Drop last line if it's a closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        logger.debug(f"Raw response: {response_text}")
        return TriageResult(
            category="other", priority="P3",
            next_action="", reasoning="",
            model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_usd=cost,
            latency_ms=latency_ms, success=False,
            error=f"JSON parse error: {e}"
        )

    # ── STEP 2: Extract required fields ──────────────────────────────────
    #
    # .get() returns None if the key doesn't exist (instead of KeyError).
    # We use .get() with a default of "" so that missing fields produce
    # a failed result rather than an exception.

    category    = parsed.get("category", "")
    priority    = parsed.get("priority", "")
    next_action = parsed.get("next_action", "")
    reasoning   = parsed.get("reasoning", "")

    # ── STEP 3: Validate against schema ──────────────────────────────────
    #
    # WHY VALIDATE AFTER PARSING?
    #
    # Even if Claude returns valid JSON, it might hallucinate a category
    # we didn't define. For example: "connectivity" instead of "network".
    # We check that the returned values are in our valid enum sets.
    #
    # We build sets of valid values for O(1) lookup:
    #   valid_categories = {"hardware", "software", ..., "other"}
    #   "hardware" in valid_categories → True  (fast dict lookup)
    #
    # If a value is invalid, we log a warning and return a failed result.
    # We don't try to "guess" what Claude meant — that would corrupt our
    # eval metrics.

    valid_categories = {c.value for c in Category}
    valid_priorities = {p.value for p in Priority}

    if category not in valid_categories:
        logger.warning(f"Claude returned invalid category: '{category}'")
        return TriageResult(
            category="other", priority="P3",
            next_action=next_action, reasoning=reasoning,
            model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_usd=cost,
            latency_ms=latency_ms, success=False,
            error=f"Invalid category: '{category}'"
        )

    if priority not in valid_priorities:
        logger.warning(f"Claude returned invalid priority: '{priority}'")
        return TriageResult(
            category=category, priority="P3",
            next_action=next_action, reasoning=reasoning,
            model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_usd=cost,
            latency_ms=latency_ms, success=False,
            error=f"Invalid priority: '{priority}'"
        )

    # ── STEP 4: Return success ────────────────────────────────────────────

    logger.debug(
        f"Classified: category={category}, priority={priority}, "
        f"tokens={input_tokens}+{output_tokens}, cost=${cost:.6f}"
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
        error=None,
    )


# ─── MAIN PREDICTOR ──────────────────────────────────────────────────────────

def predict(
    ticket_text: str,
    model: str = DEFAULT_BASELINE_MODEL,
    max_retries: int = 3,
) -> TriageResult:
    """
    Classify a single ITSM ticket using zero-shot Claude API inference.

    This is the main function you call from the eval harness and the API.
    It handles the full pipeline:
      ticket_text → API call → JSON parse → validation → TriageResult

    WHY max_retries=3?

    The API occasionally returns errors (rate limits, transient failures).
    3 retries with exponential backoff covers ~99% of transient issues
    without waiting too long. More than 3 retries suggests a persistent
    problem that needs manual intervention.

    EXPONENTIAL BACKOFF EXPLAINED:

    On retry 1: wait 2^1 = 2 seconds
    On retry 2: wait 2^2 = 4 seconds
    On retry 3: wait 2^3 = 8 seconds

    Why exponential instead of fixed? Because if the API is rate-limited,
    hammering it with immediate retries makes the problem worse. Waiting
    progressively longer gives the API time to recover.

    Parameters
    ----------
    ticket_text : str
        The raw ticket text to classify.
    model : str
        Which Claude model to use (default: claude-3-haiku, cheapest).
    max_retries : int
        Number of retry attempts on transient failures.

    Returns
    -------
    TriageResult
        The classification result with metadata.
        Check result.success to know if it worked.
    """
    # ── VALIDATE API KEY ──────────────────────────────────────────────────
    #
    # We check for the API key BEFORE making the API call. If it's missing,
    # we fail immediately with a clear error instead of getting a cryptic
    # "401 Unauthorized" from the API.
    #
    # This is called "fail fast" — surface errors as early as possible
    # with the most informative message possible.

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error(
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
        )
        return TriageResult(
            category="other", priority="P3",
            next_action="", reasoning="",
            model=model, input_tokens=0, output_tokens=0,
            cost_usd=0.0, latency_ms=0,
            success=False, error="ANTHROPIC_API_KEY not set"
        )

    # ── BUILD CLIENT ──────────────────────────────────────────────────────
    #
    # anthropic.Anthropic() creates the API client. It reads the API key
    # automatically from the ANTHROPIC_API_KEY environment variable.
    #
    # WHY CREATE THE CLIENT INSIDE predict() INSTEAD OF AT MODULE LEVEL?
    #
    # Two reasons:
    # 1. Module-level clients are created at import time. If the API key
    #    isn't set yet when the module is imported (common in tests), the
    #    client creation fails at import — before any test can set the key.
    # 2. Lazy initialization means the client is only created when you
    #    actually need it. If you import predict.py but never call predict(),
    #    no client is created and no resources are wasted.
    #
    # Tradeoff: slightly slower first call because client setup happens then.
    # For a batch eval job, this is negligible.

    client = anthropic.Anthropic(api_key=api_key)

    # ── RETRY LOOP ────────────────────────────────────────────────────────
    #
    # We attempt the API call up to max_retries + 1 times (the first
    # attempt plus max_retries retries). The enumerate() gives us both
    # the attempt number (for the wait calculation) and the loop index.

    last_error: Optional[str] = None

    for attempt in range(max_retries + 1):

        # Retry wait (exponential backoff, except on the first attempt)
        if attempt > 0:
            wait_seconds = 2 ** attempt  # 2, 4, 8 seconds
            logger.warning(f"Retry {attempt}/{max_retries} after {wait_seconds}s...")
            time.sleep(wait_seconds)

        try:
            # ── MEASURE LATENCY ───────────────────────────────────────────
            #
            # time.time() returns seconds since epoch as a float.
            # We record it before and after the API call and subtract.
            # Multiply by 1000 to convert to milliseconds.
            #
            # WHY MILLISECONDS?
            #   Because "145ms" is more readable than "0.145 seconds"
            #   in eval reports. LLM inference latency is typically
            #   100ms - 2000ms, so milliseconds is the natural unit.

            start = time.time()

            # ── THE API CALL ──────────────────────────────────────────────
            #
            # client.messages.create() is the main Anthropic API call.
            #
            # model:       Which Claude model to use.
            # max_tokens:  Maximum tokens in Claude's response. We set 512
            #              because our JSON output is ~100-200 tokens.
            #              Setting it too low truncates the response.
            #              Setting it too high wastes money (you pay for
            #              output tokens even if they're not used).
            #
            # system:      The system prompt (Claude's role and output format).
            #              Passed once. Applied to all messages in the call.
            #
            # messages:    A list of turn objects. Each turn has a "role"
            #              ("user" or "assistant") and "content" (the text).
            #              We send exactly one user message per call.
            #
            # temperature: Controls randomness. 0.0 = deterministic (always
            #              picks the highest-probability token). For
            #              classification, we want determinism — the "right"
            #              category shouldn't change between runs.
            #              Range: 0.0 (deterministic) to 1.0 (creative).

            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": build_user_prompt(ticket_text)
                    }
                ],
                temperature=0.0,  # Deterministic classification
            )

            latency_ms = int((time.time() - start) * 1000)

            # ── EXTRACT TEXT FROM RESPONSE ────────────────────────────────
            #
            # The Anthropic API returns a Message object. The actual text
            # is in response.content[0].text.
            #
            # response.content is a list because Claude can return multiple
            # content blocks (text + images in multi-modal responses).
            # For text-only responses, it's always [TextBlock(text="...")].
            # We take [0] for the first (and only) block.
            #
            # response.usage contains input_tokens and output_tokens —
            # the token counts we need for cost calculation.

            raw_text = response.content[0].text
            logger.debug(f"Raw API response (attempt {attempt+1}): {raw_text[:200]}...")

            # Parse and validate the response
            result = _parse_response(
                response_text=raw_text,
                model=model,
                usage=response.usage,
                latency_ms=latency_ms,
            )

            # If parsing failed due to bad JSON or invalid values,
            # retry (Claude might give a better response on the next attempt)
            if not result.success and attempt < max_retries:
                last_error = result.error
                logger.warning(f"Parse failed on attempt {attempt+1}: {result.error}. Retrying...")
                continue

            return result

        except anthropic.RateLimitError as e:
            # Rate limit: the API is telling us to slow down.
            # We ALWAYS retry these — they're transient by definition.
            last_error = f"Rate limit: {e}"
            logger.warning(f"Rate limit on attempt {attempt+1}. Will retry.")

        except anthropic.AuthenticationError as e:
            # Bad API key. Retrying won't fix this.
            # Return immediately with a clear error.
            logger.error(f"Authentication failed. Check your ANTHROPIC_API_KEY: {e}")
            return TriageResult(
                category="other", priority="P3",
                next_action="", reasoning="",
                model=model, input_tokens=0, output_tokens=0,
                cost_usd=0.0, latency_ms=0,
                success=False, error=f"Authentication error: {e}"
            )

        except anthropic.APIError as e:
            # Generic API error (server error, timeout, etc.)
            # Retry these — they're usually transient.
            last_error = f"API error: {e}"
            logger.warning(f"API error on attempt {attempt+1}: {e}")

        except Exception as e:
            # Catch-all for unexpected errors (network issues, etc.)
            last_error = f"Unexpected error: {e}"
            logger.error(f"Unexpected error on attempt {attempt+1}: {e}")

    # ── ALL RETRIES EXHAUSTED ─────────────────────────────────────────────
    logger.error(f"All {max_retries} retries failed. Last error: {last_error}")
    return TriageResult(
        category="other", priority="P3",
        next_action="", reasoning="",
        model=model, input_tokens=0, output_tokens=0,
        cost_usd=0.0, latency_ms=0,
        success=False, error=last_error
    )


# ─── BATCH PREDICTOR ─────────────────────────────────────────────────────────

def predict_batch(
    ticket_texts: list[str],
    model: str = DEFAULT_BASELINE_MODEL,
    max_retries: int = 3,
    show_progress: bool = True,
) -> list[TriageResult]:
    """
    Classify a list of tickets, with progress tracking.

    WHY A SEPARATE BATCH FUNCTION?

    The eval harness needs to classify hundreds of tickets. We could call
    predict() in a loop in the eval harness, but:
    1. Having a batch function centralises progress display
    2. It's easier to add rate limiting here later (sleep between calls)
    3. It makes the eval harness simpler — one function call, not a loop

    NOTE ON PARALLELISM:
    We process tickets SEQUENTIALLY (one at a time). We could use
    asyncio or concurrent.futures to parallelize, but the Anthropic API
    has rate limits. Parallel requests would hit those limits immediately
    and cause more retries than sequential calls. For a student project,
    sequential is the right choice.

    Parameters
    ----------
    ticket_texts : list[str]
        List of raw ticket texts to classify.
    model : str
        Claude model to use.
    max_retries : int
        Retries per ticket.
    show_progress : bool
        Whether to print a progress indicator.

    Returns
    -------
    list[TriageResult]
        One TriageResult per ticket, in the same order as the input.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    results = []

    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Classifying {len(ticket_texts)} tickets via {model}...",
                total=len(ticket_texts)
            )
            for text in ticket_texts:
                result = predict(text, model=model, max_retries=max_retries)
                results.append(result)
                progress.advance(task)
    else:
        for text in ticket_texts:
            result = predict(text, model=model, max_retries=max_retries)
            results.append(result)

    # Print summary
    n_success = sum(1 for r in results if r.success)
    n_failed  = len(results) - n_success
    total_cost = sum(r.cost_usd for r in results)
    avg_latency = sum(r.latency_ms for r in results) // len(results) if results else 0

    logger.info(
        f"Batch complete: {n_success}/{len(results)} successful, "
        f"{n_failed} failed, total_cost=${total_cost:.4f}, "
        f"avg_latency={avg_latency}ms"
    )

    return results
