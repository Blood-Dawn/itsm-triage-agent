"""
models/baseline/prompt.py
──────────────────────────
Prompt templates for zero-shot ITSM ticket classification via the Claude API.

WHY THIS FILE EXISTS:
    Prompts are configuration, not code. Keeping them here means:
    - Prompt changes show up as clean diffs in git
    - The API logic in predict.py never needs to change when you
      tweak the prompt
    - Anyone can read and understand exactly what we're asking Claude

HOW PROMPTS WORK WITH THE ANTHROPIC API:
    The Claude API uses a "messages" format with two roles:
      - "system": Sets Claude's role, persona, and output rules.
                  This is sent once per session. Think of it as
                  the instruction manual you hand a new employee.
      - "user":   The specific request for each ticket. Changes
                  every call.

    The model reads the system prompt first, then the user message.
    Everything in the system prompt constrains how Claude responds
    to every user message.
"""

# ─── VALID VALUES (imported from schema) ─────────────────────────────────────
#
# WHY IMPORT FROM THE SCHEMA INSTEAD OF HARDCODING STRINGS?
#
# Because if we ever add a new category to ticket.py (say, "database"),
# the prompt will automatically include it on the next run. If we
# hardcoded "hardware, software, network..." in this file, we'd have
# two places to update — and we'd inevitably forget one.
#
# This is the DRY principle: Don't Repeat Yourself. The schema is the
# single source of truth for valid categories and priorities.

from data.schema.ticket import Category, Priority

# Build comma-separated lists of valid values for the prompt.
# The list comprehension iterates over all enum members and gets
# their .value (the string "hardware", "software", etc.)
VALID_CATEGORIES = ", ".join(f'"{c.value}"' for c in Category)
VALID_PRIORITIES = ", ".join(f'"{p.value}"' for p in Priority)


# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
#
# The system prompt does four things:
#
# 1. ROLE DEFINITION — "You are an IT helpdesk triage assistant."
#    This primes Claude to use IT domain knowledge it was trained on.
#    Claude has seen thousands of IT helpdesk tickets in its training
#    data, so this role framing activates that knowledge.
#
# 2. TASK DESCRIPTION — Exactly what we want Claude to do.
#    Clear, specific instructions outperform vague ones. "Classify
#    each ticket into exactly one category" is better than "figure
#    out what type of ticket this is."
#
# 3. VALID VALUES — The exact strings Claude must use.
#    Without this, Claude might return "Networking" instead of "network",
#    "Critical" instead of "P1", or invent new categories. Giving it
#    the exhaustive list constrains its output to values we can parse.
#
# 4. OUTPUT FORMAT — The exact JSON schema we expect.
#    This is critical. We tell Claude: produce ONLY this JSON, with
#    these exact field names, in this exact order. "Only valid JSON"
#    and "no explanation outside the JSON" prevents Claude from
#    wrapping the response in prose like "Here is my analysis: {...}".
#
# WHY f-STRING FOR THE SYSTEM PROMPT?
#
# Because VALID_CATEGORIES and VALID_PRIORITIES are computed from the
# schema at import time. The f-string inserts them into the prompt
# template so the prompt always reflects the current schema.

SYSTEM_PROMPT = f"""You are an expert IT helpdesk triage assistant. Your job is to classify incoming IT support tickets and recommend the next action for the support agent.

For each ticket, you will determine:
1. The ticket CATEGORY — the type of IT problem being reported
2. The ticket PRIORITY — how urgently it needs to be addressed
3. A NEXT ACTION — a short, specific recommendation for the support agent

VALID CATEGORIES (choose exactly one):
{VALID_CATEGORIES}

Category definitions:
- "hardware": Physical device issues (laptops, monitors, keyboards, docking stations, etc.)
- "software": Application or OS issues (crashes, errors, licensing, updates, etc.)
- "network": Connectivity issues (VPN, Wi-Fi, ethernet, DNS, internet access, etc.)
- "security": Security incidents or concerns (phishing, malware, unauthorized access, etc.)
- "access": Permissions and access control (shared drives, applications, account lockouts, etc.)
- "email": Email-related issues (Outlook, delivery failures, spam, mailbox, calendar, etc.)
- "printer": Printing and scanning issues (print queues, drivers, toner, physical jams, etc.)
- "other": Any issue that does not clearly fit the above categories

VALID PRIORITIES (choose exactly one):
{VALID_PRIORITIES}

Priority definitions:
- "P1": Critical — production is down, all users affected, immediate response required
- "P2": High — major functionality broken, significant business impact, workaround may exist
- "P3": Medium — standard issue, normal SLA applies, no immediate business impact
- "P4": Low — minor inconvenience, cosmetic issue, or informational request

OUTPUT FORMAT:
Respond with ONLY a valid JSON object. No explanation, no preamble, no markdown code fences.
The JSON must have exactly these four fields in this exact order:

{{
  "reasoning": "Brief 1-2 sentence analysis of why this ticket belongs to the chosen category and priority",
  "category": <one of the valid categories above>,
  "priority": <one of the valid priorities above>,
  "next_action": "A specific, actionable 1-2 sentence recommendation for the support agent"
}}

IMPORTANT RULES:
- "reasoning" must come FIRST — think before you classify
- "category" and "priority" must be exact strings from the valid lists above
- "next_action" should be addressed to the support agent, not the user
- Do not include any text outside the JSON object
- Do not wrap the JSON in markdown code fences (no ```json)"""


# ─── USER PROMPT TEMPLATE ─────────────────────────────────────────────────────
#
# WHY IS THE USER PROMPT A FUNCTION INSTEAD OF A STRING TEMPLATE?
#
# Because functions are testable. You can write:
#   assert "VPN dropping" in build_user_prompt("VPN dropping every 5 min")
#
# A module-level string template with .format() would work too, but
# a function makes the intent clearer: "given a ticket text, produce
# a user message." It also lets us add preprocessing logic later
# (like truncating very long tickets) without changing the call site.
#
# WHY SUCH A SIMPLE USER PROMPT?
#
# Because all the complexity is in the system prompt. The user message
# just provides the ticket text. This separation of concerns means:
# - Changing the output format → edit SYSTEM_PROMPT
# - Changing how we present the ticket → edit build_user_prompt()
# - The two never interfere with each other

def build_user_prompt(ticket_text: str) -> str:
    """
    Build the user-turn message for a single ticket classification request.

    Parameters
    ----------
    ticket_text : str
        The raw ticket text (title + description combined).
        This is the 'text' field from our Ticket schema.

    Returns
    -------
    str
        The formatted user message to send to the Claude API.

    Example
    -------
    >>> prompt = build_user_prompt("Laptop won't turn on after update.")
    >>> print(prompt)
    Please classify the following IT support ticket:

    ---
    Laptop won't turn on after update.
    ---
    """
    return f"""Please classify the following IT support ticket:

---
{ticket_text.strip()}
---"""


# ─── COST REFERENCE TABLE ─────────────────────────────────────────────────────
#
# WHY STORE COSTS HERE?
#
# The eval harness will compare "zero-shot LLM cost per inference" vs
# "fine-tuned model cost per inference" (which is $0 after training).
# This table lets predict.py calculate the dollar cost of each API call
# from the token counts that the API returns in its response.
#
# WHY CLAUDE-3-HAIKU AND NOT SONNET OR OPUS?
#
# For a classification task, we don't need the most capable model.
# Classification is a pattern-matching problem that even smaller models
# handle well. Haiku is:
#   - ~10x cheaper than Sonnet per token
#   - ~3x faster response time
#   - More than capable for 8-class ticket classification
#
# Using the cheapest model that gets the job done is good engineering.
# It's also a better eval comparison: "even our cheapest baseline is
# competitive, and our fine-tuned model beats it."
#
# Prices are in USD per 1 million tokens (as of early 2026).
# Source: https://anthropic.com/pricing

MODEL_COSTS: dict[str, dict[str, float]] = {
    # claude-haiku-4-5 is the current Haiku generation (as of early 2026).
    # The old claude-3-haiku-20240307 model has been retired by Anthropic.
    # Always use the full versioned model string — Anthropic doesn't support
    # aliases like "claude-haiku-latest" in the API.
    "claude-haiku-4-5-20251001": {
        "input_per_million":  0.80,   # $0.80 per 1M input tokens
        "output_per_million": 4.00,   # $4.00 per 1M output tokens
    },
    "claude-sonnet-4-5-20250929": {
        "input_per_million":  3.00,
        "output_per_million": 15.00,
    },
    "claude-opus-4-5-20251101": {
        "input_per_million":  15.00,
        "output_per_million": 75.00,
    },
}

# The model we'll use for the baseline.
# Haiku 4.5 is the best choice here: cheap, fast, and capable enough
# for an 8-class classification task.
DEFAULT_BASELINE_MODEL = "claude-haiku-4-5-20251001"


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the USD cost of a single API call.

    Parameters
    ----------
    model : str
        The model name used for the call.
    input_tokens : int
        Number of input tokens (from the API response's usage object).
    output_tokens : int
        Number of output tokens (from the API response's usage object).

    Returns
    -------
    float
        Cost in USD for this single API call.

    Example
    -------
    >>> calculate_cost("claude-haiku-4-5-20251001", input_tokens=500, output_tokens=100)
    0.000800  # ($0.80 * 500/1M) + ($4.00 * 100/1M)
    """
    # Fall back to Haiku pricing if the model isn't in our table.
    # This prevents KeyError if Anthropic releases a new model variant.
    costs = MODEL_COSTS.get(model, MODEL_COSTS[DEFAULT_BASELINE_MODEL])

    input_cost  = (input_tokens  / 1_000_000) * costs["input_per_million"]
    output_cost = (output_tokens / 1_000_000) * costs["output_per_million"]

    return round(input_cost + output_cost, 8)  # Round to 8 decimal places
