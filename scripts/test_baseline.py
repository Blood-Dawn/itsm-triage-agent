"""
scripts/test_baseline.py
────────────────────────
Quick smoke-test for the M1 LLM baseline.

WHY THIS FILE EXISTS:
    The inline `python -c` approach we tried in PowerShell breaks badly when
    the code contains nested quotes like `t["text"]`. PowerShell's quote
    handling eats the inner quotes before Python ever sees them, causing:

        SyntaxError: '[' was never closed

    The fix is simple: put the test code in a real .py file. Python sees the
    source exactly as written, no shell escaping required. This is the
    professional way to do it anyway — scripts belong in files, not on
    command lines.

WHAT THIS SCRIPT DOES:
    1. Loads the first 3 tickets from data/raw/test.jsonl
    2. Runs each through the M1 baseline (Claude Haiku via Anthropic API)
    3. Prints a formatted summary: prediction, reasoning, cost, latency

HOW TO RUN (from the itsm-triage-agent root directory):
    python scripts/test_baseline.py

    Optional: test a different number of tickets
    python scripts/test_baseline.py --n 5

WHAT A PASSING TEST LOOKS LIKE:
    - 3/3 tickets succeed (no API errors)
    - Category and priority are valid enum values (not "null" or made-up strings)
    - Reasoning is a non-empty string
    - Cost per ticket is a small float (roughly $0.0001–$0.0005 for Haiku)
    - Latency is typically 500–2000ms per ticket

WHY WE TEST ON test.jsonl (not train.jsonl):
    The test split is the "held-out" set — data the model has never seen
    in any sense. Using it for our smoke-test means we're validating in
    the same conditions as the real eval harness will use later.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
#
# WHY THIS BLOCK?
#
# When you run `python scripts/test_baseline.py` from the project root,
# Python sets sys.path[0] to the scripts/ directory. That means imports
# like `from models.baseline.predict import predict` would fail because
# Python would look for a `models` folder inside `scripts/`, not in the
# project root.
#
# The fix: insert the project root (one level up from scripts/) at the
# front of sys.path so Python finds our packages correctly.
#
# This is the standard pattern for runnable scripts inside a package.
# An alternative is to use `python -m scripts.test_baseline` from the
# root, but that requires an __init__.py in scripts/ and is more verbose.

PROJECT_ROOT = Path(__file__).parent.parent  # scripts/ -> project root
sys.path.insert(0, str(PROJECT_ROOT))

# ─── IMPORTS (after path fix) ─────────────────────────────────────────────────
from dotenv import load_dotenv

from models.baseline.predict import predict
from models.baseline.prompt import DEFAULT_BASELINE_MODEL

# Load .env from the PROJECT_ROOT explicitly.
#
# WHY EXPLICIT PATH INSTEAD OF JUST load_dotenv()?
#
# load_dotenv() with no arguments searches upward from the *current working
# directory* (cwd). That's fine when you cd into the project root first,
# but it silently does nothing if you run the script from a different
# directory (like your home folder or the scripts/ subfolder).
#
# Using an explicit path — PROJECT_ROOT / ".env" — makes the load
# deterministic: it always finds the .env that lives next to the project
# root, no matter where you ran the script from. This follows the
# fail-fast principle: better to know immediately that the path is wrong
# than to get a cryptic "key not set" error later.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
#
# WHY override=True?
#
# load_dotenv() normally skips variables that are ALREADY set in the
# environment — even if they're set to an empty string "". In some
# execution environments (Docker containers, CI runners, the VM that
# powers this tool), the shell may pre-set API key variables to empty
# strings as placeholders. Without override=True, our real keys from
# .env would be silently ignored.
#
# override=True says: "the .env file is the authoritative source —
# overwrite whatever the shell had." That's the correct behavior for
# local development where .env is intentionally your config.


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_tickets(jsonl_path: Path, n: int, seed: int = 42) -> list[dict]:
    """
    Load a random sample of `n` tickets from a JSONL file.

    WHY RANDOM SAMPLE (not first-N)?
        test.jsonl is written in stratified order — entries are grouped by
        (category, priority). Taking the first N always gives the same
        categories and priorites, making the smoke test misleading.
        Random sampling gives a representative cross-section each run.

    WHY NOT LOAD THE WHOLE FILE FOR THE SMOKE TEST?
        test.jsonl now has ~1000 tickets. Running all of them would:
        - Cost ~$1.09 (the full M5 eval budget, wasted on a quick check)
        - Take ~30 minutes
        - Be overkill for a smoke test

        We just need 3–5 tickets to verify the API integration works.
        The full eval harness (M5) will run all of them later.

    Parameters
    ----------
    jsonl_path : Path
        Path to the .jsonl file.
    n : int
        Maximum number of tickets to load.

    Returns
    -------
    list[dict]
        List of ticket dictionaries.
    """
    import random
    all_tickets = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_tickets.append(json.loads(line))
    rng = random.Random(seed)
    return rng.sample(all_tickets, min(n, len(all_tickets)))


def print_separator(char: str = "─", width: int = 70) -> None:
    """Print a visual separator line."""
    print(char * width)


def print_result(ticket: dict, result, index: int, total: int) -> None:
    """
    Print a formatted summary of one prediction result.

    WHY A SEPARATE FUNCTION?
        Keeps the main loop clean. If we ever want to change the output
        format (e.g., add color, write to a file), we only change this
        function, not the loop.
    """
    print_separator()
    print(f"  Ticket {index}/{total}  |  Actual: {ticket['category']} / {ticket['priority']}")
    print_separator()

    # Show first 100 chars of the ticket text so we know what it's about
    preview = ticket["text"][:100].replace("\n", " ")
    print(f"  Text:       {preview}...")
    print()

    if result.success:
        # ✓ happy path
        match_cat = "✓" if result.category == ticket["category"] else "✗"
        match_pri = "✓" if result.priority == ticket["priority"] else "✗"

        print("  Prediction:")
        print(f"    Category:   {result.category}  {match_cat}  (actual: {ticket['category']})")
        print(f"    Priority:   {result.priority}  {match_pri}  (actual: {ticket['priority']})")
        print(f"    Reasoning:  {result.reasoning}")
        print(f"    Next Action:{result.next_action}")
        print()
        print("  Stats:")
        print(f"    Model:      {result.model}")
        print(f"    Tokens:     {result.input_tokens} in / {result.output_tokens} out")
        print(f"    Cost:       ${result.cost_usd:.6f}")
        print(f"    Latency:    {result.latency_ms:.0f} ms")
    else:
        # ✗ something went wrong — show the error
        print(f"  ERROR: {result.error}")

    print()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # ── CLI ──────────────────────────────────────────────────────────────────
    #
    # WHY ARGPARSE EVEN FOR A TEST SCRIPT?
    #
    # Makes it easy to test more tickets without editing the file:
    #   python scripts/test_baseline.py --n 10
    #
    # It also demonstrates good habits: even quick scripts benefit from
    # a CLI interface because you never know when you'll want to reuse them.

    parser = argparse.ArgumentParser(
        description="Smoke-test the M1 LLM baseline against held-out test tickets."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of tickets to test (default: 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_BASELINE_MODEL,
        help=f"Anthropic model to use (default: {DEFAULT_BASELINE_MODEL})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing test.jsonl (default: data/raw)",
    )
    args = parser.parse_args()

    # ── VERIFY API KEY ────────────────────────────────────────────────────────
    #
    # Check early — fail fast rather than waiting through all the ticket
    # loading, only to fail on the first API call.
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        print("\n  ERROR: ANTHROPIC_API_KEY not set or still a placeholder.")
        print("  Make sure your .env file has a real key:")
        print("    ANTHROPIC_API_KEY=sk-ant-api03-...")
        print()
        sys.exit(1)

    # ── LOAD TICKETS ─────────────────────────────────────────────────────────
    data_path = PROJECT_ROOT / args.data_dir / "test.jsonl"

    if not data_path.exists():
        print(f"\n  ERROR: test.jsonl not found at {data_path}")
        print("  Run the generator first:")
        print("    python -m data.generator.gen --n 2000 --seed 42 --output data/raw")
        print()
        sys.exit(1)

    tickets = load_tickets(data_path, args.n)
    if not tickets:
        print(f"\n  ERROR: No tickets found in {data_path}")
        sys.exit(1)

    # ── HEADER ───────────────────────────────────────────────────────────────
    print()
    print_separator("═")
    print("  M1 Baseline Smoke Test")
    print(f"  Model:   {args.model}")
    print(f"  Tickets: {len(tickets)} (from {data_path.name})")
    print_separator("═")
    print()

    # ── RUN PREDICTIONS ──────────────────────────────────────────────────────
    #
    # We call predict() one ticket at a time (not predict_batch) so we can
    # print results as they come in. predict_batch uses a Rich progress bar
    # which is better for bulk runs but hides individual outputs.

    results = []
    total_cost = 0.0
    successes = 0

    for i, ticket in enumerate(tickets, start=1):
        print(f"  Calling API for ticket {i}/{len(tickets)}...")
        result = predict(ticket["text"], model=args.model)
        results.append(result)
        print_result(ticket, result, i, len(tickets))

        if result.success:
            successes += 1
            total_cost += result.cost_usd

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print_separator("═")
    print("  SUMMARY")
    print_separator("═")
    print(f"  Passed:     {successes}/{len(tickets)}")
    print(f"  Total cost: ${total_cost:.6f}")

    if successes == len(tickets):
        print()
        print("  ✓ All predictions succeeded. M1 baseline is working!")
    else:
        failed = len(tickets) - successes
        print()
        print(f"  ✗ {failed} prediction(s) failed. Check the errors above.")

    print_separator("═")
    print()

    # Exit with non-zero code if any predictions failed.
    # This matters if you ever run this script in a CI pipeline — a failed
    # prediction should fail the build.
    sys.exit(0 if successes == len(tickets) else 1)


if __name__ == "__main__":
    main()
