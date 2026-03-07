"""
scripts/test_finetune.py
────────────────────────
Smoke test for the M3 fine-tuned model inference pipeline.

Loads the saved LoRA adapter from models/finetuned/distilbert-lora/adapter/
and runs predictions on 5 tickets from test.jsonl.

Prints category prediction, confidence score, and latency per ticket.
Compares predicted label vs ground truth so you can eyeball accuracy.

HOW TO RUN (from the itsm-triage-agent root):
    python scripts/test_finetune.py

    # Test more tickets
    python scripts/test_finetune.py --n 10

WHAT A PASSING TEST LOOKS LIKE:
    - Model loads in ~1-2s (DistilBERT + LoRA weights from disk)
    - Per-ticket latency: ~5-15ms on GPU (vs ~2000ms for M1 LLM baseline)
    - Category accuracy: 70-90%+ on these 5 tickets (after only 3 epochs)
    - No crashes, no CUDA errors

WHY THIS MATTERS FOR THE PORTFOLIO:
    This test shows you can go from raw text → prediction in ~10ms locally
    with NO API cost. That's the core selling point of fine-tuned models
    vs LLM APIs: speed, cost, and offline capability.
"""

import argparse
import json
import random
import sys
from pathlib import Path

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.finetune.predict import FinetuneResult, predict

# Default adapter path — matches where train.py saves the weights
DEFAULT_ADAPTER = PROJECT_ROOT / "models" / "finetuned" / "distilbert-lora" / "adapter"


def load_tickets(jsonl_path: Path, n: int, seed: int = 42) -> list[dict]:
    # Load ALL tickets then randomly sample n.
    #
    # WHY NOT just take the first n?
    # test.jsonl is written in stratified order — tickets are grouped by
    # (category, priority). The first N entries are always the same
    # category/priority combination, which gives a misleading accuracy
    # reading (e.g. 100% category, 0% priority because you only saw P1s).
    # Random sampling gives a representative cross-section of all 8
    # categories and 4 priorities every time you run the test.
    all_tickets = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_tickets.append(json.loads(line))

    rng = random.Random(seed)
    return rng.sample(all_tickets, min(n, len(all_tickets)))


def print_sep(char="─", width=70):
    print(char * width)


def print_result(ticket: dict, result: FinetuneResult, idx: int, total: int):
    print_sep()
    print(f"  Ticket {idx}/{total}  |  Actual: {ticket['category']} / {ticket['priority']}")
    print_sep()

    preview = ticket["text"][:100].replace("\n", " ")
    print(f"  Text:     {preview}...")
    print()

    if result.success:
        match_cat = "✓" if result.category == ticket["category"] else "✗"
        match_pri = "✓" if result.priority == ticket["priority"] else "✗"

        print(f"  Prediction:")
        print(f"    Category:    {result.category}  {match_cat}  "
              f"(actual: {ticket['category']})  confidence: {result.cat_confidence:.1%}")
        print(f"    Priority:    {result.priority}  {match_pri}  "
              f"(actual: {ticket['priority']})  confidence: {result.pri_confidence:.1%}")
        print(f"    Next Action: {result.next_action}")
        print()
        print(f"  Stats:   latency={result.latency_ms:.1f}ms  (no API cost — local model)")
    else:
        print(f"  ERROR: {result.error}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test M3 fine-tuned model on held-out test tickets."
    )
    parser.add_argument("--n",           type=int,  default=5,
                        help="Number of tickets to test (default: 5)")
    parser.add_argument("--adapter-dir", type=str,  default=str(DEFAULT_ADAPTER),
                        help=f"Path to adapter directory (default: {DEFAULT_ADAPTER})")
    parser.add_argument("--data-dir",    type=str,  default="data/raw",
                        help="Directory containing test.jsonl (default: data/raw)")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    data_path   = PROJECT_ROOT / args.data_dir / "test.jsonl"

    # ── VERIFY INPUTS ──────────────────────────────────────────────────────────
    if not adapter_dir.exists():
        print(f"\n  ERROR: Adapter not found at {adapter_dir}")
        print("  Run training first:")
        print("    python -m models.finetune.train --epochs 3 --batch-size 16")
        sys.exit(1)

    if not data_path.exists():
        print(f"\n  ERROR: test.jsonl not found at {data_path}")
        print("  Run the generator first:")
        print("    python -m data.generator.gen --n 2000 --seed 42 --output data/raw")
        sys.exit(1)

    tickets = load_tickets(data_path, args.n)

    # ── HEADER ────────────────────────────────────────────────────────────────
    print()
    print_sep("═")
    print(f"  M3 Fine-Tuned Model Smoke Test")
    print(f"  Adapter: {adapter_dir.parent.name}/{adapter_dir.name}")
    print(f"  Tickets: {len(tickets)} (from {data_path.name})")
    print_sep("═")
    print()
    print("  Loading model (this takes ~1-2s the first time)...")
    print()

    # ── RUN PREDICTIONS ───────────────────────────────────────────────────────
    results = []
    cat_correct = 0
    pri_correct = 0
    total_latency = 0.0

    for i, ticket in enumerate(tickets, start=1):
        result = predict(ticket["text"], adapter_dir=adapter_dir)
        results.append(result)
        print_result(ticket, result, i, len(tickets))

        if result.success:
            if result.category == ticket["category"]:
                cat_correct += 1
            if result.priority == ticket["priority"]:
                pri_correct += 1
            total_latency += result.latency_ms

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    n = len(tickets)
    succeeded = sum(r.success for r in results)

    print_sep("═")
    print(f"  SUMMARY")
    print_sep("═")
    print(f"  API calls:        0  (local model — no cost)")
    print(f"  Succeeded:        {succeeded}/{n}")
    if succeeded > 0:
        avg_lat = total_latency / succeeded
        print(f"  Category acc:     {cat_correct}/{succeeded} = {cat_correct/succeeded:.0%}")
        print(f"  Priority acc:     {pri_correct}/{succeeded} = {pri_correct/succeeded:.0%}")
        print(f"  Avg latency:      {avg_lat:.1f}ms/ticket")
        print()
        print("  Context: M1 LLM baseline is ~$0.001/ticket, ~2000ms latency.")
        print(f"  Fine-tuned model: $0.00/ticket, ~{avg_lat:.0f}ms latency.")
        print(f"  Speed advantage:  ~{2000/avg_lat:.0f}x faster")
    print_sep("═")
    print()

    sys.exit(0 if succeeded == n else 1)


if __name__ == "__main__":
    main()
