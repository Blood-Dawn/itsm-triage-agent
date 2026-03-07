"""
eval/run.py
───────────
M5 Evaluation harness — compare the finetuned and baseline backends
head-to-head on the held-out test set.

HOW TO RUN (from the itsm-triage-agent root):

    # Finetuned model on 200 random test tickets (fast, free, ~5s)
    python -m eval.run --backend finetuned --n 200

    # Baseline LLM on 100 tickets (~$0.11, ~3 min)
    python -m eval.run --backend baseline --n 100

    # Both backends on the same 100 tickets — true apples-to-apples comparison
    python -m eval.run --backend both --n 100

    # Full finetuned evaluation on the entire test set (1,016 tickets)
    python -m eval.run --backend finetuned --n 0

    # Change random seed (default 42 — same seed = same sample every time)
    python -m eval.run --backend both --n 100 --seed 7

Results are saved to eval/results/<timestamp>_<backend>.json automatically.

WHY A SEPARATE EVAL SCRIPT AND NOT JUST RUNNING THE API?

    The FastAPI server (M4) is for serving individual tickets interactively.
    The eval harness is for batch measurement — it needs to:
      1. Load hundreds of tickets at once
      2. Know the ground-truth labels (which the API never sees)
      3. Compute aggregate statistics across the full batch
      4. Save reproducible results to disk

    These are fundamentally different concerns from HTTP request handling.
    Separation of concerns: eval.run is for science, api.app is for users.

WHAT THIS TELLS US:

    The finetuned model was trained on synthetic data where priority is
    randomly assigned (no text signal distinguishing P1 from P3). So we
    expect category accuracy to be high (~85-95%) and priority accuracy
    to be near the P3 prior (~50%) — not because the model is broken,
    but because the training signal was weak for priority.

    The baseline LLM should perform better on priority because it can
    reason about urgency from ticket language (e.g., "production is down"
    → P1), even without explicit training.

    This difference is exactly what M5 is designed to surface. It's a
    feature, not a bug — it motivates adding urgency signals to M2 training
    data in a future milestone.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
#
# Same pattern as all other entry points: insert the project root into
# sys.path so that imports like "from eval.metrics import ..." work
# regardless of the current working directory.

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before any module that needs ANTHROPIC_API_KEY
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

from data.schema.ticket import Category, Priority
from eval.metrics import (
    ClassificationMetrics,
    EvalResult,
    LatencyStats,
    compute_classification_metrics,
    compute_latency_stats,
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

# Default adapter path (matches M4 api/app.py)
DEFAULT_ADAPTER_DIR = (
    PROJECT_ROOT / "models" / "finetuned" / "distilbert-lora" / "adapter"
)

# Where to save evaluation results
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# Label lists in a fixed order for sklearn metrics
CATEGORY_LABELS = [c.value for c in Category]  # 8 classes
PRIORITY_LABELS  = [p.value for p in Priority]  # 4 classes

console = Console()


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_test_tickets(n: int, seed: int) -> list[dict]:
    """
    Load and optionally subsample tickets from data/raw/test.jsonl.

    WHY RANDOM SUBSAMPLING INSTEAD OF FIRST-N?

    test.jsonl is sorted by category and priority (it was written out
    in that order during generation). Taking the first N tickets would
    give a biased sample — all hardware/P1 at the start. Random sampling
    with a fixed seed gives a representative, reproducible subset.

    A fixed seed means running the same command twice always evaluates
    the same tickets, making results comparable across runs.

    Parameters
    ----------
    n : int
        Number of tickets to sample. 0 means "use all".
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Parsed ticket dicts with 'text', 'category', 'priority' fields.
    """
    test_path = PROJECT_ROOT / "data" / "raw" / "test.jsonl"

    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}")
        logger.error("Run the generator first: python -m data.generator.gen --n 10000")
        sys.exit(1)

    tickets = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tickets.append(json.loads(line))

    logger.info(f"Loaded {len(tickets)} tickets from {test_path}")

    if n == 0 or n >= len(tickets):
        logger.info(f"Using full test set ({len(tickets)} tickets)")
        return tickets

    rng = random.Random(seed)
    sample = rng.sample(tickets, n)
    logger.info(f"Sampled {n}/{len(tickets)} tickets (seed={seed})")
    return sample


# ─── BACKEND RUNNERS ──────────────────────────────────────────────────────────
#
# WHY SEPARATE _run_finetuned() AND _run_baseline() FUNCTIONS?
#
# Each backend has its own result type (FinetuneResult vs TriageResult),
# its own required imports, and its own optional metadata. Isolating them
# makes it easy to add a third backend later (e.g., GPT-4o) without
# touching the finetuned code path.

def _run_finetuned(tickets: list[dict]) -> EvalResult:
    """
    Run the finetuned DistilBERT+LoRA model on a batch of tickets.

    Uses predict_batch() for efficient batched GPU inference (32 tickets
    per forward pass). Returns an EvalResult ready for printing and saving.

    WHY predict_batch() INSTEAD OF A LOOP OVER predict()?

    GPU matrix multiplications are highly parallel. Processing 32 tickets
    in one forward pass takes roughly the same time as 1 ticket. Using
    predict_batch() is ~32x more efficient and is the correct pattern for
    batch evaluation.
    """
    from models.finetune.predict import predict_batch

    adapter_dir = DEFAULT_ADAPTER_DIR
    if not adapter_dir.exists():
        logger.error(
            f"Adapter directory not found: {adapter_dir}\n"
            f"Train the model first: python -m models.finetune.train"
        )
        sys.exit(1)

    texts     = [t["text"]     for t in tickets]
    y_true_cat = [t["category"] for t in tickets]
    y_true_pri = [t["priority"] for t in tickets]

    console.print(f"\n[bold cyan]Running finetuned model on {len(texts)} tickets...[/]")

    results = predict_batch(texts, adapter_dir=adapter_dir, show_progress=True)

    # Split successes and failures
    # We only include successful predictions in metrics — failed predictions
    # would unfairly tank accuracy for API/infrastructure reasons unrelated
    # to model quality. We report the failure count separately.
    successes   = [(r, true_cat, true_pri)
                   for r, true_cat, true_pri in zip(results, y_true_cat, y_true_pri)
                   if r.success]
    n_failed    = len(results) - len(successes)

    if not successes:
        logger.error("All finetuned predictions failed — check model loading.")
        sys.exit(1)

    y_pred_cat = [r.category for r, _, _ in successes]
    y_pred_pri = [r.priority for r, _, _ in successes]
    true_cats  = [c for _, c, _ in successes]
    true_pris  = [p for _, _, p in successes]
    latencies  = [r.latency_ms for r, _, _ in successes]

    cat_metrics = compute_classification_metrics(true_cats, y_pred_cat, CATEGORY_LABELS)
    pri_metrics = compute_classification_metrics(true_pris, y_pred_pri, PRIORITY_LABELS)
    lat_stats   = compute_latency_stats(latencies)

    return EvalResult(
        backend="finetuned",
        n_total=len(results),
        n_success=len(successes),
        n_failed=n_failed,
        category=cat_metrics,
        priority=pri_metrics,
        latency=lat_stats,
        total_cost_usd=None,   # local model, no API cost
        avg_cost_usd=None,
    )


def _run_baseline(tickets: list[dict]) -> EvalResult:
    """
    Run the Claude Haiku zero-shot baseline on a batch of tickets.

    Uses predict_batch() which runs tickets SEQUENTIALLY with a progress
    bar. Sequential (not parallel) to avoid Anthropic rate limits.

    COST ESTIMATE BEFORE RUNNING:
        ~$0.00109/ticket × n tickets
        100 tickets ≈ $0.11
        200 tickets ≈ $0.22
        1016 tickets ≈ $1.11  (full test set)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error(
            "ANTHROPIC_API_KEY not set. "
            "Baseline backend requires a valid key in .env"
        )
        sys.exit(1)

    from models.baseline.predict import predict_batch

    texts      = [t["text"]     for t in tickets]
    y_true_cat = [t["category"] for t in tickets]
    y_true_pri = [t["priority"] for t in tickets]

    console.print(
        f"\n[bold yellow]Running baseline on {len(texts)} tickets "
        f"(est. cost ${len(texts) * 0.00109:.2f})...[/]"
    )

    results = predict_batch(texts, show_progress=True)

    successes = [(r, true_cat, true_pri)
                 for r, true_cat, true_pri in zip(results, y_true_cat, y_true_pri)
                 if r.success]
    n_failed  = len(results) - len(successes)

    if not successes:
        logger.error("All baseline predictions failed — check API key and connectivity.")
        sys.exit(1)

    y_pred_cat = [r.category for r, _, _ in successes]
    y_pred_pri = [r.priority for r, _, _ in successes]
    true_cats  = [c for _, c, _ in successes]
    true_pris  = [p for _, _, p in successes]
    latencies  = [r.latency_ms for r, _, _ in successes]
    costs      = [r.cost_usd   for r, _, _ in successes]

    cat_metrics = compute_classification_metrics(true_cats, y_pred_cat, CATEGORY_LABELS)
    pri_metrics = compute_classification_metrics(true_pris, y_pred_pri, PRIORITY_LABELS)
    lat_stats   = compute_latency_stats(latencies)

    total_cost = sum(costs)
    avg_cost   = total_cost / len(costs) if costs else 0.0

    return EvalResult(
        backend="baseline",
        n_total=len(results),
        n_success=len(successes),
        n_failed=n_failed,
        category=cat_metrics,
        priority=pri_metrics,
        latency=lat_stats,
        total_cost_usd=round(total_cost, 6),
        avg_cost_usd=round(avg_cost,   6),
    )


# ─── DISPLAY ──────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    """Format a 0-1 float as a percentage string."""
    return f"{v * 100:.1f}%"


def _color_pct(v: float) -> Text:
    """
    Color a percentage value based on its magnitude.

    >90%  → green (excellent)
    >70%  → yellow (acceptable)
    else  → red (needs work)

    WHY COLOR CODE?
    When scanning a comparison table, color draws the eye to problem
    areas instantly. Green/yellow/red is a universal convention.
    """
    s = _pct(v)
    if v >= 0.90:
        return Text(s, style="bold green")
    elif v >= 0.70:
        return Text(s, style="bold yellow")
    else:
        return Text(s, style="bold red")


def _print_summary_table(results: list[EvalResult]) -> None:
    """
    Print a side-by-side Rich comparison table to the console.

    WHY RICH TABLES INSTEAD OF PLAIN print()?

    Rich renders aligned, colored, bordered tables in the terminal.
    A plain print() of numbers is hard to scan. The visual alignment
    makes it immediately obvious when finetuned vs baseline diverge.
    This is the "executive summary" view — full per-class reports are
    printed separately below.
    """
    console.print()
    console.print(Panel(
        "[bold white]M5 Evaluation Results[/bold white]",
        style="bold blue",
        expand=False,
    ))

    # ── HEADLINE METRICS ──────────────────────────────────────────────────────
    tbl = Table(
        title="[bold]Accuracy and F1 Comparison[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on blue",
        min_width=70,
    )
    tbl.add_column("Metric",         style="bold", min_width=28)
    for r in results:
        tbl.add_column(r.backend.upper(), justify="center", min_width=18)

    def row(label, *values):
        tbl.add_row(label, *values)

    row("Category Accuracy",    *[_color_pct(r.category.accuracy)    for r in results])
    row("Category Macro F1",    *[_color_pct(r.category.f1_macro)    for r in results])
    row("Category Weighted F1", *[_color_pct(r.category.f1_weighted) for r in results])
    tbl.add_section()
    row("Priority Accuracy",    *[_color_pct(r.priority.accuracy)    for r in results])
    row("Priority Macro F1",    *[_color_pct(r.priority.f1_macro)    for r in results])
    row("Priority Weighted F1", *[_color_pct(r.priority.f1_weighted) for r in results])

    console.print(tbl)

    # ── LATENCY ───────────────────────────────────────────────────────────────
    lat_tbl = Table(
        title="[bold]Latency (ms)[/bold]",
        box=box.ROUNDED,
        header_style="bold white on dark_blue",
        min_width=70,
    )
    lat_tbl.add_column("Metric",  style="bold", min_width=28)
    for r in results:
        lat_tbl.add_column(r.backend.upper(), justify="center", min_width=18)

    lat_tbl.add_row("Mean",    *[f"{r.latency.mean_ms:.1f}" for r in results])
    lat_tbl.add_row("p50",     *[f"{r.latency.p50_ms:.1f}"  for r in results])
    lat_tbl.add_row("p95",     *[f"{r.latency.p95_ms:.1f}"  for r in results])
    lat_tbl.add_row("p99",     *[f"{r.latency.p99_ms:.1f}"  for r in results])

    console.print(lat_tbl)

    # ── COST (baseline only) ──────────────────────────────────────────────────
    baseline_results = [r for r in results if r.backend == "baseline"]
    if baseline_results:
        cost_tbl = Table(
            title="[bold]Cost (Baseline Only)[/bold]",
            box=box.ROUNDED,
            header_style="bold white on dark_green",
            min_width=50,
        )
        cost_tbl.add_column("Metric",        style="bold")
        cost_tbl.add_column("BASELINE",      justify="center")
        for r in baseline_results:
            cost_tbl.add_row("Total Cost",  f"${r.total_cost_usd:.4f}")
            cost_tbl.add_row("Avg/Ticket",  f"${r.avg_cost_usd:.6f}")
            cost_tbl.add_row("Tickets Run", str(r.n_success))
        console.print(cost_tbl)

    # ── COVERAGE ──────────────────────────────────────────────────────────────
    cov_tbl = Table(
        title="[bold]Coverage[/bold]",
        box=box.ROUNDED,
        header_style="bold white on dark_red",
        min_width=70,
    )
    cov_tbl.add_column("Metric",  style="bold", min_width=28)
    for r in results:
        cov_tbl.add_column(r.backend.upper(), justify="center", min_width=18)

    cov_tbl.add_row("Total Tickets",  *[str(r.n_total)   for r in results])
    cov_tbl.add_row("Succeeded",      *[str(r.n_success) for r in results])
    cov_tbl.add_row("Failed",         *[str(r.n_failed)  for r in results])

    console.print(cov_tbl)


def _print_per_class_report(result: EvalResult) -> None:
    """
    Print the full sklearn classification report for one backend.

    The classification report gives precision, recall, F1, and support
    (number of true instances) for every class. This is the detailed
    diagnostic view — useful for understanding WHERE the model fails.

    For example, if "security" has F1=0.10 you know the model is almost
    never correctly identifying security tickets, which might mean:
      - Not enough security tickets in training data
      - Security ticket language overlaps with "software" language
      - The category needs to be split more finely
    """
    console.print(f"\n[bold underline]{result.backend.upper()} — Category Report[/]")
    console.print(result.category.report)

    console.print(f"\n[bold underline]{result.backend.upper()} — Priority Report[/]")
    console.print(result.priority.report)


# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────

def _save_results(results: list[EvalResult], n: int, seed: int) -> Path:
    """
    Save evaluation results to a JSON file in eval/results/.

    WHY SAVE TO JSON?

    1. Reproducibility: You can re-read results without re-running
       inference (which costs money for the baseline).

    2. Diffing runs: Compare two JSON files to see if a model change
       improved or regressed metrics.

    3. M6 integration: If you later add CI/CD (GitHub Actions), you
       can assert that category accuracy stays above a threshold.

    The filename includes a timestamp and the backend name so multiple
    runs don't overwrite each other:
       eval/results/20260307_143022_finetuned_n200.json
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_str = "_".join(r.backend for r in results)
    filename = f"{timestamp}_{backend_str}_n{n}.json"
    out_path = RESULTS_DIR / filename

    payload = {
        "timestamp": datetime.now().isoformat(),
        "n_tickets": n,
        "seed": seed,
        "results": [r.to_dict() for r in results],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    WHY argparse INSTEAD OF CLICK OR TYPER?

    argparse is Python's built-in argument parser. No extra dependencies.
    For a script with three simple flags, argparse is the right tool.
    Click and Typer are better for large CLIs with many subcommands.
    """
    parser = argparse.ArgumentParser(
        description="M5 Evaluation harness — compare finetuned vs baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m eval.run --backend finetuned --n 200
  python -m eval.run --backend baseline --n 100
  python -m eval.run --backend both --n 100 --seed 7
  python -m eval.run --backend finetuned --n 0       # full test set
        """
    )

    parser.add_argument(
        "--backend",
        choices=["finetuned", "baseline", "both"],
        default="finetuned",
        help="Which backend(s) to evaluate (default: finetuned)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of test tickets to sample (0 = full test set, default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for ticket sampling (default: 42)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving results to eval/results/ (useful for quick checks)",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Print the full per-class classification report (verbose)",
    )

    return parser.parse_args()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the full evaluation run.

    Flow:
      1. Parse CLI args
      2. Load and sample test tickets
      3. Run the chosen backend(s)
      4. Print summary table
      5. Optionally print full per-class reports
      6. Save results to JSON
    """
    args = _parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold white]ITSM Triage Agent — M5 Evaluation[/bold white]\n"
        f"Backend: [cyan]{args.backend}[/]  |  "
        f"Tickets: [cyan]{'all' if args.n == 0 else args.n}[/]  |  "
        f"Seed: [cyan]{args.seed}[/]",
        style="blue",
    ))

    # ── Load data ─────────────────────────────────────────────────────────────
    tickets = load_test_tickets(n=args.n, seed=args.seed)
    actual_n = len(tickets)

    # ── Run backends ──────────────────────────────────────────────────────────
    results: list[EvalResult] = []

    if args.backend in ("finetuned", "both"):
        ft_result = _run_finetuned(tickets)
        results.append(ft_result)

    if args.backend in ("baseline", "both"):
        bl_result = _run_baseline(tickets)
        results.append(bl_result)

    # ── Print results ─────────────────────────────────────────────────────────
    _print_summary_table(results)

    if args.full_report:
        for r in results:
            _print_per_class_report(r)

    # ── Save ──────────────────────────────────────────────────────────────────
    if not args.no_save:
        save_path = _save_results(results, n=actual_n, seed=args.seed)
        console.print(f"\n[dim]Results saved to {save_path}[/dim]")

    console.print()


if __name__ == "__main__":
    main()
