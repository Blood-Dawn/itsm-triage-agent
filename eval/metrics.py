"""
eval/metrics.py
───────────────
Pure metric computation for the ITSM Triage Agent evaluation harness.

This module takes lists of ground-truth labels and predicted labels and
produces structured metrics objects. It has NO knowledge of how the
predictions were produced — it doesn't import any model code. That
separation makes it easy to test in isolation and easy to reuse if you
swap out a backend later.

METRICS WE COMPUTE AND WHY EACH ONE MATTERS:

Accuracy:
    The simplest metric. Fraction of tickets where the predicted label
    matches the ground truth exactly. Good for a quick headline number
    but misleading when classes are imbalanced (e.g., if 50% of tickets
    are "hardware", a model that always predicts "hardware" gets 50%
    accuracy while doing nothing useful).

Macro F1:
    F1 score computed per class, then averaged with equal weight for
    every class. This penalizes a model that ignores rare classes.
    If your model never predicts "security" (1% of tickets), macro F1
    will be noticeably lower than accuracy.

Weighted F1:
    F1 averaged by class frequency. Closer to accuracy for imbalanced
    datasets. Useful as a sanity check alongside macro F1.

Per-class F1:
    F1 broken down for each individual category/priority. This is the
    most actionable metric — if category F1 is high but security F1 is
    0.10, you know exactly where the model fails.

    F1 = 2 * (precision * recall) / (precision + recall)
    Precision = "of all tickets I labeled X, how many were really X?"
    Recall    = "of all tickets that are really X, how many did I find?"

Latency (mean, p50, p95, p99):
    Mean latency is easily distorted by outliers. Percentile latency
    gives a clearer picture:
      p50 (median): half of requests complete within this time
      p95: 95% of requests complete within this time — the "typical worst case"
      p99: 99% complete within this time — catches true outliers

    In production SLAs, you almost always see p95 or p99 commitments,
    not mean. Using percentiles here means I can speak the same language
    as production engineers in an interview.

Cost (baseline only):
    Total and per-ticket API spend. Important for the baseline backend
    where every inference costs money. Fine-tuned model cost is always $0.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ─── RESULT DATACLASSES ───────────────────────────────────────────────────────
#
# WHY DATACLASSES (not dicts) FOR METRICS?
#
# We could return a plain dict like {"accuracy": 0.92, "f1_macro": 0.88}.
# But dicts have no type hints, no IDE autocomplete, and typos fail silently
# (result["accurcy"] returns None instead of raising AttributeError).
#
# A @dataclass gives us:
#   - Named, typed fields
#   - IDE autocomplete
#   - Free __repr__ for printing/debugging
#   - asdict() to convert to a JSON-serializable dict for file output
#
# We use two nested classes: ClassificationMetrics (for one head) and
# EvalResult (wrapping both heads plus latency and cost).

@dataclass
class ClassificationMetrics:
    """
    Metrics for one classification head (category or priority).

    Fields
    ------
    accuracy : float
        Fraction of correct predictions (0.0 to 1.0).
    f1_macro : float
        Macro-averaged F1 across all classes.
    f1_weighted : float
        Frequency-weighted F1 across all classes.
    per_class_f1 : dict[str, float]
        F1 score for each individual label.
    n_total : int
        Total number of predictions evaluated.
    n_correct : int
        Number of correct predictions.
    report : str
        Full sklearn classification_report string (precision/recall/F1 per class).
    """
    accuracy:     float
    f1_macro:     float
    f1_weighted:  float
    per_class_f1: dict[str, float]
    n_total:      int
    n_correct:    int
    report:       str   = field(repr=False)  # long string, suppress from __repr__


@dataclass
class LatencyStats:
    """
    Latency distribution for a batch of inferences.

    All times in milliseconds. We track percentiles, not just mean,
    because the mean is easily skewed by slow outliers (e.g., one
    long API timeout can double the mean while p50 stays the same).
    """
    mean_ms: float
    p50_ms:  float   # median
    p95_ms:  float   # typical worst case
    p99_ms:  float   # extreme tail


@dataclass
class EvalResult:
    """
    Complete evaluation result for one backend on one test sample.

    Wraps category metrics, priority metrics, latency stats, and (for
    the baseline) cost information. This is the object that gets saved
    to JSON and printed as the summary table.

    Fields
    ------
    backend : str
        Which backend was evaluated ("finetuned" or "baseline").
    n_total : int
        Total tickets evaluated.
    n_success : int
        Tickets where the backend returned success=True.
    n_failed : int
        Tickets where the backend returned success=False (API error, etc.)
    category : ClassificationMetrics
        Metrics for the category prediction head.
    priority : ClassificationMetrics
        Metrics for the priority prediction head.
    latency : LatencyStats
        Latency distribution across all successful predictions.
    total_cost_usd : Optional[float]
        Total API cost (baseline only, None for finetuned).
    avg_cost_usd : Optional[float]
        Average cost per ticket (baseline only, None for finetuned).
    """
    backend:        str
    n_total:        int
    n_success:      int
    n_failed:       int
    category:       ClassificationMetrics
    priority:       ClassificationMetrics
    latency:        LatencyStats
    total_cost_usd: Optional[float] = None
    avg_cost_usd:   Optional[float] = None

    def to_dict(self) -> dict:
        """
        Convert this result to a JSON-serializable dict.

        WHY NOT USE asdict() DIRECTLY?
        asdict() is recursive and handles nested dataclasses, but we
        want to round floats to 4 decimal places for cleaner JSON.
        We manually asdict() and round after.
        """
        d = asdict(self)
        # Round floats for cleaner output
        def _round(obj):
            if isinstance(obj, dict):
                return {k: _round(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_round(v) for v in obj]
            if isinstance(obj, float):
                return round(obj, 4)
            return obj
        return _round(d)


# ─── CORE COMPUTATION ─────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    label_names: list[str],
) -> ClassificationMetrics:
    """
    Compute accuracy, F1, and per-class F1 for one classification head.

    WHY THE labels PARAMETER IN sklearn FUNCTIONS?

    sklearn's f1_score and classification_report take a `labels` argument
    that sets the full list of possible classes. Without it, sklearn only
    reports on classes that actually appear in y_true or y_pred during
    THIS evaluation run. If "security" happens to not appear in a small
    100-ticket sample, sklearn would silently omit it from the report and
    give an inflated macro F1 (because it's averaging over fewer classes).

    Passing our full label list ensures the report always covers all 8
    categories and all 4 priorities, even if some are absent from the sample.

    Parameters
    ----------
    y_true : list[str]
        Ground-truth labels from test.jsonl.
    y_pred : list[str]
        Predicted labels from a backend.
    label_names : list[str]
        The complete ordered list of possible labels.

    Returns
    -------
    ClassificationMetrics
        Populated metrics object.
    """
    n_total   = len(y_true)
    n_correct = sum(t == p for t, p in zip(y_true, y_pred))

    accuracy = accuracy_score(y_true, y_pred)

    # Macro: treat each class equally regardless of how many tickets it has.
    # This penalises the model for ignoring rare classes.
    f1_macro = f1_score(
        y_true, y_pred,
        average="macro",
        labels=label_names,
        zero_division=0,   # if a class has no predictions, score it 0 not NaN
    )

    # Weighted: weight each class by how many tickets it has.
    # Roughly equivalent to accuracy but penalises more on high-frequency errors.
    f1_weighted = f1_score(
        y_true, y_pred,
        average="weighted",
        labels=label_names,
        zero_division=0,
    )

    # Per-class F1: the most actionable breakdown
    per_class_f1 = {}
    for label in label_names:
        per_class_f1[label] = round(float(f1_score(
            y_true, y_pred,
            labels=[label],
            average="micro",   # micro for single-class = precision = recall = F1
            zero_division=0,
        )), 4)

    # Full text report (precision / recall / F1 / support per class)
    # This is what you'd print to the console for a full diagnostic view.
    report = classification_report(
        y_true, y_pred,
        labels=label_names,
        zero_division=0,
        digits=3,
    )

    return ClassificationMetrics(
        accuracy=round(float(accuracy), 4),
        f1_macro=round(float(f1_macro), 4),
        f1_weighted=round(float(f1_weighted), 4),
        per_class_f1=per_class_f1,
        n_total=n_total,
        n_correct=n_correct,
        report=report,
    )


def compute_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    """
    Compute latency distribution from a list of per-ticket latencies.

    WHY numpy FOR PERCENTILES?

    np.percentile() handles edge cases (empty list, duplicate values)
    gracefully and is vectorised (faster than a pure Python sort + index
    for large batches). For small lists (100-1000 tickets) the difference
    is negligible, but it's good practice.

    Parameters
    ----------
    latencies_ms : list[float]
        Per-ticket inference times in milliseconds.

    Returns
    -------
    LatencyStats
        Mean, p50, p95, p99.
    """
    if not latencies_ms:
        return LatencyStats(mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0)

    arr = np.array(latencies_ms, dtype=float)
    return LatencyStats(
        mean_ms=round(float(np.mean(arr)),              1),
        p50_ms= round(float(np.percentile(arr, 50)),    1),
        p95_ms= round(float(np.percentile(arr, 95)),    1),
        p99_ms= round(float(np.percentile(arr, 99)),    1),
    )
