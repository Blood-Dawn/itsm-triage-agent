"""
tests/test_metrics.py
─────────────────────
Unit tests for eval/metrics.py — the pure-Python metric computation module.

WHY TEST METRICS SPECIFICALLY?

    eval/metrics.py is the only module in the project that:
      1. Contains non-trivial algorithmic logic (accuracy, F1, percentiles)
      2. Has no external dependencies on torch, transformers, or API keys
      3. Can be tested in CI in under 5 seconds

    The model code (M1, M3) requires torch and GPU hardware. The API
    (M4) requires a running server. Testing those in CI would require a
    GPU runner or mocking so deeply that the tests would be meaningless.
    Testing metrics is the right CI boundary.

WHY USE pytest AND NOT unittest?

    pytest is the de-facto standard for Python testing in 2026. Compared
    to unittest:
      - Tests are plain functions, not classes with setUp/tearDown
      - assert statements work naturally (no assertEqual, assertAlmostEqual)
      - Fixtures are more composable than setUp
      - Parametrize makes it easy to run the same test with multiple inputs

HOW TO RUN:

    # From the project root
    pytest tests/ -v

    # With coverage
    pytest tests/ --cov=eval --cov-report=term-missing -v
"""

import pytest
from eval.metrics import (
    ClassificationMetrics,
    EvalResult,
    LatencyStats,
    compute_classification_metrics,
    compute_latency_stats,
)

# ─── FIXTURES ─────────────────────────────────────────────────────────────────
#
# WHY FIXTURES?
#
# Fixtures are reusable setup helpers. Instead of repeating the same
# label list and test data in every test, we define them once here and
# inject them via pytest's dependency injection (function arguments with
# the same name as a @pytest.fixture).

@pytest.fixture
def category_labels():
    """The 8 ticket categories in the fixed order used by the eval harness."""
    return ["hardware", "software", "network", "security",
            "access", "email", "printer", "other"]


@pytest.fixture
def priority_labels():
    """The 4 ticket priorities in order."""
    return ["P1", "P2", "P3", "P4"]


@pytest.fixture
def perfect_cat_predictions(category_labels):
    """20 predictions that exactly match the ground truth — should give 100% accuracy."""
    import random
    rng = random.Random(42)
    labels = rng.choices(category_labels, k=20)
    return labels, labels  # (y_true, y_pred) — identical


@pytest.fixture
def sample_latencies():
    """A realistic latency distribution with one outlier."""
    return [8.1, 9.3, 7.5, 10.2, 8.8, 9.1, 7.9, 8.5, 9.0, 350.0]


# ─── compute_classification_metrics TESTS ─────────────────────────────────────

class TestComputeClassificationMetrics:
    """Tests for the compute_classification_metrics() function."""

    def test_perfect_predictions_give_100_accuracy(self, perfect_cat_predictions, category_labels):
        """
        When y_true == y_pred exactly, accuracy and all F1 scores must be 1.0.

        This is the simplest sanity check. If this fails, the entire
        metrics module is broken.
        """
        y_true, y_pred = perfect_cat_predictions
        m = compute_classification_metrics(y_true, y_pred, category_labels)

        assert m.accuracy == 1.0,     f"Expected 1.0 accuracy, got {m.accuracy}"
        assert m.f1_macro == 1.0,     f"Expected 1.0 macro F1, got {m.f1_macro}"
        assert m.f1_weighted == 1.0,  f"Expected 1.0 weighted F1, got {m.f1_weighted}"
        assert m.n_correct == m.n_total

    def test_all_wrong_predictions_give_zero_accuracy(self, category_labels):
        """
        When every prediction is wrong, accuracy must be 0.

        We use a fixed wrong label to ensure all predictions are incorrect.
        """
        y_true = ["hardware"] * 10
        y_pred = ["software"] * 10   # all wrong

        m = compute_classification_metrics(y_true, y_pred, category_labels)

        assert m.accuracy == 0.0,   f"Expected 0.0 accuracy, got {m.accuracy}"
        assert m.n_correct == 0

    def test_partial_accuracy_is_correct(self, category_labels):
        """
        3 out of 5 correct predictions should give 60% accuracy.

        We check exact equality because accuracy_score is deterministic
        for this simple case.
        """
        y_true = ["hardware", "software", "network", "security", "access"]
        y_pred = ["hardware", "software", "network", "printer",  "email"]
        #           correct    correct     correct     wrong        wrong

        m = compute_classification_metrics(y_true, y_pred, category_labels)

        assert m.accuracy == pytest.approx(0.6, abs=1e-4)
        assert m.n_correct == 3
        assert m.n_total == 5

    def test_all_labels_present_in_per_class_f1(self, category_labels):
        """
        per_class_f1 must contain an entry for every label, even if that
        label never appears in the predictions.

        This verifies we pass the full label list to sklearn, which is the
        guard against silently omitting rare classes from the report.
        """
        # Predictions that never produce "security" or "printer"
        y_true = ["hardware", "software", "network", "access", "email"]
        y_pred = ["hardware", "software", "hardware", "access", "email"]

        m = compute_classification_metrics(y_true, y_pred, category_labels)

        for label in category_labels:
            assert label in m.per_class_f1, (
                f"Expected '{label}' in per_class_f1, but it was missing. "
                f"Keys present: {list(m.per_class_f1.keys())}"
            )

    def test_per_class_f1_values_in_range(self, category_labels):
        """
        Every per-class F1 value must be between 0.0 and 1.0 inclusive.
        """
        y_true = ["hardware", "software", "network", "security", "access"]
        y_pred = ["hardware", "software", "printer", "security", "email"]

        m = compute_classification_metrics(y_true, y_pred, category_labels)

        for label, score in m.per_class_f1.items():
            assert 0.0 <= score <= 1.0, (
                f"Per-class F1 for '{label}' is {score}, expected 0.0-1.0"
            )

    def test_report_string_is_nonempty(self, category_labels):
        """
        The classification report must be a non-empty string.
        It's saved to JSON and printed to the console — a blank report
        would be a silent failure.
        """
        y_true = ["hardware", "software", "network"]
        y_pred = ["hardware", "software", "hardware"]

        m = compute_classification_metrics(y_true, y_pred, category_labels)

        assert isinstance(m.report, str)
        assert len(m.report) > 50, "Report string is suspiciously short"

    def test_result_is_ClassificationMetrics_instance(self, category_labels):
        """The return type must be ClassificationMetrics, not a dict or tuple."""
        m = compute_classification_metrics(["hardware"], ["hardware"], category_labels)
        assert isinstance(m, ClassificationMetrics)

    @pytest.mark.parametrize("n_tickets", [1, 5, 100])
    def test_n_total_matches_input_length(self, n_tickets, category_labels):
        """n_total must always equal len(y_true), regardless of batch size."""
        import random
        rng = random.Random(n_tickets)
        y = rng.choices(category_labels, k=n_tickets)
        m = compute_classification_metrics(y, y, category_labels)
        assert m.n_total == n_tickets


# ─── compute_latency_stats TESTS ──────────────────────────────────────────────

class TestComputeLatencyStats:
    """Tests for the compute_latency_stats() function."""

    def test_percentile_ordering(self, sample_latencies):
        """
        p50 <= p95 <= p99 must always hold.

        This is a mathematical property of percentiles: higher percentiles
        are always >= lower ones for the same dataset.
        """
        s = compute_latency_stats(sample_latencies)
        assert s.p50_ms <= s.p95_ms, f"p50 ({s.p50_ms}) > p95 ({s.p95_ms})"
        assert s.p95_ms <= s.p99_ms, f"p95 ({s.p95_ms}) > p99 ({s.p99_ms})"

    def test_outlier_inflates_p99_not_p50(self, sample_latencies):
        """
        The 350ms outlier in sample_latencies should push p99 high
        but leave p50 (median) near the main cluster (~8-10ms).

        This is the whole point of percentile latency: p99 captures
        tail behavior while p50 reflects the typical experience.
        """
        s = compute_latency_stats(sample_latencies)
        assert s.p50_ms < 15.0,   f"p50 should be near median (~9ms), got {s.p50_ms}"
        assert s.p99_ms > 100.0,  f"p99 should reflect the 350ms outlier, got {s.p99_ms}"

    def test_uniform_latencies_give_equal_percentiles(self):
        """
        If all latencies are the same value, mean, p50, p95, and p99
        must all equal that value.
        """
        uniform = [42.0] * 50
        s = compute_latency_stats(uniform)

        assert s.mean_ms == pytest.approx(42.0, abs=0.1)
        assert s.p50_ms  == pytest.approx(42.0, abs=0.1)
        assert s.p95_ms  == pytest.approx(42.0, abs=0.1)
        assert s.p99_ms  == pytest.approx(42.0, abs=0.1)

    def test_empty_list_returns_zeros(self):
        """
        An empty latency list should return zeros, not raise an exception.

        In practice this would happen if all predictions failed. We want a
        graceful result rather than a crash in the eval harness.
        """
        s = compute_latency_stats([])
        assert s.mean_ms == 0.0
        assert s.p50_ms  == 0.0
        assert s.p95_ms  == 0.0
        assert s.p99_ms  == 0.0

    def test_single_value(self):
        """A single latency value should produce equal stats at all percentiles."""
        s = compute_latency_stats([77.5])
        assert s.mean_ms == pytest.approx(77.5, abs=0.1)
        assert s.p50_ms  == pytest.approx(77.5, abs=0.1)

    def test_returns_LatencyStats_instance(self, sample_latencies):
        """Return type must be LatencyStats."""
        s = compute_latency_stats(sample_latencies)
        assert isinstance(s, LatencyStats)


# ─── EvalResult TESTS ─────────────────────────────────────────────────────────

class TestEvalResult:
    """Tests for the EvalResult dataclass and its to_dict() serialisation."""

    @pytest.fixture
    def sample_eval_result(self, category_labels, priority_labels):
        """A fully-populated EvalResult for testing serialisation."""
        cat_m = compute_classification_metrics(
            ["hardware", "software", "network"],
            ["hardware", "software", "hardware"],
            category_labels,
        )
        pri_m = compute_classification_metrics(
            ["P1", "P2", "P3"],
            ["P1", "P2", "P3"],
            priority_labels,
        )
        lat = compute_latency_stats([10.0, 12.0, 11.0])
        return EvalResult(
            backend="finetuned",
            n_total=3,
            n_success=3,
            n_failed=0,
            category=cat_m,
            priority=pri_m,
            latency=lat,
            total_cost_usd=None,
            avg_cost_usd=None,
        )

    def test_to_dict_returns_dict(self, sample_eval_result):
        """to_dict() must return a plain dict."""
        d = sample_eval_result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_required_keys(self, sample_eval_result):
        """to_dict() output must contain all top-level EvalResult fields."""
        d = sample_eval_result.to_dict()
        required = {"backend", "n_total", "n_success", "n_failed",
                    "category", "priority", "latency",
                    "total_cost_usd", "avg_cost_usd"}
        assert required.issubset(d.keys()), (
            f"Missing keys: {required - d.keys()}"
        )

    def test_to_dict_nested_category_has_accuracy(self, sample_eval_result):
        """Nested ClassificationMetrics must serialise with an 'accuracy' key."""
        d = sample_eval_result.to_dict()
        assert "accuracy" in d["category"], "category dict missing 'accuracy'"
        assert "accuracy" in d["priority"], "priority dict missing 'accuracy'"

    def test_to_dict_is_json_serialisable(self, sample_eval_result):
        """The dict produced by to_dict() must round-trip through json.dumps."""
        import json
        d = sample_eval_result.to_dict()
        serialised = json.dumps(d)          # must not raise
        restored = json.loads(serialised)
        assert restored["backend"] == "finetuned"

    def test_baseline_cost_fields_are_none_for_finetuned(self, sample_eval_result):
        """
        For the finetuned backend, total_cost_usd and avg_cost_usd must be None.
        The finetuned model costs $0 — we should never populate these fields for it.
        """
        d = sample_eval_result.to_dict()
        assert d["total_cost_usd"] is None
        assert d["avg_cost_usd"]   is None
