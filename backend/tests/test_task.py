"""Tests for task detection and per-task metrics.

Everything downstream — metrics, plots, schema checks — follows from the task,
so a wrong answer here is a wrong answer everywhere. Inference is a heuristic,
which makes its edge cases worth pinning down explicitly.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.task import (
    TaskDetectionError,
    TaskType,
    class_labels,
    compute_metrics,
    detect_task,
)


class TestInference:
    """What `task: auto` concludes from a target column."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([0, 1] * 10, TaskType.BINARY),
            ([True, False] * 10, TaskType.BINARY),
            (["yes", "no"] * 10, TaskType.BINARY),
            ([0, 1, 2] * 10, TaskType.MULTICLASS),
            (["red", "green", "blue"] * 7, TaskType.MULTICLASS),
            (list(np.linspace(0.5, 99.5, 40)), TaskType.REGRESSION),
            (list(range(50)), TaskType.REGRESSION),
        ],
    )
    def test_common_targets(self, values, expected):
        """The everyday cases resolve without configuration."""
        assert detect_task(pd.Series(values)) is expected

    def test_whole_numbered_floats_are_labels_not_measurements(self):
        """A float column of 0.0/1.0 is a label set stored as floats."""
        assert detect_task(pd.Series([0.0, 1.0] * 10)) is TaskType.BINARY

    def test_fractional_floats_are_measurements_even_when_few(self):
        """Three distinct prices are still prices, not three classes."""
        assert detect_task(pd.Series([1.5, 2.5, 3.5] * 10)) is TaskType.REGRESSION

    def test_many_integer_classes_become_regression(self):
        """Beyond the cardinality ceiling, integers read as counts."""
        assert detect_task(pd.Series(range(40))) is TaskType.REGRESSION

    def test_a_constant_target_is_rejected(self):
        """One distinct value cannot be learned from, whatever the task."""
        with pytest.raises(TaskDetectionError, match="at least 2"):
            detect_task(pd.Series([1] * 10))


class TestConfiguredOverride:
    """`task:` in config.yaml wins over inference."""

    def test_regression_forced_over_integer_labels(self):
        """The escape hatch for a genuinely continuous 0/1/2 target."""
        assert (
            detect_task(pd.Series([0, 1, 2] * 10), "regression") is TaskType.REGRESSION
        )

    def test_classification_forced_over_many_values(self):
        """And the reverse: many integer labels really are classes."""
        assert (
            detect_task(pd.Series(range(40)), "classification") is TaskType.MULTICLASS
        )

    @pytest.mark.parametrize("setting", ["auto", "AUTO", None, ""])
    def test_auto_variants_all_infer(self, setting):
        """Case and blank spellings of "auto" behave the same."""
        assert detect_task(pd.Series([0, 1] * 10), setting) is TaskType.BINARY

    def test_binary_forced_on_multiclass_data_is_an_error(self):
        """Contradicting the data fails loudly rather than silently coercing."""
        with pytest.raises(TaskDetectionError, match="not 2"):
            detect_task(pd.Series([0, 1, 2] * 10), "binary_classification")

    def test_unknown_task_lists_the_valid_values(self):
        """A typo names the alternatives rather than just rejecting."""
        with pytest.raises(TaskDetectionError, match="regression"):
            detect_task(pd.Series([0, 1] * 10), "clasification")


class TestClassLabels:
    """Resolving the ordered label set for classification."""

    def test_labels_are_read_from_the_data_when_unconfigured(self):
        """Unset target_values means the data decides the label order."""
        assert class_labels(pd.Series([2, 0, 1, 0])) == [0, 1, 2]

    def test_configured_labels_are_honoured(self):
        """Configured order wins, which fixes the positive class for metrics."""
        assert class_labels(pd.Series([0, 1]), [1, 0]) == [1, 0]

    def test_mismatched_labels_are_rejected(self):
        """Catches a stale target_values after the data changed."""
        with pytest.raises(TaskDetectionError, match="do not match"):
            class_labels(pd.Series([0, 1, 2]), [0, 1])


class TestMetrics:
    """Each task reports the metrics that mean something for it."""

    def test_binary_reports_positive_class_metrics(self):
        """Binary gets the single-positive-class precision, recall, and f1."""
        y_true = pd.Series([0, 1, 0, 1])
        metrics = compute_metrics(TaskType.BINARY, y_true, [0, 1, 0, 1], [0, 1])
        assert set(metrics) == {"accuracy", "precision", "recall", "f1_score"}
        assert metrics["accuracy"] == 1.0

    def test_multiclass_reports_macro_averages(self):
        """Macro averaging so a large class cannot mask a small one."""
        y_true = pd.Series([0, 1, 2, 0, 1, 2])
        metrics = compute_metrics(TaskType.MULTICLASS, y_true, y_true, [0, 1, 2])
        assert set(metrics) == {
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        }

    def test_regression_reports_error_and_fit(self):
        """Regression gets error magnitudes plus a goodness-of-fit score."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
        metrics = compute_metrics(TaskType.REGRESSION, y_true, [1.0, 2.0, 3.0, 4.0])
        assert set(metrics) == {"rmse", "mae", "r2"}
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_rmse_is_the_root_of_mean_squared_error(self):
        """Guards the unit: an unrooted MSE would silently overstate error."""
        y_true = pd.Series([0.0, 0.0])
        metrics = compute_metrics(TaskType.REGRESSION, y_true, [3.0, 4.0])
        assert metrics["rmse"] == pytest.approx(3.5355, abs=1e-4)
        assert metrics["mae"] == pytest.approx(3.5)

    def test_metrics_are_plain_floats(self):
        """MLflow rejects numpy scalars, so every value must be a float."""
        y_true = pd.Series([0, 1, 0, 1])
        for task, pred, labels in [
            (TaskType.BINARY, [0, 1, 0, 1], [0, 1]),
            (TaskType.REGRESSION, [0.0, 1.0, 0.0, 1.0], None),
        ]:
            for value in compute_metrics(task, y_true, pred, labels).values():
                assert type(value) is float
