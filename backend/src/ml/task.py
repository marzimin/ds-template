"""What kind of problem is this, and which metrics does it deserve?

The template supports binary classification, multiclass classification, and
regression. Nearly everything downstream — which metrics to compute, which plots
make sense, whether the target should be checked against a fixed set of values —
follows from that one decision, so it is made once here and passed around.

The task is inferred from the target column by default and can be forced with
``task:`` in ``cfg/config.yaml``. Inference is a heuristic, so it always logs
what it concluded: a silent wrong guess is far worse than a noisy right one.
"""

import logging
from enum import Enum
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

logger = logging.getLogger(__name__)

#: A numeric target with at most this many distinct values is read as class
#: labels rather than measurements. Raise it if you have many classes; set
#: ``task: regression`` if your measurements are genuinely this coarse.
MAX_CLASSES_FOR_INFERENCE = 20


class TaskType(str, Enum):
    """The kind of prediction problem, which decides metrics and plots."""

    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        """True for both classification variants."""
        return self in (TaskType.BINARY, TaskType.MULTICLASS)


class TaskDetectionError(ValueError):
    """Raised when the configured task cannot apply to the given target."""


def detect_task(target: pd.Series, configured: Any = None) -> TaskType:
    """Decide which kind of problem this target represents.

    Args:
        target: The target column, nulls already removed.
        configured: ``task`` from ``cfg/config.yaml``. ``None`` or ``"auto"``
            infers; ``"classification"`` and ``"regression"`` force the family;
            an exact :class:`TaskType` value forces that specific task.

    Returns:
        The resolved task type.

    Raises:
        TaskDetectionError: If the configured task contradicts the data, or the
            value is not recognised.
    """
    n_classes = int(target.nunique(dropna=True))
    if n_classes < 2:
        raise TaskDetectionError(
            f"The target has {n_classes} distinct value(s); at least 2 are "
            "needed to learn anything."
        )

    setting = str(configured).strip().lower() if configured is not None else "auto"

    if setting in ("", "auto", "none"):
        task = _infer_task(target, n_classes)
        logger.info(
            "Task inferred as %s from the target (%d distinct values, dtype %s). "
            "Set `task:` in cfg/config.yaml to override.",
            task.value,
            n_classes,
            target.dtype,
        )
        return task

    if setting == "regression":
        return TaskType.REGRESSION

    if setting == "classification":
        return TaskType.BINARY if n_classes == 2 else TaskType.MULTICLASS

    for candidate in TaskType:
        if setting == candidate.value:
            if candidate is TaskType.BINARY and n_classes != 2:
                raise TaskDetectionError(
                    f"task is set to {candidate.value!r} but the target has "
                    f"{n_classes} distinct values, not 2."
                )
            return candidate

    raise TaskDetectionError(
        f"Unrecognised task {configured!r}. Use one of: auto, classification, "
        "regression, " + ", ".join(t.value for t in TaskType) + "."
    )


def _infer_task(target: pd.Series, n_classes: int) -> TaskType:
    """Infer the task from the target's dtype and cardinality."""
    if pd.api.types.is_bool_dtype(target):
        return TaskType.BINARY

    # Non-numeric targets are labels by definition — you cannot regress onto
    # strings or categories.
    if not pd.api.types.is_numeric_dtype(target):
        return TaskType.BINARY if n_classes == 2 else TaskType.MULTICLASS

    # Floats with fractional parts are measurements, however few there are.
    if pd.api.types.is_float_dtype(target) and not _is_whole_numbered(target):
        return TaskType.REGRESSION

    if n_classes == 2:
        return TaskType.BINARY
    if n_classes <= MAX_CLASSES_FOR_INFERENCE:
        return TaskType.MULTICLASS
    return TaskType.REGRESSION


def _is_whole_numbered(target: pd.Series) -> bool:
    """True when every value is a whole number, e.g. 1.0, 2.0, 3.0."""
    return bool((target.dropna() % 1 == 0).all())


def class_labels(target: pd.Series, configured: Any = None) -> list[Any]:
    """Return the ordered class labels for a classification target.

    Args:
        target: The target column.
        configured: ``target_values`` from config, used when set.

    Returns:
        Sorted labels, or the configured list.

    Raises:
        TaskDetectionError: If configured labels do not match the data.
    """
    observed = sorted(target.dropna().unique().tolist())
    if not configured:
        return observed

    configured_labels = list(configured)
    if set(configured_labels) != set(observed):
        raise TaskDetectionError(
            f"Configured target_values {configured_labels} do not match the "
            f"values present in the data {observed}."
        )
    return configured_labels


def compute_metrics(
    task: TaskType,
    y_true: pd.Series,
    y_pred: Any,
    labels: list[Any] | None = None,
) -> dict[str, float]:
    """Compute the metrics appropriate to this task.

    Returning a mapping rather than a fixed tuple is what lets the metric set
    differ per task without every caller changing shape. Whatever is returned is
    logged to MLflow and rendered by the dashboard, so adding a metric here makes
    it appear everywhere with no further work.

    Args:
        task: The resolved task type.
        y_true: True targets.
        y_pred: Predicted targets, aligned with ``y_true``.
        labels: Class labels, for classification.

    Returns:
        Metric name to value.
    """
    if task is TaskType.REGRESSION:
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "rmse": mse**0.5,
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}

    if task is TaskType.BINARY:
        # The positive class is the last label, matching the convention used by
        # the ROC and precision-recall curves.
        positive = (labels or sorted(pd.Series(y_true).unique().tolist()))[-1]
        shared = {"pos_label": positive, "zero_division": 0}
        metrics["precision"] = float(precision_score(y_true, y_pred, **shared))
        metrics["recall"] = float(recall_score(y_true, y_pred, **shared))
        metrics["f1_score"] = float(f1_score(y_true, y_pred, **shared))
        return metrics

    # Multiclass: macro averaging weights every class equally, so a large class
    # cannot hide poor performance on a small one.
    shared_macro: dict[str, Any] = {
        "average": "macro",
        "zero_division": 0,
        "labels": labels,
    }
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, **shared_macro))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, **shared_macro))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, **shared_macro))
    return metrics
