"""The training pipeline across all three supported task types.

These are the tests that would have failed before this template supported
anything but binary classification. Each drives the real pipeline — validation,
training, metrics, and artifact production — with only file I/O mocked.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.ml.task import TaskType
from src.ml.train_model import TrainModelPipeline

N_ROWS = 60


def _config(class_path: str, target_values=None, task=None) -> dict:
    return {
        "model_name": "m",
        "model_params": {},
        "model_registry": {"m": class_path},
        "test_size": 0.4,
        "random_state": 0,
        "stratify": False,
        "task": task,
        "target_column": "TARGET",
        "target_values": target_values,
        "data": {"dir": "data/processed", "raw_dir": "data/raw", "input_file": "i.csv"},
    }


def _frame(target: list) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "FEATURE1": np.arange(len(target), dtype=float),
            "FEATURE2": rng.rand(len(target)),
            "TARGET": target,
        }
    )


def _pipeline(df: pd.DataFrame, config: dict) -> TrainModelPipeline:
    with (
        patch("src.ml.train_model.read_config", return_value=config),
        patch("src.ml.train_model.read_data", return_value=df),
    ):
        return TrainModelPipeline()


CASES = {
    "binary": (
        _frame([0, 1] * (N_ROWS // 2)),
        _config("sklearn.ensemble.RandomForestClassifier", [0, 1]),
        TaskType.BINARY,
        {"accuracy", "precision", "recall", "f1_score"},
        [
            "confusion_matrix.png",
            "classification_report.txt",
            "roc_curve.png",
            "pr_curve.png",
        ],
    ),
    "multiclass": (
        _frame([0, 1, 2] * (N_ROWS // 3)),
        _config("sklearn.ensemble.RandomForestClassifier", [0, 1, 2]),
        TaskType.MULTICLASS,
        {"accuracy", "precision_macro", "recall_macro", "f1_macro"},
        ["confusion_matrix.png", "classification_report.txt"],
    ),
    "regression": (
        _frame(list(np.linspace(0.5, 99.5, N_ROWS))),
        _config("sklearn.ensemble.RandomForestRegressor"),
        TaskType.REGRESSION,
        {"rmse", "mae", "r2"},
        ["predicted_vs_actual.png", "residuals.png"],
    ),
}


@pytest.fixture(params=list(CASES), name="case")
def case_fixture(request, tmp_path, monkeypatch):
    """One task type, with artifact output redirected to a temp directory."""
    monkeypatch.setenv("LOCAL_PLOTS_PATH", str(tmp_path / "plots"))
    monkeypatch.setenv("LOCAL_REPORTS_PATH", str(tmp_path / "reports"))
    df, config, task, metric_keys, artifacts = CASES[request.param]
    return df, config, task, metric_keys, artifacts


def test_pipeline_supports_every_task_type(case):
    """Validation, training, metrics, and artifacts all work for each task."""
    df, config, expected_task, expected_metrics, expected_artifacts = case
    pipeline = _pipeline(df, config)

    features, target = pipeline._validate_training_data(df, "TARGET")
    assert pipeline.task is expected_task

    pipeline.train(features, target)

    assert set(pipeline.evaluate(features, target)) == expected_metrics
    produced = [path.name for path in pipeline._build_artifacts(features, target)]
    assert produced == expected_artifacts
    assert all(path for path in produced)


def test_curves_are_omitted_for_multiclass(tmp_path, monkeypatch):
    """Multiclass gets no ROC or precision-recall curve.

    A one-vs-rest curve against an arbitrary class would look plausible and be
    wrong, which is worse than having no curve at all.
    """
    monkeypatch.setenv("LOCAL_PLOTS_PATH", str(tmp_path / "plots"))
    monkeypatch.setenv("LOCAL_REPORTS_PATH", str(tmp_path / "reports"))

    df, config, *_ = CASES["multiclass"]
    pipeline = _pipeline(df, config)
    features, target = pipeline._validate_training_data(df, "TARGET")
    pipeline.train(features, target)

    names = [p.name for p in pipeline._build_artifacts(features, target)]
    assert "roc_curve.png" not in names
    assert "pr_curve.png" not in names


def test_regression_needs_no_target_values(tmp_path, monkeypatch):
    """A continuous target has no fixed value set, and none is required."""
    monkeypatch.setenv("LOCAL_PLOTS_PATH", str(tmp_path / "plots"))
    monkeypatch.setenv("LOCAL_REPORTS_PATH", str(tmp_path / "reports"))

    df, config, *_ = CASES["regression"]
    pipeline = _pipeline(df, config)
    pipeline._validate_training_data(df, "TARGET")

    assert pipeline.task is TaskType.REGRESSION
    assert pipeline.class_labels == []


def test_task_can_be_forced_against_inference():
    """`task: regression` wins over a target that looks categorical."""
    df = _frame([0, 1, 2] * (N_ROWS // 3))
    config = _config("sklearn.ensemble.RandomForestRegressor", task="regression")
    pipeline = _pipeline(df, config)

    pipeline._validate_training_data(df, "TARGET")
    assert pipeline.task is TaskType.REGRESSION
