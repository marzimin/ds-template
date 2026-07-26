from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ml.eda import EDAPipeline

_MOCK_CONFIG = {
    "data": {
        "dir": "data/processed",
        "raw_dir": "data/raw",
        "input_file": "breast_cancer.csv",
    },
    "target_column": "TARGET",
}

# Two features: one numeric, one categorical — drives plot dispatch assertions.
_DUMMY_DF = pd.DataFrame(
    {
        "FEATURE1": [1.0, 2.0, 3.0, 4.0],
        "FEATURE2": ["A", "B", "A", "B"],
        "TARGET": [0, 1, 0, 1],
    }
)

_COL_INFO = {
    "FEATURE1": {"type": "numeric", "cardinality": 4, "null_pct": 0.0},
    "FEATURE2": {"type": "categorical", "cardinality": 2, "null_pct": 0.0},
}

# Three features including a datetime column — for datetime dispatch assertions.
_DUMMY_DF_WITH_DATETIME = pd.DataFrame(
    {
        "FEATURE1": [1.0, 2.0, 3.0, 4.0],
        "FEATURE2": ["A", "B", "A", "B"],
        "DATE_COL": pd.to_datetime(
            ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]
        ),
        "TARGET": [0, 1, 0, 1],
    }
)

_COL_INFO_WITH_DATETIME = {
    "FEATURE1": {"type": "numeric", "cardinality": 4, "null_pct": 0.0},
    "FEATURE2": {"type": "categorical", "cardinality": 2, "null_pct": 0.0},
    "DATE_COL": {"type": "datetime", "cardinality": 4, "null_pct": 0.0},
}


@pytest.fixture(name="start_run_cm")
def mock_start_run_cm():
    """Reusable context-manager mock for mlflow.start_run."""
    cm = MagicMock()
    cm.__enter__.return_value = cm
    cm.__exit__.return_value = None
    return cm


@pytest.fixture(name="plot_patches")
def all_plot_patches(tmp_path):
    """Patch every plotting helper to return a dummy path and yield the mocks."""
    fake = tmp_path / "plot.png"
    fake.touch()
    patches = {
        "_plot_histogram": patch("src.ml.eda._plot_histogram", return_value=fake),
        "_plot_bar_chart": patch("src.ml.eda._plot_bar_chart", return_value=fake),
        "_plot_feature_vs_target": patch(
            "src.ml.eda._plot_feature_vs_target", return_value=fake
        ),
        "_plot_correlation_heatmap": patch(
            "src.ml.eda._plot_correlation_heatmap", return_value=fake
        ),
        "_plot_class_distribution": patch(
            "src.ml.eda._plot_class_distribution", return_value=fake
        ),
        "_plot_datetime_distribution": patch(
            "src.ml.eda._plot_datetime_distribution", return_value=fake
        ),
        "_plot_missing_values": patch(
            "src.ml.eda._plot_missing_values", return_value=fake
        ),
    }
    with (
        patches["_plot_histogram"] as mock_hist,
        patches["_plot_bar_chart"] as mock_bar,
        patches["_plot_feature_vs_target"] as mock_vs,
        patches["_plot_correlation_heatmap"] as mock_corr,
        patches["_plot_class_distribution"] as mock_cls,
        patches["_plot_datetime_distribution"] as mock_datetime,
        patches["_plot_missing_values"] as mock_mv,
    ):
        yield {
            "hist": mock_hist,
            "bar": mock_bar,
            "vs_target": mock_vs,
            "correlation": mock_corr,
            "class_dist": mock_cls,
            "datetime": mock_datetime,
            "missing": mock_mv,
        }


def _common_patches(start_run_cm):
    """Return patches shared across all full-run tests."""
    return [
        patch("src.ml.eda.read_config", return_value=_MOCK_CONFIG),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.mkdir"),
        patch("src.ml.eda.read_data", return_value=_DUMMY_DF),
        patch("src.ml.eda._detect_feature_types", return_value=_COL_INFO),
        patch("src.ml.eda.setup_mlflow"),
        patch("mlflow.active_run", return_value=None),
        patch("mlflow.start_run", return_value=start_run_cm),
        patch("mlflow.log_artifact"),
    ]


def test_run_raises_when_prepared_data_missing():
    """FileNotFoundError with a helpful message when the prepared CSV is absent."""
    with (
        patch("src.ml.eda.read_config", return_value=_MOCK_CONFIG),
        patch("src.ml.eda.setup_mlflow"),
        patch("pathlib.Path.exists", return_value=False),
    ):
        pipeline = EDAPipeline()
        with pytest.raises(FileNotFoundError, match="Run PrepareDataPipeline"):
            pipeline.run()


def test_run_dispatches_correct_plot_per_column(start_run_cm, plot_patches):
    """Histogram is used for numeric columns; bar chart for categorical ones."""
    with (
        patch("src.ml.eda.read_config", return_value=_MOCK_CONFIG),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.mkdir"),
        patch("src.ml.eda.read_data", return_value=_DUMMY_DF),
        patch("src.ml.eda._detect_feature_types", return_value=_COL_INFO),
        patch("src.ml.eda.setup_mlflow"),
        patch("mlflow.active_run", return_value=None),
        patch("mlflow.start_run", return_value=start_run_cm),
        patch("mlflow.log_artifact"),
    ):
        EDAPipeline().run()

    plot_patches["hist"].assert_called_once()
    assert plot_patches["hist"].call_args[0][1] == "FEATURE1"

    plot_patches["bar"].assert_called_once()
    assert plot_patches["bar"].call_args[0][1] == "FEATURE2"


def test_run_logs_all_artifacts_to_mlflow(start_run_cm, plot_patches):
    """Every plot path is logged to MLflow under the 'eda' artifact path.

    With 2 features: 2 distribution + 2 vs-target + 3 summary = 7 artifacts.
    """
    with (
        patch("src.ml.eda.read_config", return_value=_MOCK_CONFIG),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.mkdir"),
        patch("src.ml.eda.read_data", return_value=_DUMMY_DF),
        patch("src.ml.eda._detect_feature_types", return_value=_COL_INFO),
        patch("src.ml.eda.setup_mlflow"),
        patch("mlflow.active_run", return_value=None),
        patch("mlflow.start_run", return_value=start_run_cm),
        patch("mlflow.log_artifact") as mock_log,
    ):
        EDAPipeline().run()

    assert mock_log.call_count == 7
    for call in mock_log.call_args_list:
        assert call.kwargs["artifact_path"] == "eda"


def test_run_uses_active_mlflow_run(start_run_cm, plot_patches):
    """When an MLflow run is already active, start_run is not called."""
    with (
        patch("src.ml.eda.read_config", return_value=_MOCK_CONFIG),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.mkdir"),
        patch("src.ml.eda.read_data", return_value=_DUMMY_DF),
        patch("src.ml.eda._detect_feature_types", return_value=_COL_INFO),
        patch("src.ml.eda.setup_mlflow"),
        patch("mlflow.active_run", return_value=MagicMock()),
        patch("mlflow.start_run") as mock_start,
        patch("mlflow.log_artifact"),
    ):
        EDAPipeline().run()

    mock_start.assert_not_called()


def test_run_dispatches_datetime_plot(start_run_cm, plot_patches):
    """_plot_datetime_distribution is called for datetime-typed columns."""
    with (
        patch("src.ml.eda.read_config", return_value=_MOCK_CONFIG),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.mkdir"),
        patch("src.ml.eda.read_data", return_value=_DUMMY_DF_WITH_DATETIME),
        patch(
            "src.ml.eda._detect_feature_types",
            return_value=_COL_INFO_WITH_DATETIME,
        ),
        patch("src.ml.eda.setup_mlflow"),
        patch("mlflow.active_run", return_value=None),
        patch("mlflow.start_run", return_value=start_run_cm),
        patch("mlflow.log_artifact"),
    ):
        EDAPipeline().run()

    plot_patches["datetime"].assert_called_once()
    assert plot_patches["datetime"].call_args[0][1] == "DATE_COL"

    # Numeric and categorical columns are still dispatched correctly alongside datetime
    plot_patches["hist"].assert_called_once()
    plot_patches["bar"].assert_called_once()
