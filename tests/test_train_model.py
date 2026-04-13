from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipelines.train_model import TrainModelPipeline

_MOCK_CONFIG = {
    "model_name": "xgboost",
    "model_params": {
        "objective": "binary:logistic",
        "n_estimators": 10,
    },
    "model_registry": {
        "xgboost": "xgboost.XGBClassifier",
        "xgb": "xgboost.XGBClassifier",
        "random_forest": "sklearn.ensemble.RandomForestClassifier",
        "rf": "sklearn.ensemble.RandomForestClassifier",
    },
    "test_size": 0.5,
    "random_state": 123,
    "stratify": True,
    "target_column": "TARGET",
    "data": {
        "dir": "data/processed",
        "raw_dir": "data/raw",
        "input_file": "input_data.csv",
    },
}


@pytest.fixture(name="data")
def dummy_data_fixture():
    """Binary sample large enough for both classes to appear in train and test splits."""
    return pd.DataFrame(
        {
            "FEATURE1": list(range(20)),
            "FEATURE2": list(range(20, 40)),
            "TARGET": [0, 1] * 10,
        }
    )


@pytest.fixture(name="pipeline")
def mock_pipeline_fixture(data):
    """TrainModelPipeline with config and data I/O mocked out."""
    with (
        patch("src.pipelines.train_model.read_config", return_value=_MOCK_CONFIG),
        patch("src.pipelines.train_model.read_data", return_value=data),
    ):
        return TrainModelPipeline()


def test_pipeline_run(pipeline, data):
    """Test the pipeline's run method end-to-end (with heavy bits mocked)."""
    start_run_cm = MagicMock()
    start_run_cm.__enter__.return_value = start_run_cm
    start_run_cm.__exit__.return_value = None

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.start_run", return_value=start_run_cm),
        patch("mlflow.log_param"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_params"),
        patch("mlflow.sklearn.log_model"),
        patch("mlflow.log_artifact"),
        patch("src.pipelines.train_model.read_data", return_value=data),
        patch("src.pipelines.train_model.write_data") as mock_write,
    ):
        pipeline.run()

        assert pipeline.model is not None
        mock_write.assert_called_once()

        written_df = mock_write.call_args[0][0]
        kwargs = mock_write.call_args.kwargs

        assert "PREDICTION" in written_df.columns
        assert kwargs["file_name"] == "input_data.csv"
        assert kwargs["suffix"] == "trained"
        assert kwargs["schema_obj"] == "output_data"


def test_train_method(pipeline, data):
    """Directly test the public train method."""
    X, y = data.drop(columns=["TARGET"]), data["TARGET"]
    pipeline.train(X, y)
    assert pipeline.model is not None
