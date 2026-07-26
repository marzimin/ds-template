from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ml.train_model import TrainModelPipeline

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
        patch("src.ml.train_model.read_config", return_value=_MOCK_CONFIG),
        patch("src.ml.train_model.read_data", return_value=data),
    ):
        return TrainModelPipeline()


def test_pipeline_run(pipeline, data):
    """Test the pipeline's run method end-to-end (with heavy bits mocked)."""
    start_run_cm = MagicMock()
    start_run_cm.__enter__.return_value = start_run_cm
    start_run_cm.__exit__.return_value = None

    with (
        patch("mlflow.set_tracking_uri"),
        patch("src.ml.train_model.setup_mlflow"),
        patch("mlflow.start_run", return_value=start_run_cm),
        patch("mlflow.log_param"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_params"),
        patch("src.ml.train_model.log_model") as mock_log_model,
        patch("mlflow.log_artifact"),
        patch("src.ml.train_model.read_data", return_value=data),
        patch("src.ml.train_model.write_data") as mock_write,
    ):
        pipeline.run()

        assert pipeline.model is not None
        mock_write.assert_called_once()

        # The model must be logged with the import path that selects its MLflow
        # flavor, plus a signature for consumers to read the input schema from.
        mock_log_model.assert_called_once()
        log_kwargs = mock_log_model.call_args.kwargs
        assert log_kwargs["class_path"] == _MOCK_CONFIG["model_registry"]["xgboost"]
        assert log_kwargs["signature"] is not None
        assert log_kwargs["input_example"] is not None

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


def test_validate_training_data_rejects_non_numeric_features(pipeline):
    """Copied projects get a clear error for unencoded categorical features."""
    df = pd.DataFrame(
        {
            "FEATURE1": [1.0, 2.0, 3.0, 4.0],
            "CATEGORY": ["A", "B", "A", "B"],
            "TARGET": [0, 1, 0, 1],
        }
    )

    with pytest.raises(ValueError, match="expects all feature columns to be numeric"):
        pipeline._validate_training_data(df, "TARGET")
