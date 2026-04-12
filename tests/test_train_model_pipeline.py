from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipelines.train_model import TrainModelPipeline


@pytest.fixture(name="data")
def dummy_data_fixture():
    # Small, balanced sample for train/test split
    return pd.DataFrame(
        {"FEATURE1": [1, 2, 3, 4], "FEATURE2": [5, 6, 7, 8], "TARGET": [0, 1, 0, 1]}
    )


@pytest.fixture(name="pipeline")
def mock_pipeline_fixture(data):
    # Patch where TrainModelPipeline looks up these names (module scope)
    with (
        patch("src.pipelines.train_model.read_config") as mock_config,
        patch("src.pipelines.train_model.read_data") as mock_read,
    ):
        mock_config.return_value = {
            "model_name": "xgboost",
            "model_params": {
                "objective": "binary:logistic",
                "n_estimators": 10,
            },
            "test_size": 0.5,
            "random_state": 123,
            "stratify": True,
        }
        mock_read.return_value = data

        pipeline_instance = TrainModelPipeline()

    return pipeline_instance


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
        _, kwargs = mock_write.call_args
        written_df = mock_write.call_args[0][0]
        table_name = kwargs["table_name"]
        schema_obj = kwargs["schema_obj"]
        assert "PREDICTION" in written_df.columns
        assert table_name == "TEST_DS_TABLE_IRIS_OUTPUT"
        assert schema_obj == "output_data"


def test_train_method(pipeline, data):
    """Directly test the public train method."""
    X, y = data.drop(columns=["TARGET"]), data["TARGET"]
    pipeline.train(X, y)
    assert pipeline.model is not None
