from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ml.train_model import TrainModelPipeline

_MOCK_CONFIG = {
    "model_name": "xgb_classifier",
    "model_params": {
        "objective": "binary:logistic",
        "n_estimators": 10,
    },
    "model_registry": {
        "xgb_classifier": "xgboost.XGBClassifier",
        "rf_classifier": "sklearn.ensemble.RandomForestClassifier",
    },
    "test_size": 0.5,
    "random_state": 123,
    "stratify": True,
    "target_column": "TARGET",
    "data": {
        "dir": "data/processed",
        "raw_dir": "data/raw",
        "input_file": "breast_cancer.csv",
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
        assert (
            log_kwargs["class_path"] == _MOCK_CONFIG["model_registry"]["xgb_classifier"]
        )
        assert log_kwargs["signature"] is not None
        assert log_kwargs["input_example"] is not None

        written_df = mock_write.call_args[0][0]
        kwargs = mock_write.call_args.kwargs

        assert "PREDICTION" in written_df.columns
        assert kwargs["file_name"] == "breast_cancer.csv"
        assert kwargs["suffix"] == "trained"
        assert kwargs["schema_obj"] == "output_data"


def test_train_method(pipeline, data):
    """Directly test the public train method."""
    X, y = data.drop(columns=["TARGET"]), data["TARGET"]
    pipeline.train(X, y)
    assert pipeline.model is not None


def test_positive_class_scores_prefers_predict_proba(pipeline, data):
    """Probabilities are read for the configured positive class."""
    X, y = data.drop(columns=["TARGET"]), data["TARGET"]
    pipeline.train(X, y)
    pipeline.class_labels = [0, 1]

    scores = pipeline._positive_class_scores(X)
    assert scores is not None
    assert len(scores) == len(X)
    assert scores.between(0, 1).all()


def test_positive_class_scores_falls_back_to_decision_function(pipeline, data):
    """Estimators without predict_proba still yield ranking scores.

    ROC and PR curves need only a ranking, so an SVM-style decision_function is
    an acceptable substitute for probabilities.
    """
    X = data.drop(columns=["TARGET"])

    class _DecisionOnly:
        def decision_function(self, features):
            return pd.Series(range(len(features))).to_numpy()

    pipeline.model = _DecisionOnly()
    pipeline.class_labels = [0, 1]

    scores = pipeline._positive_class_scores(X)
    assert scores is not None
    assert len(scores) == len(X)


def test_positive_class_scores_returns_none_without_either(pipeline, data):
    """A model exposing neither method yields None rather than raising.

    _build_artifacts uses this to skip the ROC and PR curves while still
    producing the confusion matrix and classification report.
    """
    X = data.drop(columns=["TARGET"])

    class _BareModel:
        pass

    pipeline.model = _BareModel()
    pipeline.class_labels = [0, 1]

    assert pipeline._positive_class_scores(X) is None


def test_build_artifacts_skips_curves_without_scores(
    pipeline, data, tmp_path, monkeypatch
):
    """Missing scores drop the curves but keep the other two artifacts."""
    monkeypatch.setenv("LOCAL_PLOTS_PATH", str(tmp_path / "plots"))
    monkeypatch.setenv("LOCAL_REPORTS_PATH", str(tmp_path / "reports"))

    X, y = data.drop(columns=["TARGET"]), data["TARGET"]

    class _NoScores:
        def predict(self, features):
            return [0] * len(features)

    pipeline.model = _NoScores()
    pipeline.class_labels = [0, 1]

    artifacts = pipeline._build_artifacts(X, y)
    names = [p.name for p in artifacts]

    assert names == ["confusion_matrix.png", "classification_report.txt"]
    assert all(p.exists() for p in artifacts)


def test_build_artifacts_returns_nothing_without_a_model(pipeline, data):
    """The single model guard short-circuits artifact production."""
    X, y = data.drop(columns=["TARGET"]), data["TARGET"]
    pipeline.model = None

    assert pipeline._build_artifacts(X, y) == []


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
