"""Model training, evaluation, and MLflow logging."""

import importlib
import logging
import os
from pathlib import Path
from typing import Any, Optional, cast

import mlflow
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.config import read_config, resolve_project_path, target_column
from src.ml.io import read_data, write_data
from src.ml.pipeline import Pipeline
from src.ml.plots import _plot_confusion_matrix, _plot_pr_curve, _plot_roc_curve
from src.ml.tracking import (
    active_or_new_run,
    build_signature,
    log_model,
    setup_mlflow,
)
from src.schemas import normalise_column_name

logger = logging.getLogger(__name__)


class TrainModelPipeline(Pipeline):
    """End-to-end training pipeline."""

    def __init__(self, run_name: Optional[str] = None) -> None:
        """Initialise configuration and placeholders.

        Args:
            run_name: Optional MLflow run name. If not provided, a default is used.

        Raises:
            ValueError: If the configuration file cannot be loaded.
        """
        logger.info("Initializing the TrainModelPipeline.")
        self.config = read_config()
        if not self.config:
            raise ValueError("Configuration file is empty or not found.")

        self.model_name = str(self.config.get("model_name", "xgboost")).lower()
        raw_params = self.config.get("model_params", {}) or {}
        if (
            isinstance(raw_params, dict)
            and self.model_name in raw_params
            and isinstance(raw_params[self.model_name], dict)
        ):
            self.model_params = raw_params[self.model_name]
        else:
            self.model_params = raw_params
        self.run_name = run_name or "Default_Run_Name"
        self.model: Optional[BaseEstimator] = None
        self.class_labels: list[object] = []
        # Import path from model_registry; set by _build_model and used to pick
        # the matching MLflow flavor when logging.
        self.class_path: str = ""

    def run(self) -> None:
        """Execute training and log metrics, model, and plots to MLflow."""
        logger.info("Starting the training pipeline.")

        setup_mlflow()

        input_file = self.config["data"]["input_file"]
        target_col = normalise_column_name(target_column(self.config))

        df = read_data(
            file_name=input_file, suffix="prepared", schema_obj="prepared_data"
        )
        X, y = self._validate_training_data(df, target_col)

        # Fallbacks match the values shipped in cfg/config.yaml, so deleting a
        # key does not silently change behaviour.
        test_size = float(self.config.get("test_size", 0.2))
        random_state = int(self.config.get("random_state", 42))
        stratify = y if bool(self.config.get("stratify", False)) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        self.train(X_train, y_train)
        train_accuracy, train_precision, train_recall, train_f1_score = self.evaluate(
            X_train, y_train
        )
        test_accuracy, test_precision, test_recall, test_f1_score = self.evaluate(
            X_test, y_test
        )

        with active_or_new_run(self.run_name):
            mlflow.log_param("model_name", self.model_name)
            if self.model_params:
                mlflow.log_params(self.model_params)

            # log metrics
            metrics = [
                ("accuracy", train_accuracy, test_accuracy),
                ("precision", train_precision, test_precision),
                ("recall", train_recall, test_recall),
                ("f1_score", train_f1_score, test_f1_score),
            ]
            for name, train_val, test_val in metrics:
                mlflow.log_metric(f"train_{name}", train_val)
                logger.info(f"MODEL DRIFT: Train {name.capitalize()} = {train_val:.4f}")
                mlflow.log_metric(f"test_{name}", test_val)
                logger.info(f"MODEL DRIFT: Test {name.capitalize()} = {test_val:.4f}")

            # Produce every artifact, then log them in one pass — the same
            # shape EDAPipeline uses.
            for path in self._build_artifacts(X_test, y_test):
                mlflow.log_artifact(str(path), artifact_path=path.parent.name)
                logger.info("Saved %s and logged it to MLflow.", path)

            if self.model is not None:
                # A signature is logged rather than left optional: it is what
                # lets consumers discover the feature names and types a
                # prediction request must supply.
                signature = build_signature(X_train, self.model.predict(X_train))
                model_uri = log_model(
                    self.model,
                    class_path=self.class_path,
                    config=self.config,
                    signature=signature,
                    input_example=X_train.head(5),
                )
                logger.info("Model logged to %s", model_uri)

            if self.model is not None:
                df_out = df.copy()
                df_out["PREDICTION"] = self.model.predict(X)
                write_data(
                    df_out,
                    file_name=input_file,
                    suffix="trained",
                    schema_obj="output_data",
                )

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Fit the configured classifier.

        Args:
            features: Feature matrix (no target column).
            target: Target series aligned with `features`.
        """
        logger.info("Training the model.")
        self.model = self._build_model()
        self.model.fit(features, target)

    def evaluate(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[float, float, float, float]:
        """Compute accuracy, precision, recall, and f1-score on the provided data.

        Args:
            features: Feature matrix for evaluation.
            target: True labels for evaluation.

        Returns:
            Accuracy as a float in [0, 1].

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        logger.info("Evaluating the model.")
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        y_pred = self.model.predict(features)
        positive_label = self.class_labels[-1]
        accuracy = float(accuracy_score(target, y_pred))
        precision = float(
            precision_score(target, y_pred, pos_label=positive_label, zero_division=0)
        )
        recall = float(
            recall_score(target, y_pred, pos_label=positive_label, zero_division=0)
        )
        f1 = float(f1_score(target, y_pred, pos_label=positive_label, zero_division=0))

        return accuracy, precision, recall, f1

    def _validate_training_data(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Validate template assumptions before sklearn raises opaque errors."""
        if target_col not in df.columns:
            raise KeyError(
                f"Configured target_column {target_col!r} was not found in prepared "
                f"data. Available columns: {list(df.columns)}. Column names are "
                "normalised to uppercase with underscores when CSVs are read."
            )

        features = df.drop(columns=[target_col])
        target = df[target_col]
        if features.empty:
            raise ValueError(
                "Training requires at least one feature column besides the target."
            )

        non_numeric_cols = features.select_dtypes(exclude="number").columns.tolist()
        if non_numeric_cols:
            raise ValueError(
                "Default training expects all feature columns to be numeric. "
                f"Encode, drop, or transform these columns in PrepareDataPipeline: "
                f"{non_numeric_cols}."
            )

        null_counts = features.isna().sum()
        null_feature_cols = null_counts[null_counts > 0].index.tolist()
        if null_feature_cols:
            raise ValueError(
                "Default training does not impute missing feature values. "
                "Handle nulls in PrepareDataPipeline before training. Columns with "
                f"nulls: {null_feature_cols}."
            )

        if target.isna().any():
            raise ValueError(
                "Default training requires a non-null target column. Drop or fill "
                "target nulls in PrepareDataPipeline before training."
            )

        observed_labels = target.drop_duplicates().tolist()
        configured_labels = list(self.config.get("target_values") or [])
        class_labels = configured_labels or observed_labels
        if len(observed_labels) != 2:
            raise ValueError(
                "Default metrics and plots are configured for binary "
                f"classification, but target_column {target_col!r} has "
                f"{len(observed_labels)} classes: {observed_labels}. For regression or "
                "multiclass projects, replace the metrics and schema assumptions in "
                "TrainModelPipeline."
            )
        if set(class_labels) != set(observed_labels):
            raise ValueError(
                f"Configured target_values {class_labels} do not match observed "
                f"target values {observed_labels}."
            )
        self.class_labels = class_labels

        return features, target

    def _build_model(self) -> BaseEstimator:
        """Construct the model based on configuration.

        The model registry is read from ``cfg/config.yaml`` under the
        ``model_registry`` key. Each entry maps a name to a fully qualified
        class path, e.g. ``"sklearn.ensemble.RandomForestClassifier"``.

        Returns:
            An untrained sklearn-compatible estimator.

        Raises:
            ValueError: If model_name is absent from the registry or the
                class path cannot be imported.
        """
        registry: dict[str, str] = self.config.get("model_registry", {})
        class_path = registry.get(self.model_name)
        if class_path is None:
            raise ValueError(
                f"Unsupported model_name '{self.model_name}'. "
                f"Available: {list(registry.keys())}"
            )
        self.class_path = class_path
        module_name, class_name = class_path.rsplit(".", 1)
        model_cls: type[BaseEstimator] = getattr(
            importlib.import_module(module_name), class_name
        )
        return model_cls(**self.model_params)

    # Training artifacts

    def _build_artifacts(self, features: pd.DataFrame, target: pd.Series) -> list[Path]:
        """Produce the training artifacts and return their paths.

        Drawing lives in :mod:`src.ml.plots`; this method decides what to draw
        and where to put it. The caller logs the returned paths, so the model
        guard and the MLflow calls each happen once rather than once per
        artifact.

        Args:
            features: Evaluation feature matrix.
            target: True labels for evaluation.

        Returns:
            Paths to every artifact produced, in the order created.
        """
        if self.model is None:
            logger.warning("Model not available; skipping artifacts.")
            return []

        plots_dir = resolve_project_path(os.getenv("LOCAL_PLOTS_PATH", "outputs/plots"))
        reports_dir = resolve_project_path(
            os.getenv("LOCAL_REPORTS_PATH", "outputs/reports")
        )
        plots_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        y_pred = self.model.predict(features)
        positive_label = self.class_labels[-1]

        artifacts = [
            _plot_confusion_matrix(target, y_pred, self.class_labels, plots_dir),
            self._write_classification_report(target, y_pred, reports_dir),
        ]

        scores = self._positive_class_scores(features)
        if scores is None:
            logger.warning(
                "Model exposes neither predict_proba nor decision_function; "
                "skipping the ROC and precision-recall curves."
            )
        else:
            artifacts.append(_plot_roc_curve(target, scores, positive_label, plots_dir))
            artifacts.append(_plot_pr_curve(target, scores, positive_label, plots_dir))

        return artifacts

    @staticmethod
    def _write_classification_report(
        target: pd.Series, y_pred: Any, output_dir: Path
    ) -> Path:
        """Write a text classification report and return its path."""
        report = classification_report(target, y_pred, zero_division=0)
        path = output_dir / "classification_report.txt"
        path.write_text(report, encoding="utf-8")
        return path

    def _positive_class_scores(self, features: pd.DataFrame) -> Optional[pd.Series]:
        """Return positive-class scores, or None if the model exposes none.

        Prefers ``predict_proba`` and falls back to ``decision_function``. Both
        the ROC and precision-recall curves need these, so the fallback lives
        here rather than being duplicated at each call site.

        Raises:
            RuntimeError: If the model has not been trained yet.
            ValueError: If the configured positive class is absent from the
                model's own classes.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained.")

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)
            model_classes = cast(
                list[object], list(getattr(self.model, "classes_", []))
            )
            positive_label = self.class_labels[-1]
            if positive_label not in model_classes:
                raise ValueError(
                    f"Positive class {positive_label!r} was not found in model "
                    f"classes {model_classes}."
                )
            return pd.Series(proba[:, model_classes.index(positive_label)])

        if hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(features)
            if hasattr(raw_scores, "ndim") and raw_scores.ndim > 1:
                return pd.Series(raw_scores[:, 1])
            return pd.Series(raw_scores)

        return None
