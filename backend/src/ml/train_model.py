"""Model training, evaluation, and MLflow logging."""

import importlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import mlflow
import pandas as pd
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.config import read_config, resolve_project_path, target_column
from src.ml.io import read_data, write_data
from src.ml.pipeline import Pipeline
from src.ml.plots import (
    _plot_confusion_matrix,
    _plot_pr_curve,
    _plot_predicted_vs_actual,
    _plot_residuals,
    _plot_roc_curve,
)
from src.ml.task import TaskType, class_labels, compute_metrics, detect_task
from src.ml.tracking import (
    active_or_new_run,
    build_signature,
    log_model,
    setup_mlflow,
)
from src.schemas import normalise_column_name

logger = logging.getLogger(__name__)


class _Estimator(Protocol):
    """The scikit-learn estimator surface this pipeline relies on.

    `BaseEstimator` itself declares neither `fit` nor `predict` — those come
    from mixins (`ClassifierMixin`, `RegressorMixin`, ...) that vary by the
    concrete class the model registry loads at runtime. This Protocol names
    the subset every registered model is expected to implement.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> object: ...
    def predict(self, X: pd.DataFrame) -> object: ...


class TrainModelPipeline(Pipeline):
    """End-to-end training pipeline."""

    def __init__(self, run_name: str | None = None) -> None:
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

        # No default model name. A hardcoded one drifts out of step with
        # model_registry the moment a key is renamed, and then reports a model
        # the user never chose.
        configured_name = self.config.get("model_name")
        if not configured_name:
            raise ValueError(
                "model_name is not set in cfg/config.yaml. Choose one of the "
                f"keys under model_registry: "
                f"{sorted(self.config.get('model_registry', {}))}."
            )
        self.model_name = str(configured_name).lower()
        self.model_params = self._select_model_params()
        self.run_name = run_name or "Default_Run_Name"
        self.model: _Estimator | None = None
        # Resolved from the target in _validate_training_data. Defaults to
        # binary so the attribute is always a valid TaskType.
        self.task: TaskType = TaskType.BINARY
        self.class_labels: list[Any] = []
        # Import path from model_registry; set by _build_model and used to pick
        # the matching MLflow flavor when logging.
        self.class_path: str = ""

    def _select_model_params(self) -> dict[str, Any]:
        """Return the hyperparameters for the selected model.

        ``model_params`` accepts two shapes. Nested by model name lets settings
        for several models sit side by side, so switching is a one-word change
        to ``model_name``; a flat mapping applies to whichever model is chosen.

        Returns:
            Constructor arguments for the selected estimator.
        """
        params = self.config.get("model_params") or {}
        if not isinstance(params, dict):
            return {}

        nested = params.get(self.model_name)
        if isinstance(nested, dict):
            return nested

        # A flat mapping has no dict values; one that does is nested for other
        # models, so this model simply has no parameters of its own.
        if any(isinstance(value, dict) for value in params.values()):
            return {}
        return params

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

        # Stratifying needs classes to balance. On a continuous target almost
        # every value is unique, so scikit-learn would fail with "the least
        # populated class has only 1 member" — ignore the setting instead.
        stratify_requested = bool(self.config.get("stratify", False))
        if stratify_requested and not self.task.is_classification:
            logger.info("Ignoring stratify: it does not apply to %s.", self.task.value)
        stratify = y if stratify_requested and self.task.is_classification else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        self.train(X_train, y_train)
        train_metrics = self.evaluate(X_train, y_train)
        test_metrics = self.evaluate(X_test, y_test)

        with active_or_new_run(self.run_name):
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("task", self.task.value)
            if self.model_params:
                mlflow.log_params(self.model_params)

            # Whatever the task produced gets logged; adding a metric in
            # task.compute_metrics makes it appear here and in the dashboard
            # with no further change.
            for split, metrics in (("train", train_metrics), ("test", test_metrics)):
                for name, value in metrics.items():
                    mlflow.log_metric(f"{split}_{name}", value)
                    logger.info("MODEL DRIFT: %s %s = %.4f", split, name, value)

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

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
        """Compute the metrics appropriate to this task.

        Returns a mapping rather than a fixed tuple so the metric set can differ
        by task — accuracy and f1 for classification, RMSE and R² for
        regression — without any caller changing shape. Everything returned is
        logged to MLflow and rendered by the dashboard automatically.

        Args:
            features: Feature matrix for evaluation.
            target: True targets for evaluation.

        Returns:
            Metric name to value.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        logger.info("Evaluating the model.")
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        return compute_metrics(
            self.task, target, self.model.predict(features), self.class_labels
        )

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

        # Everything downstream — metrics, plots, schema checks — follows from
        # the task, so resolve it once here.
        self.task = detect_task(target, self.config.get("task"))
        self.class_labels = (
            class_labels(target, self.config.get("target_values"))
            if self.task.is_classification
            else []
        )

        return features, target

    def _build_model(self) -> _Estimator:
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
        model_cls: type[_Estimator] = getattr(
            importlib.import_module(module_name), class_name
        )
        model = model_cls(**self.model_params)
        self._check_model_suits_task(model)
        return model

    def _check_model_suits_task(self, model: _Estimator) -> None:
        """Fail early when the estimator family does not match the target.

        A regressor fitted on class labels trains without complaint and only
        fails later, when scikit-learn reports "a mix of binary and continuous
        targets" from inside the metrics — a message that never mentions the
        actual mistake. Checking here names the model, the task, and both ways
        to fix it.

        Raises:
            ValueError: If a classifier is paired with a regression target or a
                regressor with a classification target.
        """
        wants_classifier = self.task.is_classification
        if wants_classifier == is_classifier(model):
            return
        if not wants_classifier and is_regressor(model):
            return

        expected = "classifier" if wants_classifier else "regressor"
        actual = "classifier" if is_classifier(model) else "regressor"
        raise ValueError(
            f"Model {self.model_name!r} ({self.class_path}) is a {actual}, but "
            f"the target was read as {self.task.value}, which needs a "
            f"{expected}. Either choose a {expected} from model_registry in "
            "cfg/config.yaml, or set `task:` there if the target was read "
            "wrongly."
        )

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

        if self.task is TaskType.REGRESSION:
            return [
                _plot_predicted_vs_actual(target, y_pred, plots_dir),
                _plot_residuals(target, y_pred, plots_dir),
            ]

        artifacts = [
            _plot_confusion_matrix(target, y_pred, self.class_labels, plots_dir),
            self._write_classification_report(target, y_pred, reports_dir),
        ]

        # ROC and precision-recall curves are defined for two classes. Drawing
        # them for a multiclass target would silently produce a one-vs-rest
        # curve against an arbitrary class, which looks plausible and is wrong,
        # so they are restricted rather than adapted.
        if self.task is not TaskType.BINARY:
            return artifacts

        scores = self._positive_class_scores(features)
        if scores is None:
            logger.warning(
                "Model exposes neither predict_proba nor decision_function; "
                "skipping the ROC and precision-recall curves."
            )
        else:
            positive_label = self.class_labels[-1]
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

    def _positive_class_scores(self, features: pd.DataFrame) -> pd.Series | None:
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
            predict_proba = cast(
                Callable[[pd.DataFrame], Any], self.model.predict_proba
            )
            proba = predict_proba(features)
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
            decision_function = cast(
                Callable[[pd.DataFrame], Any], self.model.decision_function
            )
            raw_scores = decision_function(features)
            if hasattr(raw_scores, "ndim") and raw_scores.ndim > 1:
                return pd.Series(raw_scores[:, 1])
            return pd.Series(raw_scores)

        return None
